import importlib.metadata
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import AutoPeftModelForCausalLM, PeftModel
except ImportError:
    AutoPeftModelForCausalLM = None
    PeftModel = None


ChatMessage = Dict[str, str]


def disable_incompatible_torchao() -> None:
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return

    version_parts = tuple(int(part) for part in version.split(".")[:2] if part.isdigit())
    if version_parts >= (0, 16):
        return

    import peft.import_utils as peft_import_utils

    peft_import_utils.is_torchao_available = lambda: False
    try:
        import peft.tuners.lora.torchao as peft_lora_torchao

        peft_lora_torchao.is_torchao_available = lambda: False
    except Exception:
        pass



def apply_chat_template_no_thinking(tokenizer, messages, **kwargs):
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def strip_thinking_text(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()


@dataclass
class GenerationConfig:
    max_new_tokens: int = 160
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


@dataclass
class ConversationSession:
    session_id: str
    npc_system_prompt: str
    reward_system_prompt: str
    objective: str
    pass_score: float
    score: float = 0.0
    turn_count: int = 0
    history: List[ChatMessage] = field(default_factory=list)


class AliceLLMEngine:
    def __init__(
        self,
        model_path: str = "./qwen3-1.7b",
        adapter_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None
        self._lock = Lock()

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return

        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return

            tokenizer_path = self.adapter_path or self.model_path
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            disable_incompatible_torchao()
            model_kwargs = {"trust_remote_code": True}
            adapter_path = Path(self.adapter_path) if self.adapter_path else None
            if adapter_path and (adapter_path / "adapter_config.json").exists():
                if PeftModel is None:
                    raise ImportError("Install peft to load a text-generation adapter")
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )
                self._model = PeftModel.from_pretrained(base_model, self.adapter_path)
            elif AutoPeftModelForCausalLM is not None and (
                Path(self.model_path) / "adapter_config.json"
            ).exists():
                self._model = AutoPeftModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )
            self._model.config.pad_token_id = self._tokenizer.pad_token_id
            self._model.eval()

    def generate(
        self,
        system_prompt: str,
        history: List[ChatMessage],
        player_message: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        self._load()
        config = config or GenerationConfig()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": player_message})

        tokenized = apply_chat_template_no_thinking(
            self._tokenizer,
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        input_ids = getattr(tokenized, "input_ids", tokenized).to(self._model.device)

        generate_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.temperature > 0,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if config.temperature > 0:
            generate_kwargs["temperature"] = config.temperature

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, **generate_kwargs)

        generated_ids = output_ids[0, input_ids.shape[-1] :]
        text = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return strip_thinking_text(text)
