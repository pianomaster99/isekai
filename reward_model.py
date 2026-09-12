import random
import importlib.metadata
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from llm_engine import ChatMessage, ConversationSession


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



@dataclass
class RewardResult:
    score: float
    delta: float
    passed: bool
    reason: str


class RandomRewardModel:
    """Placeholder reward model with the same API shape as a trained scorer."""

    def __init__(self, seed: Optional[int] = None):
        self._random = random.Random(seed)

    def score_turn(
        self,
        session: ConversationSession,
        player_message: str,
        npc_reply: str,
        history: List[ChatMessage],
    ) -> RewardResult:
        del player_message, npc_reply, history

        delta = self._random.uniform(-0.1, 0.25)
        score = max(0.0, min(1.0, session.score + delta))
        passed = score >= session.pass_score

        return RewardResult(
            score=score,
            delta=delta,
            passed=passed,
            reason=(
                "random placeholder scorer; prompt retained for future reward model: "
                f"{session.reward_system_prompt[:160]}"
            ),
        )


class TrainedRewardModel:
    """Qwen sequence-regression scorer trained by train_reward_model.py."""

    def __init__(
        self,
        model_path: str = "./qwen3-1.7b",
        adapter_path: Optional[str] = None,
        max_length: int = 1536,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path or self.model_path,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        disable_incompatible_torchao()
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=1,
            problem_type="regression",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if self.adapter_path:
            if PeftModel is None:
                raise ImportError("Install peft to load a trained reward adapter")
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)

        self._model.config.pad_token_id = self._tokenizer.pad_token_id
        self._model.eval()

    def score_turn(
        self,
        session: ConversationSession,
        player_message: str,
        npc_reply: str,
        history: List[ChatMessage],
    ) -> RewardResult:
        self._load()
        prompt = self._format_prompt(session, player_message, npc_reply, history)
        inputs = self._tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            raw_score = self._model(**inputs).logits.squeeze().float().item()

        score = max(0.0, min(1.0, raw_score))
        delta = score - session.score
        return RewardResult(
            score=score,
            delta=delta,
            passed=score >= session.pass_score,
            reason=f"trained reward model score={score:.3f}",
        )

    def _format_prompt(
        self,
        session: ConversationSession,
        player_message: str,
        npc_reply: str,
        history: List[ChatMessage],
    ) -> str:
        history_text = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in history[-8:]
        )
        return (
            "You are scoring a player/NPC turn for an interactive fiction scene.\n"
            f"Objective: {session.objective}\n"
            f"Scoring criteria: {session.reward_system_prompt}\n"
            f"History:\n{history_text}\n"
            f"Player: {player_message}\n"
            f"NPC: {npc_reply}\n"
            "Return a scalar quality score from 0.0 to 1.0."
        )
