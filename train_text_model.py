import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class ChatSftDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.examples: List[Dict[str, torch.Tensor]] = []
        with Path(path).open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = item.get("messages")
                if not isinstance(messages, list):
                    raise ValueError(f"{path}:{line_number} must contain a messages list")

                tokenized = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = getattr(tokenized, "input_ids", tokenized).squeeze(0)
                self.examples.append({"input_ids": input_ids, "labels": input_ids.clone()})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_batch(batch, pad_token_id: int):
    max_length = max(example["input_ids"].shape[0] for example in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for example in batch:
        length = example["input_ids"].shape[0]
        pad_length = max_length - length
        input_ids.append(torch.nn.functional.pad(example["input_ids"], (0, pad_length), value=pad_token_id))
        labels.append(torch.nn.functional.pad(example["labels"], (0, pad_length), value=-100))
        attention_mask.append(torch.nn.functional.pad(torch.ones(length, dtype=torch.long), (0, pad_length), value=0))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Rowan NPC text model with LoRA SFT.")
    parser.add_argument("--model-path", default="./qwen3-1.7b")
    parser.add_argument("--train-file", default="datasets/rowan_ashford_sft.jsonl")
    parser.add_argument("--output-dir", default="./models/rowan-qwen3-1.7b-sft")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ChatSftDataset(args.train_file, tokenizer, args.max_length)
    if len(train_dataset) == 0:
        raise ValueError(f"No examples found in {args.train_file}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        bf16=dtype is torch.bfloat16,
        fp16=torch.cuda.is_available() and dtype is None,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
