"""
Full-parameter supervised fine-tuning (F1.4 / F2.3).

Trains **every** parameter of Qwen3-0.6B (no LoRA, no 4-bit quantisation)
on the SFT split so the model learns to emit a strict JSON decision.

Why full fine-tuning: the downstream GRPO stage updates the whole policy
and needs a checkpoint with the same parameter layout, so SFT and RL stay
in one model family instead of stacking adapters.

Intended to run on a CUDA GPU (AutoDL single 24 GB card). Deterministic
seed; all hyperparameters are logged.

Usage (on the training host):
    python3 training/src/full_sft.py \
        --data data/synthetic/sft_full.jsonl \
        --output_dir output/full_sft \
        --epochs 2 --batch_size 2

Outputs:
    <output_dir>/final/          full HF weights + tokenizer
    <output_dir>/run_config.json hyperparameters + seed
    <output_dir>/train_log.json  loss curve

Loss is computed on the assistant (JSON decision) tokens only; the
system + event prompt tokens are masked with -100.

Requirements: see training/requirements.txt (torch, transformers,
datasets, accelerate).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

from format_sft import render_chat  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", ".."))

IGNORE_INDEX = -100


@dataclass
class FullSFTConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    max_seq_length: int = 1024
    epochs: int = 2
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    seed: int = 42
    data_path: str = os.path.join(
        REPO_ROOT, "data", "synthetic", "sft_full.jsonl")
    output_dir: str = os.path.join(REPO_ROOT, "output", "full_sft")


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prompt_masked_labels(input_ids: List[int], prompt_len: int,
                         ignore_index: int = IGNORE_INDEX) -> List[int]:
    """Build labels that mask the prompt and keep the assistant tokens.

    ``prompt_len`` is clamped to ``[0, len(input_ids)]`` so a truncated
    sample never produces a longer label list than its input.
    """
    total = len(input_ids)
    cut = max(0, min(prompt_len, total))
    return [ignore_index] * cut + list(input_ids[cut:])


def load_sft_dataset(path: str):
    from datasets import Dataset

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list(records)


def encode_sft_example(example: Dict[str, Any], tokenizer,
                       max_seq_length: int) -> Dict[str, Any]:
    """Tokenise one chat sample and mask the prompt tokens."""
    messages = example["messages"]
    prompt_text = render_chat(tokenizer, messages[:-1], add_generation_prompt=True)
    full_text = render_chat(tokenizer, messages, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "chat template mismatch: prompt is not a prefix of the full conversation"
        )

    input_ids = full_ids[:max_seq_length]
    if len(prompt_ids) >= len(input_ids):
        raise ValueError("sample has no assistant tokens after truncation")

    labels = prompt_masked_labels(input_ids, len(prompt_ids))
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def parse_args() -> argparse.Namespace:
    cfg = FullSFTConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=cfg.model_name)
    parser.add_argument("--data", default=cfg.data_path)
    parser.add_argument("--output_dir", default=cfg.output_dir)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--grad_accum_steps", type=int, default=cfg.grad_accum_steps)
    parser.add_argument("--learning_rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--max_seq_length", type=int, default=cfg.max_seq_length)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    return parser.parse_args()


def main() -> None:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    args = parse_args()
    cfg = FullSFTConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        data_path=args.data,
        output_dir=args.output_dir,
    )

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU required for full-parameter SFT. "
            "See docs/cloud-training.md (AutoDL runbook)."
        )
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype)
    model.config.use_cache = False

    dataset = load_sft_dataset(cfg.data_path)
    tokenized = dataset.map(
        lambda example: encode_sft_example(example, tokenizer, cfg.max_seq_length),
        remove_columns=dataset.column_names,
        desc="tokenising SFT samples",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
        return_tensors="pt",
    )

    bf16_supported = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.output_dir, "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=1,
        save_strategy="no",
        seed=cfg.seed,
        report_to="none",
        fp16=not bf16_supported,
        bf16=bf16_supported,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer_stats = trainer.train()

    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), "optimizer": "adamw_torch", "full_finetune": True},
                  f, indent=2, ensure_ascii=False)

    log_history = [
        {k: v for k, v in step.items()
         if k in ("loss", "grad_norm", "learning_rate", "epoch")}
        for step in trainer.state.log_history
        if "loss" in step
    ]
    with open(os.path.join(cfg.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, ensure_ascii=False)

    print(f"Full SFT done. Weights -> {final_dir}")
    print(f"Final loss: {trainer_stats.training_loss}")


if __name__ == "__main__":
    main()
