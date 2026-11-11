"""GRPO training entry point for the research loop.

Loads the frozen ``grpo.py`` and calls its ``build_trainer`` unchanged; only
the module-level ``REWARD_FUNCS`` is replaced with the weighted factory and
the dataset carries the extra ``user_focus`` / ``duplicate_of`` columns.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(REPO_ROOT, "training", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grpo  # noqa: E402  (frozen)
from format_sft import render_chat  # noqa: E402
from rewards_ext import build_weighted_rewards  # noqa: E402


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(path: str, tokenizer):
    from datasets import Dataset

    records = []
    for row in _read_jsonl(path):
        records.append({
            "prompt": render_chat(tokenizer, row["messages"], add_generation_prompt=True),
            "gold_action": row["gold_action"],
            "gold_decision": row["gold_decision"],
            "event_id": row["event_id"],
            "user_focus": row.get("user_focus"),
            "duplicate_of": row.get("duplicate_of"),
        })
    return Dataset.from_list(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default="{}", help="JSON reward-weight mapping")
    return parser.parse_args()


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for GRPO")

    weights = json.loads(args.weights)
    reward_funcs = build_weighted_rewards(weights)

    grpo.set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    model.config.use_cache = False

    dataset = build_dataset(args.data, tokenizer)

    # Swap the frozen module's reward list; build_trainer reads this global.
    grpo.REWARD_FUNCS = reward_funcs
    cfg = grpo.GRPOExperimentConfig(
        model_path=args.model_path,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        temperature=args.temperature,
        seed=args.seed,
        data_path=args.data,
        output_dir=args.output_dir,
    )
    trainer = grpo.build_trainer(model, tokenizer, dataset, cfg)
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_path": args.model_path,
            "data_path": args.data,
            "output_dir": args.output_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "learning_rate": args.learning_rate,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "beta": args.beta,
            "temperature": args.temperature,
            "seed": args.seed,
            "reward_weights": weights,
            "reward_functions": [func.__name__ for func in reward_funcs],
            "train_samples": len(dataset),
            "full_finetune": True,
            "algorithm": "grpo",
        }, f, indent=2, ensure_ascii=False)

    log_history = [
        {k: v for k, v in step.items()
         if "reward" in k or k in ("loss", "epoch", "learning_rate",
                                   "completion_length", "kl")}
        for step in trainer.state.log_history
    ]
    with open(os.path.join(args.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, ensure_ascii=False)

    print(f"GRPO done. Weights -> {final_dir}")


if __name__ == "__main__":
    main()
