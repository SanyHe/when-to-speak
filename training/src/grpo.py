"""
GRPO reinforcement learning.

Starts from the full-parameter SFT checkpoint and optimises the policy
with Group Relative Policy Optimization (TRL ``GRPOTrainer``) on the
event decision task:

    prompt   -> system instruction + one event
    rollout  -> G sampled decisions per prompt
    reward   -> format validity + action correctness + channel match
                (see training/src/rewards.py)

No reward model is trained: rewards are deterministic, rule-based, and
computed from the gold decision shipped with each prompt.

Runs on a CUDA GPU only (AutoDL single 24 GB card, A100 40 GB preferred
for larger ``--num_generations``). Deterministic seed; every
hyperparameter and reward function name is logged.

Usage (on the training host):
    python3 training/src/grpo.py \
        --model_path output/full_sft/final \
        --data data/synthetic/grpo_prompts.jsonl \
        --output_dir output/grpo \
        --epochs 1 --beta 0.04 --seed 42

Outputs:
    <output_dir>/final/          full HF weights + tokenizer
    <output_dir>/run_config.json hyperparameters, seed, reward functions
    <output_dir>/train_log.json  reward / loss curve

Requirements: see training/requirements.txt (torch, transformers>=4.51,
trl>=0.19, datasets, accelerate).
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

from format_sft import render_chat  # noqa: E402
from rewards import REWARD_FUNCS  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", ".."))


@dataclass
class GRPOExperimentConfig:
    model_path: str = os.path.join(REPO_ROOT, "output", "full_sft", "final")
    max_prompt_length: int = 768
    max_completion_length: int = 200
    num_generations: int = 8
    epochs: int = 1
    batch_size: int = 2
    grad_accum_steps: int = 4
    learning_rate: float = 1e-6
    beta: float = 0.04
    temperature: float = 1.0
    seed: int = 42
    data_path: str = os.path.join(REPO_ROOT, "data", "synthetic", "grpo_prompts.jsonl")
    output_dir: str = os.path.join(REPO_ROOT, "output", "grpo")


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_grpo_dataset(path: str, tokenizer) -> Any:
    """Load prompts and render them to text with thinking mode disabled."""
    from datasets import Dataset

    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    records: List[Dict[str, Any]] = []
    for row in rows:
        records.append({
            "prompt": render_chat(tokenizer, row["messages"], add_generation_prompt=True),
            "gold_action": row["gold_action"],
            "gold_decision": row["gold_decision"],
            "event_id": row["event_id"],
        })
    return Dataset.from_list(records)


def plan_generation_batch(num_generations: int, batch_size: int,
                          grad_accum_steps: int) -> int:
    """Smallest generation batch TRL accepts for this group size.

    TRL requires ``generation_batch_size`` to be divisible by
    ``num_generations``; its default (batch x accum) is not, so G=16 with
    batch 2 x accum 4 raised "generation_batch_size (8) must be divisible by
    num_generations (16)".
    """
    default = max(1, batch_size) * max(1, grad_accum_steps)
    if default % num_generations == 0:
        return default
    return num_generations * math.ceil(default / num_generations)


def build_trainer(model, tokenizer, dataset, cfg: GRPOExperimentConfig):
    """Build a TRL GRPOTrainer across tokenizer/processing_class versions."""
    import torch
    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    bf16_supported = torch.cuda.is_bf16_supported()
    params = inspect.signature(TRLGRPOConfig.__init__).parameters
    gen_batch = plan_generation_batch(
        cfg.num_generations, cfg.batch_size, cfg.grad_accum_steps)
    grpo_kwargs = dict(
        output_dir=os.path.join(cfg.output_dir, "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=0.03,
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
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        beta=cfg.beta,
        temperature=cfg.temperature,
    )
    if "generation_batch_size" in params:
        grpo_kwargs["generation_batch_size"] = gen_batch
        print(f"[grpo] generation_batch_size={gen_batch} "
              f"for G={cfg.num_generations}")

    # TRL tracks transformers' TrainingArguments, whose newer releases removed
    # `warmup_ratio`. Derive warmup_steps, and drop any other keyword this
    # installed version no longer understands (e.g. max_prompt_length).
    if "warmup_ratio" not in params:
        steps_per_epoch = math.ceil(
            len(dataset) / max(1, cfg.batch_size * cfg.grad_accum_steps))
        total_steps = max(1, steps_per_epoch * cfg.epochs)
        grpo_kwargs.pop("warmup_ratio")
        grpo_kwargs["warmup_steps"] = max(1, int(round(total_steps * 0.03)))
    for key in [k for k in grpo_kwargs if k not in params]:
        print(f"[grpo] dropping unsupported config key: {key}")
        grpo_kwargs.pop(key)

    grpo_config = TRLGRPOConfig(**grpo_kwargs)
    trainer_kwargs = {
        "model": model,
        "reward_funcs": list(REWARD_FUNCS),
        "args": grpo_config,
        "train_dataset": dataset,
    }
    # TRL renamed the tokenizer argument to processing_class in newer releases.
    params = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return GRPOTrainer(**trainer_kwargs)


def parse_args() -> argparse.Namespace:
    cfg = GRPOExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", default=cfg.model_path,
        help="Checkpoint to start from (SFT output dir or another HF path)")
    parser.add_argument("--data", default=cfg.data_path)
    parser.add_argument("--output_dir", default=cfg.output_dir)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--grad_accum_steps", type=int, default=cfg.grad_accum_steps)
    parser.add_argument("--learning_rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--num_generations", type=int, default=cfg.num_generations)
    parser.add_argument("--max_completion_length", type=int, default=cfg.max_completion_length)
    parser.add_argument("--beta", type=float, default=cfg.beta,
                        help="KL coefficient to the SFT reference; 0 disables the reference model")
    parser.add_argument("--temperature", type=float, default=cfg.temperature)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    return parser.parse_args()


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    cfg = GRPOExperimentConfig(
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

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU required for GRPO. See docs/cloud-training.md "
            "(AutoDL runbook, section 4)."
        )
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype=dtype)
    model.config.use_cache = False

    dataset = load_grpo_dataset(cfg.data_path, tokenizer)
    trainer = build_trainer(model, tokenizer, dataset, cfg)

    trainer.train()

    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            **asdict(cfg),
            "full_finetune": True,
            "algorithm": "grpo",
            "reward_functions": [func.__name__ for func in REWARD_FUNCS],
            "train_samples": len(dataset),
        }, f, indent=2, ensure_ascii=False)

    log_history = [
        {k: v for k, v in step.items()
         if "reward" in k or k in ("loss", "epoch", "learning_rate",
                                   "completion_length", "kl")}
        for step in trainer.state.log_history
    ]
    with open(os.path.join(cfg.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, ensure_ascii=False)

    print(f"GRPO done. Weights -> {final_dir}")
    print(f"Reward functions: {[func.__name__ for func in REWARD_FUNCS]}")


if __name__ == "__main__":
    main()
