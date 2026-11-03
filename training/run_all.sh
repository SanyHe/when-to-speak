#!/usr/bin/env bash
# End-to-end cloud GPU pipeline:
#   build dataset -> full-parameter SFT -> GRPO -> real-checkpoint eval.
#
# Runs on the training host only (AutoDL, CUDA required). Local Mac runs
# stop at step 1. See docs/cloud-training.md for provisioning.
#
# Usage (from the repo root or anywhere):
#   ./training/run_all.sh
#
# Optional environment overrides:
#   BASE_MODEL=Qwen/Qwen3-0.6B   HF repo id or a local directory with the base weights
#   SEED=42                global seed
#   SFT_EPOCHS=2  SFT_BATCH=2  SFT_ACCUM=8
#   GRPO_EPOCHS=1 GRPO_BATCH=2 GRPO_GENERATIONS=8 GRPO_BETA=0.04
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-0.6B}
SEED=${SEED:-42}
SFT_EPOCHS=${SFT_EPOCHS:-2}
SFT_BATCH=${SFT_BATCH:-2}
SFT_ACCUM=${SFT_ACCUM:-8}
GRPO_EPOCHS=${GRPO_EPOCHS:-1}
GRPO_BATCH=${GRPO_BATCH:-2}
GRPO_GENERATIONS=${GRPO_GENERATIONS:-8}
GRPO_BETA=${GRPO_BETA:-0.04}

echo "=== 1/4 Build dataset (deterministic, seed=${SEED}) ==="
python3 training/src/build_dataset.py

echo "=== 2/4 Full-parameter SFT (base=${BASE_MODEL}) ==="
python3 training/src/full_sft.py \
  --model_path "${BASE_MODEL}" \
  --data data/synthetic/sft_full.jsonl \
  --output_dir output/full_sft \
  --epochs "${SFT_EPOCHS}" \
  --batch_size "${SFT_BATCH}" \
  --grad_accum_steps "${SFT_ACCUM}" \
  --learning_rate 2e-5 \
  --seed "${SEED}"

echo "=== 3/4 GRPO reinforcement learning ==="
python3 training/src/grpo.py \
  --model_path output/full_sft/final \
  --data data/synthetic/grpo_prompts.jsonl \
  --output_dir output/grpo \
  --epochs "${GRPO_EPOCHS}" \
  --batch_size "${GRPO_BATCH}" \
  --num_generations "${GRPO_GENERATIONS}" \
  --beta "${GRPO_BETA}" \
  --seed "${SEED}"

echo "=== 4/4 Real-checkpoint evaluation ==="
python3 training/src/evaluate.py --mode hf \
  --eval data/synthetic/eval_set.jsonl \
  --models base=${BASE_MODEL},sft=output/full_sft/final,sft_grpo=output/grpo/final \
  --out data/eval/eval_report_hf.json \
  --md-out data/eval/eval_report_hf.md \
  --errors-out data/eval/errors_hf.jsonl \
  --web-out web/data/traces.json

echo
echo "Pipeline finished. Weights: output/full_sft/final, output/grpo/final."
echo "Next: start the inference service with ./training/src/serve_vllm.sh"
echo "      (see docs/cloud-training.md section 6), then open the web demo."
echo "Also backfill evidence into private/feature_list.json + private/claude-progress.md"
echo "      (run_config.json, train_log.json, eval_report_hf.*, final loss/reward)."
