#!/usr/bin/env bash
# End-to-end cloud GPU pipeline (F2.3 -> F2.4 -> F2.5 -> F3.1):
#   build dataset -> full-parameter SFT -> GRPO -> real-checkpoint eval
#   -> optional MLC WebGPU compile.
#
# Runs on the training host only (AutoDL, CUDA required). Local Mac runs
# stop at step 1. See docs/cloud-training.md for provisioning.
#
# Usage (from the repo root or anywhere):
#   ./training/run_all.sh
#
# Optional environment overrides:
#   SEED=42                global seed
#   SFT_EPOCHS=2  SFT_BATCH=2  SFT_ACCUM=8
#   GRPO_EPOCHS=1 GRPO_BATCH=2 GRPO_GENERATIONS=8 GRPO_BETA=0.04
#   RUN_COMPILE=1          also compile MLC WebGPU artifacts (needs mlc-llm)
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

SEED=${SEED:-42}
SFT_EPOCHS=${SFT_EPOCHS:-2}
SFT_BATCH=${SFT_BATCH:-2}
SFT_ACCUM=${SFT_ACCUM:-8}
GRPO_EPOCHS=${GRPO_EPOCHS:-1}
GRPO_BATCH=${GRPO_BATCH:-2}
GRPO_GENERATIONS=${GRPO_GENERATIONS:-8}
GRPO_BETA=${GRPO_BETA:-0.04}
RUN_COMPILE=${RUN_COMPILE:-0}

echo "=== 1/5 Build dataset (deterministic, seed=${SEED}) ==="
python3 training/src/build_dataset.py

echo "=== 2/5 Full-parameter SFT (F2.3) ==="
python3 training/src/full_sft.py \
  --data data/synthetic/sft_full.jsonl \
  --output_dir output/full_sft \
  --epochs "${SFT_EPOCHS}" \
  --batch_size "${SFT_BATCH}" \
  --grad_accum_steps "${SFT_ACCUM}" \
  --learning_rate 2e-5 \
  --seed "${SEED}"

echo "=== 3/5 GRPO reinforcement learning (F2.4) ==="
python3 training/src/grpo.py \
  --model_path output/full_sft/final \
  --data data/synthetic/grpo_prompts.jsonl \
  --output_dir output/grpo \
  --epochs "${GRPO_EPOCHS}" \
  --batch_size "${GRPO_BATCH}" \
  --num_generations "${GRPO_GENERATIONS}" \
  --beta "${GRPO_BETA}" \
  --seed "${SEED}"

echo "=== 4/5 Real-checkpoint evaluation (F2.5) ==="
python3 training/src/evaluate.py --mode hf \
  --eval data/synthetic/eval_set.jsonl \
  --models base=Qwen/Qwen3-0.6B,sft=output/full_sft/final,sft_grpo=output/grpo/final \
  --out data/eval/eval_report_hf.json \
  --md-out data/eval/eval_report_hf.md \
  --errors-out data/eval/errors_hf.jsonl \
  --web-out web/data/traces.json

echo "=== 5/5 MLC WebGPU compile (F3.1, optional) ==="
if [ "${RUN_COMPILE}" = "1" ]; then
  ./training/src/compile_webgpu.sh output/full_sft/final dist/wts-sft wts-sft
  ./training/src/compile_webgpu.sh output/grpo/final dist/wts-grpo wts-grpo
else
  echo "skipped (set RUN_COMPILE=1 to enable; requires mlc-llm on this host)"
fi

echo
echo "Pipeline finished. Remaining manual steps:"
echo "  1. Publish dist/ artifacts to GitHub Release / Hugging Face."
echo "  2. Fill web/js/model-records.js entries and set status to \"ready\"."
echo "  3. Backfill evidence into private/feature_list.json + private/claude-progress.md"
echo "     (run_config.json, train_log.json, eval_report_hf.*, final loss/reward)."
