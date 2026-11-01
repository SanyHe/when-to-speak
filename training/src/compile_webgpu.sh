#!/usr/bin/env bash
# Compile merged HF weights into MLC WebGPU artifacts (q4f16_1) for WebLLM.
#
# Runs on the cloud GPU host only (mlc-llm + CUDA). See docs/cloud-training.md
# for the provisioning runbook. The exact mlc-llm CLI changes across releases:
# verify the subcommands with `mlc_llm --help` before the first real run and
# update this script if the interface differs.
#
# Usage:
#   ./training/src/compile_webgpu.sh <merged_dir> <out_dir> <mlc_model_name>
#
# Example:
#   ./training/src/compile_webgpu.sh output/full_sft/final dist/wts-sft wts-sft
#   ./training/src/compile_webgpu.sh output/grpo/final dist/wts-grpo wts-grpo
#
# Output: <out_dir>/params/, <out_dir>/mlc-chat-config.json and
#         <out_dir>/<mlc_model_name>-webgpu.wasm
set -euo pipefail

MERGED_DIR=${1:?usage: compile_webgpu.sh <merged_dir> <out_dir> <mlc_model_name>}
OUT_DIR=${2:?usage: compile_webgpu.sh <merged_dir> <out_dir> <mlc_model_name>}
MODEL_NAME=${3:?usage: compile_webgpu.sh <merged_dir> <out_dir> <mlc_model_name>}

if ! command -v mlc_llm >/dev/null 2>&1; then
  echo "ERROR: mlc_llm not found. Install it on the GPU host:" >&2
  echo "  pip install mlc-llm" >&2
  exit 1
fi

if [ ! -d "$MERGED_DIR" ]; then
  echo "ERROR: merged model directory not found: $MERGED_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "=== 1/3 Generate MLC config (q4f16_1) ==="
# --conv-template: confirm the correct template for the base model family
# (qwen3 / qwen2) with `mlc_llm gen_config --help` in the installed version.
mlc_llm gen_config "$MERGED_DIR" \
  --quantization q4f16_1 \
  --conv-template qwen3 \
  -o "$OUT_DIR"

echo "=== 2/3 Convert weights ==="
mlc_llm convert_weight "$MERGED_DIR" \
  --quantization q4f16_1 \
  -o "$OUT_DIR/params"

echo "=== 3/3 Compile for WebGPU ==="
mlc_llm compile "$OUT_DIR/mlc-chat-config.json" \
  --device webgpu \
  -o "$OUT_DIR/${MODEL_NAME}-webgpu.wasm"

echo
echo "Artifacts in $OUT_DIR:"
ls -lh "$OUT_DIR"
echo
echo "Next: publish params/ + wasm to a GitHub Release or HF repo, then fill"
echo "the matching entry in web/js/model-records.js and set status to \"ready\"."
