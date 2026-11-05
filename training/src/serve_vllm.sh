#!/usr/bin/env bash
# Serve a fine-tuned checkpoint with vLLM's OpenAI-compatible API.
#
# Runs on the cloud GPU host (AutoDL). The web demo calls this endpoint
# directly from the browser when "Remote inference (vLLM)" is enabled.
#
# Usage:
#   ./training/src/serve_vllm.sh [model_dir]
#
#   model_dir defaults to output/grpo/final (fall back to output/full_sft/final).
#
# Environment overrides:
#   PORT=6006              AutoDL exposes this port via its custom-service URL
#   SERVED_NAME=wts        model id the web page must send
#   API_KEY=               optional bearer token (vLLM --api-key)
#   ALLOWED_ORIGINS=*      CORS origins, space-separated or a JSON array, e.g.
#                          "http://localhost:8765 http://127.0.0.1:8765"
#   MAX_MODEL_LEN=2048     >= training max_seq_length (1024)
#   GPU_UTIL=0.6           0.6 is plenty for a 0.6B model
#   EAGER=1                pass --enforce-eager (faster startup, less VRAM)
#
# AutoDL notes:
#   1. Create the instance with a PyTorch 2.x image.
#   2. In the console open "Custom service" for port 6006; the public URL
#      looks like https://<region>-<port>.autodl.pro or a *.autodl.host link.
#   3. Paste that URL into the web page's "Endpoint" field (no /v1 suffix).
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

MODEL_PATH=${1:-output/grpo/final}
PORT=${PORT:-6006}
SERVED_NAME=${SERVED_NAME:-wts}
API_KEY=${API_KEY:-}
ALLOWED_ORIGINS=${ALLOWED_ORIGINS:-*}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
GPU_UTIL=${GPU_UTIL:-0.6}
EAGER=${EAGER:-0}

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: model directory not found: $MODEL_PATH" >&2
  echo "Run ./training/run_all.sh first (or pass the path explicitly)." >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm not installed. On the GPU host run:" >&2
  echo "  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm" >&2
  exit 1
fi

# vLLM parses --allowed-origins as JSON, so a space-separated list must be
# converted to a JSON array first. A value that is already JSON is passed
# through unchanged.
case "$ALLOWED_ORIGINS" in
  \[*) ORIGINS_JSON=$ALLOWED_ORIGINS ;;
  *) ORIGINS_JSON=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1].split()))' \
       "$ALLOWED_ORIGINS") ;;
esac

CMD=(vllm serve "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --dtype auto
  --allowed-origins "$ORIGINS_JSON")

if [ -n "$API_KEY" ]; then
  CMD+=(--api-key "$API_KEY")
fi

if [ "$EAGER" = "1" ]; then
  CMD+=(--enforce-eager)
fi

echo "Serving $MODEL_PATH as '$SERVED_NAME' on 0.0.0.0:$PORT"
echo "CORS origins: $ORIGINS_JSON"
echo "Once it is up, verify locally:"
echo "  curl -s http://127.0.0.1:$PORT/v1/models | head"
echo

"${CMD[@]}"
