#!/usr/bin/env bash
# Keepalive wrapper. Restart only after OOM / signal kills (134/137/139).
# A clean stop (0) or any other exit code ends the supervisor.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p autoresearch/logs

# Same guard as launch.sh: a preloaded sandbox library breaks CUDA in children.
case "${LD_PRELOAD:-}" in
  *sbox.so*)
    echo "note: caller has a preloaded sandbox library; the loop is started with a clean env."
    ;;
esac

while true; do
  echo "=== supervise start $(date -Iseconds) ===" >> autoresearch/logs/loop.log
  env -i \
    LD_PRELOAD= \
    SBOX_SHM_NAME= \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME="${HOME:-/root}" \
    LANG="${LANG:-en_US.UTF-8}" \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    .venv/bin/python autoresearch/loop.py "$@" >> autoresearch/logs/loop.log 2>&1
  CODE=$?
  echo "=== supervise: loop exited with $CODE ===" >> autoresearch/logs/loop.log
  if [ "$CODE" -eq 0 ]; then
    exit 0
  fi
  if [ "$CODE" -eq 134 ] || [ "$CODE" -eq 137 ] || [ "$CODE" -eq 139 ]; then
    sleep 5
    continue
  fi
  exit "$CODE"
done
