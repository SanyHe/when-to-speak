#!/usr/bin/env bash
# Show loop status, state, ledger tail, and the recent log.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== process ==="
if [ -f autoresearch/loop.pid ]; then
  PID="$(cat autoresearch/loop.pid)"
  if kill -0 "$PID" 2>/dev/null; then
    ps -o pid,ppid,sid,tty,stat,etime,pcpu,pmem,cmd -p "$PID"
  else
    echo "pid $PID is not running"
  fi
else
  echo "no autoresearch/loop.pid"
fi

echo
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

echo
echo "=== state.json ==="
cat autoresearch/state.json 2>/dev/null || echo "no state.json"

echo
echo "=== ledger (last 5) ==="
tail -n 5 autoresearch/ledger.jsonl 2>/dev/null || echo "no ledger.jsonl"

echo
echo "=== loop.log (last 30) ==="
tail -n 30 autoresearch/logs/loop.log 2>/dev/null || echo "no loop.log"
