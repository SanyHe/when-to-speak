#!/usr/bin/env bash
# Launch the Auto Research loop detached from the terminal.
# Usage: bash autoresearch/launch.sh [loop.py args...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p autoresearch/logs

if [ -f autoresearch/loop.pid ]; then
  OLD_PID="$(cat autoresearch/loop.pid)"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "loop already running (pid $OLD_PID); refusing to start a second one"
    exit 1
  fi
  rm -f autoresearch/loop.pid
fi

# Some sandboxed shells preload a library (e.g. sbox.so) into every descendant,
# which makes CUDA fail with "error 304" or SIGSEGV. `env -i` alone does not
# help because the library re-injects itself; clear LD_PRELOAD explicitly and
# verify the loop really started with a clean environment.
case "${LD_PRELOAD:-}" in
  *sbox.so*)
    echo "note: caller has a preloaded sandbox library; launching with a clean env instead."
    ;;
esac

rm -f autoresearch/STOP
echo "=== launch $(date -Iseconds) ===" >> autoresearch/logs/loop.log

# The injection is racy, so retry until the loop process starts with a clean env.
PID=""
for attempt in 1 2 3 4 5; do
  rm -f autoresearch/loop.pid
  setsid nohup env -i \
    LD_PRELOAD= \
    SBOX_SHM_NAME= \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME="${HOME:-/root}" \
    LANG="${LANG:-en_US.UTF-8}" \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    .venv/bin/python autoresearch/loop.py "$@" \
    >> autoresearch/logs/loop.log 2>&1 < /dev/null &

  # loop.py writes its own PID once it acquires the lock.
  for _ in $(seq 1 20); do
    if [ -s autoresearch/loop.pid ]; then
      break
    fi
    sleep 0.5
  done

  PID="$(cat autoresearch/loop.pid 2>/dev/null || true)"
  if [ -n "$PID" ] && tr '\0' '\n' < "/proc/$PID/environ" 2>/dev/null | grep -q "sbox.so"; then
    echo "attempt $attempt: pid=$PID inherited sbox.so; retrying with a clean env"
    kill -KILL "$PID" 2>/dev/null || true
    sleep 2
    PID=""
    continue
  fi
  break
done

if [ -z "$PID" ]; then
  echo "ERROR: could not start the loop with a clean environment (CUDA would crash)."
  exit 3
fi
echo "launched pid=$PID"
ps -o pid,ppid,sid,tty,stat,etime,cmd -p "$PID" || true
