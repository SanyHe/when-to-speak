#!/usr/bin/env bash
# init.sh — Harness startup script for the When-to-Speak repo.
#
# Installs local dev dependencies, runs verification, and prints the
# standard start command. Training itself runs on a cloud GPU (see
# docs/cloud-training.md); this script only sets up the local
# schema/data/test toolchain.

set -euo pipefail

# ---- Edit these for your project ----
INSTALL_CMD="python3 -m pip install --quiet jsonschema pytest"
VERIFY_CMD="python3 -m pytest training/tests/ -q"
START_CMD="echo 'Training runs on cloud GPU. See docs/cloud-training.md for the runbook.'"
# --------------------------------------

echo "=== init.sh: When-to-Speak harness startup ==="
echo "Working directory: $(pwd)"

# 1. Confirm we're in the repo root by checking for AGENTS.md.
if [ ! -f "AGENTS.md" ]; then
  echo "ERROR: AGENTS.md not found. Run init.sh from the repo root." >&2
  exit 1
fi

# 2. Install dependencies.
echo
echo "--- Installing dev dependencies ---"
eval "$INSTALL_CMD"

# 3. Run verification. If this fails, the baseline is broken — stop.
echo
echo "--- Running verification ---"
if eval "$VERIFY_CMD"; then
  echo
  echo "=== Verification PASSED ==="
else
  echo
  echo "=== Verification FAILED ===" >&2
  echo "Fix the baseline before doing any feature work." >&2
  exit 1
fi

# 4. Print the start command (run it if RUN_START_COMMAND=1).
echo
echo "--- Start command ---"
echo "$START_CMD"
if [ "${RUN_START_COMMAND:-0}" = "1" ]; then
  eval "$START_CMD"
fi
