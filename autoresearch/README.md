# Auto Research — interruption policy for Qwen3-0.6B

**English** | [简体中文](README.zh-CN.md)

An unattended, single-metric research loop for the "when to speak" task in this
repository. The model reads one message event and emits a strict JSON decision:

```json
{"action": "notify|digest|silence", "channel": "...", "urgency": 0.0, "reason": "...", "message": "..."}
```

The loop varies **only the training recipe**, trains from the same base model,
scores every candidate on the same held-out dev set, and stops on a fixed budget
or when a candidate clears a hard gate.

## Protocol

| Constraint | Rule |
|---|---|
| Editable surface | only [`recipe.py`](recipe.py) may define a trial difference |
| Frozen oracles | `training/src/*.py`, `data/synthetic/*`, `data/templates/*`, `autoresearch/data/dev.jsonl`, `autoresearch/data/sealed_test.jsonl`, `eval_one.py`, `gate.py` |
| Integrity | [`frozen_hashes.json`](frozen_hashes.json) stores SHA-256 for every oracle file; the loop refuses to start on a mismatch |
| Metric | dev `macro_f1` only |
| Safety | `json_valid_rate >= 0.99`, `noise_rate <= 0.02`, `miss_rate <= 0.02` |
| Base model | `Qwen/Qwen3-0.6B`, full fine-tune, no adapters |
| Sealed set | opened once, after the loop stops, for the final evaluation only |

See [`program.md`](program.md) for the full research charter.

### Gates

1. `json_valid_rate >= 0.99`
2. `noise_rate <= 0.02`
3. `miss_rate <= 0.02`
4. no regression vs the incumbent (`+0.005` tolerance on noise and miss)
5. significant improvement: `macro_f1 >= best + 0.005` with paired-bootstrap
   `P(better) >= 0.80` (2000 resamples), or `notify_recall >= best + 0.02`
   without noise regression

Gates 1–4 reject fast. A candidate that clears gate 5 is retrained with
`seed=43`; the two-seed mean must still beat the incumbent.

### Stop conditions

`STOP` file, wall clock, GPU minutes, trial budget, 3 consecutive failures,
`--patience` trials without a new best, or the goal
(`macro_f1 >= 0.97`, `noise <= 0.005`, `miss <= 0.005`, `recall >= 0.99`,
`json_valid >= 0.999`).

## Layout

```
autoresearch/
  program.md        research charter (read-only for the loop)
  recipe.py         the only editable surface: BASELINE + CANDIDATES + SEARCH_SPACE
  build_splits.py   deterministic split reproduction + leakage check
  data_ops.py       materialize SFT / GRPO data for a recipe
  rewards_ext.py    weighted reward factory on top of the frozen rewards
  train_grpo.py     GRPO entry point (overrides reward funcs, reuses frozen trainer)
  eval_one.py       dev evaluation -> predictions.jsonl / metrics.json
  gate.py           5 gates + paired bootstrap
  runner.py         one trial: materialize -> SFT -> GRPO -> eval, subprocess isolation
  loop.py           candidate selection, gating, ledger, state, stop, finalize
  launch.sh         detached start (setsid), status.sh, supervise.sh for keepalive
  freeze.py         regenerate frozen_hashes.json
```

Each trial writes `runs/<trial_id>/`; the loop appends one line per trial to
`ledger.jsonl` and atomically rewrites `state.json`. On stop it evaluates the
incumbent once on the sealed split and writes `FINAL_REPORT.md` plus
`accepted/<trial_id>.json`.

## Quickstart

Requires one CUDA GPU (≥24 GB), a CUDA build of PyTorch in the system Python,
and ~20 GB of disk for two full fine-tune checkpoints.

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install pandas pyarrow datasets
.venv/bin/pip install --no-deps trl==1.17.1
.venv/bin/pip install "huggingface-hub>=1.5,<2.0"

export HF_ENDPOINT=https://hf-mirror.com
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='autoresearch/models/Qwen3-0.6B')
"

.venv/bin/python autoresearch/build_splits.py          # dev 450 + sealed 200, leakage check
.venv/bin/python autoresearch/loop.py --smoke --max-trials 1
bash autoresearch/launch.sh --max-hours 9 --max-gpu-minutes 480 --max-trials 12 --patience 3
bash autoresearch/status.sh
```

`--smoke` runs a tiny end-to-end pass (16 SFT rows / 8 eval rows); low scores
there are expected and it writes no ledger or report.

## Data

`data/synthetic/` and `data/templates/` are the frozen source of truth.
`build_splits.py` reproduces the split exactly as the original pipeline does
(seed 42, contiguous RNG stream) and asserts byte equality against the frozen
SFT / GRPO / eval artifacts.

The `**/data/*.jsonl` dev and sealed splits are **not distributed** in this
repository: publishing the sealed rows would invalidate the held-out set. Run
`build_splits.py` to regenerate them locally. The split sizes are 800 / 300 /
200 (train / GRPO prompts / eval) plus 250 dev-extra and 200 sealed rows.

## Results

Single 96 GB GPU, one round, seven trials, 141 GPU-minutes, stop reason
`patience`.

| trial | change | dev macro-F1 | outcome |
|---|---|---|---|
| `r00_baseline` | — | 0.9264 | initial incumbent |
| `r01_sft_1ep` | SFT 1 epoch | 0.8225 | rejected: noise 4.0% |
| `r02_digest_over2` | digest ×2 | 0.8971 | rejected: miss 31.9% |
| **`r03_digest_focus2`** | digest ×2 + `{deep_work, offline}` ×2 | **0.9943** | accepted (seed 43: 0.9923, mean 0.9933) |
| `r04_contrastive_300` | r03 + 300 boundary copies | 0.9904 | rejected: not significant |
| `r05_grpo_hard2` | r03 + hard-prompt ×2 | 0.9904 | rejected: not significant |
| `r06_grpo_g16_beta02` | r03 + G=16, beta 0.02 | 0.9904 | rejected: not significant |

`r03` on the sealed split (200 rows, opened once): macro-F1 **0.9828**,
precision/recall 1.0, noise 0, miss 0, JSON validity 0.995.

See [`FINAL_REPORT.md`](FINAL_REPORT.md) and [`ledger.jsonl`](ledger.jsonl) for
the full record. Absolute paths in the ledger are normalized to `./`.

Per-trial artifacts are checked in under [`results/`](results/): the resolved
SFT/GRPO configs, the SFT loss curve, the GRPO reward curves, dev metrics and
data counts. No evaluation rows, predictions or checkpoints are included.

## Notes

- `build_splits.py` may reorder `user_focus` sampling relative to a run made
  with a different Python version; the three frozen artifacts are asserted
  byte-for-byte, so any real split drift fails loudly.
- The runner strips sandbox-injected `LD_LIBRARY_PATH` entries and starts
  detached processes with a clean environment; some IDE sandboxes preload a
  library that makes CUDA initialization fail (`error 304` / SIGSEGV).
