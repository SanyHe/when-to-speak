# Auto Research Program — When to Speak

## Mission

Improve the interruption policy of Qwen3-0.6B on the synthetic event task
without changing the task, the data contract, or the metric. The model reads
one event and emits a strict JSON decision:

```
{action: notify|digest|silence, channel, urgency, reason, message}
```

The only variable is the training recipe (data sampling + optimisation
hyperparameters), expressed in `autoresearch/recipe.py`.

## Single editable surface

**Only `autoresearch/recipe.py` may define a trial difference.** Everything else
is an oracle:

- `training/src/*.py` — training and evaluation implementation (frozen)
- `data/synthetic/*.jsonl` — gold SFT / GRPO / eval splits (frozen)
- `data/templates/*.json` — event templates (frozen)
- `autoresearch/data/dev.jsonl` — 450-row model-selection set (frozen)
- `autoresearch/data/sealed_test.jsonl` — 200-row final test set (frozen)
- `autoresearch/eval_one.py`, `autoresearch/gate.py` — metric and gate implementation

`autoresearch/frozen_hashes.json` records the SHA-256 of every oracle file. The
loop refuses to start if any hash differs.

## Single metric

Selection uses **dev macro-F1** only. Hard safety constraints:

- `json_valid_rate >= 0.99`
- `noise_rate <= 0.02`
- `miss_rate <= 0.02`

## Single base model

`autoresearch/models/Qwen3-0.6B` for every trial. No model swaps, no adapters.

## Gates (fixed, not editable by the loop)

1. `json_valid_rate >= 0.99`
2. `noise_rate <= 0.02`
3. `miss_rate <= 0.02`
4. No regression: `noise <= incumbent.noise + 0.005` and
   `miss <= incumbent.miss + 0.005`
5. Significant improvement, one of:
   - `macro_f1 >= incumbent.macro_f1 + 0.005` **and** paired-bootstrap
     `P(candidate better) >= 0.80` (2000 resamples), or
   - `notify_recall >= incumbent.notify_recall + 0.02` with noise not
     regressing.

Gates 1–4 are reject-fast and skip the bootstrap. A candidate that passes
gate 5 is retrained with `seed=43`; the two-seed mean must still beat the
incumbent, otherwise it is rejected.

## Stop conditions (any one)

- `autoresearch/STOP` exists
- wall clock > `--max-hours` (default 9)
- GPU minutes > `--max-gpu-minutes` (default 480)
- trials >= `--max-trials` (default 12)
- 3 consecutive failed trials
- `--patience` (default 3) trials without a new best
- goal reached: `macro_f1 >= 0.97`, `noise <= 0.005`, `miss <= 0.005`,
  `recall >= 0.99`, `json_valid >= 0.999`

## Sealed set

`autoresearch/data/sealed_test.jsonl` is opened **once**, after the loop stops,
to evaluate the incumbent. It is never read for model selection and never
used for training. `autoresearch/.sealed_opened` records that it was opened.

## Prohibitions

- Do not train on dev or sealed rows.
- Do not edit `training/src/` or `data/`.
- Do not run more than one loop process in this directory (flock).
- Do not change the metric, the gates, or the evaluation prompts.
- Do not change the base model.
- Do not cherry-pick a seed after seeing the result.
- Any edit to a frozen file requires: stop the loop, obtain human approval,
  delete `autoresearch/frozen_hashes.json`, then restart.

## Baseline

`r00_baseline` reproduces the existing pipeline: SFT 2 epochs, lr 2e-5,
batch 2 x accum 8; GRPO 1 epoch, lr 1e-6, beta 0.04, temperature 1.1,
G=8, max completion 200; reward weights format 1.0 / action 1.0 /
channel 0.25 / urgency 0.1 / context 0.0.

The baseline becomes the initial incumbent unconditionally.

## Pre-registered candidate queue

| trial | change | rationale |
|---|---|---|
| `r01_sft_1ep` | SFT 1 epoch | check whether 2 epochs is a lower bound |
| `r02_digest_over2` | digest x2 | baseline errors are mostly digest->silence |
| `r03_digest_focus2` | digest x2 + deep_work/offline x2 | preserve notify while lifting digest |
| `r04_contrastive_300` | r03 + 300 boundary contrastive copies | sharpen the boundary |
| `r05_grpo_hard2` | r03 + hard-prompt boost 2 | more GRPO signal on boundary prompts |
| `r06_grpo_g16_beta02` | r03 + G=16, beta 0.02 | finer advantage estimates |
| `r07_sft_lr1e5` | r03 + SFT lr 1e-5 | reduce SFT over-confidence |
| `r08_digest3_focus` | r03 + digest x3 | stronger digest prior |
| `r09_grpo_lr2e6` | r03 + GRPO lr 2e-6 | faster RL calibration |

## Reporting

On stop the loop writes `autoresearch/FINAL_REPORT.md`, the incumbent recipe to
`autoresearch/accepted/<trial_id>.json`, and one sealed evaluation to
`runs/<trial_id>/sealed_metrics.json`.
