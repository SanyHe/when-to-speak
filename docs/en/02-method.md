# Method: capability = model × harness × eval

**English** | [简体中文](../02-method.md)

> How "When to Speak" splits post-training into an acceptance-checkable loop using harness engineering.

## 1. Why a harness

The usual failure mode of post-training is not "the model is bad", it is a process out of control: data that cannot be reproduced, results that cannot be verified, web numbers that disagree with the evaluation report, last round's conclusions that cannot be handed over.

So the repo builds the harness before writing model code:

- [`AGENTS.md`](../../AGENTS.md) — working rules and definition of done
- [`harness.yaml`](../../harness.yaml) — centralised guardrails
- `init.sh` — one command to install dependencies and run verification

> Harness state files such as the feature list and progress notes live in `private/` (gitignored, not published).

The rules are plain: **one feature at a time; nothing is marked passing without reviewable evidence; strictly M1 → M2 → M3.**

## 2. Data contract first

`training/src/schema.py` defines the JSON Schemas for `event` and `decision`:

- Fixed fields, `additionalProperties: false`, no extra keys.
- Timestamps must be ISO-8601 with a timezone.
- Cross-field rules live in the schema via `if/then/else`: when action is `silence`, channel must be `none` and message must be empty.

That turns "valid JSON output" into a unit-testable hard contract rather than a post-training hope.

## 3. Synthetic data and the rule expert

This stage uses no real data. It comes from 38 event templates (`data/templates/`), covering 5 sources × 3 actions:

1. **Deterministic rule expert** (`expert_rules.py`) produces gold labels: duplicates / marketing / routine automation → silence; time-sensitive and direct → notify; everything else → digest; during deep work, non-urgent notify is downgraded to digest.
2. **Template combinations** enumerate `(sender, subject, snippet)` tuples as the unit of near-duplicate splitting, so eval and training never share a combination.
3. **Boundary samples** are exported to `boundary_review.jsonl`; after human review they flow back as gold through `--reviews`, closing the data iteration loop.
4. **Fixed random seed**, the whole dataset rebuilds with one command: `python3 training/src/build_dataset.py`.

Scale (configurable):

| Data | Size | Purpose |
|---|---|---|
| SFT | 800 | Expert demonstrations: format + basic decisions |
| GRPO prompts | 300 | Reinforcement learning: each event carries a gold decision for the reward functions |
| Eval | 200 | Held-out evaluation, never trained on |

## 4. Training

- Base model: Qwen3-0.6B; **full-parameter fine-tuning** (no LoRA, no 4-bit quantization, no Unsloth), with SFT and RL sharing the same parameter layout.
- **SFT**: input event + instruction, output the gold decision; loss is computed on the assistant span only (prompt masked to -100), teaching strict JSON and the three actions first.
- **GRPO**: reinforcement learning on the full-SFT checkpoint. Each prompt samples G decisions, scored by deterministic rule rewards:
  - Format reward: strict schema pass +1.0, parseable JSON only +0.5, otherwise 0;
  - Action reward: correct action +1.0; unnecessary notify -0.5; missed notify -1.0; digest↔silence confusion -0.15;
  - Channel reward: correct action and channel +0.25;
  - Urgency reward: correct action and urgency +0.10 (format and channel rewards saturate after SFT, so a finer signal is needed to separate samples of the same prompt).
  The miss penalty is twice the noise penalty, so noise drops without sacrificing recall. Rules live in `training/src/rewards.py` and are unit-tested locally.
- Training runs only on a **cloud GPU in China** (single AutoDL card), never locally; hyperparameters, seeds, logs and checkpoints are all persisted.

Runbook: [`cloud-training.md`](cloud-training.md).

## 5. Evaluation

base / SFT / SFT+GRPO are compared on the same held-out events:

- Precision / recall (notify as the positive class)
- Noise rate (should not notify but did / all events)
- Miss rate (should notify but did not / events that should notify)
- Three-class macro-F1, JSON validity, confusion matrix
- Every non-exact prediction is archived to `errors.jsonl` as input for the next data and training round

Metric definitions and current results: [`03-eval.md`](03-eval.md).

## 6. Web delivery

- `web/` is a pure static site reading `web/data/traces.json` (events + gold + per-model decisions + metrics), replaying them one by one and accumulating stats live.
- Three-model switching, single step / autoplay / reset, noise-rate and miss-rate panels.
- Real models plug in through the vLLM service on AutoDL (OpenAI-compatible API) called directly from the page; by default it replays mock traces offline with zero network requests.
- Earlier stages used **deterministic mock traces** to validate the whole chain; every mock number is labelled as such in the UI and docs, and the real weights replaced them after training.

## 7. Known deviations

The harness default is "enter M2 only after M1 passes". Before GPU capacity was available, this project completed the GPU-independent parts (data pipeline, evaluator, web page) and validated the chain with mock decisions; training features stayed `blocked` rather than being faked as `passing`. That deviation is recorded in the session log in `private/claude-progress.md`.
