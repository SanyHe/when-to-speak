# AGENTS.md — When to Speak

> This is the harness entry file. Read it first at the start of every session.
> It defines how the agent works on this repo. Following these rules is what
> makes the work reliable across sessions.

## 1. Startup Workflow (do this before writing any code)

1. Read [`private/claude-progress.md`](./private/claude-progress.md) — find the **Current Verified State** and the **Next best action**.
2. Read [`private/feature_list.json`](./private/feature_list.json) — identify the feature currently `in_progress` or the highest-priority `not_started` feature. Only **one** feature may be `in_progress` at a time.
3. Read the sections relevant to that feature in [`docs/02-method.md`](./docs/02-method.md)
   (data contract, rules, training) and [`docs/03-eval.md`](./docs/03-eval.md) (metrics, milestones).
4. Confirm the environment with `./init.sh` (installs dev deps, runs verification). If verification fails, fix the baseline before doing anything else.

> Harness state files (`private/claude-progress.md`, `private/feature_list.json`, `private/PLAN.md`)
> live in `private/` — they are gitignored and never published to GitHub. A fresh clone will not
> have them; they are local working state for the maintainer's agent sessions.

## 2. Working Rules

- **One feature at a time.** Never start a new feature while another is `in_progress`.
- **Evidence before marking `passing`.** A feature is `passing` only when:
  - Its `verification` steps have been run and recorded in `evidence`.
  - The output is reproducible (fixed seeds, deterministic scripts).
  - No other feature regressed (full `pytest training/tests/` still green).
- **Stop agents from declaring victory too early.** If you cannot run the verification (e.g. no GPU for training), mark the feature `blocked` with a documented reason — do not mark it `passing`.
- **Strict milestone order.** M1 must pass before M2; M2 before M3. Do not skip ahead.
- **Scope discipline.** Implement only the selected feature. Do not pre-build M2/M3 infrastructure while in M1.
- **Code comments in English.** All source comments must be in English.
- **No secrets in the repo.** Tokens/keys use environment variables only. Synthetic data only; real data stays in `private/` (gitignored).

## 3. Definition of Done for a Feature

A feature is done when all of the following are true:

1. `private/feature_list.json` status is `passing` and `evidence` is filled with concrete, reproducible proof (test output, file paths, metrics).
2. `private/claude-progress.md` has a new session record with: Goal, Completed, Verification run, Evidence recorded, Known risks, Next best action.
3. `pytest training/tests/` passes (for any feature touching training/data code).
4. No half-finished work is left uncommitted or unrecorded.

## 4. Project Commands & Paths

| Purpose | Command |
|---|---|
| Install dev deps | `pip install jsonschema pytest` |
| Verify (tests) | `python3 -m pytest training/tests/ -v` |
| Regenerate seed events | `python3 training/src/gen_events.py` |
| Label events (expert) | `python3 training/src/expert_rules.py` |
| Format SFT data | `python3 training/src/format_sft.py` |
| Full SFT (cloud only) | see [`docs/cloud-training.md`](./docs/cloud-training.md) §3 |
| GRPO RL (cloud only) | see [`docs/cloud-training.md`](./docs/cloud-training.md) §4 |
| Full pipeline (cloud only) | `./training/run_all.sh`（数据集 → SFT → GRPO → 评测） |
| Serve inference (cloud only) | `./training/src/serve_vllm.sh`（vLLM OpenAI 兼容 API） |

Key paths:
- Schemas & validation: [`training/src/schema.py`](./training/src/schema.py)
- Rule expert: [`training/src/expert_rules.py`](./training/src/expert_rules.py)
- Data generator: [`training/src/gen_events.py`](./training/src/gen_events.py)
- Full-parameter SFT: [`training/src/full_sft.py`](./training/src/full_sft.py)
- GRPO training: [`training/src/grpo.py`](./training/src/grpo.py)
- GRPO rewards: [`training/src/rewards.py`](./training/src/rewards.py)
- Synthetic data: [`data/synthetic/`](./data/synthetic/)

## 5. Training Compute

Training runs on a **domestic cloud GPU** (AutoDL / 火山引擎 / 阿里云 PAI).
It does **not** run locally. The pipeline is **full-parameter SFT**
(no LoRA / no quantisation) followed by **GRPO** reinforcement learning.
See [`docs/cloud-training.md`](./docs/cloud-training.md) for the
provisioning and runbook. Local work is limited to: schema, data generation, expert
rules, data formatting, reward functions, and tests.

## 6. End-of-Session Checklist

Before ending a session:

- [ ] `./init.sh` verification still passes.
- [ ] `private/feature_list.json` reflects the true state (no false `passing`).
- [ ] `private/claude-progress.md` is updated with the session record and next best action.
- [ ] No `in_progress` feature is left half-done without a recorded handoff.
- [ ] Any blocker is documented in the feature's `notes` and in `private/claude-progress.md`.
