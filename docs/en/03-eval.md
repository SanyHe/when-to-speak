# Evaluation: turning "considerate" into numbers

**English** | [简体中文](../03-eval.md)

> Every metric can be recomputed with one command, `python3 training/src/evaluate.py`; the web page accumulates the exact same formulas live.

## 1. Metric definitions

With `notify` as the positive class:

| Metric | Definition | Intuition |
|---|---|---|
| Precision | Share of predicted notify that truly should notify | Accuracy of interruptions |
| Recall | Share of should-notify events correctly notified | Important events are not missed |
| Noise rate | Events notified although they should not be / all events | How often the user is interrupted for nothing |
| Miss rate | Events that should notify but did not / events that should notify | How often something important is dropped |
| Macro-F1 | Arithmetic mean of the three-class F1 | Overall quality under class imbalance |
| JSON validity | Share of outputs that parse strictly and match the schema | Reliability of tool-call formatting |

The core acceptance is not "higher accuracy", but:

> **A significant drop in noise rate without a rise in miss rate.**

## 2. Real results

Eval set: 200 held-out events (`data/synthetic/eval_set.jsonl`), measured with
`--mode hf` on the fine-tuned weights (see `data/eval/eval_report_hf.json`):

| Model | JSON validity | Precision | Recall | Noise rate | Miss rate | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| base | 0.00% | 0.190 | 1.000 | 81.00% | 0.00% | 0.106 | 19.00% |
| SFT | 99.50% | 0.816 | 0.816 | 3.50% | 18.42% | 0.899 | 91.50% |
| SFT + GRPO | 99.50% | 0.837 | 0.947 | 3.50% | 5.26% | 0.924 | 93.00% |

How to read it:

1. **SFT fixes the format first**: JSON validity goes from 0% (base never emits parseable JSON) to 99.5%, clearing the ≥95% M1 bar.
2. **SFT already goes quiet**: noise rate drops from 81.0% to 3.5%, but recall stays at 0.816 — it buys silence too aggressively, missing 18.4% of events that should notify.
3. **GRPO calibrates the boundary**: with noise unchanged at 3.5%, recall rises 0.816 → 0.947 and the miss rate halves from 18.42% to 5.26%, lifting macro-F1 0.899 → 0.924. This is exactly what the asymmetric penalties target (miss -1.0 vs noise -0.5).

The earlier M1 base-vs-SFT comparison on the 100-event seed set is in
[`data/eval/m1_base_vs_sft.md`](../../data/eval/m1_base_vs_sft.md).

### Historical mock pipeline check

Before GPU capacity was available, a deterministic mock policy produced the numbers below solely to validate the data → evaluation → web chain. They are **not** model results and are superseded by the table above.

| Model | JSON validity | Precision | Recall | Noise rate | Miss rate | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| Base | 85.50% | 0.330 | 0.895 | 34.50% | 10.53% | 0.633 | 63.50% |
| SFT | 98.50% | 0.714 | 0.921 | 7.00% | 7.89% | 0.898 | 91.50% |
| SFT + GRPO | 99.00% | 0.841 | 0.974 | 3.50% | 2.63% | 0.950 | 96.00% |

## 3. Error archive

`evaluate.py` writes every non-exact prediction to `data/eval/errors.jsonl` with these fields:

- `model` / `event_id` / `error_type`
- `gold` / `predicted` (full decision)
- `json_valid` / `raw` (raw model output)

| error_type | Meaning | User-facing effect |
|---|---|---|
| `over_notify` | Notified although it should not | Needless interruption |
| `missed_notify` | Should notify but stayed silent / digested | Important event dropped |
| `invalid_json` | Unparseable output (falls back to notify) | Unreliable format |
| `wrong_action` | digest / silence inverted | Wrong summarisation strategy |
| `wrong_channel` | Right action, wrong channel | Notification lands in the wrong place |

That archive is the direct input for the next data and training round: every failing event should become a new eval case.

## 4. Web consistency

The web page (`web/`) reads `web/data/traces.json` and accumulates the same formulas as `compute_metrics()` in the frontend.

- After stepping to the last event, the panel values must match the corresponding report exactly: mock traces against `data/eval/eval_report.json`, real-model traces against `data/eval/eval_report_hf.json`.
- `web/data/traces.json` is a generated artifact. CI reads its `_meta.mode`: under mock it regenerates the traces and diffs them field by field, while under hf (real checkpoints) it cannot reproduce them on a CPU runner and instead verifies that the web metric formulas match `eval_report_hf.json`.

## 5. Real-model evaluation protocol (AutoDL)

After training, run this on the GPU host:

```bash
python3 training/src/evaluate.py --mode hf \
  --eval data/synthetic/eval_set.jsonl \
  --models base=Qwen/Qwen3-0.6B,\
sft=output/full_sft/final,\
sft_grpo=output/grpo/final \
  --out data/eval/eval_report_hf.json \
  --md-out data/eval/eval_report_hf.md \
  --errors-out data/eval/errors_hf.jsonl \
  --web-out web/data/traces.json
```

`output/grpo/` is the released model (second GRPO round: G=16, temperature 1.1, new urgency reward);
the first round (G=8) is kept in `output/grpo_v1/` for comparison — it tied with SFT on the same 200 events.

Then backfill the following into the evidence section of `private/feature_list.json`:

- `eval_report_hf.json` (three-model metrics)
- `errors_hf.jsonl` (error counts and type distribution)
- training logs and hyperparameters (`run_config.json`, `train_log.json`)

## 6. Limitations

- The mock policy only validates the chain, not real model behaviour; all mock numbers were explicitly labelled in the UI and docs.
- The GRPO gain comes from the second round, not the first: round one averaged `frac_reward_zero_std` 0.667
  (two-thirds of prompts had 8 rollouts with identical reward, leaving no advantage) and tied with SFT.
  Raising G to 16, temperature to 1.1 and adding the urgency reward brought that metric down to 0.573,
  with macro-F1 0.924, recall 0.947, miss rate 5.26% and noise unchanged at 3.50%. Single seed, 200 samples:
  a +0.025 macro-F1 may still contain sampling noise.
- With only 200 eval events the per-class sample count is small, so metrics carry sampling variance; more templates would raise confidence.
- Gold labels come from deterministic rules, which are conservative on boundaries such as "indirect but urgent"; `boundary_review.jsonl` keeps the human review path open.
- Strict JSON validity of a real model can be worse than the mock assumption; the harness fallback (parse failure → notify) raises the noise rate, which is exactly what SFT has to fix first.
