# Evaluation Report

- Mode: **mock**
- Eval set: `data/synthetic/seed_labeled.jsonl` (100 events)
- Note: MOCK VALUES: deterministic mock policies, not trained checkpoints. Re-measure on the AutoDL GPU and replace this evidence.

| Model | JSON valid | Precision (notify) | Recall (notify) | Noise rate | Miss rate | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| base | 81.00% | 0.476 | 0.938 | 33.00% | 6.25% | 0.657 | 65.00% |
| sft | 100.00% | 0.780 | 1.000 | 9.00% | 0.00% | 0.915 | 91.00% |

> Noise rate = wrong notify / all events. Miss rate = missed notify / all should-notify events.
> Error cases are archived in the accompanying report JSON for the next iteration.
