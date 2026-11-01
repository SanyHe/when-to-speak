# Evaluation Report

- Mode: **mock**
- Eval set: `data/synthetic/eval_set.jsonl` (200 events)
- Note: MOCK VALUES: deterministic mock policies, not trained checkpoints. Re-measure on the AutoDL GPU and replace this evidence.

| Model | JSON valid | Precision (notify) | Recall (notify) | Noise rate | Miss rate | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| base | 85.50% | 0.330 | 0.895 | 34.50% | 10.53% | 0.633 | 63.50% |
| sft | 98.50% | 0.714 | 0.921 | 7.00% | 7.89% | 0.898 | 91.50% |
| sft_grpo | 99.00% | 0.841 | 0.974 | 3.50% | 2.63% | 0.950 | 96.00% |

> Noise rate = wrong notify / all events. Miss rate = missed notify / all should-notify events.
> Error cases are archived in the accompanying report JSON for the next iteration.
