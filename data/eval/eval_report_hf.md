# Evaluation Report

- Mode: **hf**
- Eval set: `data/synthetic/eval_set.jsonl` (200 events)
- Note: REAL VALUES: measured on cloud GPU checkpoints.

| Model | JSON valid | Precision (notify) | Recall (notify) | Noise rate | Miss rate | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| base | 0.00% | 0.190 | 1.000 | 81.00% | 0.00% | 0.106 | 19.00% |
| sft | 99.50% | 0.816 | 0.816 | 3.50% | 18.42% | 0.899 | 91.50% |
| sft_grpo | 99.50% | 0.837 | 0.947 | 3.50% | 5.26% | 0.924 | 93.00% |

> Noise rate = wrong notify / all events. Miss rate = missed notify / all should-notify events.
> Error cases are archived in the accompanying report JSON for the next iteration.
