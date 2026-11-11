# Auto Research — Final Report

- Stop reason: **patience**
- Finished at: 2026-11-11T21:48:19
- Git commit: `92440f83e6071d3970e0b93d992dab48b52b0004`
- Trials run: 7
- GPU minutes: 141.1

## Incumbent

- Trial: `r03_digest_focus2` (seed 42)
- Dev (450): macro-F1 **0.9943**, recall 1.0000, precision 1.0000, noise 0.0000, miss 0.0000, json 0.9978
- Sealed (200): macro-F1 **0.9828**, recall 1.0000, precision 1.0000, noise 0.0000, miss 0.0000, json 0.9950

## Best recipe

```json
{
  "name": "r03_digest_focus2",
  "sft": {
    "epochs": 2,
    "batch_size": 2,
    "grad_accum_steps": 8,
    "learning_rate": 2e-05,
    "seed": 42
  },
  "data": {
    "focus_oversample": {
      "deep_work": 2,
      "offline": 2
    },
    "action_oversample": {
      "digest": 2
    },
    "contrastive_limit": 0,
    "grpo_focus_oversample": {}
  },
  "grpo": {
    "enabled": true,
    "epochs": 1,
    "batch_size": 2,
    "grad_accum_steps": 4,
    "learning_rate": 1e-06,
    "beta": 0.04,
    "temperature": 1.1,
    "num_generations": 8,
    "max_completion_length": 200,
    "seed": 42,
    "prompt_mode": "all",
    "hard_boost": 1,
    "reward_weights": {
      "format": 1.0,
      "action": 1.0,
      "channel": 0.25,
      "urgency": 0.1,
      "context": 0.0
    }
  }
}
```

## Trial summary

| trial | decision | macro-F1 | recall | noise | miss | json | GPU min |
|---|---|---|---|---|---|---|---|
| r00_baseline | accepted_baseline | 0.9264 | 0.9043 | 0.0200 | 0.0957 | 0.9978 | 14.9 |
| r01_sft_1ep | rejected | 0.8225 | 1.0000 | 0.0400 | 0.0000 | 1.0000 | 14.9 |
| r02_digest_over2 | rejected | 0.8971 | 0.6809 | 0.0000 | 0.3191 | 1.0000 | 16.1 |
| r03_digest_focus2 | accepted | 0.9943 | 1.0000 | 0.0000 | 0.0000 | 0.9978 | 17.7 |
| r04_contrastive_300 | rejected | 0.9904 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 18.0 |
| r05_grpo_hard2 | rejected | 0.9904 | 1.0000 | 0.0000 | 0.0000 | 0.9978 | 20.3 |
| r06_grpo_g16_beta02 | rejected | 0.9904 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 21.1 |
