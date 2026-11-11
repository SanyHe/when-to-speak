"""The single editable surface of the Auto Research loop.

Only values in this file may differ between trials. The training scripts,
gold data, metric implementation, and evaluation sets are frozen.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List


BASELINE: Dict[str, Any] = {
    "name": "r00_baseline",
    "sft": {
        "epochs": 2,
        "batch_size": 2,
        "grad_accum_steps": 8,
        "learning_rate": 2e-5,
        "seed": 42,
    },
    "data": {
        "focus_oversample": {},
        "action_oversample": {},
        "contrastive_limit": 0,
        "grpo_focus_oversample": {},
    },
    "grpo": {
        "enabled": True,
        "epochs": 1,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "learning_rate": 1e-6,
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
            "context": 0.0,
        },
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _recipe(name: str, **overrides: Any) -> Dict[str, Any]:
    recipe = _merge(BASELINE, overrides)
    recipe["name"] = name
    return recipe


CANDIDATES: List[Dict[str, Any]] = [
    _recipe("r01_sft_1ep", sft={"epochs": 1}),
    _recipe("r02_digest_over2", data={"action_oversample": {"digest": 2}}),
    _recipe(
        "r03_digest_focus2",
        data={
            "action_oversample": {"digest": 2},
            "focus_oversample": {"deep_work": 2, "offline": 2},
        },
    ),
    _recipe(
        "r04_contrastive_300",
        data={
            "action_oversample": {"digest": 2},
            "focus_oversample": {"deep_work": 2, "offline": 2},
            "contrastive_limit": 300,
        },
    ),
    _recipe(
        "r05_grpo_hard2",
        data={
            "action_oversample": {"digest": 2},
            "focus_oversample": {"deep_work": 2, "offline": 2},
        },
        grpo={"hard_boost": 2},
    ),
    _recipe(
        "r06_grpo_g16_beta02",
        data={
            "action_oversample": {"digest": 2},
            "focus_oversample": {"deep_work": 2, "offline": 2},
        },
        grpo={"num_generations": 16, "beta": 0.02},
    ),
    _recipe(
        "r07_sft_lr1e5",
        sft={"learning_rate": 1e-5},
        data={
            "action_oversample": {"digest": 2},
            "focus_oversample": {"deep_work": 2, "offline": 2},
        },
    ),
    _recipe(
        "r08_digest3_focus",
        data={
            "action_oversample": {"digest": 3},
            "focus_oversample": {"deep_work": 2, "offline": 2},
        },
    ),
    _recipe(
        "r09_grpo_lr2e6",
        data={
            "action_oversample": {"digest": 2},
            "focus_oversample": {"deep_work": 2, "offline": 2},
        },
        grpo={"learning_rate": 2e-6},
    ),
]

# Declared mutation space. The loop runs the pre-registered queue; this table
# documents which values a future round may explore.
SEARCH_SPACE: Dict[str, Any] = {
    "sft.epochs": [1, 2, 3],
    "sft.learning_rate": [1e-5, 2e-5, 3e-5],
    "data.action_oversample": {"notify": [1, 2], "digest": [1, 2, 3], "silence": [1, 2]},
    "data.focus_oversample": {"deep_work": [1, 2], "available": [1, 2], "offline": [1, 2]},
    "data.contrastive_limit": [0, 150, 300],
    "grpo.num_generations": [8, 16],
    "grpo.beta": [0.02, 0.04, 0.08],
    "grpo.learning_rate": [1e-6, 2e-6],
    "grpo.hard_boost": [1, 2],
    "grpo.reward_weights": {
        "format": [0.5, 1.0],
        "action": [1.0, 1.5],
        "channel": [0.0, 0.25],
        "urgency": [0.0, 0.1],
        "context": [0.0, 0.1],
    },
}
