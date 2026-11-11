"""Weighted GRPO reward factory.

Five components: format / action / channel / urgency / context. Each is the
frozen implementation from ``training/src/rewards.py`` (except ``context``,
which is new) multiplied by the weight declared in ``recipe.py``.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(REPO_ROOT, "training", "src")
sys.path.insert(0, SRC)

from rewards import (  # noqa: E402
    action_reward,
    channel_reward,
    extract_completion_text,
    format_reward,
    parse_decision,
    urgency_reward,
)

BASE_REWARDS = {
    "format": format_reward,
    "action": action_reward,
    "channel": channel_reward,
    "urgency": urgency_reward,
}


def context_reward(completions: List[Any],
                   user_focus: Optional[List[str]] = None,
                   duplicate_of: Optional[List[Optional[str]]] = None,
                   gold_action: Optional[List[str]] = None,
                   gold_decision: Optional[List[Dict[str, Any]]] = None,
                   **kwargs: Any) -> List[float]:
    """Reward decisions that respect the event context.

    +1 when the action matches gold and the context rule holds:
      - duplicate_of set  -> silence
      - user_focus deep_work -> not notify
      - otherwise -> neutral (1.0)
    """
    count = len(completions)
    focuses = user_focus if user_focus is not None else [None] * count
    duplicates = duplicate_of if duplicate_of is not None else [None] * count
    actions = gold_action if gold_action is not None else [None] * count
    decisions = gold_decision if gold_decision is not None else [None] * count
    if not (len(focuses) == len(duplicates) == len(actions) == len(decisions) == count):
        raise ValueError("context_reward received misaligned columns")

    rewards: List[float] = []
    for index, completion in enumerate(completions):
        valid, pred = parse_decision(extract_completion_text(completion))
        gold = actions[index]
        if gold is None and isinstance(decisions[index], dict):
            gold = decisions[index].get("action")
        if not valid or gold is None or pred["action"] != gold:
            rewards.append(0.0)
            continue
        if duplicates[index] is not None:
            rewards.append(1.0 if pred["action"] == "silence" else 0.0)
        elif focuses[index] == "deep_work":
            rewards.append(1.0 if pred["action"] != "notify" else 0.0)
        else:
            rewards.append(1.0)
    return rewards


def _scaled(func, weight: float):
    def wrapped(completions, **kwargs):
        return [weight * value for value in func(completions, **kwargs)]

    wrapped.__name__ = f"{func.__name__}_x{weight:g}"
    wrapped.__doc__ = f"{func.__doc__} (weight={weight})"
    return wrapped


def build_weighted_rewards(weights: Dict[str, float]) -> List[Any]:
    """Return one weighted callable per non-zero reward component."""
    funcs = dict(BASE_REWARDS)
    funcs["context"] = context_reward
    unknown = set(weights) - set(funcs)
    if unknown:
        raise ValueError(f"unknown reward weights: {sorted(unknown)}")
    reward_funcs = []
    for name, func in funcs.items():
        weight = float(weights.get(name, 0.0))
        if weight != 0.0:
            reward_funcs.append(_scaled(func, weight))
    if not reward_funcs:
        raise ValueError("at least one reward weight must be non-zero")
    return reward_funcs
