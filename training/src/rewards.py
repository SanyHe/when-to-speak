"""
GRPO reward functions (F2.4).

Pure-Python and dependency-free (only the JSON schema helpers), so the
reward logic can be unit-tested locally while the GRPO run itself needs a
cloud GPU.

Reward is the sum of three functions (TRL sums every function in
``reward_funcs``):

    format_reward   1.0 strict schema-valid JSON | 0.5 parseable JSON
                    object that violates the schema | 0.0 otherwise
    action_reward   +1.0 exact action | -0.5 noisy notify | -1.0 missed
                    notify | -0.15 digest<->silence confusion | 0.0 if the
                    completion cannot be parsed
    channel_reward  +0.25 when both action and channel match the gold

The asymmetric penalties encode the product rule: a missed urgent event
costs twice as much as a needless notification, so GRPO is pushed to keep
recall while reducing noise.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

from schema import ACTIONS, validate_decision  # noqa: E402

FORMAT_VALID = 1.0
FORMAT_PARSEABLE = 0.5
ACTION_EXACT = 1.0
ACTION_NOISE = -0.5
ACTION_MISSED = -1.0
ACTION_ADJACENT = -0.15
CHANNEL_EXACT = 0.25


def extract_completion_text(completion: Any) -> str:
    """Return the assistant text from a string or a chat-message list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if not completion:
            return ""
        last = completion[-1]
        if isinstance(last, dict):
            return "" if last.get("content") is None else str(last["content"])
        return str(last)
    return "" if completion is None else str(completion)


def _json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse ``text`` as a single JSON object, else return None."""
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def parse_decision(text: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Strictly parse a decision: schema-valid JSON object only."""
    obj = _json_object(text)
    if obj is None:
        return False, None
    result = validate_decision(obj)
    if not result.valid:
        return False, None
    return True, obj


def _aligned(values: Optional[List[Any]], count: int, name: str) -> List[Any]:
    """Return a per-completion list; raise on a length mismatch."""
    if values is None:
        return [None] * count
    if len(values) != count:
        raise ValueError(f"{name} has {len(values)} values for {count} completions")
    return list(values)


def _gold_action_of(action: Any, decision: Any) -> Optional[str]:
    if action in ACTIONS:
        return action
    if isinstance(decision, dict):
        candidate = decision.get("action")
        if candidate in ACTIONS:
            return candidate
    return None


def format_reward(completions: List[Any], **kwargs: Any) -> List[float]:
    """Reward strict JSON output (partial credit for parseable JSON)."""
    rewards: List[float] = []
    for completion in completions:
        text = extract_completion_text(completion)
        if parse_decision(text)[0]:
            rewards.append(FORMAT_VALID)
        elif _json_object(text) is not None:
            rewards.append(FORMAT_PARSEABLE)
        else:
            rewards.append(0.0)
    return rewards


def action_reward(completions: List[Any], gold_action: Optional[List[str]] = None,
                  gold_decision: Optional[List[Dict[str, Any]]] = None,
                  **kwargs: Any) -> List[float]:
    """Reward the three-way action decision with safety asymmetry."""
    count = len(completions)
    actions = _aligned(gold_action, count, "gold_action")
    decisions = _aligned(gold_decision, count, "gold_decision")

    rewards: List[float] = []
    for index, completion in enumerate(completions):
        valid, pred = parse_decision(extract_completion_text(completion))
        gold = _gold_action_of(actions[index], decisions[index])
        if not valid or gold is None:
            rewards.append(0.0)
            continue
        if pred["action"] == gold:
            rewards.append(ACTION_EXACT)
        elif pred["action"] == "notify":
            rewards.append(ACTION_NOISE)
        elif gold == "notify":
            rewards.append(ACTION_MISSED)
        else:
            rewards.append(ACTION_ADJACENT)
    return rewards


def channel_reward(completions: List[Any],
                   gold_decision: Optional[List[Dict[str, Any]]] = None,
                   gold_action: Optional[List[str]] = None,
                   **kwargs: Any) -> List[float]:
    """Reward the delivery channel when the action is already correct."""
    count = len(completions)
    decisions = _aligned(gold_decision, count, "gold_decision")
    actions = _aligned(gold_action, count, "gold_action")

    rewards: List[float] = []
    for index, completion in enumerate(completions):
        valid, pred = parse_decision(extract_completion_text(completion))
        gold_decision = decisions[index]
        if not valid or not isinstance(gold_decision, dict):
            rewards.append(0.0)
            continue
        if pred["action"] != gold_decision.get("action"):
            rewards.append(0.0)
            continue
        rewards.append(CHANNEL_EXACT if pred["channel"] == gold_decision.get("channel") else 0.0)
    return rewards


REWARD_FUNCS = [format_reward, action_reward, channel_reward]


def reward_breakdown(completion: Any, gold_action: str,
                     gold_decision: Dict[str, Any]) -> Dict[str, float]:
    """Score one completion and return every component plus the total."""
    parts = {
        func.__name__: func(
            [completion], gold_action=[gold_action], gold_decision=[gold_decision],
        )[0]
        for func in REWARD_FUNCS
    }
    return {**parts, "total": round(sum(parts.values()), 4)}
