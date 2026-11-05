"""
Unit tests for the GRPO reward functions.

These run locally with no GPU: the rewards are pure Python, so the exact
scoring rules used during cloud training are verifiable here.
"""
import json
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))

from rewards import (  # noqa: E402
    ACTION_ADJACENT, ACTION_EXACT, ACTION_MISSED, ACTION_NOISE,
    CHANNEL_EXACT, FORMAT_PARSEABLE, FORMAT_VALID, URGENCY_BONUS,
    action_reward, channel_reward, extract_completion_text, format_reward,
    reward_breakdown, urgency_reward,
)

_SILENCE = {"action": "silence", "channel": "none", "urgency": "low",
            "reason": "重复通知", "message": ""}
_DIGEST = {"action": "digest", "channel": "email", "urgency": "normal",
           "reason": "可汇总", "message": "周报摘要"}
_NOTIFY = {"action": "notify", "channel": "push", "urgency": "urgent",
           "reason": "P0 故障", "message": "请确认处理方案"}


def _as_message(decision):
    return [{"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)}]


def test_extract_completion_text_handles_strings_and_messages():
    assert extract_completion_text('{"action": "silence"}') == '{"action": "silence"}'
    assert extract_completion_text(_as_message(_SILENCE)) == json.dumps(_SILENCE, ensure_ascii=False)
    assert extract_completion_text(None) == ""
    assert extract_completion_text([]) == ""


def test_format_reward_grades_strict_json():
    good = json.dumps(_SILENCE, ensure_ascii=False)
    schema_invalid = json.dumps({k: v for k, v in _SILENCE.items() if k != "message"})
    prose = "Sure! Here is my decision:\n" + good
    fenced = "```json\n" + good + "\n```"

    rewards = format_reward([good, schema_invalid, prose, fenced])
    assert rewards == [FORMAT_VALID, FORMAT_PARSEABLE, 0.0, 0.0]


def test_format_reward_accepts_chat_message_completions():
    assert format_reward([_as_message(_NOTIFY)]) == [FORMAT_VALID]


def test_action_reward_exact_match():
    assert action_reward([_as_message(_NOTIFY)], gold_action=["notify"]) == [ACTION_EXACT]


def test_action_reward_noise_penalty():
    assert action_reward([_as_message(_NOTIFY)], gold_action=["silence"]) == [ACTION_NOISE]


def test_action_reward_miss_penalty_is_harsher_than_noise():
    assert action_reward([_as_message(_SILENCE)], gold_action=["notify"]) == [ACTION_MISSED]
    assert ACTION_MISSED < ACTION_NOISE


def test_action_reward_adjacent_confusion():
    assert action_reward([_as_message(_SILENCE)], gold_action=["digest"]) == [ACTION_ADJACENT]
    assert action_reward([_as_message(_DIGEST)], gold_action=["silence"]) == [ACTION_ADJACENT]


def test_action_reward_zero_on_invalid_json():
    assert action_reward(["not json"], gold_action=["notify"]) == [0.0]


def test_action_reward_falls_back_to_gold_decision():
    rewards = action_reward([_as_message(_DIGEST)], gold_decision=[_DIGEST])
    assert rewards == [ACTION_EXACT]


def test_channel_reward_requires_matching_action_and_channel():
    assert channel_reward([_as_message(_NOTIFY)], gold_decision=[_NOTIFY]) == [CHANNEL_EXACT]
    wrong_channel = {**_NOTIFY, "channel": "email"}
    assert channel_reward([_as_message(wrong_channel)], gold_decision=[_NOTIFY]) == [0.0]
    assert channel_reward([_as_message(_SILENCE)], gold_decision=[_NOTIFY]) == [0.0]


def test_urgency_reward_bonus_requires_matching_action_and_urgency():
    assert urgency_reward([_as_message(_NOTIFY)], gold_decision=[_NOTIFY]) == [URGENCY_BONUS]
    wrong_urgency = {**_NOTIFY, "urgency": "normal"}
    assert urgency_reward([_as_message(wrong_urgency)], gold_decision=[_NOTIFY]) == [0.0]
    # A wrong action never earns the urgency bonus, even with the right urgency.
    assert urgency_reward([_as_message(_SILENCE)], gold_decision=[_NOTIFY]) == [0.0]
    assert urgency_reward(["not json"], gold_decision=[_NOTIFY]) == [0.0]


def test_reward_breakdown_sums_components():
    breakdown = reward_breakdown(json.dumps(_NOTIFY, ensure_ascii=False),
                                 "notify", _NOTIFY)
    assert breakdown["format_reward"] == FORMAT_VALID
    assert breakdown["action_reward"] == ACTION_EXACT
    assert breakdown["channel_reward"] == CHANNEL_EXACT
    assert breakdown["urgency_reward"] == URGENCY_BONUS
    assert breakdown["total"] == 2.35


def test_length_mismatch_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        action_reward([json.dumps(_NOTIFY)], gold_action=["notify", "silence"])
