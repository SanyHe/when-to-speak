"""
Tests for the evaluation harness (F1.5 / F2.5).

Covers:
  - metric math on hand-built records
  - the mock model comparison satisfies the M1/M2 acceptance direction
    (SFT JSON validity >= 95%; GRPO noise down and miss rate not up)
  - web traces are consistent with the reported metrics
"""
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))

from evaluate import (  # noqa: E402
    build_web_traces, compute_metrics, evaluate_mock, load_labeled,
)

_EVAL = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "synthetic", "eval_set.jsonl",
)

_ACTIONS = ("notify", "digest", "silence")


def _decision(action):
    if action == "silence":
        return {"action": "silence", "channel": "none", "urgency": "low", "reason": "r", "message": ""}
    return {
        "action": action,
        "channel": "im",
        "urgency": "urgent" if action == "notify" else "normal",
        "reason": "r",
        "message": "m",
    }


def _record(gold, pred, json_valid=True):
    return {
        "event_id": "evt",
        "gold": _decision(gold),
        "pred": _decision(pred),
        "raw": "{}",
        "json_valid": json_valid,
    }


def test_metric_math_on_known_records():
    records = [
        _record("notify", "notify"),
        _record("notify", "silence"),
        _record("digest", "notify"),
        _record("silence", "silence"),
    ]
    m = compute_metrics(records)
    assert m["total"] == 4
    assert m["notify_precision"] == 0.5     # 1 TP / 2 predicted notify
    assert m["notify_recall"] == 0.5        # 1 TP / 2 should-notify
    assert m["noise_rate"] == 0.25          # 1 wrong notify / 4 events
    assert m["miss_rate"] == 0.5            # 1 missed / 2 should-notify
    assert m["accuracy"] == 0.5
    assert m["json_valid_rate"] == 1.0


def test_invalid_json_lowers_validity_rate():
    records = [_record("notify", "notify", json_valid=False)]
    assert compute_metrics(records)["json_valid_rate"] == 0.0


def test_mock_comparison_satisfies_acceptance_direction():
    items = load_labeled(_EVAL)
    results = evaluate_mock(items, ["base", "sft", "sft_grpo"])
    metrics = {model: compute_metrics(records) for model, records in results.items()}

    assert metrics["sft"]["json_valid_rate"] >= 0.95
    assert metrics["sft"]["noise_rate"] < metrics["base"]["noise_rate"]
    assert metrics["sft_grpo"]["noise_rate"] < metrics["sft"]["noise_rate"]
    assert metrics["sft_grpo"]["miss_rate"] <= metrics["sft"]["miss_rate"]


def test_web_traces_match_metrics():
    items = load_labeled(_EVAL)
    results = evaluate_mock(items, ["base", "sft", "sft_grpo"])
    traces = build_web_traces(items, results, "mock")

    assert len(traces["events"]) == len(items)
    assert [m["id"] for m in traces["models"]] == ["base", "sft", "sft_grpo"]
    for model_id, records in results.items():
        assert traces["metrics"][model_id] == compute_metrics(records)

    first = traces["events"][0]
    assert first["pred"]["base"]["json_valid"] in (True, False)
    assert first["gold"]["action"] in _ACTIONS
