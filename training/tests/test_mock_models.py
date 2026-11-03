"""
Unit tests for the deterministic mock model policies.

The mock policies stand in for the cloud-GPU checkpoints. Tests pin the
properties that make them usable as evidence:
  - deterministic output per (model, event)
  - every effective decision is schema-valid
  - JSON validity rates match the documented mock behaviour
  - strict parsing rejects prose / fenced / extra-key outputs
"""
import json
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))

from mock_models import MODEL_SPECS, mock_output, parse_model_output  # noqa: E402
from schema import validate_decision  # noqa: E402

_LABELED = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "synthetic", "seed_labeled.jsonl",
)


def _events():
    with open(_LABELED, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    return [{k: v for k, v in item.items() if k not in ("decision", "boundary")} for item in items]


def test_mock_output_is_deterministic():
    event = _events()[0]
    for model_id in MODEL_SPECS:
        assert mock_output(model_id, event) == mock_output(model_id, event)


def test_effective_decisions_are_schema_valid():
    events = _events()
    for model_id in MODEL_SPECS:
        for event in events:
            out = mock_output(model_id, event)
            result = validate_decision(out["decision"])
            assert result.valid, (model_id, event["event_id"], result.errors)


def test_json_validity_rates_match_mock_behaviour():
    events = _events()
    rates = {}
    for model_id in MODEL_SPECS:
        valid = sum(1 for event in events if mock_output(model_id, event)["json_valid"])
        rates[model_id] = valid / len(events)

    assert rates["base"] < 0.92
    assert rates["sft"] >= 0.95
    assert rates["sft_grpo"] >= 0.98


def test_parse_rejects_invalid_outputs():
    assert parse_model_output("Sure! Here is my decision:\n{}")[0] is False
    assert parse_model_output("```json\n{\"action\": \"silence\"}\n```")[0] is False
    extra_key = {
        "action": "silence", "channel": "none", "urgency": "low",
        "reason": "r", "message": "", "extra": 1,
    }
    assert parse_model_output(json.dumps(extra_key))[0] is False
    assert parse_model_output("not json")[0] is False


def test_parse_accepts_strict_decision():
    valid = {
        "action": "notify", "channel": "im", "urgency": "urgent",
        "reason": "P0 故障", "message": "请确认处理方案",
    }
    ok, decision = parse_model_output(json.dumps(valid, ensure_ascii=False))
    assert ok is True
    assert decision == valid
