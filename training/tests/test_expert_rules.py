"""
Unit tests for the rule-based expert (F1.3).

Verifies:
  - every emitted decision passes the decision schema
  - determinism: same event -> same decision
  - key rule branches: duplicate -> silence, marketing -> silence,
    urgent+directed -> notify, deep_work downgrade
  - boundary flag is correct for clear vs ambiguous events
"""
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))

from expert_rules import decide, is_boundary  # noqa: E402
from schema import validate_decision  # noqa: E402


def _event(**overrides):
    base = {
        "event_id": "evt_x",
        "source": "email",
        "sender": "a@b.com",
        "subject": "hello",
        "snippet": "world",
        "mentions_user": False,
        "is_direct": False,
        "time_sensitive": False,
        "received_at": "2026-10-31T09:00:00+08:00",
        "context": {
            "user_focus": "available",
            "recent_notifications": 0,
            "duplicate_of": None,
        },
    }
    # Merge context overrides into the base context before applying the rest,
    # so unspecified context sub-fields keep their defaults.
    ctx_override = overrides.pop("context", None)
    if isinstance(ctx_override, dict):
        base["context"] = {**base["context"], **ctx_override}
    base.update(overrides)
    return base


def test_decision_always_schema_valid():
    cases = [
        _event(),
        _event(source="im", mentions_user=True, is_direct=True, time_sensitive=True),
        _event(source="email", subject="限时促销", snippet="全场五折"),
        _event(context={"duplicate_of": "evt_0"}),
        _event(source="task", time_sensitive=True),
        _event(source="routine", context={"user_focus": "deep_work"}),
    ]
    for e in cases:
        d = decide(e)
        r = validate_decision(d)
        assert r.valid, (e["event_id"], r.errors)


def test_determinism():
    e = _event(source="im", mentions_user=True, is_direct=True, time_sensitive=True)
    assert decide(e) == decide(e)


def test_duplicate_is_silence():
    e = _event(context={"duplicate_of": "evt_0"})
    d = decide(e)
    assert d["action"] == "silence"
    assert d["channel"] == "none"
    assert d["message"] == ""


def test_marketing_undirected_is_silence():
    e = _event(subject="限时优惠", snippet="全场五折抢购",
               mentions_user=False, is_direct=False, time_sensitive=False)
    assert decide(e)["action"] == "silence"


def test_marketing_directed_is_not_silence():
    # Marketing but directly addressed to the user -> not silence (notify/digest).
    e = _event(subject="限时优惠", snippet="全场五折",
               mentions_user=True, is_direct=True, time_sensitive=False)
    assert decide(e)["action"] != "silence"


def test_urgent_and_directed_is_notify():
    e = _event(source="email", mentions_user=True, is_direct=True,
               time_sensitive=True)
    d = decide(e)
    assert d["action"] == "notify"
    assert d["urgency"] == "urgent"


def test_direct_personal_message_is_notify():
    e = _event(source="im", mentions_user=True, is_direct=True,
               time_sensitive=False)
    assert decide(e)["action"] == "notify"


def test_automated_routine_undirected_is_silence():
    e = _event(source="routine", mentions_user=False, is_direct=False,
               time_sensitive=False)
    assert decide(e)["action"] == "silence"


def test_deep_work_downgrades_non_urgent_notify_to_digest():
    # direct personal message (notify, normal) during deep_work -> digest
    e = _event(source="im", mentions_user=True, is_direct=True,
               time_sensitive=False, context={"user_focus": "deep_work"})
    d = decide(e)
    assert d["action"] == "digest"


def test_deep_work_keeps_urgent_notify():
    # urgent notify stays notify even in deep_work
    e = _event(source="email", mentions_user=True, is_direct=True,
               time_sensitive=True, context={"user_focus": "deep_work"})
    assert decide(e)["action"] == "notify"


def test_time_sensitive_undirected_is_digest():
    e = _event(source="task", time_sensitive=True,
               mentions_user=False, is_direct=False)
    assert decide(e)["action"] == "digest"


def test_boundary_flag_for_clear_vs_ambiguous():
    # Clear silence (duplicate) -> not boundary
    assert not is_boundary(_event(context={"duplicate_of": "evt_0"}))
    # Clear notify (urgent + directed) -> not boundary
    assert not is_boundary(_event(mentions_user=True, is_direct=True,
                                  time_sensitive=True))
    # Ambiguous (time-sensitive but undirected) -> boundary
    assert is_boundary(_event(source="task", time_sensitive=True,
                              mentions_user=False, is_direct=False))
