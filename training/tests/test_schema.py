"""
Unit tests for the data contract (event / decision schemas).

Covers the acceptance criteria for F1.1:
  - a group of valid samples passes validation
  - a group of invalid samples fails with meaningful error paths
  - cross-field business rules (silence <-> channel/message) are enforced
"""
import os
import sys

# Make training/src importable regardless of the pytest invocation cwd.
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))

from schema import (  # noqa: E402
    ACTIONS,
    CHANNELS,
    DECISION_SCHEMA,
    EVENT_SCHEMA,
    SOURCES,
    URGENCIES,
    USER_FOCUS_STATES,
    validate_decision,
    validate_event,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def make_event(**overrides):
    """Return a fully-valid event, with optional field overrides."""
    base = {
        "event_id": "evt_0001",
        "source": "email",
        "sender": "boss@company.com",
        "subject": "线上故障需要你确认",
        "snippet": "生产环境 P0 告警，需要你 10 分钟内确认处理方案。",
        "mentions_user": True,
        "is_direct": True,
        "time_sensitive": True,
        "received_at": "2026-10-31T09:12:00+08:00",
        "context": {
            "user_focus": "deep_work",
            "recent_notifications": 2,
            "duplicate_of": None,
        },
    }
    base.update(overrides)
    return base


def make_decision(**overrides):
    """Return a fully-valid decision, with optional field overrides."""
    base = {
        "action": "notify",
        "channel": "im",
        "urgency": "urgent",
        "reason": "P0 生产故障且直接点名，时效性强",
        "message": "收到 P0 告警，需要你在 10 分钟内确认处理方案。",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Enum / schema sanity
# ---------------------------------------------------------------------------

def test_enum_values_match_plan():
    assert SOURCES == ("email", "im", "task", "routine", "system")
    assert USER_FOCUS_STATES == ("deep_work", "available", "offline")
    assert ACTIONS == ("notify", "digest", "silence")
    assert CHANNELS == ("im", "email", "push", "none")
    assert URGENCIES == ("urgent", "normal", "low")


def test_schemas_are_strict_no_extra_fields():
    assert EVENT_SCHEMA["additionalProperties"] is False
    assert DECISION_SCHEMA["additionalProperties"] is False
    assert EVENT_SCHEMA["properties"]["context"]["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Valid event cases
# ---------------------------------------------------------------------------

def test_valid_event_passes():
    result = validate_event(make_event())
    assert result.valid, result.errors


def test_valid_event_with_duplicate_of_string_passes():
    event = make_event(context={
        "user_focus": "available",
        "recent_notifications": 0,
        "duplicate_of": "evt_0000",
    })
    assert validate_event(event).valid


def test_valid_event_all_sources_and_focus_states():
    for src in SOURCES:
        for focus in USER_FOCUS_STATES:
            event = make_event(source=src, context={
                "user_focus": focus,
                "recent_notifications": 5,
                "duplicate_of": None,
            })
            assert validate_event(event).valid, (src, focus)


def test_valid_event_zulu_timezone_passes():
    assert validate_event(make_event(received_at="2026-10-31T01:12:00Z")).valid


def test_valid_event_fractional_seconds_passes():
    assert validate_event(
        make_event(received_at="2026-10-31T09:12:00.123+08:00")
    ).valid


# ---------------------------------------------------------------------------
# Invalid event cases
# ---------------------------------------------------------------------------

def test_event_missing_required_field_fails():
    event = make_event()
    del event["event_id"]
    result = validate_event(event)
    assert not result.valid
    assert any("event_id" in e for e in result.errors)


def test_event_missing_context_subfield_fails():
    ctx = {"user_focus": "deep_work", "recent_notifications": 0}  # no duplicate_of
    result = validate_event(make_event(context=ctx))
    assert not result.valid
    assert any("duplicate_of" in e for e in result.errors)


def test_event_wrong_type_fails():
    result = validate_event(make_event(event_id=123))
    assert not result.valid


def test_event_invalid_source_enum_fails():
    result = validate_event(make_event(source="slack"))
    assert not result.valid


def test_event_invalid_focus_enum_fails():
    ctx = {"user_focus": "dnd", "recent_notifications": 0, "duplicate_of": None}
    assert not validate_event(make_event(context=ctx)).valid


def test_event_extra_field_fails():
    result = validate_event(make_event(extra="nope"))
    assert not result.valid


def test_event_extra_context_field_fails():
    ctx = {
        "user_focus": "deep_work",
        "recent_notifications": 0,
        "duplicate_of": None,
        "bogus": 1,
    }
    assert not validate_event(make_event(context=ctx)).valid


def test_event_negative_recent_notifications_fails():
    ctx = {"user_focus": "deep_work", "recent_notifications": -1, "duplicate_of": None}
    assert not validate_event(make_event(context=ctx)).valid


def test_event_bad_timestamp_format_fails():
    for bad_ts in ["2026-10-31", "not-a-time", "2026-10-31T09:12:00", ""]:
        result = validate_event(make_event(received_at=bad_ts))
        assert not result.valid, bad_ts


def test_event_duplicate_of_must_be_string_or_null():
    ctx = {"user_focus": "deep_work", "recent_notifications": 0, "duplicate_of": 99}
    assert not validate_event(make_event(context=ctx)).valid


def test_event_non_object_input_fails():
    assert not validate_event("not an object").valid
    assert not validate_event(None).valid
    assert not validate_event(42).valid


# ---------------------------------------------------------------------------
# Valid decision cases
# ---------------------------------------------------------------------------

def test_valid_notify_decision_passes():
    assert validate_decision(make_decision()).valid


def test_valid_digest_decision_passes():
    d = make_decision(action="digest", channel="email", urgency="normal")
    assert validate_decision(d).valid


def test_valid_silence_decision_passes():
    d = make_decision(
        action="silence", channel="none", urgency="low", message="",
    )
    assert validate_decision(d).valid


def test_valid_decision_all_action_channel_combos():
    # notify / digest allow im | email | push; silence requires none.
    for action in ("notify", "digest"):
        for channel in ("im", "email", "push"):
            d = make_decision(action=action, channel=channel)
            assert validate_decision(d).valid, (action, channel)
    assert validate_decision(
        make_decision(action="silence", channel="none", message="")
    ).valid


# ---------------------------------------------------------------------------
# Invalid decision cases
# ---------------------------------------------------------------------------

def test_decision_missing_field_fails():
    d = make_decision()
    del d["action"]
    assert not validate_decision(d).valid


def test_decision_invalid_action_enum_fails():
    assert not validate_decision(make_decision(action="ping")).valid


def test_decision_extra_field_fails():
    assert not validate_decision(make_decision(extra=1)).valid


def test_decision_silence_requires_channel_none():
    d = make_decision(action="silence", channel="im", message="")
    result = validate_decision(d)
    assert not result.valid
    assert any("channel" in e for e in result.errors)


def test_decision_silence_requires_empty_message():
    d = make_decision(action="silence", channel="none", message="still talking")
    result = validate_decision(d)
    assert not result.valid
    assert any("message" in e for e in result.errors)


def test_decision_non_silence_rejects_channel_none():
    d = make_decision(action="notify", channel="none")
    assert not validate_decision(d).valid


def test_decision_non_silence_rejects_empty_message():
    d = make_decision(action="digest", channel="email", message="")
    assert not validate_decision(d).valid


def test_decision_non_object_input_fails():
    assert not validate_decision([]).valid
    assert not validate_decision(None).valid
