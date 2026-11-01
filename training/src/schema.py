"""
Data contract for the When-to-Speak demo.

Defines JSON schemas and validation helpers for:
  - event:    the input event arriving at the always-on agent
  - decision: the structured output decision produced by the model / expert

Schemas follow docs/PLAN.md section 4. All objects are strict JSON:
fixed keys, UTF-8, no extra fields allowed.

Cross-field business rules are encoded inside the schemas themselves
(via JSON Schema if/then/else) so that a single validator enforces
both structural and semantic constraints.
"""
from __future__ import annotations

from typing import Any, List

import jsonschema

# ---------------------------------------------------------------------------
# Field enums — single source of truth, reused by the data generator and the
# rule-based expert (see PLAN 4.3 for the allowed values).
# ---------------------------------------------------------------------------

SOURCES = ("email", "im", "task", "routine", "system")
USER_FOCUS_STATES = ("deep_work", "available", "offline")
ACTIONS = ("notify", "digest", "silence")
CHANNELS = ("im", "email", "push", "none")
URGENCIES = ("urgent", "normal", "low")

# ISO 8601 datetime: YYYY-MM-DDTHH:MM:SS[.ffffff](Z | +/-HH:MM).
# A timezone designator is required so events are orderable across sources.
_ISO8601_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"


# ---------------------------------------------------------------------------
# Event schema (input to the agent)
# ---------------------------------------------------------------------------

EVENT_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Event",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "event_id",
        "source",
        "sender",
        "subject",
        "snippet",
        "mentions_user",
        "is_direct",
        "time_sensitive",
        "received_at",
        "context",
    ],
    "properties": {
        "event_id": {"type": "string", "minLength": 1},
        "source": {"type": "string", "enum": list(SOURCES)},
        "sender": {"type": "string"},
        "subject": {"type": "string"},
        "snippet": {"type": "string"},
        "mentions_user": {"type": "boolean"},
        "is_direct": {"type": "boolean"},
        "time_sensitive": {"type": "boolean"},
        "received_at": {"type": "string", "pattern": _ISO8601_PATTERN},
        "context": {
            "type": "object",
            "additionalProperties": False,
            "required": ["user_focus", "recent_notifications", "duplicate_of"],
            "properties": {
                "user_focus": {"type": "string", "enum": list(USER_FOCUS_STATES)},
                "recent_notifications": {"type": "integer", "minimum": 0},
                "duplicate_of": {"type": ["string", "null"]},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Decision schema (output of the agent)
# ---------------------------------------------------------------------------

DECISION_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Decision",
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "channel", "urgency", "reason", "message"],
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "channel": {"type": "string", "enum": list(CHANNELS)},
        "urgency": {"type": "string", "enum": list(URGENCIES)},
        "reason": {"type": "string"},
        "message": {"type": "string"},
    },
    # Cross-field business rules (PLAN 4.3):
    #   action == "silence"  => channel == "none" AND message == ""
    #   action != "silence"  => channel != "none" AND message != ""
    "allOf": [
        {
            "if": {"properties": {"action": {"const": "silence"}}},
            "then": {
                "properties": {
                    "channel": {"const": "none"},
                    "message": {"const": ""},
                }
            },
            "else": {
                "properties": {
                    "channel": {"enum": ["im", "email", "push"]},
                    "message": {"type": "string", "minLength": 1},
                }
            },
        }
    ],
}


# ---------------------------------------------------------------------------
# Validation API
# ---------------------------------------------------------------------------

class ValidationResult:
    """Structured outcome of validating an object against a schema.

    Truthy when valid. ``errors`` always holds a list of human-readable
    messages pointing at the offending field path.
    """

    def __init__(self, valid: bool, errors: List[str] | None = None) -> None:
        self.valid = valid
        self.errors = errors or []

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        if self.valid:
            return "ValidationResult(valid=True)"
        return f"ValidationResult(valid=False, errors={self.errors!r})"


def _validate(obj: Any, schema: dict) -> ValidationResult:
    """Run the Draft7 validator and collect every violation as a string."""
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path))
    messages: List[str] = []
    for err in errors:
        loc = ".".join(str(p) for p in err.absolute_path) or "<root>"
        messages.append(f"{loc}: {err.message}")
    return ValidationResult(valid=not messages, errors=messages)


def validate_event(obj: Any) -> ValidationResult:
    """Validate an input event against the event schema."""
    return _validate(obj, EVENT_SCHEMA)


def validate_decision(obj: Any) -> ValidationResult:
    """Validate an output decision against the decision schema."""
    return _validate(obj, DECISION_SCHEMA)
