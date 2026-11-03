"""
Acceptance tests for the seed event generator.

Verifies:
  - one-click rebuild produces exactly the expected count
  - every generated event passes the event schema
  - no duplicate event payloads
  - all 5 sources and all 3 expected_action classes are covered
"""
import json
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))

import gen_events  # noqa: E402
from schema import validate_event  # noqa: E402


def test_generate_one_hundred_events_all_valid_and_unique():
    templates = gen_events.load_templates(gen_events.TEMPLATES_PATH)
    events = gen_events.generate(templates, count=100, seed=gen_events.SEED)

    assert len(events) == 100
    assert len({e["event_id"] for e in events}) == 100

    # Every event passes the schema contract.
    for e in events:
        r = validate_event(e)
        assert r.valid, (e["event_id"], r.errors)

    # No two events are identical.
    sigs = {json.dumps(e, sort_keys=True, ensure_ascii=False) for e in events}
    assert len(sigs) == 100


def test_all_sources_and_expected_actions_covered():
    templates = gen_events.load_templates(gen_events.TEMPLATES_PATH)
    events = gen_events.generate(templates, count=100, seed=gen_events.SEED)

    sources = {e["source"] for e in events}
    assert sources == {"email", "im", "task", "routine", "system"}

    expected_actions = {t["expected_action"] for t in templates}
    assert expected_actions == {"notify", "digest", "silence"}


def test_output_is_reproducible():
    templates = gen_events.load_templates(gen_events.TEMPLATES_PATH)
    a = gen_events.generate(templates, count=100, seed=gen_events.SEED)
    b = gen_events.generate(templates, count=100, seed=gen_events.SEED)
    assert a == b
