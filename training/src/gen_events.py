"""
Synthetic seed event generator.

Reads template specs from ``data/templates/seed_templates.json`` and emits a
deterministic, reproducible set of events to ``data/synthetic/seed_events.jsonl``.

Usage (from repo root):
    python3 training/src/gen_events.py                 # default 100 events
    python3 training/src/gen_events.py --count 150     # custom count

Each generated event is validated against the event schema before being
written; the script aborts if any event fails validation or if duplicates
are detected.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

# Make training/src importable (schema module).
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

from schema import USER_FOCUS_STATES, validate_event  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", ".."))
TEMPLATES_PATH = os.path.join(REPO_ROOT, "data", "templates", "seed_templates.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "synthetic", "seed_events.jsonl")

SEED = 42
DEFAULT_COUNT = 100

# Time window for received_at: a 7-day range ending "now" (generation time).
TIME_WINDOW_DAYS = 7


# Fixed base time so generated timestamps are deterministic across runs.
# Chosen as a stable epoch anchor; the relative ordering within the window
# is what matters for the demo.
BASE_TIME = datetime(2026, 10, 31, 0, 0, 0, tzinfo=timezone.utc)


def load_templates(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["templates"]


def pick(rng: random.Random, pool: List[str]) -> str:
    return rng.choice(pool)


def random_timestamp(rng: random.Random, base: datetime, window_days: int) -> str:
    """Return an ISO-8601 timestamp with offset within ``window_days`` of ``base``."""
    offset_seconds = rng.randint(0, window_days * 24 * 3600)
    ts = base - timedelta(seconds=offset_seconds)
    # Use +08:00 timezone to match the documented example and keep offsets stable.
    return ts.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S+08:00")


def build_event(
    rng: random.Random,
    template: Dict[str, Any],
    event_id: str,
    base_time: datetime,
) -> Dict[str, Any]:
    """Materialize one fully-specified event from a template."""
    event: Dict[str, Any] = {
        "event_id": event_id,
        "source": template["source"],
        "sender": pick(rng, template["sender_pool"]),
        "subject": pick(rng, template["subject_pool"]),
        "snippet": pick(rng, template["snippet_pool"]),
        "mentions_user": template["mentions_user"],
        "is_direct": template["is_direct"],
        "time_sensitive": template["time_sensitive"],
        "received_at": random_timestamp(rng, base_time, TIME_WINDOW_DAYS),
        "context": {
            "user_focus": rng.choice(USER_FOCUS_STATES),
            "recent_notifications": rng.randint(0, 5),
            "duplicate_of": None,
        },
    }
    return event


def generate(templates: List[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    events: List[Dict[str, Any]] = []
    seen: set = set()

    for i in range(count):
        template = rng.choice(templates)
        event_id = f"evt_{i:04d}"
        event = build_event(rng, template, event_id, BASE_TIME)

        # Validate against the schema before accepting.
        result = validate_event(event)
        if not result.valid:
            raise RuntimeError(
                f"Generated invalid event {event_id}: {result.errors}"
            )

        # Duplicate check: compare the full event payload (excluding event_id
        # and received_at, which are inherently unique, would be too lenient;
        # we compare the entire dict to guarantee no identical events).
        signature = json.dumps(event, sort_keys=True, ensure_ascii=False)
        if signature in seen:
            # Retry once with a fresh randomization to avoid duplicates.
            event = build_event(rng, template, event_id, base_time)
            signature = json.dumps(event, sort_keys=True, ensure_ascii=False)
            if signature in seen:
                raise RuntimeError(f"Duplicate event generated for {event_id}")
        seen.add(signature)
        events.append(event)

    return events


def write_jsonl(events: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def summarize(events: List[Dict[str, Any]], templates: List[Dict[str, Any]]) -> None:
    from collections import Counter

    src_dist = Counter(e["source"] for e in events)
    # Map event -> expected_action via the template's source+fixed fields.
    # Simpler: count expected_action from templates used (we don't store
    # template_id on events, so recompute by matching source+mentions+...).
    tpl_by_id = {t["template_id"]: t for t in templates}
    # expected_action distribution is best reported from templates themselves.
    expected_dist = Counter(t["expected_action"] for t in templates)

    print(f"Generated {len(events)} events -> {OUTPUT_PATH}")
    print(f"Source distribution: {dict(src_dist)}")
    print(f"Template expected_action coverage: {dict(expected_dist)}")
    # Verify all 3 expected actions are represented in templates.
    missing = {"notify", "digest", "silence"} - set(expected_dist)
    if missing:
        print(f"WARNING: templates missing expected_action: {missing}")
    else:
        print("All 3 expected_action classes present in templates.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help="number of events to generate (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    templates = load_templates(TEMPLATES_PATH)
    events = generate(templates, args.count, args.seed)
    write_jsonl(events, OUTPUT_PATH)
    summarize(events, templates)


if __name__ == "__main__":
    main()
