"""
Synthetic dataset builder (F2.1 / F2.2).

Builds the full M2 dataset from templates in one reproducible run:

    seed_templates.json  +  extra_templates.json
        -> deterministic event materialization (one event per template combo)
        -> rule-expert gold labels (F1.3), optional human-review overrides
        -> intentional duplicate events (gold = silence)
        -> split into SFT / GRPO / Eval with no near-duplicate leakage
        -> data/synthetic/{sft_full,grpo_prompts,eval_set}.jsonl

Design notes
------------
* Every (template, sender, subject, snippet) combo is a *unique* near-dup
  key. Combos are shuffled once and allocated to exactly one split, so the
  Eval set can never share a near-duplicate with the training sets.
* Each split gets exactly the configured number of events (duplicates
  replace tail events, keeping the count stable).
* GRPO prompts keep the gold decision so the deterministic reward
  functions can score rollouts without a reward model.
* Boundary samples are exported for review; a review file with non-null
  ``reviewed_decision`` entries is applied on the next build.

Usage (repo root):
    python3 training/src/build_dataset.py
    python3 training/src/build_dataset.py --reviews data/synthetic/boundary_review.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

import gen_events  # noqa: E402
from expert_rules import label_events  # noqa: E402
from format_sft import SYSTEM_PROMPT, to_sft_sample  # noqa: E402
from schema import USER_FOCUS_STATES, validate_decision, validate_event  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", ".."))
SEED_TEMPLATES = os.path.join(REPO_ROOT, "data", "templates", "seed_templates.json")
EXTRA_TEMPLATES = os.path.join(REPO_ROOT, "data", "templates", "extra_templates.json")
OUT_DIR = os.path.join(REPO_ROOT, "data", "synthetic")

DEFAULT_SIZES = {"sft": 800, "grpo": 300, "eval": 200}
DUPLICATE_RATE = 0.10
BOUNDARY_EXPORT_LIMIT = 60


# ---------------------------------------------------------------------------
# Templates and event materialization
# ---------------------------------------------------------------------------

def load_all_templates() -> List[Dict[str, Any]]:
    """Merge the M1 seed templates with the M2 extra templates."""
    templates = []
    for path in (SEED_TEMPLATES, EXTRA_TEMPLATES):
        with open(path, "r", encoding="utf-8") as f:
            templates.extend(json.load(f)["templates"])
    return templates


def iter_combos(templates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enumerate every unique (template, sender, subject, snippet) combo."""
    combos = []
    for template in templates:
        for sender in template["sender_pool"]:
            for subject in template["subject_pool"]:
                for snippet in template["snippet_pool"]:
                    combos.append({
                        "template": template,
                        "sender": sender,
                        "subject": subject,
                        "snippet": snippet,
                    })
    return combos


def _event_from_combo(rng: random.Random, combo: Dict[str, Any], event_id: str) -> Dict[str, Any]:
    template = combo["template"]
    event = {
        "event_id": event_id,
        "source": template["source"],
        "sender": combo["sender"],
        "subject": combo["subject"],
        "snippet": combo["snippet"],
        "mentions_user": template["mentions_user"],
        "is_direct": template["is_direct"],
        "time_sensitive": template["time_sensitive"],
        "received_at": gen_events.random_timestamp(rng, gen_events.BASE_TIME, gen_events.TIME_WINDOW_DAYS),
        "context": {
            "user_focus": rng.choice(USER_FOCUS_STATES),
            "recent_notifications": rng.randint(0, 5),
            "duplicate_of": None,
        },
    }
    result = validate_event(event)
    if not result.valid:
        raise RuntimeError(f"invalid generated event {event_id}: {result.errors}")
    return event


def near_dup_key(item: Dict[str, Any]) -> Optional[str]:
    """Near-duplicate key: source + sender + subject + snippet, normalized.

    Intentional duplicate events (``duplicate_of`` set) are excluded: they
    are part of the same split by construction and must not trip the
    cross-split leakage check.
    """
    if item.get("context", {}).get("duplicate_of"):
        return None
    text = "|".join([
        item["source"], item["sender"], item["subject"], item["snippet"],
    ])
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def _add_minutes(timestamp: str, minutes: int) -> str:
    dt = datetime.fromisoformat(timestamp)
    return (dt + timedelta(minutes=minutes)).isoformat(timespec="seconds")


def inject_duplicates(events: List[Dict[str, Any]], rng: random.Random,
                      rate: float = DUPLICATE_RATE) -> List[Dict[str, Any]]:
    """Append deterministic duplicate events, replacing tail events.

    A duplicate keeps the payload of an earlier event, gets a fresh id and
    a later timestamp, and points ``duplicate_of`` at the original. The
    rule expert labels it as silence, which gives the Eval set real
    "already handled / repeated" cases.
    """
    count = max(1, int(len(events) * rate))
    originals = events[:count]
    kept = events[:-count]
    duplicates = []
    for index, original in enumerate(originals):
        duplicate = copy.deepcopy(original)
        duplicate["event_id"] = f"{original['event_id']}_dup{index:02d}"
        duplicate["received_at"] = _add_minutes(original["received_at"], rng.randint(3, 20))
        duplicate["context"]["duplicate_of"] = original["event_id"]
        result = validate_event(duplicate)
        if not result.valid:
            raise RuntimeError(f"invalid duplicate event {duplicate['event_id']}: {result.errors}")
        duplicates.append(duplicate)
    result = kept + duplicates
    result.sort(key=lambda e: (e["received_at"], e["event_id"]))
    return result


# ---------------------------------------------------------------------------
# GRPO prompts
# ---------------------------------------------------------------------------

def build_grpo_prompts(labeled: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn each GRPO-split event into a prompt with its gold decision.

    The prompt contains only system + user messages; the model must
    generate the decision. ``gold_action`` / ``gold_decision`` are kept
    as extra columns so the reward functions can score rollouts.
    """
    prompts = []
    for item in labeled:
        event = {k: v for k, v in item.items() if k not in ("decision", "boundary", "reviewed")}
        gold = item["decision"]
        prompts.append({
            "event_id": item["event_id"],
            "source": item["source"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(event, ensure_ascii=False)},
            ],
            "gold_action": gold["action"],
            "gold_decision": gold,
        })
    return prompts


# ---------------------------------------------------------------------------
# Human-review write-back
# ---------------------------------------------------------------------------

def apply_reviews(labeled: List[Dict[str, Any]], reviews_path: str) -> int:
    """Override gold decisions with reviewed decisions from a review file."""
    if not reviews_path or not os.path.exists(reviews_path):
        return 0
    by_id: Dict[str, Dict[str, Any]] = {}
    with open(reviews_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            review = json.loads(line)
            decision = review.get("reviewed_decision")
            if not decision:
                continue
            result = validate_decision(decision)
            if not result.valid:
                raise RuntimeError(
                    f"invalid reviewed decision for {review.get('event_id')}: {result.errors}"
                )
            by_id[review["event_id"]] = decision

    overrides = 0
    for item in labeled:
        if item["event_id"] in by_id:
            item["decision"] = by_id[item["event_id"]]
            item["reviewed"] = True
            item["boundary"] = False
            overrides += 1
    return overrides


def export_boundary_review(labeled_items: List[Dict[str, Any]], path: str,
                           limit: int = BOUNDARY_EXPORT_LIMIT) -> int:
    """Export boundary samples with an empty slot for human review."""
    boundary = [item for item in labeled_items if item.get("boundary")][:limit]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in boundary:
            entry = {
                "event_id": item["event_id"],
                "source": item["source"],
                "subject": item["subject"],
                "snippet": item["snippet"],
                "rule_decision": item["decision"],
                "reviewed_decision": None,
                "reviewer": None,
                "status": "pending",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(boundary)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _write_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _cross_split_collisions(splits: Dict[str, List[Dict[str, Any]]]) -> int:
    keys = {
        name: {k for k in (near_dup_key(e) for e in events) if k}
        for name, events in splits.items()
    }
    names = list(keys)
    collisions = 0
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            collisions += len(keys[left] & keys[right])
    return collisions


def build(seed: int = 42, sizes: Optional[Dict[str, int]] = None,
          reviews_path: str = "", export_boundary: bool = True) -> Dict[str, Any]:
    """Build the full dataset and write all artifacts.

    Returns a summary dict (also written to dataset_stats.json).
    """
    sizes = sizes or DEFAULT_SIZES
    total = sum(sizes.values())

    templates = load_all_templates()
    combos = iter_combos(templates)
    if len(combos) < total:
        raise RuntimeError(
            f"not enough unique template combos: {len(combos)} < {total}; "
            "extend the template pools before building."
        )

    rng = random.Random(seed)
    rng.shuffle(combos)

    splits: Dict[str, List[Dict[str, Any]]] = {}
    cursor = 0
    for name in ("sft", "grpo", "eval"):
        size = sizes[name]
        chunk = combos[cursor:cursor + size]
        cursor += size
        prefix = {"sft": "s", "grpo": "g", "eval": "e"}[name]
        events = [
            _event_from_combo(rng, combo, f"evt_{prefix}{index:04d}")
            for index, combo in enumerate(chunk)
        ]
        splits[name] = inject_duplicates(events, rng)

    overrides = 0
    labeled: Dict[str, List[Dict[str, Any]]] = {}
    for name, events in splits.items():
        labeled[name] = label_events(events)
        overrides += apply_reviews(labeled[name], reviews_path)

    collisions = _cross_split_collisions(splits)
    if collisions:
        raise RuntimeError(f"near-duplicate leakage across splits: {collisions}")

    sft_samples = [to_sft_sample(item) for item in labeled["sft"]]
    grpo_prompts = build_grpo_prompts(labeled["grpo"])

    _write_jsonl(sft_samples, os.path.join(OUT_DIR, "sft_full.jsonl"))
    _write_jsonl(grpo_prompts, os.path.join(OUT_DIR, "grpo_prompts.jsonl"))
    _write_jsonl(labeled["eval"], os.path.join(OUT_DIR, "eval_set.jsonl"))

    boundary_exported = 0
    if export_boundary:
        boundary_exported = export_boundary_review(
            labeled["sft"] + labeled["eval"],
            os.path.join(OUT_DIR, "boundary_review.jsonl"),
        )

    stats = {
        "seed": seed,
        "templates": len(templates),
        "combos_available": len(combos),
        "sizes": {"sft": len(sft_samples), "grpo_prompts": len(grpo_prompts),
                  "eval": len(labeled["eval"])},
        "action_dist": {
            name: dict(Counter(item["decision"]["action"] for item in items))
            for name, items in labeled.items()
        },
        "source_dist": {
            "eval": dict(Counter(item["source"] for item in labeled["eval"]))
        },
        "intentional_duplicates": {
            name: sum(1 for item in items if item["context"]["duplicate_of"])
            for name, items in labeled.items()
        },
        "boundary": {
            name: sum(1 for item in items if item.get("boundary"))
            for name, items in labeled.items()
        },
        "reviewed_overrides": overrides,
        "boundary_exported": boundary_exported,
        "near_duplicate_cross_split": collisions,
    }
    with open(os.path.join(OUT_DIR, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-size", type=int, default=DEFAULT_SIZES["sft"])
    parser.add_argument("--grpo-size", type=int, default=DEFAULT_SIZES["grpo"])
    parser.add_argument("--eval-size", type=int, default=DEFAULT_SIZES["eval"])
    parser.add_argument("--reviews", default="", help="boundary review JSONL with reviewed decisions")
    args = parser.parse_args()

    stats = build(
        seed=args.seed,
        sizes={"sft": args.sft_size, "grpo": args.grpo_size, "eval": args.eval_size},
        reviews_path=args.reviews,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
