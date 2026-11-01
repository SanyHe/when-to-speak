"""
Rule-based expert that produces gold labels for events (F1.3).

The expert is a *deterministic* decision function: given the same event it
always returns the same decision. It encodes simple heuristics derived from
PLAN section 5.2 (source, mentions_user, is_direct, time_sensitive,
marketing keywords, user_focus, duplicate_of).

Events that are NOT covered by the strong rules are flagged as "boundary"
and can be exported for human / strong-model review; the reviewed labels can
later be written back as gold.

The expert never returns a decision that fails the decision schema.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

from schema import validate_decision  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", ".."))

# Marketing / mass-notification keywords. Matched case-insensitively against
# subject + snippet.
MARKETING_KEYWORDS = (
    "促销", "优惠", "限时", "折扣", "优惠券", "抢购", "升级",
    "免费试用", "newsletter", "promo", "limited", "sale",
)

# Sources that produce automated, low-value notifications when not directed.
AUTOMATED_SOURCES = ("routine", "task")


def _channel_for(source: str, action: str) -> str:
    """Map source + action to a delivery channel (PLAN 4.3)."""
    if action == "silence":
        return "none"
    if source == "im":
        return "im"
    if source == "email":
        return "email"
    return "push"  # task / routine / system


def _message_for(event: Dict[str, Any], action: str) -> str:
    """Build a concise notification message; empty for silence."""
    if action == "silence":
        return ""
    return f"{event['subject']}：{event['snippet']}"


def _make_decision(action: str, channel: str, urgency: str,
                   reason: str, message: str) -> Dict[str, Any]:
    return {
        "action": action,
        "channel": channel,
        "urgency": urgency,
        "reason": reason,
        "message": message,
    }


def decide(event: Dict[str, Any]) -> Dict[str, Any]:
    """Return the gold decision for ``event`` (deterministic).

    The returned dict always passes ``validate_decision``.
    """
    source = event["source"]
    mentions = event["mentions_user"]
    direct = event["is_direct"]
    time_sens = event["time_sensitive"]
    duplicate = event["context"]["duplicate_of"]
    focus = event["context"]["user_focus"]
    text = f"{event['subject']} {event['snippet']}".lower()

    directed = mentions or direct
    is_marketing = any(kw in text for kw in MARKETING_KEYWORDS)

    # ---- Strong SILENCE rules ----
    if duplicate is not None:
        return _make_decision(
            "silence", "none", "low",
            "重复事件，不重复打扰", "",
        )
    if is_marketing and not directed and not time_sens:
        return _make_decision(
            "silence", "none", "low",
            "营销类群发，无需打扰", "",
        )
    if source in AUTOMATED_SOURCES and not directed and not time_sens:
        return _make_decision(
            "silence", "none", "low",
            "例行自动化通知，无需打扰", "",
        )

    # ---- Strong NOTIFY rules ----
    if time_sens and directed:
        action, urgency, reason = "notify", "urgent", "时效性强且直接关联用户"
    elif mentions and direct:
        action, urgency, reason = "notify", "normal", "直接发给用户的消息"
    # ---- DIGEST (softer) rules ----
    elif time_sens:
        action, urgency, reason = (
            "digest", "normal", "时效信息但非直接发给用户，进入汇总",
        )
    elif source in ("email", "im") and not directed:
        action, urgency, reason = "digest", "low", "非直接消息，进入汇总"
    else:
        action, urgency, reason = "digest", "low", "默认进入汇总"

    # Deep-work protection: downgrade non-urgent notify to digest so the
    # agent does not interrupt focused work unless it is truly urgent.
    if action == "notify" and urgency != "urgent" and focus == "deep_work":
        action = "digest"
        urgency = "normal"
        reason = reason + "；用户处于 deep_work，降级为汇总"

    channel = _channel_for(source, action)
    message = _message_for(event, action)
    return _make_decision(action, channel, urgency, reason, message)


def is_boundary(event: Dict[str, Any]) -> bool:
    """True when the event is NOT covered by a strong rule.

    Boundary events are candidates for human / strong-model review because
    the deterministic heuristic is less confident about them.
    """
    source = event["source"]
    mentions = event["mentions_user"]
    direct = event["is_direct"]
    time_sens = event["time_sensitive"]
    duplicate = event["context"]["duplicate_of"]
    directed = mentions or direct
    text = f"{event['subject']} {event['snippet']}".lower()
    is_marketing = any(kw in text for kw in MARKETING_KEYWORDS)

    clear_silence = (
        duplicate is not None
        or (is_marketing and not directed and not time_sens)
        or (source in AUTOMATED_SOURCES and not directed and not time_sens)
    )
    clear_notify = (time_sens and directed) or (mentions and direct)
    return not (clear_silence or clear_notify)


def label_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach a gold ``decision`` and ``boundary`` flag to each event."""
    labeled = []
    for event in events:
        decision = decide(event)
        # Guarantee every emitted decision is schema-valid.
        result = validate_decision(decision)
        if not result.valid:
            raise RuntimeError(
                f"Expert produced invalid decision for {event['event_id']}: "
                f"{result.errors}"
            )
        labeled.append({
            **event,
            "decision": decision,
            "boundary": is_boundary(event),
        })
    return labeled


# ---------------------------------------------------------------------------
# CLI: label a JSONL of events and (optionally) export boundary samples.
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=os.path.join(REPO_ROOT, "data", "synthetic", "seed_events.jsonl"),
        help="input events JSONL",
    )
    parser.add_argument(
        "--output", default=os.path.join(REPO_ROOT, "data", "synthetic", "seed_labeled.jsonl"),
        help="output labeled events JSONL",
    )
    parser.add_argument(
        "--boundary", default=os.path.join(REPO_ROOT, "data", "synthetic", "boundary.jsonl"),
        help="output boundary-sample JSONL (set to empty string to skip)",
    )
    args = parser.parse_args()

    events = _read_jsonl(args.input)
    labeled = label_events(events)
    _write_jsonl(labeled, args.output)

    boundary_path = args.boundary
    if boundary_path:
        boundary_items = [item for item in labeled if item["boundary"]]
        _write_jsonl(boundary_items, boundary_path)
        print(f"Boundary samples: {len(boundary_items)} -> {boundary_path}")

    from collections import Counter
    action_dist = Counter(item["decision"]["action"] for item in labeled)
    print(f"Labeled {len(labeled)} events -> {args.output}")
    print(f"Action distribution: {dict(action_dist)}")


if __name__ == "__main__":
    main()
