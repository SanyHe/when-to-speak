"""Materialize SFT / GRPO data for one recipe.

The frozen splits are read from ``autoresearch/data/*_labeled.jsonl`` (regenerated
byte-identically from the original pipeline) and only the sampling policy is
applied here. Gold labels are never changed.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(REPO_ROOT, "training", "src")
sys.path.insert(0, SRC)

from format_sft import SYSTEM_PROMPT, to_sft_sample  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "autoresearch", "data")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _factor(mapping: Dict[str, int], key: str) -> int:
    return max(1, int(mapping.get(key, 1)))


def build_sft(recipe: Dict[str, Any], out_path: str, smoke: bool = False) -> Dict[str, Any]:
    """Apply action/focus oversampling and optional boundary contrastive copies."""
    items = _read_jsonl(os.path.join(DATA_DIR, "sft_labeled.jsonl"))
    if smoke:
        items = items[:16]

    action_map = recipe["data"].get("action_oversample", {})
    focus_map = recipe["data"].get("focus_oversample", {})

    samples: List[Dict[str, Any]] = []
    for item in items:
        action = item["decision"]["action"]
        focus = item["context"]["user_focus"]
        repeats = _factor(action_map, action) * _factor(focus_map, focus)
        for _ in range(repeats):
            samples.append(to_sft_sample(item))

    contrastive_limit = int(recipe["data"].get("contrastive_limit", 0))
    contrastive_added = 0
    if contrastive_limit > 0 and not smoke:
        pool = [
            item for item in items
            if item.get("boundary") and item["decision"]["action"] in ("digest", "silence")
        ]
        cursor = 0
        while pool and contrastive_added < contrastive_limit:
            samples.append(to_sft_sample(pool[cursor % len(pool)]))
            cursor += 1
            contrastive_added += 1

    _write_jsonl(out_path, samples)
    return {
        "base_rows": len(items),
        "rows": len(samples),
        "contrastive_added": contrastive_added,
        "action_dist": dict(Counter(
            json.loads(s["messages"][-1]["content"])["action"] for s in samples)),
        "focus_dist": dict(Counter(
            json.loads(s["messages"][1]["content"])["context"]["user_focus"]
            for s in samples)),
    }


def build_grpo(recipe: Dict[str, Any], out_path: str, smoke: bool = False) -> Dict[str, Any]:
    """Build GRPO prompts with hard-prompt boost and focus oversampling."""
    items = _read_jsonl(os.path.join(DATA_DIR, "grpo_labeled.jsonl"))
    if smoke:
        items = items[:8]

    grpo_cfg = recipe["grpo"]
    prompt_mode = grpo_cfg.get("prompt_mode", "all")
    if prompt_mode == "hard":
        items = [item for item in items if item.get("boundary")]
    elif prompt_mode != "all":
        raise ValueError(f"unknown grpo.prompt_mode: {prompt_mode}")

    focus_map = recipe["data"].get("grpo_focus_oversample", {})
    hard_boost = max(1, int(grpo_cfg.get("hard_boost", 1)))

    rows: List[Dict[str, Any]] = []
    for item in items:
        event = {k: v for k, v in item.items()
                 if k not in ("decision", "boundary", "reviewed")}
        gold = item["decision"]
        hard = bool(item.get("boundary"))
        repeats = _factor(focus_map, event["context"]["user_focus"])
        if hard:
            repeats *= hard_boost
        for _ in range(repeats):
            rows.append({
                "event_id": item["event_id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(event, ensure_ascii=False)},
                ],
                "gold_action": gold["action"],
                "gold_decision": gold,
                "user_focus": event["context"]["user_focus"],
                "duplicate_of": event["context"]["duplicate_of"],
                "boundary": hard,
            })

    _write_jsonl(out_path, rows)
    return {
        "base_rows": len(items),
        "rows": len(rows),
        "hard_prompts": sum(1 for row in rows if row["boundary"]),
        "action_dist": dict(Counter(row["gold_action"] for row in rows)),
        "focus_dist": dict(Counter(row["user_focus"] for row in rows)),
    }


def materialize(recipe: Dict[str, Any], run_dir: str, smoke: bool = False) -> Dict[str, Any]:
    """Write ``sft.jsonl`` / ``grpo.jsonl`` and return their statistics."""
    os.makedirs(run_dir, exist_ok=True)
    sft_stats = build_sft(recipe, os.path.join(run_dir, "sft.jsonl"), smoke=smoke)
    grpo_stats = build_grpo(recipe, os.path.join(run_dir, "grpo.jsonl"), smoke=smoke)
    stats = {"sft": sft_stats, "grpo": grpo_stats, "smoke": smoke}
    with open(os.path.join(run_dir, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return stats


if __name__ == "__main__":
    from recipe import BASELINE

    print(json.dumps(materialize(BASELINE, os.path.join(REPO_ROOT, "autoresearch", "runs", "_dry")),
                     indent=2))
