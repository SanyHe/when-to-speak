"""Evaluate one checkpoint on the dev set and write predictions + metrics."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(REPO_ROOT, "training", "src")
sys.path.insert(0, SRC)

import evaluate as ev  # noqa: E402  (frozen)
from schema import SOURCES  # noqa: E402


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _slice_metrics(items: List[Dict[str, Any]], records: List[Dict[str, Any]],
                   predicate) -> Dict[str, Any]:
    indices = [i for i, item in enumerate(items) if predicate(item)]
    if not indices:
        return {}
    return ev.compute_metrics([records[i] for i in indices])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    items = ev.load_labeled(args.eval_file)
    if args.max_items > 0:
        items = items[:args.max_items]

    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    results = ev.evaluate_hf({"model": args.model}, items)
    records = results["model"]
    elapsed = time.time() - started

    metrics = ev.compute_metrics(records)
    slices = {
        "source": {
            source: _slice_metrics(items, records,
                                   lambda item, s=source: item["source"] == s)
            for source in SOURCES
        },
        "boundary": _slice_metrics(items, records,
                                   lambda item: bool(item.get("boundary"))),
        "non_boundary": _slice_metrics(items, records,
                                       lambda item: not item.get("boundary")),
    }

    _write_jsonl(
        os.path.join(args.out_dir, "predictions.jsonl"),
        [{
            "event_id": record["event_id"],
            "gold": record["gold"],
            "pred": record["pred"],
            "json_valid": record["json_valid"],
            "raw": record["raw"],
        } for record in records],
    )
    _write_json(os.path.join(args.out_dir, "metrics.json"), {
        "model": args.model,
        "eval_file": args.eval_file,
        "eval_items": len(items),
        "elapsed_seconds": round(elapsed, 2),
        "metrics": metrics,
        "slices": slices,
    })
    print("EVAL_METRICS " + json.dumps({
        "macro_f1": metrics["macro_f1"],
        "noise_rate": metrics["noise_rate"],
        "miss_rate": metrics["miss_rate"],
        "notify_recall": metrics["notify_recall"],
        "json_valid_rate": metrics["json_valid_rate"],
        "elapsed_seconds": round(elapsed, 1),
    }))


if __name__ == "__main__":
    main()
