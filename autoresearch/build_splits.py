"""Materialize the research dev / sealed splits.

Replays the frozen pipeline exactly:

    templates -> iter_combos() -> seed(42) shuffle
    combos[0:800]     -> SFT      (frozen, already on disk)
    combos[800:1100]  -> GRPO     (frozen, already on disk)
    combos[1100:1300] -> eval     (frozen, already on disk)
    combos[1300:1550] -> dev_extra (new)
    combos[1550:1750] -> sealed    (new)

``dev.jsonl`` = frozen eval 200 + dev_extra 250 = 450 rows.
``sealed_test.jsonl`` = 200 rows; only opened once after the loop stops.

The script also writes ``sft_labeled.jsonl`` / ``grpo_labeled.jsonl`` so the
materializer can read action / focus labels without re-deriving them, and
asserts that the regenerated first 200 eval rows are byte-identical to the
frozen ``data/synthetic/eval_set.jsonl``.

Usage:
    .venv/bin/python autoresearch/build_splits.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(REPO_ROOT, "training", "src")
sys.path.insert(0, SRC)

import build_dataset as bd  # noqa: E402
from expert_rules import label_events  # noqa: E402
from format_sft import to_sft_sample  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "autoresearch", "data")

SPLITS = [
    ("sft", 800, "s"),
    ("grpo", 300, "g"),
    ("eval", 200, "e"),
    ("dev_extra", 250, "d"),
    ("sealed", 200, "t"),
]


def _write_jsonl(path: str, items) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    templates = bd.load_all_templates()
    combos = bd.iter_combos(templates)
    expected = sum(size for _, size, _ in SPLITS)
    if len(combos) != expected:
        raise RuntimeError(f"expected {expected} combos, found {len(combos)}")

    rng = random.Random(42)
    rng.shuffle(combos)

    labeled = {}
    cursor = 0
    for name, size, prefix in SPLITS:
        chunk = combos[cursor:cursor + size]
        cursor += size
        events = [
            bd._event_from_combo(rng, combo, f"evt_{prefix}{index:04d}")
            for index, combo in enumerate(chunk)
        ]
        events = bd.inject_duplicates(events, rng)
        labeled[name] = label_events(events)

    # --- reproducibility check against the frozen artifacts -----------------
    frozen_eval = _read_jsonl(os.path.join(REPO_ROOT, "data", "synthetic", "eval_set.jsonl"))
    if labeled["eval"] != frozen_eval:
        for i, (a, b) in enumerate(zip(labeled["eval"], frozen_eval)):
            if a != b:
                raise RuntimeError(f"eval row {i} differs from frozen eval_set.jsonl")
        raise RuntimeError("eval split length differs from frozen eval_set.jsonl")

    frozen_sft = _read_jsonl(os.path.join(REPO_ROOT, "data", "synthetic", "sft_full.jsonl"))
    regen_sft = [to_sft_sample(item) for item in labeled["sft"]]
    if regen_sft != frozen_sft:
        raise RuntimeError("SFT split does not match frozen sft_full.jsonl")

    frozen_grpo = _read_jsonl(os.path.join(REPO_ROOT, "data", "synthetic", "grpo_prompts.jsonl"))
    regen_grpo = bd.build_grpo_prompts(labeled["grpo"])
    if regen_grpo != frozen_grpo:
        raise RuntimeError("GRPO split does not match frozen grpo_prompts.jsonl")

    # --- leakage check -------------------------------------------------------
    def key_set(items):
        return {k for k in (bd.near_dup_key(e) for e in items) if k}

    dev = labeled["eval"] + labeled["dev_extra"]
    sealed = labeled["sealed"]
    train_keys = key_set(labeled["sft"]) | key_set(labeled["grpo"])
    dev_keys = key_set(dev)
    sealed_keys = key_set(sealed)
    leaks = {
        "dev_vs_train": len(dev_keys & train_keys),
        "sealed_vs_train": len(sealed_keys & train_keys),
        "dev_vs_sealed": len(dev_keys & sealed_keys),
    }
    if any(leaks.values()):
        raise RuntimeError(f"near-duplicate leakage detected: {leaks}")

    _write_jsonl(os.path.join(OUT_DIR, "dev.jsonl"), dev)
    _write_jsonl(os.path.join(OUT_DIR, "sealed_test.jsonl"), sealed)
    _write_jsonl(os.path.join(OUT_DIR, "sft_labeled.jsonl"), labeled["sft"])
    _write_jsonl(os.path.join(OUT_DIR, "grpo_labeled.jsonl"), labeled["grpo"])
    _write_jsonl(os.path.join(OUT_DIR, "dev_extra_labeled.jsonl"), labeled["dev_extra"])

    def dist(items):
        return dict(Counter(item["decision"]["action"] for item in items))

    stats = {
        "seed": 42,
        "combos": len(combos),
        "split_sizes": {name: len(labeled[name]) for name, _, _ in SPLITS},
        "dev_size": len(dev),
        "sealed_size": len(sealed),
        "dev_action_dist": dist(dev),
        "dev_extra_action_dist": dist(labeled["dev_extra"]),
        "sealed_action_dist": dist(sealed),
        "dev_boundary": sum(1 for item in dev if item.get("boundary")),
        "sealed_boundary": sum(1 for item in sealed if item.get("boundary")),
        "leakage": leaks,
        "reproduced_frozen_eval": True,
        "reproduced_frozen_sft": True,
        "reproduced_frozen_grpo": True,
        "dev_sha256": _sha256(os.path.join(OUT_DIR, "dev.jsonl")),
        "sealed_sha256": _sha256(os.path.join(OUT_DIR, "sealed_test.jsonl")),
    }
    with open(os.path.join(OUT_DIR, "split_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
