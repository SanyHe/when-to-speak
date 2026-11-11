"""Record the SHA-256 of every oracle file consumed by the research loop."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(REPO_ROOT, "autoresearch", "frozen_hashes.json")

FILES = [
    "training/src/build_dataset.py",
    "training/src/evaluate.py",
    "training/src/expert_rules.py",
    "training/src/format_sft.py",
    "training/src/full_sft.py",
    "training/src/gen_events.py",
    "training/src/grpo.py",
    "training/src/rewards.py",
    "training/src/schema.py",
    "training/src/mock_models.py",
    "data/synthetic/sft_full.jsonl",
    "data/synthetic/grpo_prompts.jsonl",
    "data/synthetic/eval_set.jsonl",
    "data/templates/seed_templates.json",
    "data/templates/extra_templates.json",
    "autoresearch/data/dev.jsonl",
    "autoresearch/data/sealed_test.jsonl",
    "autoresearch/data/sft_labeled.jsonl",
    "autoresearch/data/grpo_labeled.jsonl",
    "autoresearch/data/dev_extra_labeled.jsonl",
    "autoresearch/eval_one.py",
    "autoresearch/gate.py",
    "autoresearch/program.md",
]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = {}
    for rel_path in FILES:
        path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(path):
            raise SystemExit(f"missing oracle file: {rel_path}")
        files[rel_path] = sha256(path)
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        git_sha = "unknown"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "files": files,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
