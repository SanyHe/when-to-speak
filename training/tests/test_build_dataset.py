"""
Acceptance tests for the M2 dataset builder (F2.1 / F2.2).

Verifies sizes, gold validity, split isolation, reproducibility, GRPO
prompt integrity, and the boundary-review write-back path.
"""
import json
import os
import sys
import tempfile

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))

import build_dataset  # noqa: E402
from schema import validate_decision  # noqa: E402


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_dataset_sizes_and_no_leakage():
    stats = build_dataset.build()
    assert stats["sizes"] == {"sft": 800, "grpo_prompts": 300, "eval": 200}
    assert stats["templates"] >= 30
    assert stats["near_duplicate_cross_split"] == 0

    for split in ("sft", "grpo", "eval"):
        assert set(stats["action_dist"][split]) == {"notify", "digest", "silence"}

    eval_items = _read_jsonl(os.path.join(build_dataset.OUT_DIR, "eval_set.jsonl"))
    for item in eval_items:
        assert validate_decision(item["decision"]).valid


def test_sft_samples_are_chat_records():
    samples = _read_jsonl(os.path.join(build_dataset.OUT_DIR, "sft_full.jsonl"))
    assert len(samples) == 800
    for sample in samples:
        roles = [m["role"] for m in sample["messages"]]
        assert roles == ["system", "user", "assistant"]
        decision = json.loads(sample["messages"][2]["content"])
        assert validate_decision(decision).valid


def test_grpo_prompts_are_prompt_only_with_gold():
    prompts = _read_jsonl(os.path.join(build_dataset.OUT_DIR, "grpo_prompts.jsonl"))
    assert len(prompts) == 300
    event_ids = set()
    for prompt in prompts:
        assert [m["role"] for m in prompt["messages"]] == ["system", "user"]
        gold = prompt["gold_decision"]
        assert validate_decision(gold).valid
        assert prompt["gold_action"] == gold["action"]
        assert prompt["event_id"] not in event_ids
        event_ids.add(prompt["event_id"])


def test_build_is_reproducible():
    first = build_dataset.build()
    with open(os.path.join(build_dataset.OUT_DIR, "sft_full.jsonl"), "rb") as f:
        first_bytes = f.read()
    second = build_dataset.build()
    with open(os.path.join(build_dataset.OUT_DIR, "sft_full.jsonl"), "rb") as f:
        second_bytes = f.read()
    assert first == second
    assert first_bytes == second_bytes


def test_review_write_back_overrides_gold():
    eval_items = _read_jsonl(os.path.join(build_dataset.OUT_DIR, "eval_set.jsonl"))
    entry = next(item for item in eval_items if item.get("boundary"))
    reviewed = {
        "action": "notify",
        "channel": "push",
        "urgency": "normal",
        "reason": "人工复核：应当通知",
        "message": "复核后的通知内容",
    }
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({
            "event_id": entry["event_id"],
            "reviewed_decision": reviewed,
        }, ensure_ascii=False) + "\n")
        review_path = f.name

    try:
        overrides = build_dataset.apply_reviews(eval_items, review_path)
        assert overrides == 1
        updated = next(i for i in eval_items if i["event_id"] == entry["event_id"])
        assert updated["decision"] == reviewed
        assert updated["boundary"] is False
    finally:
        os.unlink(review_path)
