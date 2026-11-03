"""
Format labeled events into SFT training samples.

Each labeled event (event + gold decision) becomes a chat-style sample:
    system  : task instruction (output strict JSON decision)
    user    : the event as JSON
    assistant: the gold decision as JSON

Only the assistant (decision) tokens contribute to the loss; the prompt
(system + user) is masked. This is handled by Unsloth/TRL's SFTTrainer
via the chat template + response masking.

Run locally (no GPU needed):
    python3 training/src/format_sft.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", ".."))

SYSTEM_PROMPT = (
    "You are an always-on agent that decides whether and how to notify the "
    "user about an incoming event. Given a single event object, respond with "
    "exactly one strict JSON object and nothing else. The JSON must contain "
    "exactly these keys: action, channel, urgency, reason, message.\n"
    "- action: one of notify | digest | silence\n"
    "- channel: one of im | email | push | none (use none only for silence)\n"
    "- urgency: one of urgent | normal | low\n"
    "- reason: a short justification string\n"
    "- message: the notification text (empty string when action is silence)\n"
    "Do not add any extra keys or text outside the JSON object."
)

DEFAULT_INPUT = os.path.join(REPO_ROOT, "data", "synthetic", "seed_labeled.jsonl")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "data", "synthetic", "sft_seed.jsonl")


def render_chat(tokenizer, messages: list, add_generation_prompt: bool) -> str:
    """Render chat messages with the tokenizer's template.

    Qwen3 defaults to thinking mode; training and evaluation must both
    disable it so the supervised target and the rollouts stay in the same
    format. Older tokenizers without the flag fall back to the plain call.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def to_sft_sample(labeled_event: dict) -> dict:
    """Convert a labeled event into a chat-style SFT sample."""
    event = {
        k: v for k, v in labeled_event.items()
        if k not in ("decision", "boundary")
    }
    decision = labeled_event["decision"]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(event, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        labeled = [json.loads(line) for line in f if line.strip()]

    samples = [to_sft_sample(e) for e in labeled]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} SFT samples -> {args.output}")


if __name__ == "__main__":
    main()
