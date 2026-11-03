"""
Deterministic mock model policies.

The real base / full-SFT / SFT+GRPO checkpoints are trained on a cloud GPU
(AutoDL) and are not available in the local toolchain. To unblock the
evaluation harness, the web player, and the end-to-end pipeline, this
module provides three *mock* policies with fixed, reproducible behaviour:

    base      - verbose, over-notifies, frequently emits malformed JSON
    sft       - learned the JSON contract, but still over-notifies on
                borderline events and misses a few urgent ones
    sft_grpo  - closest to the gold policy; lowest noise and miss rate

All mock behaviour is a pure function of (model_id, event_id): the same
event always yields the same output. The numbers are MOCK values — the
real model metrics must be re-measured on the cloud GPU (see
docs/cloud-training.md) and written back to private/feature_list.json.

The mock policies are intentionally simple: a hash-derived random draw
plus a perturbation of the rule-expert gold decision. This keeps the
demo honest (it is clearly labelled as mock) while exercising every
code path the real models will use.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

from expert_rules import _channel_for, decide  # noqa: E402
from schema import validate_decision  # noqa: E402

# Mock behaviour knobs. These are MOCK target rates, not measured metrics.
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "base": {
        "label": "Base",
        "tagline": "Qwen3-0.6B 原始模型（mock 行为）",
        "tagline_en": "Original Qwen3-0.6B (mock behaviour)",
        "description": "未后训练：格式不稳定，倾向过度打扰。",
        "description_en": "Not post-trained: unstable formatting, tends to over-notify.",
        "invalid_rate": 0.15,
        "over_notify": 0.34,
        "miss_notify": 0.10,
        "wrong_channel": 0.08,
        "verbose": True,
    },
    "sft": {
        "label": "SFT",
        "tagline": "全参监督微调（mock 行为）",
        "tagline_en": "Full-parameter supervised fine-tuning (mock behaviour)",
        "description": "学会严格 JSON 与基础决策，边界场景仍偏吵。",
        "description_en": "Learned strict JSON and basic decisions; still noisy on edge cases.",
        "invalid_rate": 0.02,
        "over_notify": 0.10,
        "miss_notify": 0.06,
        "wrong_channel": 0.05,
        "verbose": False,
    },
    "sft_grpo": {
        "label": "SFT + GRPO",
        "tagline": "GRPO 强化学习后的校准模型（mock 行为）",
        "tagline_en": "GRPO-calibrated model (mock behaviour)",
        "description": "打扰阈值校准：噪音显著下降，漏报不上升。",
        "description_en": "Calibrated interruption threshold: much less noise, no extra misses.",
        "invalid_rate": 0.005,
        "over_notify": 0.03,
        "miss_notify": 0.04,
        "wrong_channel": 0.01,
        "verbose": False,
    },
}

DEFAULT_MODELS = ("base", "sft", "sft_grpo")

_INVALID_RAW_VARIANTS = (
    "Sure! Here is my decision:\n{json}",
    '{{"action": "notify", "channel": "im", "urgency": "normal"',
    "```json\n{json}\n```",
)


def model_ids() -> tuple:
    """Return the canonical model ids in display order."""
    return DEFAULT_MODELS


def _draw(model_id: str, event_id: str, salt: str) -> float:
    """Stable pseudo-random value in [0, 1) for (model, event, salt)."""
    digest = hashlib.sha256(
        f"{model_id}|{event_id}|{salt}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _notify_channel(event: Dict[str, Any]) -> str:
    return _channel_for(event["source"], "notify")


def _notify_message(event: Dict[str, Any], verbose: bool) -> str:
    message = f"{event['subject']}：{event['snippet']}"
    if verbose:
        message += " 建议尽快查看，如需我继续跟进请直接回复。"
    return message


def _over_notify(event: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
    """Plausible but wrong 'interrupt now' decision."""
    return {
        "action": "notify",
        "channel": _notify_channel(event),
        "urgency": "urgent" if event["time_sensitive"] else "normal",
        "reason": "担心遗漏重要信息，选择立即通知",
        "message": _notify_message(event, verbose),
    }


def _silence(reason: str) -> Dict[str, Any]:
    return {
        "action": "silence",
        "channel": "none",
        "urgency": "low",
        "reason": reason,
        "message": "",
    }


def _digest(event: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
    return {
        "action": "digest",
        "channel": _channel_for(event["source"], "digest"),
        "urgency": "normal",
        "reason": "不急于打断，放入周期性汇总",
        "message": _notify_message(event, verbose),
    }


def _with_wrong_channel(decision: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the action but deliver through a plausible wrong channel."""
    if decision["action"] == "silence":
        return decision
    current = decision["channel"]
    alternatives = [c for c in ("im", "email", "push") if c != current]
    index = int(_draw("channel", event["event_id"], "pick") * len(alternatives))
    alt = alternatives[min(index, len(alternatives) - 1)]
    return {**decision, "channel": alt}


def _effective_decision(model_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
    """Return the schema-valid decision the mock model means to emit."""
    spec = MODEL_SPECS[model_id]
    gold = decide(event)
    eid = event["event_id"]

    decision: Dict[str, Any]
    if gold["action"] != "notify" and _draw(model_id, eid, "over") < spec["over_notify"]:
        decision = _over_notify(event, spec["verbose"])
    elif gold["action"] == "notify" and _draw(model_id, eid, "miss") < spec["miss_notify"]:
        decision = _silence("判断为可延后，暂不打扰")
    else:
        decision = dict(gold)
        if spec["verbose"]:
            decision["message"] = _notify_message(event, True) if decision["action"] != "silence" else ""
            decision["reason"] = decision["reason"] + "；为了不遗漏先提醒你"

    if (
        decision["action"] != "silence"
        and _draw(model_id, eid, "channel") < spec["wrong_channel"]
    ):
        decision = _with_wrong_channel(decision, event)

    return decision


def fallback_decision(event: Dict[str, Any]) -> Dict[str, Any]:
    """Error-recovery decision used when the model output cannot be parsed.

    The demo harness defaults to notifying rather than silently dropping an
    event: a malformed output is treated as an over-notification, which is
    exactly the failure mode the post-training is meant to reduce.
    """
    return {
        "action": "notify",
        "channel": _notify_channel(event),
        "urgency": "normal",
        "reason": "模型输出无法解析，回退为通知",
        "message": f"{event['subject']}：{event['snippet']}",
    }


def _render_raw(model_id: str, event: Dict[str, Any], decision: Dict[str, Any]) -> str:
    """Render the raw model text, injecting malformed output deterministically."""
    spec = MODEL_SPECS[model_id]
    payload = json.dumps(decision, ensure_ascii=False)
    if _draw(model_id, event["event_id"], "invalid") < spec["invalid_rate"]:
        variant = int(_draw(model_id, event["event_id"], "invalid_kind") * len(_INVALID_RAW_VARIANTS))
        raw = _INVALID_RAW_VARIANTS[min(variant, len(_INVALID_RAW_VARIANTS) - 1)]
        return raw.replace("{json}", payload)
    return payload


def mock_output(model_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
    """Produce one mock model output for ``event``.

    Returns ``raw`` (the text the model would emit), ``decision`` (the
    effective schema-valid decision after harness parsing/recovery) and
    ``json_valid`` (whether ``raw`` parsed as a strict JSON decision).
    """
    if model_id not in MODEL_SPECS:
        raise KeyError(f"unknown mock model: {model_id}")
    decision = _effective_decision(model_id, event)
    raw = _render_raw(model_id, event, decision)
    valid, parsed = parse_model_output(raw)
    return {
        "raw": raw,
        "decision": parsed if valid else fallback_decision(event),
        "json_valid": valid,
    }


def parse_model_output(raw: str) -> tuple:
    """Strictly parse a model output into a decision.

    Returns ``(valid, decision)``. Strict means: the whole string must be
    a single JSON object that passes the decision schema. Markdown fences,
    prose prefixes, truncated objects and extra keys are rejected.
    """
    text = raw.strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, None
    result = validate_decision(obj)
    if not result.valid:
        return False, None
    return True, obj


def main() -> None:
    """Smoke test: print one mock output per model for a sample event."""
    sample = {
        "event_id": "evt_smoke",
        "source": "email",
        "sender": "boss@company.com",
        "subject": "线上故障需要你确认",
        "snippet": "生产环境 P0 告警，需要你 10 分钟内确认处理方案。",
        "mentions_user": True,
        "is_direct": True,
        "time_sensitive": True,
        "received_at": "2026-10-31T09:12:00+08:00",
        "context": {"user_focus": "deep_work", "recent_notifications": 2, "duplicate_of": None},
    }
    for model_id in model_ids():
        out = mock_output(model_id, sample)
        print(f"[{model_id}] json_valid={out['json_valid']} decision={out['decision']}")


if __name__ == "__main__":
    main()
