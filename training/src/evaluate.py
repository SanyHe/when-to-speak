"""
Evaluation harness for When-to-Speak (F1.5 / F2.5).

Computes the metrics defined in docs/PLAN.md section 7 on a held-out,
gold-labelled event set and compares models:

    precision / recall / macro-F1   (3-class, plus notify-specific)
    noise rate                      (wrong notify / total events)
    miss rate                       (notified events missed / should-notify)
    JSON validity rate

Two modes:
    --mode mock   deterministic mock policies (default; no GPU required)
    --mode hf     real HuggingFace checkpoints (run on the cloud GPU)

Outputs:
    data/eval/eval_report.json / m1_base_vs_sft.json
    data/eval/eval_report.md    (human-readable comparison table)
    data/eval/errors.jsonl      (archived error cases for the next iteration)
    web/data/traces.json        (events + per-model decisions for the player)

Usage (repo root):
    python3 training/src/evaluate.py \
        --eval data/synthetic/eval_set.jsonl \
        --models base,sft,sft_grpo \
        --out data/eval/eval_report.json \
        --md-out data/eval/eval_report.md \
        --errors-out data/eval/errors.jsonl \
        --web-out web/data/traces.json

    # F1.5 base vs SFT on the 100 seed events (mock):
    python3 training/src/evaluate.py \
        --eval data/synthetic/seed_labeled.jsonl \
        --models base,sft \
        --out data/eval/m1_base_vs_sft.json

    # Real checkpoints (cloud GPU only):
    python3 training/src/evaluate.py --mode hf \
        --models base=Qwen/Qwen3-0.6B,sft=output/full_sft/final,sft_grpo=output/grpo/final \
        --out data/eval/eval_report_hf.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

from mock_models import MODEL_SPECS, mock_output, parse_model_output  # noqa: E402
from schema import ACTIONS, validate_decision  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", ".."))

ACTION_ORDER = list(ACTIONS)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute classification + safety metrics for one model.

    ``records`` items: {event_id, gold, pred, json_valid}.
    ``gold`` / ``pred`` are schema-valid decision dicts.
    """
    total = len(records)
    correct = sum(1 for r in records if r["gold"]["action"] == r["pred"]["action"])
    valid = sum(1 for r in records if r.get("json_valid", True))

    notifiable = sum(1 for r in records if r["gold"]["action"] == "notify")
    predicted_notify = sum(1 for r in records if r["pred"]["action"] == "notify")
    true_notify = sum(
        1 for r in records
        if r["gold"]["action"] == "notify" and r["pred"]["action"] == "notify"
    )
    noisy = sum(
        1 for r in records
        if r["gold"]["action"] != "notify" and r["pred"]["action"] == "notify"
    )
    missed = sum(
        1 for r in records
        if r["gold"]["action"] == "notify" and r["pred"]["action"] != "notify"
    )

    per_class: Dict[str, Dict[str, float]] = {}
    for action in ACTION_ORDER:
        support = sum(1 for r in records if r["gold"]["action"] == action)
        predicted = sum(1 for r in records if r["pred"]["action"] == action)
        tp = sum(
            1 for r in records
            if r["gold"]["action"] == action and r["pred"]["action"] == action
        )
        precision = _safe_div(tp, predicted)
        recall = _safe_div(tp, support)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[action] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "predicted": predicted,
        }

    macro_f1 = sum(per_class[a]["f1"] for a in ACTION_ORDER) / len(ACTION_ORDER)

    confusion = {
        f"{gold}->{pred}": count
        for gold in ACTION_ORDER
        for pred in ACTION_ORDER
        if (count := sum(
            1 for r in records
            if r["gold"]["action"] == gold and r["pred"]["action"] == pred
        ))
    }

    return {
        "total": total,
        "json_valid_rate": round(_safe_div(valid, total), 4),
        "accuracy": round(_safe_div(correct, total), 4),
        "macro_f1": round(macro_f1, 4),
        "notify_precision": round(_safe_div(true_notify, predicted_notify), 4),
        "notify_recall": round(_safe_div(true_notify, notifiable), 4),
        "noise_rate": round(_safe_div(noisy, total), 4),
        "miss_rate": round(_safe_div(missed, notifiable), 4),
        "counts": {
            "should_notify": notifiable,
            "predicted_notify": predicted_notify,
            "true_notify": true_notify,
            "noisy_notify": noisy,
            "missed_notify": missed,
            "invalid_json": total - valid,
        },
        "per_class": per_class,
        "confusion": confusion,
    }


# ---------------------------------------------------------------------------
# Evaluation drivers
# ---------------------------------------------------------------------------

def load_labeled(path: str) -> List[Dict[str, Any]]:
    """Load gold-labelled events (each line carries a ``decision`` field)."""
    with open(path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if not items:
        raise RuntimeError(f"empty eval file: {path}")
    for item in items:
        if "decision" not in item:
            raise RuntimeError(f"eval item {item.get('event_id')} has no gold decision")
        result = validate_decision(item["decision"])
        if not result.valid:
            raise RuntimeError(f"invalid gold decision in {item['event_id']}: {result.errors}")
    return items


def event_of(item: Dict[str, Any]) -> Dict[str, Any]:
    """Strip label-only fields, keeping the raw event payload."""
    return {k: v for k, v in item.items() if k not in ("decision", "boundary", "reviewed")}


def evaluate_mock(items: Iterable[Dict[str, Any]],
                  model_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Run the deterministic mock policies over the eval set."""
    results: Dict[str, List[Dict[str, Any]]] = {}
    for model_id in model_ids:
        records = []
        for item in items:
            out = mock_output(model_id, event_of(item))
            records.append({
                "event_id": item["event_id"],
                "gold": item["decision"],
                "pred": out["decision"],
                "raw": out["raw"],
                "json_valid": out["json_valid"],
            })
        results[model_id] = records
    return results


def evaluate_hf(specs: Dict[str, str], items: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Run real checkpoints (requires transformers + torch on a GPU host)."""
    import torch  # noqa: F401  (lazy import: mock mode must stay dependency-free)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from format_sft import SYSTEM_PROMPT, render_chat

    results: Dict[str, List[Dict[str, Any]]] = {}
    for model_id, model_path in specs.items():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        model.eval()
        records = []
        for item in items:
            event = event_of(item)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(event, ensure_ascii=False)},
            ]
            prompt = render_chat(tokenizer, messages, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            ).strip()
            valid, decision = parse_model_output(raw)
            if not valid:
                from mock_models import fallback_decision
                decision = fallback_decision(event)
            records.append({
                "event_id": item["event_id"],
                "gold": item["decision"],
                "pred": decision,
                "raw": raw,
                "json_valid": valid,
            })
        results[model_id] = records
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _error_type(gold: Dict[str, Any], pred: Dict[str, Any], json_valid: bool) -> str:
    if not json_valid:
        return "invalid_json"
    if gold["action"] == pred["action"]:
        return "wrong_channel" if gold["channel"] != pred["channel"] else "exact_match"
    if pred["action"] == "notify":
        return "over_notify"
    if gold["action"] == "notify":
        return "missed_notify"
    return "wrong_action"


def build_error_cases(results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Archive every non-exact prediction for the next data/training round."""
    cases = []
    for model_id, records in results.items():
        for record in records:
            error_type = _error_type(record["gold"], record["pred"], record["json_valid"])
            if error_type == "exact_match":
                continue
            cases.append({
                "model": model_id,
                "event_id": record["event_id"],
                "error_type": error_type,
                "gold": record["gold"],
                "predicted": record["pred"],
                "json_valid": record["json_valid"],
                "raw": record["raw"],
            })
    return cases


def build_report(mode: str, eval_path: str, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    metrics = {model_id: compute_metrics(records) for model_id, records in results.items()}
    report = {
        "_meta": {
            "mode": mode,
            "eval_file": os.path.relpath(eval_path, REPO_ROOT),
            "total": len(next(iter(results.values()))) if results else 0,
            "model_ids": list(results.keys()),
            "note": (
                "MOCK VALUES: deterministic mock policies, not trained checkpoints. "
                "Re-measure on the AutoDL GPU and replace this evidence."
                if mode == "mock" else
                "REAL VALUES: measured on cloud GPU checkpoints."
            ),
        },
        "metrics": metrics,
        "error_cases": build_error_cases(results),
    }
    return report


def render_markdown(report: Dict[str, Any]) -> str:
    meta = report["_meta"]
    lines = [
        "# Evaluation Report",
        "",
        f"- Mode: **{meta['mode']}**",
        f"- Eval set: `{meta['eval_file']}` ({meta['total']} events)",
        f"- Note: {meta['note']}",
        "",
        "| Model | JSON valid | Precision (notify) | Recall (notify) | Noise rate | Miss rate | Macro-F1 | Accuracy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model_id, m in report["metrics"].items():
        lines.append(
            f"| {model_id} | {m['json_valid_rate']:.2%} | {m['notify_precision']:.3f} | "
            f"{m['notify_recall']:.3f} | {m['noise_rate']:.2%} | {m['miss_rate']:.2%} | "
            f"{m['macro_f1']:.3f} | {m['accuracy']:.2%} |"
        )
    lines += [
        "",
        "> Noise rate = wrong notify / all events. Miss rate = missed notify / all should-notify events.",
        "> Error cases are archived in the accompanying report JSON for the next iteration.",
        "",
    ]
    return "\n".join(lines)


def build_web_traces(items: List[Dict[str, Any]],
                     results: Dict[str, List[Dict[str, Any]]],
                     mode: str) -> Dict[str, Any]:
    """Bundle events + per-model decisions + metrics for the static player."""
    by_model: Dict[str, List[Dict[str, Any]]] = results
    events = []
    for index, item in enumerate(items):
        entry = {
            "event": event_of(item),
            "gold": item["decision"],
            "pred": {},
        }
        for model_id, records in by_model.items():
            record = records[index]
            entry["pred"][model_id] = {
                "raw": record["raw"],
                "decision": record["pred"],
                "json_valid": record["json_valid"],
            }
        events.append(entry)

    return {
        "_meta": {
            "mode": mode,
            "total": len(items),
            "note": (
                "Mock decision traces (deterministic). Compiled WebLLM model "
                "records will replace these once F3.1 is unblocked."
                if mode == "mock" else
                "Real checkpoint decision traces (cloud GPU)."
            ),
        },
        "models": [
            {
                "id": model_id,
                "label": MODEL_SPECS.get(model_id, {}).get("label", model_id),
                "tagline": MODEL_SPECS.get(model_id, {}).get("tagline", ""),
                "tagline_en": MODEL_SPECS.get(model_id, {}).get("tagline_en", ""),
                "description": MODEL_SPECS.get(model_id, {}).get("description", ""),
                "description_en": MODEL_SPECS.get(model_id, {}).get("description_en", ""),
            }
            for model_id in by_model.keys()
        ],
        "events": events,
        "metrics": {m: compute_metrics(r) for m, r in by_model.items()},
    }


def _write_json(path: str, obj: Any, indent: int | None = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def _write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _parse_model_arg(raw: str) -> Tuple[List[str], Dict[str, str]]:
    """Parse ``--models``: ``base,sft`` (mock) or ``sft=path,...`` (hf)."""
    ids: List[str] = []
    specs: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            model_id, path = part.split("=", 1)
            ids.append(model_id)
            specs[model_id] = path
        else:
            ids.append(part)
    return ids, specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "hf"), default="mock")
    parser.add_argument("--eval", default=os.path.join(
        REPO_ROOT, "data", "synthetic", "eval_set.jsonl"))
    parser.add_argument("--models", default=",".join(MODEL_SPECS.keys()))
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "data", "eval", "eval_report.json"))
    parser.add_argument("--md-out", default="")
    parser.add_argument("--errors-out", default="")
    parser.add_argument("--web-out", default="")
    args = parser.parse_args()

    model_ids, hf_specs = _parse_model_arg(args.models)
    unknown = [m for m in model_ids if m not in MODEL_SPECS]
    if unknown:
        raise SystemExit(f"unknown model id(s): {unknown}; allowed: {list(MODEL_SPECS)}")

    items = load_labeled(args.eval)
    if args.mode == "mock":
        results = evaluate_mock(items, model_ids)
    else:
        if not hf_specs:
            raise SystemExit("--mode hf requires --models id=path entries")
        results = evaluate_hf(hf_specs, items)

    report = build_report(args.mode, args.eval, results)
    _write_json(args.out, report)

    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write(render_markdown(report))
    if args.errors_out:
        _write_jsonl(args.errors_out, report["error_cases"])
    if args.web_out:
        _write_json(args.web_out, build_web_traces(items, results, args.mode), indent=None)

    print(f"Mode: {args.mode} | eval: {len(items)} events")
    for model_id, m in report["metrics"].items():
        print(
            f"  {model_id:<8} json_valid={m['json_valid_rate']:.2%} "
            f"P={m['notify_precision']:.3f} R={m['notify_recall']:.3f} "
            f"noise={m['noise_rate']:.2%} miss={m['miss_rate']:.2%} "
            f"macroF1={m['macro_f1']:.3f} acc={m['accuracy']:.2%}"
        )
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
