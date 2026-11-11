"""Fixed acceptance gates and the paired bootstrap.

Gates 1-4 are reject-fast. Gate 5 requires a real improvement over the
incumbent, confirmed by a paired bootstrap over the same dev events.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from schema import ACTIONS

ACTION_ORDER = list(ACTIONS)


def _f1(records: List[Dict[str, Any]], indices: List[int]) -> float:
    total = len(indices)
    f1_sum = 0.0
    for action in ACTION_ORDER:
        tp = support = predicted = 0
        for index in indices:
            record = records[index]
            gold = record["gold"]["action"]
            pred = record["pred"]["action"]
            if gold == action:
                support += 1
            if pred == action:
                predicted += 1
            if gold == action and pred == action:
                tp += 1
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1_sum += (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return f1_sum / len(ACTION_ORDER) if total else 0.0


def paired_bootstrap(candidate: List[Dict[str, Any]],
                     incumbent: List[Dict[str, Any]],
                     n_boot: int = 2000,
                     seed: int = 2026) -> float:
    """P(candidate macro-F1 > incumbent macro-F1) on paired event resamples."""
    by_id = {record["event_id"]: record for record in incumbent}
    cand = [record for record in candidate if record["event_id"] in by_id]
    inc = [by_id[record["event_id"]] for record in cand]
    if not cand:
        return 0.0
    # Both lists are now aligned by event_id.
    aligned_inc = inc
    aligned_cand = cand
    n = len(aligned_cand)

    rng = random.Random(seed)
    better = 0
    for _ in range(n_boot):
        indices = [rng.randrange(n) for _ in range(n)]
        if _f1(aligned_cand, indices) > _f1(aligned_inc, indices):
            better += 1
    return better / n_boot


def evaluate_gates(metrics: Dict[str, Any],
                   incumbent: Optional[Dict[str, Any]],
                   candidate_records: Optional[List[Dict[str, Any]]] = None,
                   incumbent_records: Optional[List[Dict[str, Any]]] = None,
                   n_boot: int = 2000) -> Dict[str, Any]:
    """Run gates 1-5 and return a decision record."""
    reasons: List[str] = []
    gate_results: Dict[str, Any] = {}

    gate_results["g1_json"] = metrics["json_valid_rate"] >= 0.99
    if not gate_results["g1_json"]:
        reasons.append(f"json_valid_rate {metrics['json_valid_rate']:.4f} < 0.99")

    gate_results["g2_noise"] = metrics["noise_rate"] <= 0.02
    if not gate_results["g2_noise"]:
        reasons.append(f"noise_rate {metrics['noise_rate']:.4f} > 0.02")

    gate_results["g3_miss"] = metrics["miss_rate"] <= 0.02
    if not gate_results["g3_miss"]:
        reasons.append(f"miss_rate {metrics['miss_rate']:.4f} > 0.02")

    if incumbent is None:
        return {
            "passed": all(gate_results.values()),
            "reasons": reasons,
            "gates": gate_results,
            "p_better": None,
            "delta_macro_f1": None,
        }

    gate_results["g4_no_regression"] = (
        metrics["noise_rate"] <= incumbent["noise_rate"] + 0.005
        and metrics["miss_rate"] <= incumbent["miss_rate"] + 0.005
    )
    if not gate_results["g4_no_regression"]:
        reasons.append(
            "regression: noise {:.4f} vs {:.4f}, miss {:.4f} vs {:.4f}".format(
                metrics["noise_rate"], incumbent["noise_rate"],
                metrics["miss_rate"], incumbent["miss_rate"],
            )
        )

    if not all(gate_results.values()):
        return {
            "passed": False,
            "reasons": reasons,
            "gates": gate_results,
            "p_better": None,
            "delta_macro_f1": round(metrics["macro_f1"] - incumbent["macro_f1"], 4),
        }

    p_better = None
    if candidate_records is not None and incumbent_records is not None:
        p_better = paired_bootstrap(candidate_records, incumbent_records, n_boot=n_boot)

    f1_gain = metrics["macro_f1"] >= incumbent["macro_f1"] + 0.005
    boot_ok = p_better is not None and p_better >= 0.80
    recall_gain = (
        metrics["notify_recall"] >= incumbent["notify_recall"] + 0.02
        and metrics["noise_rate"] <= incumbent["noise_rate"] + 0.005
    )
    gate_results["g5_significant"] = (f1_gain and boot_ok) or recall_gain
    if not gate_results["g5_significant"]:
        reasons.append(
            "no significant gain: macro_f1 {:.4f} vs {:.4f} (need +0.005, p={}), "
            "notify_recall {:.4f} vs {:.4f} (need +0.02)".format(
                metrics["macro_f1"], incumbent["macro_f1"],
                "n/a" if p_better is None else f"{p_better:.3f}",
                metrics["notify_recall"], incumbent["notify_recall"],
            )
        )

    return {
        "passed": all(gate_results.values()),
        "reasons": reasons,
        "gates": gate_results,
        "p_better": p_better,
        "delta_macro_f1": round(metrics["macro_f1"] - incumbent["macro_f1"], 4),
    }


def confirmation_passes(primary: Dict[str, Any], confirmation: Dict[str, Any],
                        incumbent: Dict[str, Any]) -> Dict[str, Any]:
    """Two-seed confirmation: both seeds safe, mean macro-F1 beats incumbent."""
    reasons: List[str] = []
    for name, metrics in (("primary", primary), ("confirmation", confirmation)):
        if metrics["json_valid_rate"] < 0.99:
            reasons.append(f"{name}: json_valid_rate < 0.99")
        if metrics["noise_rate"] > 0.02:
            reasons.append(f"{name}: noise_rate > 0.02")
        if metrics["miss_rate"] > 0.02:
            reasons.append(f"{name}: miss_rate > 0.02")
        if metrics["noise_rate"] > incumbent["noise_rate"] + 0.005:
            reasons.append(f"{name}: noise regressed")
        if metrics["miss_rate"] > incumbent["miss_rate"] + 0.005:
            reasons.append(f"{name}: miss regressed")

    mean_f1 = (primary["macro_f1"] + confirmation["macro_f1"]) / 2
    if mean_f1 <= incumbent["macro_f1"]:
        reasons.append(
            f"two-seed mean macro_f1 {mean_f1:.4f} does not beat incumbent "
            f"{incumbent['macro_f1']:.4f}"
        )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "mean_macro_f1": round(mean_f1, 4),
    }
