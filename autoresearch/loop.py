"""Auto Research main loop.

Selects a recipe from the pre-registered queue, runs one trial, applies the
frozen gates, records the ledger entry, and checks every stop condition.
Only one loop process may run in this directory (flock).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

RESEARCH = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(RESEARCH, ".."))
sys.path.insert(0, RESEARCH)
sys.path.insert(0, os.path.join(REPO_ROOT, "training", "src"))

import gate  # noqa: E402
import recipe as recipe_mod  # noqa: E402
import runner  # noqa: E402

LEDGER = os.path.join(RESEARCH, "ledger.jsonl")
STATE = os.path.join(RESEARCH, "state.json")
STOP = os.path.join(RESEARCH, "STOP")
LOCK = os.path.join(RESEARCH, "loop.lock")
PID_FILE = os.path.join(RESEARCH, "loop.pid")
RUNS = os.path.join(RESEARCH, "runs")
ACCEPTED = os.path.join(RESEARCH, "accepted")
LOGS = os.path.join(RESEARCH, "logs")
FROZEN = os.path.join(RESEARCH, "frozen_hashes.json")
SEALED = os.path.join(RESEARCH, "data", "sealed_test.jsonl")
SEALED_MARKER = os.path.join(RESEARCH, ".sealed_opened")

GOAL = {
    "macro_f1": 0.97,
    "noise_rate": 0.005,
    "miss_rate": 0.005,
    "notify_recall": 0.99,
    "json_valid_rate": 0.999,
}

_HALT = False


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _write_json_atomic(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_ledger(entry: Dict[str, Any]) -> None:
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def verify_frozen() -> None:
    if not os.path.exists(FROZEN):
        raise SystemExit(f"missing {FROZEN}; run autoresearch/freeze.py first")
    with open(FROZEN, "r", encoding="utf-8") as f:
        frozen = json.load(f)
    mismatches = []
    for rel_path, expected in frozen["files"].items():
        path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(path):
            mismatches.append(f"{rel_path}: missing")
            continue
        actual = _sha256(path)
        if actual != expected:
            mismatches.append(f"{rel_path}: {actual[:12]} != {expected[:12]}")
    if mismatches:
        raise SystemExit(
            "FROZEN FILE MISMATCH — stop, get human approval, delete "
            "autoresearch/frozen_hashes.json, then restart:\n  " + "\n  ".join(mismatches)
        )
    log(f"frozen hashes verified ({len(frozen['files'])} files)")


def _load_predictions(run_dir: str) -> Optional[List[Dict[str, Any]]]:
    path = os.path.join(run_dir, "eval", "predictions.jsonl")
    if not os.path.exists(path):
        return None
    return _read_jsonl(path)


def _prune_weights(run_dir: str) -> None:
    for stage in ("sft", "grpo"):
        final = os.path.join(run_dir, stage, "final")
        if os.path.isdir(final):
            shutil.rmtree(final, ignore_errors=True)
            log(f"pruned weights: {os.path.relpath(final, REPO_ROOT)}")


def _goal_reached(metrics: Dict[str, Any]) -> bool:
    return (
        metrics["macro_f1"] >= GOAL["macro_f1"]
        and metrics["noise_rate"] <= GOAL["noise_rate"]
        and metrics["miss_rate"] <= GOAL["miss_rate"]
        and metrics["notify_recall"] >= GOAL["notify_recall"]
        and metrics["json_valid_rate"] >= GOAL["json_valid_rate"]
    )


def _incumbent_from_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trial_id": entry["trial_id"],
        "recipe": entry["recipe"],
        "metrics": entry["metrics"],
        "run_dir": entry["run_dir"],
        "model_for_eval": entry.get("model_for_eval"),
        "seed": entry.get("seed", 42),
    }


def init_state() -> Dict[str, Any]:
    """Restore state from state.json, or rebuild it from the ledger."""
    if os.path.exists(STATE):
        with open(STATE, "r", encoding="utf-8") as f:
            state = json.load(f)
        state.setdefault("trials_run", [])
        state.setdefault("gpu_minutes", 0.0)
        state.setdefault("no_improve", 0)
        state.setdefault("consecutive_failures", 0)
        return state

    ledger = _read_jsonl(LEDGER)
    incumbent = None
    trials_run: List[str] = []
    no_improve = 0
    consecutive_failures = 0
    gpu_minutes = 0.0
    for entry in ledger:
        trials_run.append(entry["trial_id"])
        gpu_minutes += float(entry.get("gpu_minutes", 0.0))
        if entry.get("decision") in ("accepted", "accepted_baseline"):
            incumbent = _incumbent_from_entry(entry)
            no_improve = 0
            consecutive_failures = 0
        else:
            no_improve += 1
            if entry.get("status") == "failed":
                consecutive_failures += 1
            else:
                consecutive_failures = 0
    return {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "incumbent": incumbent,
        "trials_run": trials_run,
        "gpu_minutes": round(gpu_minutes, 2),
        "no_improve": no_improve,
        "consecutive_failures": consecutive_failures,
        "last_trial": trials_run[-1] if trials_run else None,
    }


def _check_stop(state: Dict[str, Any], args: argparse.Namespace,
                started: float) -> Optional[str]:
    if os.path.exists(STOP):
        return "stop_file"
    incumbent = state.get("incumbent")
    if incumbent and _goal_reached(incumbent["metrics"]):
        return "goal_reached"
    if (time.time() - started) / 3600.0 > args.max_hours:
        return "max_hours"
    if state["gpu_minutes"] > args.max_gpu_minutes:
        return "max_gpu_minutes"
    if len(state["trials_run"]) >= args.max_trials:
        return "max_trials"
    if state["consecutive_failures"] >= 3:
        return "consecutive_failures"
    if state["no_improve"] >= args.patience:
        return "patience"
    return None


def _update_after(state: Dict[str, Any], entry: Dict[str, Any],
                  result: Dict[str, Any], confirmation: Optional[Dict[str, Any]]) -> None:
    state["trials_run"].append(entry["trial_id"])
    state["gpu_minutes"] = round(
        state["gpu_minutes"] + float(result.get("gpu_minutes", 0.0))
        + float((confirmation or {}).get("gpu_minutes", 0.0)), 2)
    state["last_trial"] = entry["trial_id"]
    if entry["decision"] in ("accepted", "accepted_baseline"):
        state["incumbent"] = _incumbent_from_entry(entry)
        state["no_improve"] = 0
        state["consecutive_failures"] = 0
    else:
        state["no_improve"] += 1
        state["consecutive_failures"] = (
            state["consecutive_failures"] + 1 if entry["status"] == "failed" else 0
        )
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json_atomic(STATE, state)


def _base_entry(result: Dict[str, Any], decision: str,
                reasons: List[str]) -> Dict[str, Any]:
    return {
        "trial_id": result["trial_id"],
        "recipe": result["recipe"],
        "status": result["status"],
        "decision": decision,
        "metrics": result.get("metrics"),
        "data_stats": result.get("data_stats"),
        "stage_seconds": result.get("stage_seconds"),
        "gpu_minutes": result.get("gpu_minutes", 0.0),
        "seed": result.get("seed", 42),
        "git_sha": _git_sha(),
        "reasons": reasons,
        "run_dir": result["run_dir"],
        "model_for_eval": result.get("model_for_eval"),
        "confirmation": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _run_sealed(incumbent: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Evaluate the incumbent on the sealed set exactly once."""
    run_dir = incumbent["run_dir"]
    metrics_path = os.path.join(run_dir, "sealed_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(SEALED_MARKER):
        log("sealed set was already opened; refusing to open it again")
        return None

    model_path = incumbent.get("model_for_eval")
    if not model_path or not os.path.isdir(model_path):
        log(f"cannot run sealed eval: weights missing at {model_path}")
        return None

    log(f"sealed evaluation (one time) for {incumbent['trial_id']}")
    out_dir = os.path.join(run_dir, "sealed_eval")
    result = runner.run_cmd([
        runner.PYTHON, os.path.join(RESEARCH, "eval_one.py"),
        "--model", model_path,
        "--eval-file", SEALED,
        "--out-dir", out_dir,
    ], os.path.join(run_dir, "logs", "sealed_eval.log"), 3600)
    if result["exit_code"] != 0:
        log(f"sealed eval failed: exit {result['exit_code']}")
        return None

    with open(os.path.join(out_dir, "metrics.json"), "r", encoding="utf-8") as f:
        payload = json.load(f)
    sealed = {
        "trial_id": incumbent["trial_id"],
        "eval_file": SEALED,
        "metrics": payload["metrics"],
        "slices": payload.get("slices", {}),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(sealed, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(SEALED_MARKER, "w", encoding="utf-8") as f:
        f.write(json.dumps({"trial_id": incumbent["trial_id"],
                            "timestamp": sealed["timestamp"]}) + "\n")
    return sealed


def finalize(state: Dict[str, Any], stop_reason: str) -> int:
    incumbent = state.get("incumbent")
    os.makedirs(ACCEPTED, exist_ok=True)

    sealed = _run_sealed(incumbent, state) if incumbent else None

    if incumbent:
        with open(os.path.join(ACCEPTED, f"{incumbent['trial_id']}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(incumbent["recipe"], f, indent=2, ensure_ascii=False)
            f.write("\n")

    lines = [
        "# Auto Research — Final Report",
        "",
        f"- Stop reason: **{stop_reason}**",
        f"- Finished at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Git commit: `{_git_sha()}`",
        f"- Trials run: {len(state['trials_run'])}",
        f"- GPU minutes: {state['gpu_minutes']:.1f}",
    ]
    if incumbent:
        m = incumbent["metrics"]
        lines += [
            "",
            "## Incumbent",
            "",
            f"- Trial: `{incumbent['trial_id']}` (seed {incumbent.get('seed', 42)})",
            f"- Dev (450): macro-F1 **{m['macro_f1']:.4f}**, recall "
            f"{m['notify_recall']:.4f}, precision {m['notify_precision']:.4f}, "
            f"noise {m['noise_rate']:.4f}, miss {m['miss_rate']:.4f}, "
            f"json {m['json_valid_rate']:.4f}",
        ]
        if sealed:
            s = sealed["metrics"]
            lines += [
                f"- Sealed (200): macro-F1 **{s['macro_f1']:.4f}**, recall "
                f"{s['notify_recall']:.4f}, precision {s['notify_precision']:.4f}, "
                f"noise {s['noise_rate']:.4f}, miss {s['miss_rate']:.4f}, "
                f"json {s['json_valid_rate']:.4f}",
            ]
        elif os.path.exists(SEALED_MARKER):
            lines.append("- Sealed (200): not re-opened (already used once)")
        lines += [
            "",
            "## Best recipe",
            "",
            "```json",
            json.dumps(incumbent["recipe"], indent=2, ensure_ascii=False),
            "```",
        ]

    lines += [
        "",
        "## Trial summary",
        "",
        "| trial | decision | macro-F1 | recall | noise | miss | json | GPU min |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in _read_jsonl(LEDGER):
        m = entry.get("metrics") or {}
        lines.append(
            f"| {entry['trial_id']} | {entry.get('decision', '')} | "
            f"{m.get('macro_f1', float('nan')):.4f} | "
            f"{m.get('notify_recall', float('nan')):.4f} | "
            f"{m.get('noise_rate', float('nan')):.4f} | "
            f"{m.get('miss_rate', float('nan')):.4f} | "
            f"{m.get('json_valid_rate', float('nan')):.4f} | "
            f"{entry.get('gpu_minutes', 0):.1f} |"
        )
    lines.append("")

    with open(os.path.join(RESEARCH, "FINAL_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    state["status"] = "finished"
    state["stop_reason"] = stop_reason
    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    if sealed:
        state["sealed_metrics"] = sealed["metrics"]
    _write_json_atomic(STATE, state)
    log(f"finalized: {stop_reason} (report: autoresearch/FINAL_REPORT.md)")
    return 0


def _handle_signal(signum, frame) -> None:  # noqa: ARG001
    global _HALT
    _HALT = True
    with open(STOP, "w", encoding="utf-8") as f:
        f.write(f"signal {signum}\n")
    log(f"signal {signum}: STOP written, finishing the current stage")


def run_smoke(args: argparse.Namespace) -> int:
    verify_frozen()
    run_dir = os.path.join(RUNS, "_smoke")
    result = runner.run_trial(recipe_mod.BASELINE, run_dir, seed=args.seed, smoke=True)
    log(f"smoke status={result['status']} metrics={result.get('metrics')}")
    if result["status"] != "ok":
        log(f"smoke error: {result.get('error')}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-hours", type=float, default=9.0)
    parser.add_argument("--max-gpu-minutes", type=float, default=480.0)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(ACCEPTED, exist_ok=True)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if args.smoke:
        return run_smoke(args)

    lock_handle = open(LOCK, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another loop already holds autoresearch/loop.lock; exiting")
        return 3
    with open(PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    try:
        verify_frozen()
        state = init_state()
        if state.get("status") == "finished":
            log(f"state is finished ({state.get('stop_reason')}); nothing to do")
            return 0

        started = time.time()
        log(f"loop start pid={os.getpid()} git={_git_sha()} "
            f"trials_run={len(state['trials_run'])} gpu_min={state['gpu_minutes']:.1f}")

        if "r00_baseline" not in state["trials_run"]:
            log("running baseline r00_baseline")
            result = runner.run_trial(
                recipe_mod.BASELINE, os.path.join(RUNS, "r00_baseline"), args.seed)
            entry = _base_entry(result, "accepted_baseline" if result["status"] == "ok" else "failed",
                                [] if result["status"] == "ok" else [result.get("error", "failed")])
            _append_ledger(entry)
            _update_after(state, entry, result, None)
            if result["status"] != "ok":
                log(f"baseline failed: {result.get('error')}")
                return finalize(state, "baseline_failed")
            log(f"baseline dev macro_f1={result['metrics']['macro_f1']:.4f} "
                f"noise={result['metrics']['noise_rate']:.4f} "
                f"miss={result['metrics']['miss_rate']:.4f}")
        else:
            if not state.get("incumbent"):
                log("ledger has no incumbent; cannot continue")
                return finalize(state, "no_incumbent")
            log(f"resuming with incumbent {state['incumbent']['trial_id']}")

        for recipe in recipe_mod.CANDIDATES:
            reason = _check_stop(state, args, started)
            if reason or _HALT:
                return finalize(state, reason or "stop_file")

            if recipe["name"] in state["trials_run"]:
                log(f"skip {recipe['name']} (already in ledger)")
                continue

            incumbent = state["incumbent"]
            run_dir = os.path.join(RUNS, recipe["name"])
            log(f"trial {recipe['name']} start")
            result = runner.run_trial(recipe, run_dir, args.seed)

            if result["status"] != "ok":
                entry = _base_entry(result, "failed", [result.get("error", "failed")])
                _append_ledger(entry)
                _update_after(state, entry, result, None)
                _prune_weights(run_dir)
                log(f"trial {recipe['name']} FAILED: {result.get('error')}")
                reason = _check_stop(state, args, started)
                if reason:
                    return finalize(state, reason)
                continue

            candidate_preds = _load_predictions(run_dir)
            incumbent_preds = _load_predictions(incumbent["run_dir"])
            decision = gate.evaluate_gates(
                result["metrics"], incumbent["metrics"],
                candidate_preds, incumbent_preds)

            confirmation = None
            confirmed_payload = None
            if decision["passed"]:
                log(f"trial {recipe['name']} passed gates; confirmation seed=43")
                confirmation_dir = run_dir + "__seed43"
                confirmation = runner.run_trial(recipe, confirmation_dir, 43)
                if confirmation["status"] == "ok":
                    check = gate.confirmation_passes(
                        result["metrics"], confirmation["metrics"], incumbent["metrics"])
                    confirmed_payload = {
                        "trial_id": confirmation["trial_id"],
                        "seed": 43,
                        "metrics": confirmation["metrics"],
                        "mean_macro_f1": check["mean_macro_f1"],
                        "passed": check["passed"],
                        "reasons": check["reasons"],
                        "run_dir": confirmation_dir,
                    }
                    if check["passed"]:
                        decision["passed"] = True
                        decision["confirmation"] = confirmed_payload
                    else:
                        decision["passed"] = False
                        decision["reasons"].extend(check["reasons"])
                else:
                    confirmed_payload = {
                        "trial_id": confirmation["trial_id"],
                        "seed": 43,
                        "status": "failed",
                        "error": confirmation.get("error"),
                        "passed": False,
                    }
                    decision["passed"] = False
                    decision["reasons"].append("confirmation run failed")

            if decision["passed"]:
                old_incumbent = state["incumbent"]
                entry = _base_entry(result, "accepted", decision["reasons"])
                entry["gates"] = decision["gates"]
                entry["p_better"] = decision.get("p_better")
                entry["confirmation"] = confirmed_payload
                _append_ledger(entry)
                _update_after(state, entry, result, confirmation)
                _prune_weights(confirmation_dir)
                if old_incumbent["run_dir"] != run_dir:
                    _prune_weights(old_incumbent["run_dir"])
                log(f"trial {recipe['name']} ACCEPTED "
                    f"macro_f1={result['metrics']['macro_f1']:.4f} "
                    f"mean2={confirmed_payload['mean_macro_f1']:.4f}")
            else:
                entry = _base_entry(result, "rejected", decision["reasons"])
                entry["gates"] = decision["gates"]
                entry["p_better"] = decision.get("p_better")
                entry["confirmation"] = confirmed_payload
                _append_ledger(entry)
                _update_after(state, entry, result, confirmation)
                _prune_weights(run_dir)
                if confirmation is not None:
                    _prune_weights(confirmation["run_dir"])
                log(f"trial {recipe['name']} REJECTED: {'; '.join(decision['reasons'])}")

            reason = _check_stop(state, args, started)
            if reason:
                return finalize(state, reason)

        return finalize(state, _check_stop(state, args, started) or "queue_exhausted")

    finally:
        try:
            if os.path.exists(PID_FILE):
                with open(PID_FILE, "r", encoding="utf-8") as f:
                    if f.read().strip() == str(os.getpid()):
                        os.remove(PID_FILE)
        except OSError:
            pass
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    sys.exit(main())
