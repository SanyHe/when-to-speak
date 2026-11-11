"""Execute one trial: materialize -> SFT -> GRPO -> dev evaluation.

Every heavy stage runs in its own process group with a timeout and a cleaned
environment. Some sandboxed execution environments prepend their own library
directories to LD_LIBRARY_PATH; those entries are dropped so the system
libsqlite3 is used (otherwise ``import trl`` fails).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(REPO_ROOT, "training", "src")
RESEARCH = os.path.join(REPO_ROOT, "autoresearch")
DATA_DIR = os.path.join(RESEARCH, "data")
MODEL_PATH = os.path.join(RESEARCH, "models", "Qwen3-0.6B")
PYTHON = os.path.join(REPO_ROOT, ".venv", "bin", "python")

# Library paths injected by sandboxed editors / execution wrappers.
# Override with a comma-separated SANDBOX_LIB_MARKERS if your environment
# needs different markers.
SANDBOX_LIB_MARKERS = tuple(
    marker for marker in os.environ.get("SANDBOX_LIB_MARKERS", "sbox,ai-agent").split(",")
    if marker
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_ops  # noqa: E402


def clean_env() -> Dict[str, str]:
    env = os.environ.copy()
    parts = [
        part for part in env.get("LD_LIBRARY_PATH", "").split(":")
        if part and not any(marker in part for marker in SANDBOX_LIB_MARKERS)
    ]
    if parts:
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _tail(path: str, lines: int = 40) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except FileNotFoundError:
        return ""


def run_cmd(cmd: List[str], log_path: str, timeout: int) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    started = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=clean_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait()
    return {
        "cmd": cmd,
        "exit_code": code,
        "seconds": round(time.time() - started, 1),
        "timed_out": timed_out,
        "log_tail": _tail(log_path),
    }


TRANSIENT_CUDA_MARKERS = (
    "CUDA initialization",
    "Error 304",
    "CUDA GPU required",
)


def run_stage(cmd: List[str], log_path: str, timeout: int,
              attempts: int = 6) -> Dict[str, Any]:
    """Run a stage, retrying only transient CUDA initialisation failures.

    The host GPU is occasionally preempted; a fresh process usually recovers
    within a couple of minutes. Back off up to 60s between attempts.
    """
    result: Dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        result = run_cmd(cmd, log_path, timeout)
        if result["exit_code"] == 0:
            result["attempts"] = attempt
            return result
        transient = any(marker in result["log_tail"] for marker in TRANSIENT_CUDA_MARKERS)
        if not transient or attempt == attempts:
            result["attempts"] = attempt
            return result
        time.sleep(min(20 * attempt, 60))
    return result


def run_trial(recipe: Dict[str, Any], run_dir: str, seed: int,
              smoke: bool = False) -> Dict[str, Any]:
    """Run one full trial and return a result record (never raises)."""
    started = time.time()
    stages: Dict[str, Any] = {}
    os.makedirs(run_dir, exist_ok=True)
    log_dir = os.path.join(run_dir, "logs")

    try:
        t0 = time.time()
        data_stats = data_ops.materialize(recipe, run_dir, smoke=smoke)
        stages["materialize"] = round(time.time() - t0, 2)
    except Exception as exc:  # noqa: BLE001
        return _failed(recipe, run_dir, seed, stages, f"materialize: {exc}", started)

    sft_cfg = recipe["sft"]
    sft_timeout = 1800 if smoke else 3600
    sft = run_stage([
        PYTHON, os.path.join(SRC, "full_sft.py"),
        "--model_path", MODEL_PATH,
        "--data", os.path.join(run_dir, "sft.jsonl"),
        "--output_dir", os.path.join(run_dir, "sft"),
        "--epochs", str(sft_cfg["epochs"]),
        "--batch_size", str(sft_cfg["batch_size"]),
        "--grad_accum_steps", str(sft_cfg["grad_accum_steps"]),
        "--learning_rate", str(sft_cfg["learning_rate"]),
        "--seed", str(seed),
    ], os.path.join(log_dir, "sft.log"), sft_timeout)
    stages["sft"] = sft
    if sft["exit_code"] != 0:
        return _failed(recipe, run_dir, seed, stages,
                       f"sft exit {sft['exit_code']}", started, data_stats)

    model_for_eval = os.path.join(run_dir, "sft", "final")
    grpo_cfg = recipe["grpo"]
    if grpo_cfg.get("enabled", True):
        grpo_timeout = 3600 if smoke else 7200
        grpo = run_stage([
            PYTHON, os.path.join(RESEARCH, "train_grpo.py"),
            "--model_path", os.path.join(run_dir, "sft", "final"),
            "--data", os.path.join(run_dir, "grpo.jsonl"),
            "--output_dir", os.path.join(run_dir, "grpo"),
            "--epochs", str(grpo_cfg["epochs"]),
            "--batch_size", str(grpo_cfg["batch_size"]),
            "--grad_accum_steps", str(grpo_cfg["grad_accum_steps"]),
            "--learning_rate", str(grpo_cfg["learning_rate"]),
            "--num_generations", str(grpo_cfg["num_generations"]),
            "--max_completion_length", str(grpo_cfg["max_completion_length"]),
            "--beta", str(grpo_cfg["beta"]),
            "--temperature", str(grpo_cfg["temperature"]),
            "--seed", str(seed),
            "--weights", json.dumps(grpo_cfg["reward_weights"]),
        ], os.path.join(log_dir, "grpo.log"), grpo_timeout)
        stages["grpo"] = grpo
        if grpo["exit_code"] != 0:
            return _failed(recipe, run_dir, seed, stages,
                           f"grpo exit {grpo['exit_code']}", started, data_stats)
        model_for_eval = os.path.join(run_dir, "grpo", "final")

    eval_file = os.path.join(DATA_DIR, "dev.jsonl")
    max_items = 0
    if smoke:
        eval_file = os.path.join(run_dir, "dev_smoke.jsonl")
        _write_smoke_dev(eval_file, 8)
        max_items = 8

    eval_timeout = 900 if smoke else 3600
    eval_result = run_stage([
        PYTHON, os.path.join(RESEARCH, "eval_one.py"),
        "--model", model_for_eval,
        "--eval-file", eval_file,
        "--out-dir", os.path.join(run_dir, "eval"),
        "--max-items", str(max_items),
    ], os.path.join(log_dir, "eval.log"), eval_timeout)
    stages["eval"] = eval_result
    if eval_result["exit_code"] != 0:
        return _failed(recipe, run_dir, seed, stages,
                       f"eval exit {eval_result['exit_code']}", started, data_stats)

    with open(os.path.join(run_dir, "eval", "metrics.json"), "r", encoding="utf-8") as f:
        eval_payload = json.load(f)

    elapsed = time.time() - started
    result = {
        "trial_id": recipe["name"] + ("" if seed == 42 else f"__seed{seed}"),
        "recipe": recipe,
        "seed": seed,
        "status": "ok",
        "run_dir": run_dir,
        "model_for_eval": model_for_eval,
        "metrics": eval_payload["metrics"],
        "slices": eval_payload.get("slices", {}),
        "data_stats": data_stats,
        "stage_seconds": stages,
        "wall_seconds": round(elapsed, 1),
        "gpu_minutes": round(elapsed / 60.0, 2),
        "smoke": smoke,
    }
    _write_json(os.path.join(run_dir, "trial_result.json"), result)
    return result


def _write_smoke_dev(path: str, count: int) -> None:
    with open(os.path.join(DATA_DIR, "dev.jsonl"), "r", encoding="utf-8") as f:
        rows = [line for line in f if line.strip()][:count]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(rows)


def _write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _failed(recipe: Dict[str, Any], run_dir: str, seed: int,
            stages: Dict[str, Any], error: str, started: float,
            data_stats: Dict[str, Any] | None = None) -> Dict[str, Any]:
    result = {
        "trial_id": recipe["name"] + ("" if seed == 42 else f"__seed{seed}"),
        "recipe": recipe,
        "seed": seed,
        "status": "failed",
        "run_dir": run_dir,
        "error": error,
        "data_stats": data_stats or {},
        "stage_seconds": stages,
        "wall_seconds": round(time.time() - started, 1),
        "gpu_minutes": round((time.time() - started) / 60.0, 2),
    }
    _write_json(os.path.join(run_dir, "trial_result.json"), result)
    return result


if __name__ == "__main__":
    from recipe import BASELINE

    print(json.dumps(run_trial(BASELINE, os.path.join(RESEARCH, "runs", "_manual"),
                               seed=42, smoke=True), indent=2)[:4000])
