#!/usr/bin/env python3
"""NeurIPS night 2 pipeline: runs AFTER R1-Distill probe steering finishes.

Deploy and run:
    scp scripts/neurips_night2.py root@<pod>:/workspace/s1s2/scripts/
    ssh root@<pod> "cd /workspace/s1s2 && nohup python scripts/neurips_night2.py &"

Resume after failure (skips completed jobs automatically):
    python scripts/neurips_night2.py

Skip / only:
    python scripts/neurips_night2.py --skip olmo32b_full
    python scripts/neurips_night2.py --only qwen_within_cot olmo7b_probe_steering

Priority order (estimated ~28h total):
    1. Qwen within-CoT extraction              ~8h GPU
    2. OLMo-32B-Instruct full pipeline         ~16h GPU
    3. OLMo-7B probe steering (L24)            ~4h GPU
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Environment — set before any HF/torch imports
# ---------------------------------------------------------------------------
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOG_FILE = Path("/workspace/neurips_night2.log")
STATE_FILE = Path("/workspace/neurips_night2_state.json")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_fh = None


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    global _log_fh
    if _log_fh is None:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _log_fh = open(LOG_FILE, "a")  # noqa: SIM115
    _log_fh.write(line + "\n")
    _log_fh.flush()


def banner(msg: str) -> None:
    sep = "=" * 72
    log(sep)
    log(msg)
    log(sep)


# ---------------------------------------------------------------------------
# GPU / CUDA helpers
# ---------------------------------------------------------------------------
def flush_cuda() -> None:
    """Aggressively free VRAM between jobs."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1e9
            if allocated > 0.5:
                log(f"  WARNING: {allocated:.1f} GB still allocated after flush")
    except Exception:
        pass


def check_gpu() -> None:
    """Verify CUDA is available and report VRAM. Exits on failure."""
    try:
        import torch
    except ImportError:
        log("FATAL: torch not installed")
        sys.exit(1)

    if not torch.cuda.is_available():
        log("FATAL: CUDA not available")
        sys.exit(1)

    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        log(f"  GPU {i}: {name} -- {total:.1f} GB VRAM")

    # Quick allocation smoke test
    try:
        x = torch.zeros(1, device="cuda")
        del x
        torch.cuda.empty_cache()
    except RuntimeError as e:
        log(f"FATAL: CUDA allocation test failed: {e}")
        sys.exit(1)

    log(f"GPU check passed: {torch.cuda.device_count()} device(s)")


# ---------------------------------------------------------------------------
# State management (checkpoint / resume)
# ---------------------------------------------------------------------------
def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def mark_completed(state: dict[str, Any], job_name: str, elapsed: float) -> None:
    state["completed"][job_name] = {
        "finished_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
    }
    state["failed"].pop(job_name, None)
    save_state(state)


def mark_failed(state: dict[str, Any], job_name: str, error: str) -> None:
    state["failed"][job_name] = {
        "failed_at": datetime.now(UTC).isoformat(),
        "error": error[:2000],
    }
    save_state(state)


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------
def run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command, streaming output to stdout and the log file."""
    log(f"  CMD: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd or PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    output_lines: list[str] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        output_lines.append(line)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {line}", flush=True)
        if _log_fh is not None:
            _log_fh.write(f"  [{ts}] {line}\n")
            _log_fh.flush()

    proc.wait()
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="\n".join(output_lines),
    )


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

def job_qwen_within_cot() -> None:
    """Job 1: Qwen within-CoT extraction (~8h GPU).

    Extract activations at multiple timepoints within the chain-of-thought
    for Qwen3-8B. Tests whether the S1/S2 direction evolves during generation.
    """
    result = run_cmd([
        sys.executable, str(SCRIPTS_DIR / "extract_qwen_toggle.py"),
        "--within-cot",
        "--model", "Qwen/Qwen3-8B",
        "--output", "data/activations/qwen3_8b_think_within_cot.h5",
        "--max-new-tokens", "4096",
        "--cache-dir", "/workspace/hf_cache",
    ])
    if result.returncode != 0:
        raise RuntimeError(
            f"extract_qwen_toggle.py exited with code {result.returncode}"
        )


def job_olmo32b_full() -> None:
    """Job 2: OLMo-32B-Instruct full pipeline (~16h GPU).

    Full pipeline: behavioral validation, activation extraction,
    probing, and cross-model comparison for the 32B scale model.
    """
    result = run_cmd([
        sys.executable, str(SCRIPTS_DIR / "run_olmo32b_full.py"),
    ])
    if result.returncode != 0:
        raise RuntimeError(
            f"run_olmo32b_full.py exited with code {result.returncode}"
        )


def job_olmo7b_probe_steering() -> None:
    """Job 3: OLMo-7B probe steering at layer 24 (~4h GPU).

    Causal intervention: train probes on S1/S2 direction at layer 24
    for OLMo-3-7B-Instruct, then steer generations along that direction.
    """
    result = run_cmd([
        sys.executable, str(SCRIPTS_DIR / "run_probe_steering.py"),
        "--model", "allenai/OLMo-3-7B-Instruct",
        "--h5-path", "data/activations/olmo3_instruct.h5",
        "--target-layer", "24",
        "--max-new-tokens", "128",
        "--output", "results/causal/probe_steering_olmo7b_l24.json",
    ])
    if result.returncode != 0:
        raise RuntimeError(
            f"run_probe_steering.py (OLMo-7B) exited with code {result.returncode}"
        )


# ---------------------------------------------------------------------------
# Job registry — priority order
# ---------------------------------------------------------------------------
JOBS: list[tuple[str, str, callable]] = [
    ("qwen_within_cot",       "Qwen within-CoT extraction",        job_qwen_within_cot),
    ("olmo32b_full",           "OLMo-32B-Instruct full pipeline",  job_olmo32b_full),
    ("olmo7b_probe_steering",  "OLMo-7B probe steering (L24)",     job_olmo7b_probe_steering),
]

# Estimated runtimes for the summary table (hours)
ESTIMATED_HOURS: dict[str, float] = {
    "qwen_within_cot": 8.0,
    "olmo32b_full": 16.0,
    "olmo7b_probe_steering": 4.0,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeurIPS night 2 pipeline: run remaining experiments with checkpoint/resume.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        metavar="JOB",
        help=(
            "Job names to skip. Available: "
            + ", ".join(name for name, _, _ in JOBS)
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        metavar="JOB",
        help="Run ONLY these jobs (in registry order). Overrides --skip.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear checkpoint state and re-run all jobs from scratch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing anything.",
    )
    parser.add_argument(
        "--wait-for-r1",
        action="store_true",
        help="Poll until R1-Distill probe steering process exits, then start.",
    )
    args = parser.parse_args()

    # Validate job names
    valid_names = {name for name, _, _ in JOBS}
    for name in args.skip + args.only:
        if name not in valid_names:
            print(f"ERROR: unknown job '{name}'. Valid: {sorted(valid_names)}")
            sys.exit(1)

    # --wait-for-r1: block until the R1 steering process is gone
    if args.wait_for_r1:
        _wait_for_r1_steering()

    banner("NEURIPS NIGHT 2 PIPELINE -- STARTING")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"State file:   {STATE_FILE}")
    log(f"Log file:     {LOG_FILE}")
    log(f"Python:       {sys.executable}")

    # GPU check
    check_gpu()

    # Load or reset state
    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        log("Checkpoint state reset.")
    state = load_state()

    # Determine which jobs to run
    skip_set = set(args.skip)
    only_set = set(args.only) if args.only else None

    jobs_to_run: list[tuple[str, str, callable]] = []
    for name, description, fn in JOBS:
        if only_set is not None and name not in only_set:
            log(f"  FILTERED (--only): {name}")
            continue
        if name in skip_set:
            log(f"  SKIPPED (--skip):  {name}")
            continue
        if name in state["completed"]:
            prev = state["completed"][name]
            log(f"  ALREADY DONE:      {name} (took {prev['elapsed_seconds']:.0f}s)")
            continue
        jobs_to_run.append((name, description, fn))

    if not jobs_to_run:
        banner("NO JOBS TO RUN -- all completed or skipped")
        return

    # Print queue with estimated total
    total_est = sum(ESTIMATED_HOURS.get(n, 0) for n, _, _ in jobs_to_run)
    log(f"\nJobs queued ({len(jobs_to_run)}), estimated ~{total_est:.0f}h total:")
    for name, description, _ in jobs_to_run:
        est = ESTIMATED_HOURS.get(name, 0)
        log(f"  - {name}: {description} (~{est:.0f}h)")
    log("")

    if args.dry_run:
        banner("DRY RUN -- exiting without executing")
        return

    # Execute jobs
    t_pipeline = time.time()
    results_summary: list[dict[str, Any]] = []

    for i, (name, description, fn) in enumerate(jobs_to_run, 1):
        banner(f"JOB {i}/{len(jobs_to_run)}: {description} [{name}]")
        t_job = time.time()

        try:
            fn()
            elapsed = time.time() - t_job
            mark_completed(state, name, elapsed)
            status = "OK"
            log(f"  Completed in {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
        except Exception as e:
            elapsed = time.time() - t_job
            tb = traceback.format_exc()
            mark_failed(state, name, tb)
            status = "FAILED"
            log(f"  FAILED after {elapsed:.0f}s: {e}")
            log(f"  Traceback:\n{tb}")

        results_summary.append({
            "job": name,
            "description": description,
            "status": status,
            "elapsed_seconds": round(elapsed, 1),
        })

        # Flush VRAM between all jobs regardless of outcome
        flush_cuda()

    # ---------------------------------------------------------------------------
    # Final summary table
    # ---------------------------------------------------------------------------
    total_elapsed = time.time() - t_pipeline
    banner("NEURIPS NIGHT 2 PIPELINE COMPLETE")
    log(f"Total wall time: {total_elapsed:.0f}s ({total_elapsed / 3600:.1f}h)")
    log("")
    log(f"{'Job':<28} {'Status':<10} {'Actual':>10} {'Est':>8}")
    log(f"{'-' * 28} {'-' * 10} {'-' * 10} {'-' * 8}")
    for r in results_summary:
        actual_h = r["elapsed_seconds"] / 3600
        est_h = ESTIMATED_HOURS.get(r["job"], 0)
        actual_str = f"{actual_h:.1f}h"
        est_str = f"~{est_h:.0f}h"
        log(f"{r['job']:<28} {r['status']:<10} {actual_str:>10} {est_str:>8}")

    n_ok = sum(1 for r in results_summary if r["status"] == "OK")
    n_fail = sum(1 for r in results_summary if r["status"] == "FAILED")
    log(f"\n{n_ok} succeeded, {n_fail} failed out of {len(results_summary)} jobs")

    if n_fail > 0:
        log("\nFailed jobs can be retried by re-running this script.")
        log("Completed jobs will be skipped automatically via checkpoint.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Wait for R1 steering to finish
# ---------------------------------------------------------------------------
def _wait_for_r1_steering() -> None:
    """Poll every 60s until no run_probe_steering.py process is running."""
    import shutil

    banner("WAITING FOR R1-DISTILL PROBE STEERING TO FINISH")

    poll_interval = 60  # seconds
    while True:
        # Check if any probe steering process is still alive
        result = subprocess.run(
            ["pgrep", "-f", "run_probe_steering.py"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # No matching process found -- R1 steering is done
            log("R1-Distill probe steering process not found -- proceeding.")
            break

        pids = result.stdout.strip().split("\n")
        # Filter out our own process
        own_pid = str(os.getpid())
        other_pids = [p for p in pids if p and p != own_pid]

        if not other_pids:
            log("R1-Distill probe steering process not found -- proceeding.")
            break

        ts = datetime.now().strftime("%H:%M:%S")
        print(
            f"  [{ts}] R1 steering still running (PIDs: {', '.join(other_pids)}). "
            f"Polling again in {poll_interval}s...",
            flush=True,
        )
        time.sleep(poll_interval)

    # Give CUDA a moment to release memory after the prior job exits
    time.sleep(5)
    flush_cuda()


if __name__ == "__main__":
    main()
