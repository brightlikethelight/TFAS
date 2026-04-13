#!/usr/bin/env python3
"""One-shot GPU deployment orchestrator for all pending s1s2 work.

Runs every pending GPU job sequentially on a B200 pod, with checkpoint/resume,
per-job timing, and failure isolation.

Deploy and run:
    scp scripts/run_all_gpu.py root@<pod>:/workspace/s1s2/scripts/
    ssh root@<pod> "cd /workspace/s1s2 && python scripts/run_all_gpu.py"

Resume after failure (skips completed jobs automatically):
    python scripts/run_all_gpu.py

Skip specific jobs:
    python scripts/run_all_gpu.py --skip sae_goodfire new_items

Reset checkpoint state and run everything fresh:
    python scripts/run_all_gpu.py --reset
"""
from __future__ import annotations

import argparse
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
# Environment — set before any imports that touch HF/torch
# ---------------------------------------------------------------------------
os.environ["HF_HOME"] = "/workspace/hf_cache"
# Set HF_TOKEN via environment variable or .env file — never hardcode
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
STATE_FILE = Path("/workspace/gpu_pipeline_state.json")
LOG_FILE = Path("/workspace/gpu_pipeline_log.txt")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_fh = None


def log(msg: str) -> None:
    """Print to stdout and append to the persistent log file."""
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
# GPU check
# ---------------------------------------------------------------------------
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

    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        log(f"  GPU {i}: {name} — {total:.1f} GB VRAM")

    # Quick allocation test
    try:
        x = torch.zeros(1, device="cuda")
        del x
        torch.cuda.empty_cache()
    except RuntimeError as e:
        log(f"FATAL: CUDA allocation test failed: {e}")
        sys.exit(1)

    log(f"GPU check passed: {n_gpus} device(s)")


# ---------------------------------------------------------------------------
# Checkpoint state
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
    # Clear from failed if it was there from a prior run
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
        # Write to both stdout and log
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

def job_sae_goodfire() -> None:
    """Job 1: SAE Goodfire L19 analysis (~30 min).

    Runs the self-contained SAE analysis script on existing Llama activations.
    """
    result = run_cmd(
        [sys.executable, str(SCRIPTS_DIR / "run_sae_goodfire.py")],
    )
    if result.returncode != 0:
        raise RuntimeError(f"run_sae_goodfire.py exited with code {result.returncode}")


def job_new_items() -> None:
    """Job 2: New items behavioral validation (~1 hr).

    Runs Llama + R1-Distill on sunk cost and natural frequency items.
    """
    result = run_cmd(
        [sys.executable, str(SCRIPTS_DIR / "run_new_items.py")],
    )
    if result.returncode != 0:
        raise RuntimeError(f"run_new_items.py exited with code {result.returncode}")


def job_olmo3_full() -> None:
    """Job 3: OLMo-3-7B pair pipeline (~3 hr).

    Downloads OLMo-3-7B-Instruct and OLMo-3-7B-Think, runs behavioral
    validation, extracts activations, runs probes.
    """
    result = run_cmd(
        [sys.executable, str(SCRIPTS_DIR / "run_olmo3_full.py")],
    )
    if result.returncode != 0:
        raise RuntimeError(f"run_olmo3_full.py exited with code {result.returncode}")


def job_reextract_activations() -> None:
    """Job 4: Re-extract activations for expanded benchmark (~2 hr).

    Benchmark grew from 330 to 380+ items. Re-run extraction for Llama
    and R1-Distill on all items.
    """
    models = [
        {
            "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
            "output": "data/activations/llama31_8b_instruct.h5",
            "max_new_tokens": "256",
        },
        {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "output": "data/activations/r1_distill_llama.h5",
            "max_new_tokens": "2048",
        },
    ]
    for m in models:
        log(f"  Extracting activations: {m['model']}")
        result = run_cmd([
            sys.executable, str(SCRIPTS_DIR / "extract_real.py"),
            "--model", m["model"],
            "--output", m["output"],
            "--max-new-tokens", m["max_new_tokens"],
        ])
        if result.returncode != 0:
            raise RuntimeError(
                f"extract_real.py failed for {m['model']} "
                f"(exit code {result.returncode})"
            )
        # Free VRAM between models
        _flush_cuda()


def job_attention_entropy() -> None:
    """Job 5: Attention entropy extraction (~1 hr).

    Extract per-head attention metrics for Llama and R1-Distill using
    eager attention (not Flash Attention) so the full attention matrix
    is materialized.
    """
    models = [
        {
            "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
            "output": "results/attention/llama31_attention.json",
        },
        {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "output": "results/attention/r1_distill_attention.json",
        },
    ]
    for m in models:
        log(f"  Extracting attention: {m['model']}")
        result = run_cmd([
            sys.executable, str(SCRIPTS_DIR / "extract_attention.py"),
            "--model", m["model"],
            "--output", m["output"],
        ])
        if result.returncode != 0:
            raise RuntimeError(
                f"extract_attention.py failed for {m['model']} "
                f"(exit code {result.returncode})"
            )
        _flush_cuda()


def job_bootstrap_cis() -> None:
    """Job 6: Bootstrap confidence intervals (~30 min, CPU-bound).

    Compute bootstrap CIs for probe AUCs using compute_bootstrap_cis.py,
    which reads activations directly from HDF5 and runs 1000-resample
    bootstrap per layer. Runs on each model's activation file.
    """
    h5_files = [
        "data/activations/llama31_8b_instruct.h5",
        "data/activations/r1_distill_llama.h5",
    ]
    for h5 in h5_files:
        h5_path = PROJECT_ROOT / h5
        if not h5_path.exists():
            log(f"  WARNING: {h5} not found, skipping bootstrap for it")
            continue
        log(f"  Bootstrap CIs for {h5}")
        result = run_cmd([
            sys.executable, str(SCRIPTS_DIR / "compute_bootstrap_cis.py"),
            "--h5-path", h5,
            "--output-dir", "results/bootstrap_cis/",
            "--n-bootstrap", "1000",
        ])
        if result.returncode != 0:
            raise RuntimeError(
                f"compute_bootstrap_cis.py failed for {h5} "
                f"(exit code {result.returncode})"
            )


def _flush_cuda() -> None:
    """Best-effort CUDA cache flush between models."""
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Job registry — order matters (priority order from spec)
# ---------------------------------------------------------------------------
JOBS: list[tuple[str, str, callable]] = [
    ("sae_goodfire",          "SAE Goodfire L19 analysis",                job_sae_goodfire),
    ("new_items",             "New items behavioral validation",          job_new_items),
    ("olmo3_full",            "OLMo-3-7B full pipeline",                 job_olmo3_full),
    ("reextract_activations", "Re-extract activations (expanded bench)",  job_reextract_activations),
    ("attention_entropy",     "Attention entropy extraction",             job_attention_entropy),
    ("bootstrap_cis",         "Bootstrap confidence intervals",           job_bootstrap_cis),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all pending s1s2 GPU jobs sequentially with checkpoint/resume.",
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
    args = parser.parse_args()

    # Validate job names
    valid_names = {name for name, _, _ in JOBS}
    for name in args.skip + args.only:
        if name not in valid_names:
            print(f"ERROR: unknown job '{name}'. Valid: {sorted(valid_names)}")
            sys.exit(1)

    banner("S1/S2 GPU PIPELINE — STARTING")
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
            log(f"  ALREADY DONE:      {name} (took {prev['elapsed_seconds']}s)")
            continue
        jobs_to_run.append((name, description, fn))

    if not jobs_to_run:
        banner("NO JOBS TO RUN — all completed or skipped")
        return

    log(f"\nJobs queued ({len(jobs_to_run)}):")
    for name, description, _ in jobs_to_run:
        log(f"  - {name}: {description}")
    log("")

    if args.dry_run:
        banner("DRY RUN — exiting without executing")
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
            log(f"  Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
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

        # Flush VRAM between jobs regardless of success/failure
        _flush_cuda()

    # Final report
    total_elapsed = time.time() - t_pipeline
    banner("PIPELINE COMPLETE")
    log(f"Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min, {total_elapsed/3600:.1f} hr)")
    log("")
    log(f"{'Job':<28} {'Status':<10} {'Time':>10}")
    log(f"{'-'*28} {'-'*10} {'-'*10}")
    for r in results_summary:
        t_str = f"{r['elapsed_seconds']:.0f}s"
        log(f"{r['job']:<28} {r['status']:<10} {t_str:>10}")

    n_ok = sum(1 for r in results_summary if r["status"] == "OK")
    n_fail = sum(1 for r in results_summary if r["status"] == "FAILED")
    log(f"\n{n_ok} succeeded, {n_fail} failed out of {len(results_summary)} jobs")

    if n_fail > 0:
        log("\nFailed jobs can be retried by re-running this script.")
        log("Completed jobs will be skipped automatically.")
        sys.exit(1)


if __name__ == "__main__":
    main()
