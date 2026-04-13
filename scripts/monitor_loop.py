#!/usr/bin/env python3
"""Continuous GPU monitor for s1s2 B200 pod.

SSHes to the pod every 120s, reads GPU state + pipeline progress,
appends a one-line status to /tmp/gpu_status.log and stdout.

Auto-restart: if GPU idles (0% for 3 consecutive checks) with incomplete
jobs, restarts the pipeline. If all jobs done and GPU idle, downloads
results and exits.

Usage:
    python scripts/monitor_loop.py                    # foreground
    nohup python scripts/monitor_loop.py &            # background
    python scripts/monitor_loop.py --interval 60      # faster polling
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SSH_CMD = [
    "ssh",
    "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
    "root@198.13.252.84",
    "-p", "44933",
    "-i", str(Path.home() / ".ssh" / "runpod_key"),
]

LOG_FILE = Path("/tmp/gpu_status.log")
REMOTE_STATE = "/workspace/gpu_pipeline_state.json"
REMOTE_PIPELINE_LOG = "/workspace/gpu_pipeline.log"
REMOTE_OLMO_LOG = "/workspace/olmo3_log.txt"
REMOTE_PROJECT = "/workspace/s1s2"
LOCAL_RESULTS = Path(__file__).resolve().parent.parent / "results_downloaded"

ALL_JOBS = [
    "sae_goodfire",
    "new_items",
    "olmo3_full",
    "reextract_activations",
    "attention_entropy",
    "bootstrap_cis",
]

IDLE_THRESHOLD = 3  # percent
IDLE_CHECKS_FOR_RESTART = 3
IDLE_CHECKS_FOR_DONE_EXIT = 3


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------
def ssh_exec(cmd: str, timeout: int = 30) -> tuple[int, str]:
    """Run a command on the pod via SSH. Returns (returncode, stdout)."""
    try:
        result = subprocess.run(
            [*SSH_CMD, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return -1, "SSH_TIMEOUT"
    except Exception as e:
        return -1, f"SSH_ERROR: {e}"


def scp_download(remote_path: str, local_path: Path) -> bool:
    """Download a file/dir from pod via SCP."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    scp_cmd = [
        "scp", "-r",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-P", "44933",
        "-i", str(Path.home() / ".ssh" / "runpod_key"),
        f"root@198.13.252.84:{remote_path}",
        str(local_path),
    ]
    try:
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pod queries
# ---------------------------------------------------------------------------
def get_gpu_status() -> dict | None:
    """Query nvidia-smi. Returns dict with util, mem_used, mem_total, temp."""
    rc, out = ssh_exec(
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu "
        "--format=csv,noheader,nounits"
    )
    if rc != 0 or not out:
        return None
    try:
        parts = [p.strip() for p in out.split(",")]
        return {
            "util": int(parts[0]),
            "mem_used_mib": int(parts[1]),
            "mem_total_mib": int(parts[2]),
            "temp_c": int(parts[3]),
        }
    except (ValueError, IndexError):
        return None


def get_pipeline_state() -> dict:
    """Read the pipeline state JSON from the pod."""
    rc, out = ssh_exec(f"cat {REMOTE_STATE} 2>/dev/null")
    if rc != 0 or not out:
        return {"completed": {}, "failed": {}}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"completed": {}, "failed": {}}


def get_last_log_line() -> str:
    """Get the last meaningful log line from pipeline or OLMo log."""
    # Try the OLMo-specific log first (has item progress)
    rc, out = ssh_exec(f"tail -1 {REMOTE_OLMO_LOG} 2>/dev/null")
    if rc == 0 and out and "[" in out:
        return out.strip()
    # Fall back to pipeline log
    rc, out = ssh_exec(
        f"grep -E '\\[\\d+/\\d+\\]|JOB |Completed|FAILED|PIPELINE' "
        f"{REMOTE_PIPELINE_LOG} 2>/dev/null | tail -1"
    )
    if rc == 0 and out:
        return out.strip()
    return "(no log)"


def get_running_jobs() -> list[str]:
    """Check what python processes are running on the pod."""
    rc, out = ssh_exec("ps aux | grep python | grep -v grep | grep -v wandb")
    if rc != 0:
        return []
    lines = out.strip().split("\n") if out.strip() else []
    jobs = []
    for line in lines:
        if "run_olmo3_full" in line:
            jobs.append("olmo3_full")
        elif "run_all_gpu" in line:
            jobs.append("pipeline_orchestrator")
        elif "run_sae_goodfire" in line:
            jobs.append("sae_goodfire")
        elif "run_new_items" in line:
            jobs.append("new_items")
        elif "extract_real" in line:
            jobs.append("reextract_activations")
        elif "extract_attention" in line:
            jobs.append("attention_entropy")
        elif "compute_bootstrap" in line:
            jobs.append("bootstrap_cis")
        elif "train_grpo" in line:
            jobs.append("grpo_training")
        elif "hf_hub_download" in line or "goodfire" in line.lower():
            jobs.append("sae_download")
    return jobs


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def restart_pipeline() -> bool:
    """Restart run_all_gpu.py on the pod (resumes from checkpoint)."""
    log_print("ACTION: Restarting pipeline on pod...")
    # Clear failed jobs so they retry
    rc, _ = ssh_exec(
        f"cd {REMOTE_PROJECT} && python3 -c \""
        "import json; "
        f"state = json.load(open('{REMOTE_STATE}')); "
        "state['failed'] = {}; "
        f"json.dump(state, open('{REMOTE_STATE}', 'w'), indent=2); "
        "print('Reset failed jobs. Completed:', list(state['completed'].keys()))"
        "\""
    )
    if rc != 0:
        log_print("  WARNING: Could not reset failed jobs state")

    rc, out = ssh_exec(
        f"cd {REMOTE_PROJECT} && "
        f"nohup python scripts/run_all_gpu.py >> {REMOTE_PIPELINE_LOG} 2>&1 & "
        "echo 'STARTED'"
    )
    success = "STARTED" in out
    if success:
        log_print("  Pipeline restart issued.")
    else:
        log_print(f"  Pipeline restart FAILED: {out}")
    return success


def download_results() -> bool:
    """Download results directory from pod."""
    log_print("ACTION: Downloading results from pod...")
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)

    paths_to_download = [
        (f"{REMOTE_PROJECT}/results/", LOCAL_RESULTS / "results"),
        (f"{REMOTE_PROJECT}/data/activations/", LOCAL_RESULTS / "activations"),
        (REMOTE_STATE, LOCAL_RESULTS / "gpu_pipeline_state.json"),
        (REMOTE_PIPELINE_LOG, LOCAL_RESULTS / "gpu_pipeline.log"),
    ]

    all_ok = True
    for remote, local in paths_to_download:
        log_print(f"  Downloading {remote} -> {local}")
        if not scp_download(remote, local):
            log_print(f"  WARNING: Failed to download {remote}")
            all_ok = False

    if all_ok:
        log_print(f"  All results downloaded to {LOCAL_RESULTS}")
    return all_ok


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_print(msg: str) -> None:
    """Print to stdout and append to /tmp/gpu_status.log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def format_oneliner(
    gpu: dict | None,
    state: dict,
    running: list[str],
    last_log: str,
) -> str:
    """Format a single status line."""
    if gpu is None:
        return "CONNECTION_FAILED"

    completed = list(state.get("completed", {}).keys())
    failed = list(state.get("failed", {}).keys())
    n_done = len(completed)
    n_total = len(ALL_JOBS)
    mem_gb = gpu["mem_used_mib"] / 1024

    active = ", ".join(running) if running else "idle"
    status_parts = [
        f"GPU:{gpu['util']:3d}%",
        f"VRAM:{mem_gb:.0f}/{gpu['mem_total_mib']/1024:.0f}GB",
        f"T:{gpu['temp_c']}C",
        f"Jobs:{n_done}/{n_total}done",
    ]
    if failed:
        status_parts.append(f"FAIL:{','.join(failed)}")
    status_parts.append(f"Active:[{active}]")

    # Truncate last log to fit
    max_log = 80
    log_short = last_log[:max_log] + "..." if len(last_log) > max_log else last_log
    status_parts.append(f"Log:{log_short}")

    return " | ".join(status_parts)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="GPU monitor for s1s2 B200 pod")
    parser.add_argument("--interval", type=int, default=120, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Single check, no loop")
    args = parser.parse_args()

    log_print("=" * 72)
    log_print("s1s2 GPU MONITOR STARTED")
    log_print(f"  Interval: {args.interval}s | Log: {LOG_FILE}")
    log_print(f"  Idle restart: GPU <{IDLE_THRESHOLD}%% for {IDLE_CHECKS_FOR_RESTART} checks")
    log_print("=" * 72)

    idle_count = 0
    check_num = 0

    while True:
        check_num += 1

        # Gather state
        gpu = get_gpu_status()
        state = get_pipeline_state()
        running = get_running_jobs()
        last_log = get_last_log_line()

        # Format and log
        oneliner = format_oneliner(gpu, state, running, last_log)
        log_print(oneliner)

        if gpu is None:
            log_print("  WARNING: Could not reach pod. Will retry.")
            if args.once:
                sys.exit(1)
            time.sleep(args.interval)
            continue

        completed = set(state.get("completed", {}).keys())
        all_done = completed >= set(ALL_JOBS)
        gpu_idle = gpu["util"] <= IDLE_THRESHOLD

        # Track consecutive idle checks
        if gpu_idle:
            idle_count += 1
        else:
            idle_count = 0

        # Decision logic
        if all_done and gpu_idle and idle_count >= IDLE_CHECKS_FOR_DONE_EXIT:
            log_print("ALL JOBS COMPLETE and GPU idle. Downloading results and exiting.")
            download_results()
            log_print("MONITOR EXIT: All done.")
            sys.exit(0)

        if gpu_idle and idle_count >= IDLE_CHECKS_FOR_RESTART and not all_done:
            # Check if the pipeline orchestrator is already running
            if "pipeline_orchestrator" not in running:
                remaining = set(ALL_JOBS) - completed
                log_print(
                    f"GPU idle for {idle_count} checks. "
                    f"Remaining jobs: {sorted(remaining)}. Restarting pipeline."
                )
                restart_pipeline()
                idle_count = 0
            else:
                log_print(
                    f"GPU idle ({idle_count} checks) but pipeline_orchestrator still running. "
                    "Waiting for it to launch a job."
                )

        if args.once:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
