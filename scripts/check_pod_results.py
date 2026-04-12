#!/usr/bin/env python3
"""Check and download results from a RunPod GPU pod.

Lists all result files on the pod, shows sizes, downloads new/updated
files to the local results/ directory, and prints a pipeline status summary.

Usage:
    python scripts/check_pod_results.py "ssh root@IP -p PORT -i KEY"
    python scripts/check_pod_results.py "ssh root@IP -p PORT -i KEY" --download
    python scripts/check_pod_results.py "ssh root@IP -p PORT -i KEY" --download --include-activations
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_RESULTS = PROJECT_ROOT / "results"
REMOTE_DIR = "/workspace/s1s2"
STATE_FILE = "/workspace/gpu_pipeline_state.json"
LOG_FILE = "/workspace/gpu_pipeline_log.txt"

# Directories on the pod that contain results worth downloading
RESULT_DIRS = [
    "results/",
    "outputs/",
    "figures/",
]

# Large directories that are only downloaded with --include-activations
LARGE_DIRS = [
    "data/activations/",
]


# ---------------------------------------------------------------------------
# SSH / SCP helpers
# ---------------------------------------------------------------------------
def parse_ssh_cmd(ssh_cmd: str) -> tuple[list[str], str, list[str]]:
    """Parse SSH command string into (ssh_args, user_host, scp_opts).

    Given: "ssh root@1.2.3.4 -p 12345 -i ~/.ssh/key"
    Returns:
        ssh_args  = ["ssh", "root@1.2.3.4", "-p", "12345", "-i", "~/.ssh/key"]
        user_host = "root@1.2.3.4"
        scp_opts  = ["-P", "12345", "-i", "~/.ssh/key"]
    """
    parts = shlex.split(ssh_cmd)
    ssh_args = parts  # Full SSH command as list

    user_host = ""
    scp_opts: list[str] = []
    i = 1  # skip "ssh"
    while i < len(parts):
        if parts[i] == "-p":
            scp_opts.extend(["-P", parts[i + 1]])
            i += 2
        elif parts[i] == "-i":
            scp_opts.extend(["-i", parts[i + 1]])
            i += 2
        elif parts[i] == "-o":
            scp_opts.extend(["-o", parts[i + 1]])
            i += 2
        elif parts[i].startswith("-"):
            scp_opts.append(parts[i])
            i += 1
        else:
            user_host = parts[i]
            i += 1

    return ssh_args, user_host, scp_opts


def ssh_exec(ssh_args: list[str], cmd: str, timeout: int = 30) -> str | None:
    """Execute a command on the pod via SSH. Returns stdout or None on failure."""
    full_cmd = ssh_args + [
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        cmd,
    ]
    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def scp_download(
    user_host: str,
    scp_opts: list[str],
    remote_path: str,
    local_path: str,
    recursive: bool = True,
) -> bool:
    """Download a file/dir from the pod. Returns True on success."""
    cmd = ["scp", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no"]
    cmd.extend(scp_opts)
    if recursive:
        cmd.append("-r")
    cmd.append(f"{user_host}:{remote_path}")
    cmd.append(local_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Pod interrogation
# ---------------------------------------------------------------------------
def get_pipeline_state(ssh_args: list[str]) -> dict[str, Any]:
    """Read the pipeline state JSON from the pod."""
    raw = ssh_exec(ssh_args, f"cat {STATE_FILE} 2>/dev/null")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def list_remote_files(ssh_args: list[str], remote_dir: str) -> list[dict[str, Any]]:
    """List files in a remote directory with sizes and modification times.

    Returns list of dicts: {path, size_bytes, size_human, mtime}
    """
    cmd = (
        f"find {REMOTE_DIR}/{remote_dir} -type f "
        f"-exec stat -c '%s %Y %n' {{}} \\; 2>/dev/null"
    )
    raw = ssh_exec(ssh_args, cmd, timeout=60)
    if not raw:
        return []

    files = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(" ", 2)
        if len(parts) != 3:
            continue
        size_bytes = int(parts[0])
        mtime = datetime.fromtimestamp(int(parts[1]))
        full_path = parts[2]
        # Make relative to REMOTE_DIR
        rel_path = full_path.replace(f"{REMOTE_DIR}/", "", 1)

        files.append({
            "path": rel_path,
            "full_remote_path": full_path,
            "size_bytes": size_bytes,
            "size_human": _human_size(size_bytes),
            "mtime": mtime.isoformat(),
        })

    return sorted(files, key=lambda f: f["path"])


def _human_size(nbytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} PB"


def get_gpu_status(ssh_args: list[str]) -> str | None:
    """Get current GPU utilization from the pod."""
    return ssh_exec(
        ssh_args,
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader",
    )


def get_log_tail(ssh_args: list[str], lines: int = 20) -> str | None:
    """Get the tail of the pipeline log."""
    return ssh_exec(ssh_args, f"tail -{lines} {LOG_FILE} 2>/dev/null")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_pipeline_status(state: dict[str, Any]) -> None:
    """Print pipeline job status summary."""
    print_section("Pipeline Status")

    all_jobs = [
        ("sae_goodfire", "SAE Goodfire L19 analysis"),
        ("new_items", "New items behavioral validation"),
        ("olmo3_full", "OLMo-3-7B full pipeline"),
        ("reextract_activations", "Re-extract activations"),
        ("attention_entropy", "Attention entropy extraction"),
        ("bootstrap_cis", "Bootstrap confidence intervals"),
    ]

    completed = state.get("completed", {})
    failed = state.get("failed", {})

    n_done = 0
    n_fail = 0
    for name, desc in all_jobs:
        if name in completed:
            info = completed[name]
            elapsed = info.get("elapsed_seconds", "?")
            finished = info.get("finished_at", "?")
            print(f"  [DONE]    {name:<28} {elapsed:>8}s   finished: {finished}")
            n_done += 1
        elif name in failed:
            info = failed[name]
            error = info.get("error", "unknown")[:80]
            print(f"  [FAILED]  {name:<28} error: {error}")
            n_fail += 1
        else:
            print(f"  [PENDING] {name:<28}")

    n_pending = len(all_jobs) - n_done - n_fail
    print(f"\n  Summary: {n_done} done, {n_fail} failed, {n_pending} pending")


def print_file_listing(files: list[dict[str, Any]], category: str) -> None:
    """Print a table of files."""
    if not files:
        print(f"  (no files in {category})")
        return

    total_bytes = sum(f["size_bytes"] for f in files)
    print(f"  {len(files)} files, {_human_size(total_bytes)} total\n")

    for f in files:
        print(f"    {f['size_human']:>10}  {f['path']}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_results(
    user_host: str,
    scp_opts: list[str],
    ssh_args: list[str],
    include_activations: bool = False,
) -> None:
    """Download result files from the pod to local results/ directory."""
    print_section("Downloading Results")

    dirs_to_download = list(RESULT_DIRS)
    if include_activations:
        dirs_to_download.extend(LARGE_DIRS)

    for remote_dir in dirs_to_download:
        # Check if the remote dir exists
        check = ssh_exec(ssh_args, f"test -d {REMOTE_DIR}/{remote_dir} && echo yes")
        if not check or "yes" not in check:
            print(f"  Skipping {remote_dir} (not found on pod)")
            continue

        local_dir = PROJECT_ROOT / remote_dir
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading {remote_dir} -> {local_dir}")
        ok = scp_download(
            user_host,
            scp_opts,
            f"{REMOTE_DIR}/{remote_dir}",
            str(local_dir.parent),
            recursive=True,
        )
        if ok:
            print(f"    OK")
        else:
            print(f"    FAILED")

    # Also grab the pipeline state and log
    print(f"  Downloading pipeline state and log...")
    state_local = PROJECT_ROOT / "results" / "gpu_pipeline_state.json"
    state_local.parent.mkdir(parents=True, exist_ok=True)
    scp_download(user_host, scp_opts, STATE_FILE, str(state_local), recursive=False)
    log_local = PROJECT_ROOT / "results" / "gpu_pipeline_log.txt"
    scp_download(user_host, scp_opts, LOG_FILE, str(log_local), recursive=False)

    print(f"\n  Results downloaded to: {PROJECT_ROOT / 'results'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check and download s1s2 results from a RunPod GPU pod.",
    )
    parser.add_argument(
        "ssh_cmd",
        help='Full SSH command, e.g. "ssh root@IP -p PORT -i KEY"',
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download result files to local results/ directory",
    )
    parser.add_argument(
        "--include-activations",
        action="store_true",
        help="Also download activation HDF5 files (can be very large)",
    )
    parser.add_argument(
        "--log-lines",
        type=int,
        default=20,
        help="Number of log lines to show (default: 20)",
    )
    args = parser.parse_args()

    ssh_args, user_host, scp_opts = parse_ssh_cmd(args.ssh_cmd)

    # Test connection
    print("Connecting to pod...")
    test = ssh_exec(ssh_args, "echo ok")
    if test is None or "ok" not in test:
        print("ERROR: Cannot connect to pod. Check your SSH command.")
        sys.exit(1)
    print("Connected.\n")

    # GPU status
    print_section("GPU Status")
    gpu = get_gpu_status(ssh_args)
    if gpu:
        print(f"  {gpu.strip()}")
    else:
        print("  Could not query GPU (pod may be stopped)")

    # Pipeline state
    state = get_pipeline_state(ssh_args)
    if state:
        print_pipeline_status(state)
    else:
        print_section("Pipeline Status")
        print("  No pipeline state found (pipeline has not run yet)")

    # List result files
    for remote_dir in RESULT_DIRS:
        print_section(f"Files: {remote_dir}")
        files = list_remote_files(ssh_args, remote_dir)
        print_file_listing(files, remote_dir)

    # Activation files (just show sizes, don't list every file)
    print_section("Activation Files")
    act_files = list_remote_files(ssh_args, "data/activations/")
    if act_files:
        total = sum(f["size_bytes"] for f in act_files)
        print(f"  {len(act_files)} files, {_human_size(total)} total")
        for f in act_files:
            print(f"    {f['size_human']:>10}  {f['path']}")
    else:
        print("  No activation files found")

    # Log tail
    print_section("Pipeline Log (last %d lines)" % args.log_lines)
    log_tail = get_log_tail(ssh_args, args.log_lines)
    if log_tail:
        for line in log_tail.strip().split("\n"):
            print(f"  {line}")
    else:
        print("  No log file found")

    # Download if requested
    if args.download:
        download_results(user_host, scp_opts, ssh_args, args.include_activations)

    print("")


if __name__ == "__main__":
    main()
