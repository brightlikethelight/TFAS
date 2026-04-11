#!/usr/bin/env python3
"""Checkpoint-aware extraction wrapper for preemptible SLURM jobs.

The core insight: extraction processes one model at a time. After each model
completes, we write a checkpoint marker. If the job is preempted (SIGUSR1
from SLURM), we finish the current model, save, and exit cleanly. The next
job invocation skips already-completed models.

Usage:
    python deploy/checkpoint_extract.py \
        --config configs/extract.yaml \
        --checkpoint-dir /n/holyscratch01/$USER/s1s2/checkpoints \
        --output-dir /n/holyscratch01/$USER/s1s2/activations

    # Override which models to extract:
    python deploy/checkpoint_extract.py \
        --config configs/extract.yaml \
        --models llama-3.1-8b-instruct r1-distill-llama-8b \
        --checkpoint-dir ./checkpoints \
        --output-dir ./activations
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
from beartype import beartype

_REPO = Path(__file__).resolve().parent.parent

# ---- Preemption signal handling ---------------------------------------------

_preempted = False


def _handle_preempt(signum: int, frame: object) -> None:
    """SIGUSR1 handler — SLURM sends this before wall-time preemption."""
    global _preempted
    _preempted = True
    print(
        "[CHECKPOINT] Preemption signal received (SIGUSR1). "
        "Will finish current model then exit cleanly.",
        flush=True,
    )


# Register handler. SIGUSR1 is sent by SLURM --signal=B:USR1@120.
signal.signal(signal.SIGUSR1, _handle_preempt)


# ---- Checkpoint management --------------------------------------------------


@beartype
def checkpoint_path(checkpoint_dir: Path, model_key: str) -> Path:
    """Path to the checkpoint marker for a given model."""
    return checkpoint_dir / f"{model_key}.done"


@beartype
def is_model_complete(checkpoint_dir: Path, model_key: str) -> bool:
    """Check if a model's extraction has already been checkpointed."""
    return checkpoint_path(checkpoint_dir, model_key).exists()


@beartype
def mark_model_complete(
    checkpoint_dir: Path, model_key: str, elapsed_s: float, output_path: str
) -> None:
    """Write a checkpoint marker after successful extraction."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    marker = checkpoint_path(checkpoint_dir, model_key)
    marker.write_text(
        json.dumps({
            "model_key": model_key,
            "elapsed_s": elapsed_s,
            "output_path": output_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    )
    print(f"[CHECKPOINT] Marked {model_key} as complete: {marker}", flush=True)


@beartype
def load_extract_config(config_path: Path) -> dict:
    """Load the Hydra extract config as a plain dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@beartype
def load_model_registry(config_path: Path) -> dict[str, dict]:
    """Load the model registry from configs/models.yaml."""
    models_path = config_path.parent / "models.yaml"
    with open(models_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["models"]


# ---- Extraction runner ------------------------------------------------------


@beartype
def run_extraction_for_model(
    model_key: str,
    config_path: Path,
    output_dir: Path,
    run_name: str,
) -> bool:
    """Run extraction for a single model via the Hydra CLI.

    Returns True if extraction succeeded, False otherwise.
    """
    output_path = output_dir / f"{run_name}.h5"
    cmd = [
        sys.executable,
        str(_REPO / "scripts" / "extract_all.py"),
        f"run_name={run_name}",
        f"models_to_extract=[{model_key}]",
        f"output_dir={output_dir}",
        # Hydra writes its own output dir; redirect it so it doesn't collide
        f"hydra.run.dir={output_dir}/hydra/{model_key}",
    ]
    print(f"\n{'='*60}", flush=True)
    print(f"[EXTRACT] Starting: {model_key}", flush=True)
    print(f"[EXTRACT] Command: {' '.join(cmd)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    result = subprocess.run(cmd, cwd=str(_REPO))
    return result.returncode == 0


# ---- Main -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Checkpoint-aware extraction for preemptible SLURM jobs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO / "configs" / "extract.yaml",
        help="Path to extraction config (default: configs/extract.yaml)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory for checkpoint markers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for HDF5 output files",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model keys to extract (default: from extract.yaml models_to_extract)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="main",
        help="Run name for the HDF5 output file (default: main)",
    )
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to extract
    if args.models:
        model_keys = args.models
    else:
        cfg = load_extract_config(args.config)
        model_keys = cfg.get("models_to_extract", [])

    if not model_keys:
        print("ERROR: no models specified. Use --models or set models_to_extract in config.")
        return 1

    registry = load_model_registry(args.config)
    unknown = [m for m in model_keys if m not in registry]
    if unknown:
        print(f"ERROR: unknown model keys: {unknown}")
        print(f"Valid keys: {list(registry.keys())}")
        return 1

    # Process models one at a time with checkpoint gating
    n_total = len(model_keys)
    n_skipped = 0
    n_done = 0
    n_failed = 0

    for i, model_key in enumerate(model_keys, 1):
        if _preempted:
            print(f"\n[CHECKPOINT] Preempted before starting {model_key}. Exiting cleanly.")
            break

        if is_model_complete(args.checkpoint_dir, model_key):
            print(f"[SKIP] {model_key} ({i}/{n_total}) — already checkpointed.")
            n_skipped += 1
            continue

        t0 = time.time()
        success = run_extraction_for_model(
            model_key=model_key,
            config_path=args.config,
            output_dir=args.output_dir,
            run_name=args.run_name,
        )
        elapsed = time.time() - t0

        if success:
            output_path = str(args.output_dir / f"{args.run_name}.h5")
            mark_model_complete(args.checkpoint_dir, model_key, elapsed, output_path)
            n_done += 1
            print(f"[DONE] {model_key} ({i}/{n_total}) in {elapsed:.0f}s", flush=True)
        else:
            n_failed += 1
            print(f"[FAIL] {model_key} ({i}/{n_total}) after {elapsed:.0f}s", flush=True)

        if _preempted:
            print(f"\n[CHECKPOINT] Preempted after {model_key}. Exiting cleanly.")
            break

    # Summary
    print(f"\n{'='*60}")
    print(f"Checkpoint extraction summary:")
    print(f"  Total models:    {n_total}")
    print(f"  Skipped (done):  {n_skipped}")
    print(f"  Completed now:   {n_done}")
    print(f"  Failed:          {n_failed}")
    print(f"  Remaining:       {n_total - n_skipped - n_done - n_failed}")
    if _preempted:
        print(f"  Status: PREEMPTED — resubmit job to continue.")
    elif n_failed > 0:
        print(f"  Status: PARTIAL FAILURE")
    elif n_skipped + n_done == n_total:
        print(f"  Status: ALL COMPLETE")
    print(f"{'='*60}")

    if n_failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
