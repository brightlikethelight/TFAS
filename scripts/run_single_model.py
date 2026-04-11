#!/usr/bin/env python3
"""Extract one model and run CPU analyses on its cached activations.

Useful for incremental progress on preemptible jobs: get one model done
and show preliminary results while waiting for the full run.

Usage::

    python scripts/run_single_model.py llama-3.1-8b-instruct
    python scripts/run_single_model.py r1-distill-llama-8b --no-extract
    python scripts/run_single_model.py llama-3.1-8b-instruct --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the src layout works without ``pip install -e .``.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# OpenMP guard for macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ALL_MODELS = [
    "llama-3.1-8b-instruct",
    "gemma-2-9b-it",
    "r1-distill-llama-8b",
    "r1-distill-qwen-7b",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract + analyze a single model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Runs: extract -> probes -> attention -> geometry for the given model.\n"
            "Use --no-extract to skip extraction when activations already exist."
        ),
    )
    parser.add_argument(
        "model",
        choices=ALL_MODELS,
        help="Model key to process.",
    )
    parser.add_argument(
        "--extract",
        dest="include_extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the extraction stage (default: yes).",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip stages with valid checkpoints.",
    )
    parser.add_argument(
        "--stop-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop on first failure.",
    )
    parser.add_argument(
        "--activations",
        default="data/activations/main.h5",
        help="Path to the shared HDF5 activations file.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for result JSONs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=".pipeline_checkpoints",
        help="Directory for checkpoint markers.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Global RNG seed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from s1s2.pipeline import PipelineConfig, run_single_model
    from s1s2.utils.seed import set_global_seed

    config = PipelineConfig(
        stages=[],  # driven by run_single_model internally
        models=[args.model],
        activations_path=args.activations,
        results_dir=args.results_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        skip_completed=args.skip_completed,
        stop_on_error=args.stop_on_error,
    )

    set_global_seed(config.seed, deterministic_torch=False)

    report = run_single_model(
        model_key=args.model,
        config=config,
        repo_root=_ROOT,
        include_extract=args.include_extract,
    )
    return 1 if report.any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
