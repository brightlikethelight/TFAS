#!/usr/bin/env python3
"""Full experiment orchestrator for the s1s2 project.

Runs the entire pipeline from benchmark validation through extraction,
analysis, and figure generation. Each stage writes a JSON checkpoint so
interrupted runs resume where they left off.

Usage::

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --stages extract,probes,figures
    python scripts/run_pipeline.py --models llama-3.1-8b-instruct,r1-distill-llama-8b
    python scripts/run_pipeline.py --no-skip-completed --stop-on-error
    python scripts/run_pipeline.py --clean-checkpoints

Stage ordering:

    validate -> extract -> probes -> sae -> attention -> geometry
    -> causal -> metacog -> figures
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full s1s2 analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Stage order: validate -> extract -> probes -> sae -> attention\n"
            "             -> geometry -> causal -> metacog -> figures\n"
            "\n"
            "Checkpoints are written to .pipeline_checkpoints/ and invalidated\n"
            "automatically when config parameters (models, seed, paths) change."
        ),
    )
    parser.add_argument(
        "--stages",
        default="all",
        help=(
            "Comma-separated list of stages to run, or 'all'. "
            "Valid: validate,extract,probes,sae,attention,geometry,causal,metacog,figures"
        ),
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys, or 'all' for the full set.",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip stages that have a valid checkpoint (default: on).",
    )
    parser.add_argument(
        "--stop-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop the pipeline on the first stage failure (default: off).",
    )
    parser.add_argument(
        "--activations",
        default="data/activations/main.h5",
        help="Path to the shared HDF5 activations file.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for per-workstream result JSONs.",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory for generated paper figures.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=".pipeline_checkpoints",
        help="Directory for stage checkpoint markers.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Global RNG seed.",
    )
    parser.add_argument(
        "--clean-checkpoints",
        action="store_true",
        help="Remove all checkpoint files and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from s1s2.pipeline import ALL_STAGES, Pipeline, PipelineConfig
    from s1s2.utils.seed import set_global_seed

    # Resolve models
    all_models = [
        "llama-3.1-8b-instruct",
        "gemma-2-9b-it",
        "r1-distill-llama-8b",
        "r1-distill-qwen-7b",
    ]
    if args.models == "all":
        models = all_models
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Resolve stages
    if args.stages == "all":
        stages = list(ALL_STAGES)
    else:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    config = PipelineConfig(
        stages=stages,
        models=models,
        activations_path=args.activations,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        skip_completed=args.skip_completed,
        stop_on_error=args.stop_on_error,
    )

    pipeline = Pipeline(config, repo_root=_ROOT)

    # --clean-checkpoints: remove and exit
    if args.clean_checkpoints:
        n = pipeline.clean_checkpoints()
        print(f"Removed {n} checkpoint file(s).")
        return 0

    set_global_seed(config.seed, deterministic_torch=False)

    report = pipeline.run()
    return 1 if report.any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
