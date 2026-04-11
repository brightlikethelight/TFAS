#!/usr/bin/env python3
"""Launch the s1s2 interactive analysis dashboard.

Usage::

    python scripts/run_dashboard.py                    # real results
    python scripts/run_dashboard.py --synthetic        # demo with synthetic data
    python scripts/run_dashboard.py --port 7860        # custom port
    python scripts/run_dashboard.py --config configs/dashboard.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running without `pip install -e .` in dev environments.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# OpenMP guard for macOS where torch + numpy fight over libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _load_config(config_path: str | None) -> dict:
    """Load dashboard config from YAML, returning an empty dict on failure."""
    if config_path is None:
        return {}
    p = Path(config_path)
    if not p.exists():
        print(f"Warning: config file {p} not found, using defaults.")
        return {}
    try:
        import yaml

        with p.open() as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        print("Warning: pyyaml not installed; ignoring config file.")
        return {}
    except Exception as exc:
        print(f"Warning: failed to load config {p}: {exc}")
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the s1s2 interactive analysis dashboard.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic demo data instead of real results.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve the dashboard on (default: 7860).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to the results/ directory. Defaults to results/ in the repo root.",
    )
    parser.add_argument(
        "--activations-path",
        type=str,
        default=None,
        help="Path to the HDF5 activation cache (reserved for future use).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a dashboard YAML config file.",
    )
    args = parser.parse_args()

    # Merge CLI args with config file (CLI takes precedence).
    cfg = _load_config(args.config)
    results_dir = args.results_dir or cfg.get("results_dir")
    activations_path = args.activations_path or cfg.get("activations_path")
    synthetic = args.synthetic or cfg.get("synthetic", False)
    port = args.port if args.port != 7860 else cfg.get("port", 7860)

    try:
        from s1s2.dashboard.app import create_app
    except ImportError as exc:
        print(
            f"Failed to import dashboard: {exc}\n"
            "Install the dashboard extras: pip install -e '.[dashboard]'"
        )
        return 1

    app = create_app(
        results_dir=results_dir,
        activations_path=activations_path,
        synthetic=synthetic,
    )

    mode = "synthetic demo" if synthetic else "real results"
    print(f"Starting s1s2 dashboard ({mode}) on port {port}...")
    app.launch(
        server_port=port,
        share=args.share,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
