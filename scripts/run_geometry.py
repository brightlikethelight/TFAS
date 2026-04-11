#!/usr/bin/env python3
"""Driver script for the representational-geometry workstream.

Thin wrapper around :func:`s1s2.geometry.cli.main` so users can run::

    python scripts/run_geometry.py
    python scripts/run_geometry.py models_to_analyze=[llama-3.1-8b-instruct]

All flags are Hydra overrides against ``configs/geometry.yaml``.
"""

from __future__ import annotations

import os
import sys

# Ensure the src layout works even when running without ``pip install -e .``.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _run() -> None:
    try:
        from s1s2.geometry.cli import main
    except ImportError as exc:
        sys.stderr.write(
            "[run_geometry] failed to import s1s2.geometry.cli: "
            f"{exc}\n"
            "Install the package in editable mode first:\n"
            "    pip install -e .[dev]\n"
            "If the geometry CLI module has not been built yet, ask the\n"
            "workstream owner — this script is only a thin wrapper.\n"
        )
        raise SystemExit(1) from exc
    main()


if __name__ == "__main__":
    _run()
