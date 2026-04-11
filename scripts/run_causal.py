#!/usr/bin/env python3
"""Driver script for the causal-intervention workstream.

Thin wrapper around :func:`s1s2.causal.cli.main` so users can run::

    python scripts/run_causal.py
    python scripts/run_causal.py models_to_analyze=[llama-3.1-8b-instruct]

All flags are Hydra overrides against ``configs/causal.yaml``.
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
        from s1s2.causal.cli import main
    except ImportError as exc:
        sys.stderr.write(
            "[run_causal] failed to import s1s2.causal.cli: "
            f"{exc}\n"
            "Install the package in editable mode first:\n"
            "    pip install -e .[dev]\n"
            "If the causal CLI module has not been built yet, ask the\n"
            "workstream owner — this script is only a thin wrapper.\n"
        )
        raise SystemExit(1) from exc
    main()


if __name__ == "__main__":
    _run()
