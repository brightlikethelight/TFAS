#!/usr/bin/env python3
"""Driver script for the metacognitive-monitoring workstream.

Thin wrapper around :func:`s1s2.metacog.cli.main` so users can run::

    python scripts/run_metacog.py
    python scripts/run_metacog.py models_to_analyze=[r1-distill-llama-8b]

All flags are Hydra overrides against ``configs/metacog.yaml``.
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
        from s1s2.metacog.cli import main
    except ImportError as exc:
        sys.stderr.write(
            "[run_metacog] failed to import s1s2.metacog.cli: "
            f"{exc}\n"
            "Install the package in editable mode first:\n"
            "    pip install -e .[dev]\n"
            "If the metacog CLI module has not been built yet, ask the\n"
            "workstream owner — this script is only a thin wrapper.\n"
        )
        raise SystemExit(1) from exc
    main()


if __name__ == "__main__":
    _run()
