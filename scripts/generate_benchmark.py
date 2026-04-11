#!/usr/bin/env python3
"""Standalone driver for building ``data/benchmark/benchmark.jsonl``.

This is a thin wrapper around :func:`s1s2.benchmark.build.build_full_benchmark`
plus :func:`s1s2.benchmark.validate.validate_benchmark`. It exists so the
benchmark file can be regenerated without remembering the package layout::

    python scripts/generate_benchmark.py
    python scripts/generate_benchmark.py --output /tmp/scratch.jsonl
    python scripts/generate_benchmark.py --expand-paraphrases

The script aborts non-zero if validation fails after writing, but the
file is left in place so a developer can inspect it.

This is intentionally separate from the Hydra-driven training scripts:
the benchmark is a static asset, not an experiment, so we don't need
the Hydra config-merge machinery here. The Hydra config
``configs/benchmark.yaml`` is consumed by ``s1s2.benchmark.cli`` instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when invoked directly without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the cognitive-bias benchmark JSONL."
    )
    parser.add_argument(
        "--output",
        default=str(_ROOT / "data" / "benchmark" / "benchmark.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (reserved; current generators are deterministic).",
    )
    parser.add_argument(
        "--expand-paraphrases",
        action="store_true",
        help=(
            "Write each paraphrase as its own JSONL record. By default we "
            "keep paraphrases nested inside the primary item so the on-disk "
            "benchmark stays compact."
        ),
    )
    args = parser.parse_args(argv)

    from s1s2.benchmark.build import build_full_benchmark, write_jsonl
    from s1s2.benchmark.templates import expand_paraphrases
    from s1s2.benchmark.validate import _print_report, validate_benchmark

    items = build_full_benchmark(seed=args.seed)
    if args.expand_paraphrases:
        items = expand_paraphrases(items)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(items, out)
    print(f"Wrote {len(items)} records to {out}")

    report = validate_benchmark(out)
    _print_report(report, out)
    if not report.ok:
        print(
            "VALIDATION FAILED. The file has been written but is "
            "structurally broken; investigate the errors above.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
