#!/usr/bin/env python3
"""CLI entry point for generating the unified s1s2 results report.

Usage::

    python scripts/generate_report.py --results-dir results/ --output report.md
    python scripts/generate_report.py --results-dir results/ --output-dir reports/
    python scripts/generate_report.py --results-dir results/ --json-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the src layout works without pip install -e .
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a unified report from per-workstream s1s2 results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root directory containing per-workstream result subdirectories (default: results/).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the Markdown report output (e.g., report.md).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write both report.json and report.md into.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        default=False,
        help="Only output JSON (no Markdown).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    from s1s2.report import generate_report

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: results directory '{results_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    report = generate_report(results_dir)

    if args.output_dir:
        report.save(Path(args.output_dir))
        print(f"Report saved to {args.output_dir}/report.json and report.md")
    elif args.json_only:
        if args.output:
            Path(args.output).write_text(report.to_json())
            print(f"JSON report written to {args.output}")
        else:
            print(report.to_json())
    else:
        md = report.to_markdown()
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(md)
            print(f"Markdown report written to {args.output}")
        else:
            print(md)


if __name__ == "__main__":
    main()
