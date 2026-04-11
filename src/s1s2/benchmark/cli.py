"""Command-line entry point for benchmark validation, stats and generation.

Three subcommands:

* ``validate`` -- run :func:`s1s2.benchmark.validate.validate_benchmark`
  on the on-disk JSONL file and print the full report. Exits non-zero
  if any structural error is found.
* ``stats`` -- print per-category counts, conflict ratios and
  difficulty distribution. Does not raise on missing target counts.
* ``generate`` -- (re)build ``data/benchmark/benchmark.jsonl`` from the
  template and generator modules, validate the result, and refuse to
  overwrite the file if validation fails.

Invocation::

    python -m s1s2.benchmark.cli validate
    python -m s1s2.benchmark.cli validate --path data/benchmark/benchmark.jsonl
    python -m s1s2.benchmark.cli stats
    python -m s1s2.benchmark.cli generate --output data/benchmark/benchmark.jsonl

We use :mod:`argparse` rather than ``click`` so the CLI has no
dependency outside the standard library and the project's already-
declared deps. The Hydra config (``configs/benchmark.yaml``) is
read manually for default paths so this CLI does NOT require a
running Hydra app context (which is convenient for tests and for
calling from other scripts).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from s1s2.benchmark.loader import BenchmarkItem, load_benchmark
from s1s2.benchmark.validate import (
    _print_report,
    validate_benchmark,
)
from s1s2.utils.types import ALL_CATEGORIES

_DEFAULT_PATH = Path("data/benchmark/benchmark.jsonl")


# --------------------------------------------------------------------- #
# subcommand handlers                                                   #
# --------------------------------------------------------------------- #


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"benchmark file not found: {path}", file=sys.stderr)
        return 2
    report = validate_benchmark(path)
    _print_report(report, path)
    if args.json:
        json.dump(
            {
                "ok": report.ok,
                "errors": report.errors,
                "warnings": report.warnings,
                "stats": report.stats,
            },
            sys.stdout,
            default=str,
            indent=2,
        )
        sys.stdout.write("\n")
    return 0 if report.ok else 1


def cmd_stats(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"benchmark file not found: {path}", file=sys.stderr)
        return 2
    items = load_benchmark(path)
    print(f"Loaded {len(items)} records from {path}")
    _print_stats_table(items)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Heavy import deferred so `python -m s1s2.benchmark.cli validate`
    # never imports the generator module (and so a syntax error in
    # the generator file does not break the validator path).
    from s1s2.benchmark.build import build_full_benchmark, write_jsonl

    items = build_full_benchmark(seed=args.seed)
    if args.expand_paraphrases:
        from s1s2.benchmark.templates import expand_paraphrases

        items = expand_paraphrases(items)

    write_jsonl(items, output)
    print(f"Wrote {len(items)} records to {output}")

    report = validate_benchmark(output)
    _print_report(report, output)
    if not report.ok:
        print(
            "REGENERATION FAILED VALIDATION; benchmark left in place but "
            "is structurally broken. Inspect errors above.",
            file=sys.stderr,
        )
        return 1
    _print_stats_table(items)
    return 0


# --------------------------------------------------------------------- #
# helpers                                                               #
# --------------------------------------------------------------------- #


def _print_stats_table(items: list[BenchmarkItem]) -> None:
    """Render a per-category breakdown of the loaded items."""
    by_cat: dict[str, dict[str, Any]] = {
        cat: {"conflict": 0, "control": 0, "paraphrases": 0, "subcats": Counter()}
        for cat in ALL_CATEGORIES
    }
    for it in items:
        slot = by_cat[it.category]
        if "__p" in it.id:
            slot["paraphrases"] += 1
            continue
        slot["conflict" if it.conflict else "control"] += 1
        slot["paraphrases"] += len(it.paraphrases)
        slot["subcats"][it.subcategory] += 1

    print()
    header = f"{'category':<14}{'conflict':>10}{'control':>10}{'paraphrases':>14}{'subcategories':>30}"
    print(header)
    print("-" * len(header))
    for cat in ALL_CATEGORIES:
        slot = by_cat[cat]
        sub_strs = ",".join(f"{k}={v}" for k, v in sorted(slot["subcats"].items()))
        print(
            f"{cat:<14}"
            f"{slot['conflict']:>10}"
            f"{slot['control']:>10}"
            f"{slot['paraphrases']:>14}"
            f"  {sub_strs[:28]}"
        )
    primaries = sum(slot["conflict"] + slot["control"] for slot in by_cat.values())
    print("-" * len(header))
    print(
        f"TOTAL primaries: {primaries}; matched pairs: "
        f"{sum(slot['conflict'] for slot in by_cat.values())}"
    )


# --------------------------------------------------------------------- #
# argparse wiring                                                       #
# --------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="s1s2.benchmark.cli",
        description="Cognitive-bias benchmark validate/stats/generate CLI.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="run the structural validator")
    p_val.add_argument(
        "--path", default=str(_DEFAULT_PATH), help="benchmark JSONL path"
    )
    p_val.add_argument(
        "--json",
        action="store_true",
        help="emit a JSON copy of the report after the human-readable one",
    )
    p_val.set_defaults(func=cmd_validate)

    p_stats = sub.add_parser("stats", help="print per-category counts")
    p_stats.add_argument(
        "--path", default=str(_DEFAULT_PATH), help="benchmark JSONL path"
    )
    p_stats.set_defaults(func=cmd_stats)

    p_gen = sub.add_parser("generate", help="build benchmark.jsonl")
    p_gen.add_argument(
        "--output", default=str(_DEFAULT_PATH), help="output JSONL path"
    )
    p_gen.add_argument("--seed", type=int, default=0, help="RNG seed")
    p_gen.add_argument(
        "--expand-paraphrases",
        action="store_true",
        help="write each paraphrase as its own JSONL record (sibling)",
    )
    p_gen.set_defaults(func=cmd_generate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
