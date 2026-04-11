"""Cognitive bias benchmark loading, generation, and validation.

Public API:
    - :class:`BenchmarkItem` -- typed record
    - :func:`load_benchmark` -- read a JSONL benchmark file
    - :func:`iter_matched_pairs` -- iterate (conflict, control) pairs
    - :func:`filter_by_category` / :func:`filter_conflict` -- common filters
    - :func:`validate_benchmark` -- full structural validation
    - :mod:`generators` -- non-CRT generators
    - :mod:`templates` -- CRT generators (existing)
    - :mod:`build` -- canonical benchmark assembly
"""

from __future__ import annotations

from s1s2.benchmark.loader import (
    BenchmarkItem,
    filter_by_category,
    filter_conflict,
    group_by_matched_pair,
    iter_matched_pairs,
    load_benchmark,
)
from s1s2.benchmark.validate import ValidationReport, validate_benchmark

__all__ = [
    "BenchmarkItem",
    "ValidationReport",
    "filter_by_category",
    "filter_conflict",
    "group_by_matched_pair",
    "iter_matched_pairs",
    "load_benchmark",
    "validate_benchmark",
]
