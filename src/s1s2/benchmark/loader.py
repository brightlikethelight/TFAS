"""Typed loading of the cognitive bias benchmark.

The on-disk format is JSON-Lines (``benchmark.jsonl``). This module wraps
that file into dataclass instances and exposes the few filter / pairing
helpers the rest of the codebase needs. All schema validation is
centralised in :mod:`s1s2.benchmark.validate`; this loader only performs
the minimum sanity required to produce typed objects.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from beartype import beartype

from s1s2.utils.types import ALL_CATEGORIES, TaskCategory

_REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "category",
    "subcategory",
    "conflict",
    "difficulty",
    "prompt",
    "system_prompt",
    "correct_answer",
    "lure_answer",
    "answer_pattern",
    "lure_pattern",
    "matched_pair_id",
    "source",
    "provenance_note",
    "paraphrases",
)

_ALLOWED_SOURCES: frozenset[str] = frozenset(
    {"novel", "hagendorff_2023", "template", "adapted"}
)


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """One benchmark line, deserialised.

    Matches the schema in ``data/benchmark/SCHEMA.md``. Frozen so loaded
    items can be reused across workstreams without accidental mutation.
    """

    id: str
    category: TaskCategory
    subcategory: str
    conflict: bool
    difficulty: int
    prompt: str
    system_prompt: str | None
    correct_answer: str
    lure_answer: str
    answer_pattern: str
    lure_pattern: str
    matched_pair_id: str
    source: str
    provenance_note: str
    paraphrases: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict preserving the schema field order."""
        d = asdict(self)
        d["paraphrases"] = list(self.paraphrases)
        return d


@beartype
def _coerce_item(raw: dict[str, Any], line_number: int) -> BenchmarkItem:
    """Convert a raw JSON dict into a :class:`BenchmarkItem`.

    Raises :class:`ValueError` on missing fields or obviously-wrong types.
    The heavy semantic checks (pair-matching, regex round-trip, …) live in
    :mod:`s1s2.benchmark.validate`.
    """
    missing = [k for k in _REQUIRED_FIELDS if k not in raw]
    if missing:
        raise ValueError(f"line {line_number}: missing fields {missing}")

    cat = raw["category"]
    if cat not in ALL_CATEGORIES:
        raise ValueError(
            f"line {line_number}: category {cat!r} not in {list(ALL_CATEGORIES)}"
        )

    src = raw["source"]
    if src not in _ALLOWED_SOURCES:
        raise ValueError(
            f"line {line_number}: source {src!r} not in {sorted(_ALLOWED_SOURCES)}"
        )

    if not isinstance(raw["difficulty"], int) or not (1 <= raw["difficulty"] <= 5):
        raise ValueError(
            f"line {line_number}: difficulty must be int in [1,5], got {raw['difficulty']!r}"
        )

    if not isinstance(raw["paraphrases"], list) or not all(
        isinstance(p, str) for p in raw["paraphrases"]
    ):
        raise ValueError(f"line {line_number}: paraphrases must be a list of strings")

    sys_prompt = raw["system_prompt"]
    if sys_prompt is not None and not isinstance(sys_prompt, str):
        raise ValueError(
            f"line {line_number}: system_prompt must be str or null, got {type(sys_prompt)!r}"
        )

    return BenchmarkItem(
        id=str(raw["id"]),
        category=cat,  # type: ignore[arg-type]
        subcategory=str(raw["subcategory"]),
        conflict=bool(raw["conflict"]),
        difficulty=int(raw["difficulty"]),
        prompt=str(raw["prompt"]),
        system_prompt=sys_prompt,
        correct_answer=str(raw["correct_answer"]),
        lure_answer=str(raw["lure_answer"]),
        answer_pattern=str(raw["answer_pattern"]),
        lure_pattern=str(raw["lure_pattern"]),
        matched_pair_id=str(raw["matched_pair_id"]),
        source=str(raw["source"]),
        provenance_note=str(raw["provenance_note"]),
        paraphrases=tuple(raw["paraphrases"]),
    )


@beartype
def load_benchmark(path: str | Path) -> list[BenchmarkItem]:
    """Load a JSONL benchmark file into typed :class:`BenchmarkItem`\\ s.

    Performs minimal per-line schema checks. For the full cross-item
    validation suite (pair matching, category counts, regex round-trip,
    …), run :func:`s1s2.benchmark.validate.validate_benchmark`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    items: list[BenchmarkItem] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {lineno}: malformed JSON: {e}") from e
            items.append(_coerce_item(raw, lineno))
    return items


@beartype
def iter_matched_pairs(
    items: Iterable[BenchmarkItem],
) -> Iterator[tuple[BenchmarkItem, BenchmarkItem]]:
    """Yield ``(conflict_item, control_item)`` pairs from a benchmark.

    Pairs are keyed by ``matched_pair_id``. If an id has more than one
    conflict record (paraphrases) the *primary* (non-suffixed) id is
    preferred; otherwise the first conflict is paired with the first
    control. Pairs missing a counterpart are skipped silently — use
    :mod:`s1s2.benchmark.validate` to flag them.
    """
    by_pair: dict[str, dict[str, list[BenchmarkItem]]] = {}
    for it in items:
        slot = by_pair.setdefault(
            it.matched_pair_id, {"conflict": [], "control": []}
        )
        key = "conflict" if it.conflict else "control"
        slot[key].append(it)

    for _, group in by_pair.items():
        if not group["conflict"] or not group["control"]:
            continue
        conflict = _pick_primary(group["conflict"])
        control = _pick_primary(group["control"])
        yield conflict, control


def _pick_primary(candidates: list[BenchmarkItem]) -> BenchmarkItem:
    """Return the primary (non-paraphrase) item from a list of siblings.

    Primaries do not carry the ``__p<N>`` suffix introduced by the
    paraphrase expander; fall back to the first candidate otherwise.
    """
    for c in candidates:
        if "__p" not in c.id:
            return c
    return candidates[0]


@beartype
def filter_by_category(
    items: Iterable[BenchmarkItem], category: TaskCategory
) -> list[BenchmarkItem]:
    """Return only items with the given category."""
    return [it for it in items if it.category == category]


@beartype
def filter_conflict(
    items: Iterable[BenchmarkItem], only_conflict: bool = True
) -> list[BenchmarkItem]:
    """Return only conflict (or only control) items."""
    return [it for it in items if it.conflict is only_conflict]


@beartype
def group_by_matched_pair(
    items: Iterable[BenchmarkItem],
) -> dict[str, list[BenchmarkItem]]:
    """Group items by ``matched_pair_id``.

    Useful for sanity-checking that every conflict item has a paired
    control before running downstream analyses.
    """
    out: dict[str, list[BenchmarkItem]] = {}
    for it in items:
        out.setdefault(it.matched_pair_id, []).append(it)
    return out
