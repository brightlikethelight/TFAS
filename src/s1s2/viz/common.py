"""Shared helpers for paper figure generators.

This module collects the small utility functions every per-figure module
needs: loading a tree of per-workstream result JSONs, saving a figure in
one or more formats with consistent DPI/bounds, and translating raw JSON
payloads into the tiny data shapes the workstream plot functions expect.

We keep this surface deliberately small — each helper does one thing and
is trivially unit-testable without any matplotlib backend. Per-figure
modules then build on top of these primitives, so the "wiring" code in
``paper_figures.py`` stays short and readable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from beartype import beartype

from s1s2.utils.logging import get_logger

logger = get_logger("s1s2.viz.common")

__all__ = [
    "DEFAULT_DPI",
    "FIG_SIZE_DOUBLE_COLUMN",
    "FIG_SIZE_DOUBLE_ROW",
    "FIG_SIZE_SINGLE_COLUMN",
    "FIG_SIZE_SQUARE",
    "load_json_tree",
    "load_single_json",
    "require_dir",
    "resolve_output_path",
    "save_figure",
]


# --------------------------------------------------------------------------- #
# Canonical figure sizes (inches)                                              #
# --------------------------------------------------------------------------- #

#: 3.5in wide — fits one column of a typical two-column paper.
FIG_SIZE_SINGLE_COLUMN: tuple[float, float] = (3.5, 2.4)
#: 7.2in wide — fits the full text width of a two-column paper.
FIG_SIZE_DOUBLE_COLUMN: tuple[float, float] = (7.2, 3.2)
#: 7.2in wide, 5.4in tall — full-width figure with larger vertical extent
#: (used for ridge plots and 2x2 panels).
FIG_SIZE_DOUBLE_ROW: tuple[float, float] = (7.2, 5.4)
#: Square figure — useful for volcano plots or PCA scatter.
FIG_SIZE_SQUARE: tuple[float, float] = (4.5, 4.5)

#: Publication-grade DPI. Matches the figures.py default.
DEFAULT_DPI: int = 300


# --------------------------------------------------------------------------- #
# I/O helpers                                                                  #
# --------------------------------------------------------------------------- #


@beartype
def require_dir(path: str | Path, *, label: str = "directory") -> Path:
    """Return ``path`` as a :class:`Path`, raising ``FileNotFoundError`` if missing.

    The ``label`` is only used in the error message; supplying a human-
    readable name ("probe results") improves the user-facing report when
    :func:`make_paper_figures` catches the exception and logs it.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


@beartype
def load_json_tree(root: str | Path) -> list[dict[str, Any]]:
    """Recursively load every ``*.json`` under ``root``.

    Malformed JSON files are skipped with a warning rather than blowing
    up the entire sweep — in practice this lets you keep partial results
    around (e.g. a half-written file from a crashed run) without having
    to delete them before regenerating figures.

    Raises ``FileNotFoundError`` if ``root`` does not exist. Returns an
    empty list if the directory exists but contains no JSON files — the
    caller can then decide whether to skip the figure.
    """
    rpath = require_dir(root, label="results directory")
    out: list[dict[str, Any]] = []
    for p in sorted(rpath.rglob("*.json")):
        try:
            with p.open() as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("failed to parse %s: %s", p, exc)
            continue
        if isinstance(payload, dict):
            out.append(payload)
        elif isinstance(payload, list):
            # Some workstreams write a top-level list. Wrap each entry
            # into a dict with a synthetic source field for traceability.
            for i, entry in enumerate(payload):
                if isinstance(entry, dict):
                    out.append(entry)
                else:
                    logger.debug("skipping non-dict list entry in %s[%d]", p, i)
        else:
            logger.debug("skipping non-dict/list JSON %s", p)
    return out


@beartype
def load_single_json(path: str | Path) -> dict[str, Any]:
    """Load exactly one JSON file. Raises ``FileNotFoundError`` on miss."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open() as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict at top level of {p}, got {type(payload).__name__}")
    return payload


@beartype
def resolve_output_path(
    output_dir: str | Path,
    name: str,
    formats: Iterable[str],
) -> Path:
    """Compute the primary output path for a named figure.

    The "primary" path is the one returned to callers; it uses the first
    format in ``formats``. The directory is created if missing. This
    helper exists so every figure module can compute its output path the
    same way without repeating directory-creation boilerplate.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    formats = tuple(formats)
    if not formats:
        raise ValueError("formats must be non-empty")
    primary = formats[0].lstrip(".")
    return out / f"{name}.{primary}"


@beartype
def save_figure(
    fig: Any,
    primary_path: str | Path,
    formats: Iterable[str] = ("pdf",),
    *,
    dpi: int = DEFAULT_DPI,
    close: bool = True,
) -> Path:
    """Save ``fig`` in one or more formats and return the primary path.

    If ``formats`` lists multiple formats, additional copies are saved as
    siblings of ``primary_path`` (same stem, different extension). When
    ``close`` is True (default), the figure is closed afterwards so it
    does not leak memory across the sweep — set to False only if you
    need to inspect the figure interactively in a notebook.
    """
    # Lazy import so this module is usable even in environments where
    # matplotlib is not yet imported (e.g. the config sanity-check tests).
    import matplotlib.pyplot as plt

    primary_path = Path(primary_path)
    primary_path.parent.mkdir(parents=True, exist_ok=True)

    formats = tuple(f.lstrip(".") for f in formats)
    if not formats:
        raise ValueError("formats must be non-empty")

    # The primary path's suffix is authoritative for the first format.
    primary_fmt = primary_path.suffix.lstrip(".") or formats[0]
    extras = [f for f in formats if f != primary_fmt]

    fig.savefig(primary_path, dpi=dpi, bbox_inches="tight")
    for fmt in extras:
        fig.savefig(primary_path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return primary_path


@beartype
def format_list(fmt_config: str | Iterable[str]) -> list[str]:
    """Translate a ``"pdf" | "png" | "both"`` selector into a concrete format list.

    Keeping this conversion in one place lets the main entry point accept
    either the compact user-facing string or an explicit list.
    """
    if isinstance(fmt_config, str):
        f = fmt_config.lower().strip()
        if f == "both":
            return ["pdf", "png"]
        return [f]
    return [str(x).lstrip(".").lower() for x in fmt_config]


@beartype
def filter_by_models(
    records: list[dict[str, Any]],
    models: Iterable[str] | None,
) -> list[dict[str, Any]]:
    """Keep only records whose ``model`` field is in ``models``.

    If ``models`` is None, returns ``records`` unchanged — this is the
    common "all models" code path. The filter is a no-op on records
    without a ``model`` field so it does not accidentally drop
    benchmark-level JSONs that lack per-model provenance.
    """
    if models is None:
        return records
    allowed = set(models)
    return [r for r in records if ("model" not in r) or (r["model"] in allowed)]
