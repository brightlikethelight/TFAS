"""Figure 1 — benchmark schematic.

This is the only figure in the paper that is NOT derived from model
experimental results; it is built directly from the benchmark JSONL. We
produce a two-panel figure:

- **Left**: horizontal bar chart of items per category (grouped by
  conflict vs control). Gives the reader an at-a-glance sense of the
  coverage of the benchmark — how many CRT items, how many base-rate,
  etc. — without opening the dataset.
- **Right**: 2-3 example items (conflict + matched control pair) with
  the S1 lure visually highlighted in the conflict prompt. This shows
  rather than tells what we mean by "conflict" and "matched control".

Design notes:

* We never import the real :mod:`s1s2.benchmark` loader at module scope,
  so this figure can be generated from a raw JSONL file alone. The
  loader is only imported in the hot path inside
  :func:`_load_benchmark_jsonl`, and we fall back to plain ``json`` if
  the full loader is unavailable (e.g. in a minimal CI environment).
* Conflict and control bars use the "s1" and "s2" semantic colors from
  :mod:`s1s2.viz.theme` so they match every other figure in the paper.
* We pick examples deterministically (by sorted ID) so the figure is
  reproducible.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from textwrap import fill
from typing import Any

from beartype import beartype

from s1s2.viz.common import (
    FIG_SIZE_DOUBLE_COLUMN,
    format_list,
    resolve_output_path,
    save_figure,
)
from s1s2.viz.paper_figures_config import PaperFiguresConfig
from s1s2.viz.theme import COLORS, set_paper_theme

__all__ = ["make_figure_1_benchmark"]


_CATEGORY_ORDER: tuple[str, ...] = (
    "crt",
    "base_rate",
    "syllogism",
    "anchoring",
    "framing",
    "conjunction",
    "arithmetic",
)


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #


@beartype
def _load_benchmark_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a benchmark JSONL file into a list of dicts.

    We use plain ``json.loads`` per line rather than the typed
    :class:`BenchmarkItem` loader because this figure only needs a tiny
    subset of the fields and we want the module to be importable even
    when the full benchmark package has a syntax error.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"benchmark JSONL not found: {p}")
    items: list[dict[str, Any]] = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines rather than crash the whole figure.
                continue
    if not items:
        raise ValueError(f"benchmark JSONL at {p} is empty (no valid items)")
    return items


@beartype
def _category_counts(
    items: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Return ``{category: {"conflict": n, "control": n}}``.

    Categories are pulled verbatim from the item. Unknown categories are
    kept so new categories added to the benchmark automatically appear.
    """
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"conflict": 0, "control": 0})
    for it in items:
        cat = str(it.get("category", "unknown"))
        key = "conflict" if it.get("conflict", False) else "control"
        out[cat][key] += 1
    return dict(out)


@beartype
def _pick_examples(
    items: list[dict[str, Any]],
    *,
    n_pairs: int = 2,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return up to ``n_pairs`` (conflict, control) pairs sharing a ``matched_pair_id``.

    Items are sorted by category then by id so the output is stable
    across runs. If the benchmark has fewer than ``n_pairs`` valid pairs
    we return whatever is available (an empty list is tolerated by the
    plotter, which just skips the right panel).
    """
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        mpid = it.get("matched_pair_id")
        if isinstance(mpid, str) and mpid:
            by_pair[mpid].append(it)

    # Sort by (category, id) for determinism.
    pair_ids = sorted(
        by_pair.keys(),
        key=lambda k: (by_pair[k][0].get("category", ""), k),
    )

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for pid in pair_ids:
        bucket = by_pair[pid]
        conflict = next((x for x in bucket if x.get("conflict")), None)
        control = next((x for x in bucket if not x.get("conflict")), None)
        if conflict is not None and control is not None:
            pairs.append((conflict, control))
        if len(pairs) >= n_pairs:
            break
    return pairs


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #


def _plot_category_bars(ax, counts: dict[str, dict[str, int]]) -> None:
    """Draw a horizontal grouped bar chart of conflict/control counts per category."""
    # Preserve canonical order; append any unknown categories at the bottom.
    known = [c for c in _CATEGORY_ORDER if c in counts]
    unknown = sorted(c for c in counts if c not in _CATEGORY_ORDER)
    cats = known + unknown

    conflict_counts = [counts[c]["conflict"] for c in cats]
    control_counts = [counts[c]["control"] for c in cats]

    import numpy as np

    y = np.arange(len(cats))
    h = 0.38
    ax.barh(
        y - h / 2,
        conflict_counts,
        height=h,
        color=COLORS["s1"],
        label="conflict (S1 lure)",
    )
    ax.barh(
        y + h / 2,
        control_counts,
        height=h,
        color=COLORS["s2"],
        label="control (no lure)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in cats])
    ax.set_xlabel("items")
    ax.set_title("(a) benchmark composition", fontsize=10, loc="left")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)


def _plot_example_pairs(ax, pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    """Render 2-3 (conflict, control) pairs as labeled text blocks.

    The conflict example has its ``lure_answer`` highlighted in the
    border color. This is intentionally text-heavy — the goal is for a
    reader to immediately see what an "S1 lure" looks like compared to
    its matched control.
    """
    ax.set_axis_off()
    if not pairs:
        ax.text(
            0.5,
            0.5,
            "no example pairs available",
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
        ax.set_title("(b) example items", fontsize=10, loc="left")
        return

    ax.set_title("(b) example items", fontsize=10, loc="left")
    # Lay out examples vertically. Each pair takes a slice of the axes
    # from top to bottom.
    n_pairs = len(pairs)
    slice_h = 1.0 / n_pairs
    for i, (conflict, control) in enumerate(pairs):
        y_top = 1.0 - i * slice_h

        cat = str(conflict.get("category", "?")).replace("_", " ")
        cpt = str(conflict.get("prompt", ""))
        lure = str(conflict.get("lure_answer", ""))
        correct = str(conflict.get("correct_answer", ""))
        ctl_pt = str(control.get("prompt", ""))

        # Wrap the prompt so it fits in the panel.
        conflict_txt = fill(cpt, width=55)
        control_txt = fill(ctl_pt, width=55)

        header = f"[{cat}]  correct={correct!r}  lure={lure!r}"
        body = (
            f"conflict: {conflict_txt}\n"
            f"control:  {control_txt}"
        )
        ax.text(
            0.02,
            y_top - 0.05 * slice_h,
            header,
            ha="left",
            va="top",
            fontsize=8,
            color=COLORS["s1"],
            weight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.02,
            y_top - 0.20 * slice_h,
            body,
            ha="left",
            va="top",
            fontsize=7.5,
            family="monospace",
            transform=ax.transAxes,
        )
        if i < n_pairs - 1:
            # Draw a faint divider in axes coordinates manually — axhline
            # only accepts data coordinates, so we use Line2D + transAxes
            # via ax.plot with the transform set explicitly.
            sep_y = y_top - slice_h + 0.02
            ax.plot(
                [0.02, 0.98],
                [sep_y, sep_y],
                color="lightgray",
                linestyle="-",
                linewidth=0.5,
                transform=ax.transAxes,
            )


@beartype
def make_figure_1_benchmark(cfg: PaperFiguresConfig) -> Path:
    """Generate Figure 1 (benchmark schematic) and return the output path.

    Reads the benchmark JSONL at ``cfg.benchmark_path``, computes the
    per-category counts, picks deterministic example pairs, and writes
    the two-panel figure to ``cfg.output_dir``.
    """
    import matplotlib.pyplot as plt

    set_paper_theme()

    items = _load_benchmark_jsonl(cfg.benchmark_path)
    counts = _category_counts(items)
    pairs = _pick_examples(items, n_pairs=2)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIG_SIZE_DOUBLE_COLUMN,
        gridspec_kw={"width_ratios": [1.0, 1.3]},
    )
    _plot_category_bars(axes[0], counts)
    _plot_example_pairs(axes[1], pairs)

    fig.suptitle(
        "Figure 1: S1/S2 cognitive-bias benchmark",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    formats = format_list(cfg.format)
    out_path = resolve_output_path(cfg.output_dir, "figure_1_benchmark", formats)
    return save_figure(fig, out_path, formats, dpi=cfg.dpi)
