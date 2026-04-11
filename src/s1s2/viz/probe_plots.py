"""Layer-wise probe accuracy curves.

Produces the headline figure for the probing workstream: per-layer ROC-AUC
with bootstrap CI ribbons, one line per (model, probe_type) combination,
faceted or filtered by target and position.

The module is importable (for notebooks) and has a ``__main__`` with argparse
so you can run::

    python -m s1s2.viz.probe_plots \
        --results-dir results/probes --target task_type --position P0 \
        --out figures/probe_layer_curves_task_type_P0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype

__all__ = [
    "load_results_tree",
    "main",
    "plot_layer_accuracy_curves",
]


# Canonical palette — standard models blue, reasoning models orange — lazily
# imported from matplotlib/seaborn inside the plot function so this module is
# importable even in CPU-only headless environments where only the JSON
# loading bits are needed.

STANDARD_MODELS = {"llama-3.1-8b-instruct", "gemma-2-9b-it"}
REASONING_MODELS = {"r1-distill-llama-8b", "r1-distill-qwen-7b"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@beartype
def load_results_tree(results_dir: str | Path) -> list[dict[str, Any]]:
    """Walk ``results_dir`` and return every probe JSON.

    Expected layout: ``results_dir/{model}/{target}/layer_NN_pos_{pos}.json``.
    """
    rd = Path(results_dir)
    if not rd.exists():
        raise FileNotFoundError(rd)
    out: list[dict[str, Any]] = []
    for model_dir in sorted(rd.iterdir()):
        if not model_dir.is_dir():
            continue
        for target_dir in sorted(model_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            for f in sorted(target_dir.glob("layer_*.json")):
                with f.open() as fh:
                    d = json.load(fh)
                out.append(d)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _model_color(model: str, probe: str) -> str:
    """Return a color for a (model, probe) combination.

    Reasoning models are warm, standard models are cool. Probe type sets the
    lightness within each palette.
    """
    cool = ["#1f4e79", "#2e75b6", "#8faadc", "#bdd7ee"]
    warm = ["#c55a11", "#ed7d31", "#f4b183", "#fbe5d6"]
    palette = warm if model in REASONING_MODELS else cool
    probe_order = {"logistic": 0, "mlp": 1, "mass_mean": 2, "ccs": 3}
    return palette[probe_order.get(probe, 0) % len(palette)]


@beartype
def plot_layer_accuracy_curves(
    results: list[dict[str, Any]],
    outpath: str | Path,
    target: str,
    position: str,
    probes: tuple[str, ...] = ("logistic",),
    show_ci: bool = True,
    show_significance_band: bool = True,
) -> Path:
    """Plot ROC-AUC vs layer for every (model, probe) slice matching ``target``/``position``.

    Parameters
    ----------
    results
        Output of :func:`load_results_tree` (or a manually constructed list).
    outpath
        Output path stem — ``.pdf`` and ``.png`` will both be written.
    target, position
        Filter the results to this slice.
    probes
        Which probe types to draw. One line per (model, probe).
    show_ci
        Whether to draw bootstrap CI ribbons around each line.
    show_significance_band
        Whether to draw the 95th-percentile permutation-null band as a dashed
        horizontal line (the statistical significance threshold).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)

    filtered = [
        r for r in results
        if r.get("target") == target and r.get("position") == position
    ]
    if not filtered:
        raise ValueError(
            f"no results match target={target!r} position={position!r}"
        )

    # Organise: per (model, probe) -> sorted list of (layer, auc, ci_lo, ci_hi)
    by_key: dict[tuple[str, str], list[tuple[int, float, float, float, float]]] = {}
    null_95_by_key: dict[tuple[str, str], list[float]] = {}
    for r in filtered:
        model = r["model"]
        for probe in probes:
            probe_dict = r.get("probes", {}).get(probe)
            if probe_dict is None:
                continue
            summary = probe_dict.get("summary", {})
            auc = summary.get("roc_auc")
            if auc is None:
                continue
            lo = summary.get("roc_auc_ci_lower", auc)
            hi = summary.get("roc_auc_ci_upper", auc)
            sel = summary.get("selectivity", np.nan)
            null_95 = summary.get("permutation_null_auc_95", np.nan)
            by_key.setdefault((model, probe), []).append(
                (int(r["layer"]), float(auc), float(lo), float(hi), float(sel))
            )
            null_95_by_key.setdefault((model, probe), []).append(float(null_95))

    if not by_key:
        raise ValueError(
            f"no probe entries matched {probes=}; "
            f"results contain probes: {sorted({p for r in filtered for p in r.get('probes', {})})}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, probe), rows in by_key.items():
        rows.sort(key=lambda t: t[0])
        layers = np.array([t[0] for t in rows])
        aucs = np.array([t[1] for t in rows])
        lows = np.array([t[2] for t in rows])
        highs = np.array([t[3] for t in rows])
        color = _model_color(model, probe)
        label = f"{model} ({probe})"
        ax.plot(layers, aucs, "-o", color=color, label=label, markersize=4, linewidth=1.8)
        if show_ci:
            ax.fill_between(layers, lows, highs, color=color, alpha=0.18, linewidth=0)

    # Chance line.
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="chance")

    # Significance band: use the mean null_95 across ALL plotted series for
    # the target/position slice. This is a single scalar and gives a visual
    # reference for "this layer is above what random labels would give".
    if show_significance_band:
        all_null = [v for rows in null_95_by_key.values() for v in rows if np.isfinite(v)]
        if all_null:
            null_thresh = float(np.mean(all_null))
            ax.axhline(
                null_thresh,
                color="crimson",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"null 95th pct ({null_thresh:.3f})",
            )

    ax.set_xlabel("Layer")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Layer-wise probe accuracy ({target}, {position})")
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9)
    fig.tight_layout()

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    pdf = outpath.with_suffix(".pdf")
    png = outpath.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot layer-wise probe accuracy curves.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/probes"),
        help="Root of probe result JSON tree.",
    )
    p.add_argument("--target", type=str, default="task_type")
    p.add_argument("--position", type=str, default="P0")
    p.add_argument(
        "--probes",
        nargs="+",
        default=["logistic"],
        help="Which probe types to draw as separate lines.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path stem (no extension). Defaults to figures/probe_layer_curves_<target>_<position>.",
    )
    p.add_argument("--no-ci", action="store_true", help="Suppress CI ribbons.")
    p.add_argument("--no-null", action="store_true", help="Suppress null band.")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    results = load_results_tree(args.results_dir)
    out = args.out or Path("figures") / f"probe_layer_curves_{args.target}_{args.position}"
    plot_layer_accuracy_curves(
        results=results,
        outpath=out,
        target=args.target,
        position=args.position,
        probes=tuple(args.probes),
        show_ci=not args.no_ci,
        show_significance_band=not args.no_null,
    )
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


if __name__ == "__main__":  # pragma: no cover
    main()
