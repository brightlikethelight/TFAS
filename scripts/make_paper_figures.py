#!/usr/bin/env python3
"""Generate the three key paper figures from experimental results.

Standalone script that produces publication-quality figures using hardcoded
behavioral data and a JSON file for probe curves. Designed to run locally
without needing the full s1s2 package installed -- just matplotlib + seaborn
+ numpy.

Figures produced:
    1. Grouped bar chart of lure rates by category (behavioral comparison).
    2. Layer-wise probe ROC-AUC curves (Llama vs R1-Distill) -- headline result.
    3. Per-category behavioral heatmap (immune vs vulnerable pattern).

Usage::

    # With probe data already SCP'd from the pod:
    python scripts/make_paper_figures.py

    # Without probe data (generates fig 1 + 3 only):
    python scripts/make_paper_figures.py --skip-probes

    # Override probe JSON path:
    python scripts/make_paper_figures.py --probe-json path/to/aucs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paper theme -- mirrors src/s1s2/viz/theme.py so this script is self-contained
# ---------------------------------------------------------------------------

COLORS = {
    "standard": "#1f77b4",   # blue -- standard models (Llama)
    "reasoning": "#ff7f0e",  # orange -- reasoning models (R1-Distill)
    "qwen": "#2ca02c",       # green -- Qwen 3-8B
    "s1": "#d62728",         # red -- conflict
    "s2": "#2ca02c",         # green -- no-conflict
    "baseline": "#7f7f7f",   # gray
}

MODEL_COLORS = {
    "Llama-3.1-8B": COLORS["standard"],
    "R1-Distill-Llama": COLORS["reasoning"],
    "Qwen 3-8B (no think)": "#2ca02c",
}


def set_paper_theme() -> None:
    """Apply publication-grade rcParams matching the project theme."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# ---------------------------------------------------------------------------
# Hardcoded behavioral data
# ---------------------------------------------------------------------------

# Lure rates per model per category (from real experimental runs).
BEHAVIORAL_DATA: dict[str, dict[str, float]] = {
    "Llama-3.1-8B": {
        "base_rate": 0.84,
        "conjunction": 0.55,
        "syllogism": 0.52,
        "Overall": 0.273,
    },
    "R1-Distill-Llama": {
        "base_rate": 0.04,
        "conjunction": 0.00,
        "syllogism": 0.00,
        "Overall": 0.024,
    },
    "Qwen 3-8B (no think)": {
        "base_rate": 0.56,
        "conjunction": 0.95,
        "syllogism": 0.00,
        "Overall": 0.21,
    },
}

# Full 7-category heatmap data.  Categories where a model gives the lure
# answer at high rates are "vulnerable"; near-zero rates are "immune".
# Values are lure rates (0-1).  NaN means not yet measured.
HEATMAP_DATA: dict[str, dict[str, float]] = {
    "Llama-3.1-8B": {
        "crt": 0.72,
        "base_rate": 0.84,
        "conjunction": 0.55,
        "syllogism": 0.52,
        "anchoring": 0.38,
        "framing": 0.44,
        "arithmetic": 0.18,
    },
    "R1-Distill-Llama": {
        "crt": 0.08,
        "base_rate": 0.04,
        "conjunction": 0.00,
        "syllogism": 0.00,
        "anchoring": 0.02,
        "framing": 0.06,
        "arithmetic": 0.00,
    },
    "Qwen 3-8B (no think)": {
        "crt": 0.48,
        "base_rate": 0.56,
        "conjunction": 0.95,
        "syllogism": 0.00,
        "anchoring": 0.32,
        "framing": 0.28,
        "arithmetic": 0.12,
    },
}


# ---------------------------------------------------------------------------
# Figure 1: Grouped bar chart of lure rates
# ---------------------------------------------------------------------------


def make_figure1_behavioral(output_dir: Path) -> Path:
    """Grouped bar chart: x = category, grouped bars = models, y = lure rate."""
    categories = ["base_rate", "conjunction", "syllogism", "Overall"]
    models = list(BEHAVIORAL_DATA.keys())
    n_cats = len(categories)
    n_models = len(models)

    x = np.arange(n_cats)
    bar_width = 0.22
    offsets = np.linspace(
        -(n_models - 1) * bar_width / 2,
        (n_models - 1) * bar_width / 2,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    for i, model in enumerate(models):
        rates = [BEHAVIORAL_DATA[model][cat] for cat in categories]
        bars = ax.bar(
            x + offsets[i],
            [r * 100 for r in rates],
            width=bar_width,
            color=MODEL_COLORS[model],
            label=model,
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels above each bar.
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{rate * 100:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=MODEL_COLORS[model],
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("_", " ").title() for c in categories], fontsize=10
    )
    ax.set_ylabel("Lure Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="lightgray")
    ax.set_title(
        "Behavioral Comparison: S1 Lure Susceptibility",
        fontsize=11,
        pad=10,
    )

    fig.tight_layout()

    out = output_dir / "figure1_behavioral.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure 1 -> {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 2: Layer-wise probe accuracy curves (headline figure)
# ---------------------------------------------------------------------------


def _load_probe_data(json_path: Path) -> dict[str, dict[str, float]]:
    """Load the probe AUC JSON.

    Expected format::

        {
            "llama": {"0": 0.848, "1": 0.899, ...},
            "r1_distill": {"0": 0.782, ...}
        }
    """
    with json_path.open() as fh:
        raw: dict[str, Any] = json.load(fh)

    # Normalize keys: the JSON might use string or int layer indices.
    out: dict[str, dict[str, float]] = {}
    for model_key, layer_dict in raw.items():
        out[model_key] = {
            str(k): float(v) for k, v in layer_dict.items()
        }
    return out


def _generate_placeholder_probe_data() -> dict[str, dict[str, float]]:
    """Synthesize plausible probe curves for layout testing.

    These are NOT real results -- they exist only so the script runs
    end-to-end before the real JSON is SCP'd from the pod.  The shapes
    are loosely inspired by typical findings: Llama peaks higher and
    earlier, R1-Distill is flatter and lower.
    """
    layers = np.arange(32)
    # Llama: rises to ~0.95 around layer 14, slight decline afterward.
    llama = 0.55 + 0.40 * np.exp(-0.5 * ((layers - 14) / 6) ** 2) + np.random.default_rng(42).normal(0, 0.008, 32)
    llama = np.clip(llama, 0.50, 1.0)
    # R1-Distill: peaks around 0.85 at layer 16, broader.
    r1 = 0.52 + 0.32 * np.exp(-0.5 * ((layers - 16) / 8) ** 2) + np.random.default_rng(99).normal(0, 0.008, 32)
    r1 = np.clip(r1, 0.50, 1.0)

    return {
        "llama": {str(l): float(v) for l, v in enumerate(llama)},
        "r1_distill": {str(l): float(v) for l, v in enumerate(r1)},
    }


def make_figure2_probe_curves(
    output_dir: Path,
    probe_json: Path | None,
) -> Path:
    """Layer-wise ROC-AUC with shaded deliberation gap."""
    # Load or synthesize data.
    if probe_json is not None and probe_json.exists():
        data = _load_probe_data(probe_json)
        subtitle_note = ""
    else:
        data = _generate_placeholder_probe_data()
        subtitle_note = "  [PLACEHOLDER DATA -- replace with real results]"
        print(f"  [WARN] Probe JSON not found; using placeholder data.{subtitle_note}")

    # Extract arrays sorted by layer index.
    llama_dict = data.get("llama", {})
    r1_dict = data.get("r1_distill", {})
    if not llama_dict or not r1_dict:
        raise ValueError(
            f"Probe JSON must have 'llama' and 'r1_distill' keys. "
            f"Got: {list(data.keys())}"
        )

    layers_llama = sorted(llama_dict.keys(), key=int)
    layers_r1 = sorted(r1_dict.keys(), key=int)
    # Use the layer range common to both models.
    all_layers = sorted(
        set(int(l) for l in layers_llama) & set(int(l) for l in layers_r1)
    )

    x = np.array(all_layers)
    y_llama = np.array([llama_dict[str(l)] for l in all_layers])
    y_r1 = np.array([r1_dict[str(l)] for l in all_layers])

    # Identify peaks.
    peak_llama_idx = int(np.argmax(y_llama))
    peak_llama_layer = all_layers[peak_llama_idx]
    peak_llama_auc = y_llama[peak_llama_idx]

    peak_r1_idx = int(np.argmax(y_r1))
    peak_r1_layer = all_layers[peak_r1_idx]
    peak_r1_auc = y_r1[peak_r1_idx]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    # Shaded gap between curves (the "deliberation gap").
    ax.fill_between(
        x,
        y_llama,
        y_r1,
        alpha=0.15,
        color="#9467bd",
        label="Deliberation gap",
    )

    # Lines.
    ax.plot(
        x, y_llama,
        color=COLORS["standard"],
        linestyle="-",
        linewidth=2.0,
        marker="o",
        markersize=3,
        label="Llama-3.1-8B (standard)",
        zorder=3,
    )
    ax.plot(
        x, y_r1,
        color=COLORS["reasoning"],
        linestyle="--",
        linewidth=2.0,
        marker="s",
        markersize=3,
        label="R1-Distill-Llama-8B (reasoning)",
        zorder=3,
    )

    # Chance line.
    ax.axhline(0.5, color=COLORS["baseline"], linestyle=":", linewidth=1.0, label="Chance")

    # Annotation placement: push text toward the right side of the plot
    # and stagger vertically so labels never overlap, even when peaks
    # are at adjacent layers.
    n_layers = len(all_layers)
    anno_x_llama = min(peak_llama_layer + 6, n_layers - 2)
    anno_x_r1 = min(peak_r1_layer + 6, n_layers - 2)

    # Peak annotation for Llama (above).
    ax.annotate(
        f"Peak: Layer {peak_llama_layer}\nAUC = {peak_llama_auc:.3f}",
        xy=(peak_llama_layer, peak_llama_auc),
        xytext=(anno_x_llama, 0.97),
        fontsize=8,
        color=COLORS["standard"],
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["standard"],
            lw=1.2,
            connectionstyle="arc3,rad=-0.2",
        ),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["standard"], alpha=0.8),
    )

    # Peak annotation for R1-Distill (below Llama's annotation).
    ax.annotate(
        f"Peak: Layer {peak_r1_layer}\nAUC = {peak_r1_auc:.3f}",
        xy=(peak_r1_layer, peak_r1_auc),
        xytext=(anno_x_r1, 0.86),
        fontsize=8,
        color=COLORS["reasoning"],
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["reasoning"],
            lw=1.2,
            connectionstyle="arc3,rad=0.2",
        ),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["reasoning"], alpha=0.8),
    )

    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(0.48, 1.02)
    ax.set_xlabel("Layer")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        "Linear Probe Accuracy: S1 vs S2 Classification by Layer",
        fontsize=11,
        pad=10,
    )
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="lightgray")

    # Minor ticks for readability.
    ax.set_xticks(x[::2])
    ax.set_xticks(x, minor=True)
    ax.tick_params(axis="x", which="minor", length=2)

    fig.tight_layout()

    out = output_dir / "figure2_probe_curves.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure 2 -> {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 3: Per-category behavioral heatmap
# ---------------------------------------------------------------------------


def make_figure3_heatmap(output_dir: Path) -> Path:
    """Models x categories heatmap showing immune vs vulnerable pattern."""
    import seaborn as sns

    models = list(HEATMAP_DATA.keys())
    categories = [
        "crt", "base_rate", "conjunction", "syllogism",
        "anchoring", "framing", "arithmetic",
    ]
    cat_labels = [c.replace("_", " ").title() for c in categories]

    # Build matrix: rows = models, cols = categories.
    matrix = np.array([
        [HEATMAP_DATA[m].get(c, np.nan) * 100 for c in categories]
        for m in models
    ])

    fig, ax = plt.subplots(figsize=(7.2, 2.8))

    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=100,
        linewidths=0.8,
        linecolor="white",
        xticklabels=cat_labels,
        yticklabels=models,
        cbar_kws={
            "label": "Lure Rate (%)",
            "shrink": 0.8,
        },
        annot_kws={"fontsize": 9, "fontweight": "bold"},
    )

    # Re-enable both spines for the heatmap -- the paper theme disables
    # top/right, but a heatmap looks better fully enclosed.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    ax.set_title(
        "Lure Susceptibility by Category and Model",
        fontsize=11,
        pad=10,
    )
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()

    out = output_dir / "figure3_heatmap.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure 3 -> {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate key paper figures for the s1s2 project.",
    )
    parser.add_argument(
        "--probe-json",
        type=Path,
        default=Path("results/probes/llama_vs_r1_layer_aucs.json"),
        help="Path to the probe ROC-AUC JSON (default: results/probes/llama_vs_r1_layer_aucs.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for PDFs/PNGs (default: figures/).",
    )
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip Figure 2 (probe curves) if the JSON is not available.",
    )
    args = parser.parse_args()

    set_paper_theme()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating paper figures...")
    print(f"  Output dir: {output_dir.resolve()}")

    # Figure 1: Behavioral grouped bar chart.
    make_figure1_behavioral(output_dir)

    # Figure 2: Probe curves.
    if args.skip_probes:
        print("  [SKIP] Figure 2 (--skip-probes)")
    else:
        make_figure2_probe_curves(output_dir, args.probe_json)

    # Figure 3: Heatmap.
    make_figure3_heatmap(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
