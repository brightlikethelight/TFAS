#!/usr/bin/env python3
"""Analyze attention entropy results: Llama-3.1-8B vs R1-Distill-Llama-8B.

Loads per-head attention entropy data from B200 pod results and produces:
  1. Layer-wise conflict vs control entropy profiles
  2. Identification of S2-specialized heads (from pre-computed BH-FDR tests)
  3. Figure 5: two-panel normalized entropy plot
  4. JSON summary with key statistics for the paper

The pre-computed analysis section in each JSON already has BH-FDR corrected
q-values and rank-biserial correlations per head. We leverage those rather
than re-running the tests.

Usage::

    python scripts/analyze_attention.py
    python scripts/analyze_attention.py --data-dir results_pod/attention --output-dir results/attention
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Project theme (matches make_paper_figures.py conventions)
# ---------------------------------------------------------------------------

COLORS = {
    "conflict": "#1f77b4",   # blue
    "control": "#ff7f0e",    # orange
    "gap_fill": "#9467bd",   # purple
}


def set_paper_theme() -> None:
    """Publication-grade rcParams matching the project theme."""
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
# Data loading
# ---------------------------------------------------------------------------


def load_attention_data(path: Path) -> dict[str, Any]:
    """Load a full attention JSON file. ~81MB, takes a few seconds."""
    print(f"  Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)...")
    with open(path) as f:
        data = json.load(f)
    n = len(data["items"])
    print(f"    {data['model']}: {n} items, {data['n_layers']}L x {data['n_heads']}H")
    return data


# ---------------------------------------------------------------------------
# Layer-wise entropy computation from raw item data
# ---------------------------------------------------------------------------


def compute_layer_profiles(
    data: dict[str, Any],
    metric: str = "norm_entropy",
) -> dict[str, NDArray[np.float64]]:
    """Compute mean per-layer metric for conflict vs control items.

    Returns dict with keys:
        conflict_mean, control_mean: (n_layers,)
        conflict_sem, control_sem: (n_layers,) standard error of mean
        gap: conflict_mean - control_mean
    """
    n_layers = data["n_layers"]
    n_heads = data["n_heads"]

    conflict_vals: list[NDArray] = []
    control_vals: list[NDArray] = []

    for item in data["items"]:
        # item['metrics'][metric] is (n_layers, n_heads)
        arr = np.array(item["metrics"][metric], dtype=np.float64)
        assert arr.shape == (n_layers, n_heads), (
            f"Unexpected shape {arr.shape} for {item['id']}"
        )
        # Mean across heads within each layer -> (n_layers,)
        layer_means = arr.mean(axis=1)
        if item["conflict"]:
            conflict_vals.append(layer_means)
        else:
            control_vals.append(layer_means)

    # Stack: (n_items, n_layers)
    conflict_mat = np.stack(conflict_vals)
    control_mat = np.stack(control_vals)

    return {
        "conflict_mean": conflict_mat.mean(axis=0),
        "control_mean": control_mat.mean(axis=0),
        "conflict_sem": conflict_mat.std(axis=0, ddof=1) / np.sqrt(conflict_mat.shape[0]),
        "control_sem": control_mat.std(axis=0, ddof=1) / np.sqrt(control_mat.shape[0]),
        "gap": conflict_mat.mean(axis=0) - control_mat.mean(axis=0),
        "n_conflict": conflict_mat.shape[0],
        "n_control": control_mat.shape[0],
    }


# ---------------------------------------------------------------------------
# Analysis of pre-computed statistical tests
# ---------------------------------------------------------------------------


def analyze_head_tests(
    data: dict[str, Any],
    metric: str = "norm_entropy",
) -> dict[str, Any]:
    """Extract per-head BH-FDR test results and compute layer-wise significance.

    Uses the pre-computed analysis.query_head_tests section which has
    per-head Mann-Whitney U tests with BH-FDR correction.
    """
    analysis = data["analysis"]
    n_layers = data["n_layers"]
    n_heads = data["n_heads"]

    qht = analysis["query_head_tests"][metric]
    tests = qht["tests"]

    n_significant = qht["n_significant"]
    n_total = qht["n_tests"]

    # Build layer x head significance and effect size matrices
    sig_matrix = np.zeros((n_layers, n_heads), dtype=bool)
    effect_matrix = np.zeros((n_layers, n_heads), dtype=np.float64)
    qval_matrix = np.ones((n_layers, n_heads), dtype=np.float64)

    for t in tests:
        sig_matrix[t["layer"], t["head"]] = t["significant"]
        effect_matrix[t["layer"], t["head"]] = t["r_rb"]
        qval_matrix[t["layer"], t["head"]] = t["q_value"]

    # Layer-wise: number of significant heads per layer
    sig_per_layer = sig_matrix.sum(axis=1)  # (n_layers,)

    # Heads where conflict > control (positive r_rb AND significant)
    conflict_higher = sig_matrix & (effect_matrix > 0)
    control_higher = sig_matrix & (effect_matrix < 0)

    n_conflict_higher = int(conflict_higher.sum())
    n_control_higher = int(control_higher.sum())

    # Layer with most significant heads
    peak_sig_layer = int(np.argmax(sig_per_layer))

    return {
        "metric": metric,
        "fdr_method": qht["fdr_method"],
        "alpha": qht["alpha"],
        "n_significant": n_significant,
        "n_total": n_total,
        "proportion_significant": n_significant / n_total,
        "n_conflict_higher": n_conflict_higher,
        "n_control_higher": n_control_higher,
        "sig_per_layer": sig_per_layer.tolist(),
        "peak_sig_layer": peak_sig_layer,
        "peak_sig_count": int(sig_per_layer[peak_sig_layer]),
        "sig_matrix": sig_matrix,
        "effect_matrix": effect_matrix,
    }


def analyze_s2_heads(data: dict[str, Any]) -> dict[str, Any]:
    """Extract S2-specialized heads from pre-computed analysis."""
    s2h = data["analysis"]["s2_specialized_heads"]
    qh = s2h["query_head"]
    kv = s2h["kv_group"]

    qh_heads = qh["heads"]
    conflict_higher_heads = [h for h in qh_heads if h["direction"] == "conflict_higher"]
    control_higher_heads = [h for h in qh_heads if h["direction"] == "control_higher"]

    # Layer distribution of S2-specialized heads
    layers = [h["layer"] for h in qh_heads]
    layer_counts = np.zeros(data["n_layers"], dtype=int)
    for l in layers:
        layer_counts[l] += 1

    return {
        "criteria": qh["criteria"],
        "n_specialized": qh["n_specialized"],
        "n_total": qh["n_total"],
        "proportion": qh["proportion"],
        "n_conflict_higher": len(conflict_higher_heads),
        "n_control_higher": len(control_higher_heads),
        "layer_distribution": layer_counts.tolist(),
        "heads": qh_heads,
        "kv_group": {
            "n_specialized": kv["n_specialized"],
            "n_total": kv["n_total"],
            "proportion": kv["proportion"],
        },
    }


def find_peak_gap(
    profiles: dict[str, NDArray],
    head_tests: dict[str, Any],
) -> dict[str, Any]:
    """Find where the conflict-control entropy gap peaks.

    Only considers layers where a meaningful number of heads are significant,
    to avoid reporting a spurious peak in a noisy layer.
    """
    gap = profiles["gap"]
    sig_per_layer = np.array(head_tests["sig_per_layer"])

    # Absolute gap
    abs_gap = np.abs(gap)
    peak_layer = int(np.argmax(abs_gap))
    peak_magnitude = float(abs_gap[peak_layer])

    # Also find peak among layers with >= 50% heads significant
    n_heads = head_tests["n_total"] // len(sig_per_layer)
    robust_mask = sig_per_layer >= (n_heads * 0.5)
    if robust_mask.any():
        robust_gap = np.where(robust_mask, abs_gap, 0)
        robust_peak = int(np.argmax(robust_gap))
    else:
        robust_peak = peak_layer

    return {
        "peak_layer": peak_layer,
        "peak_gap": peak_magnitude,
        "peak_gap_signed": float(gap[peak_layer]),
        "robust_peak_layer": robust_peak,
        "robust_peak_gap": float(abs_gap[robust_peak]),
    }


def identify_significant_layers(
    head_tests: dict[str, Any],
    n_heads: int,
    threshold: float = 0.25,
) -> NDArray[np.bool_]:
    """Mark layers where >= threshold fraction of heads are significant.

    Used for shading in the figure. A layer is 'significant' if at least
    threshold * n_heads individual heads have q < alpha.
    """
    sig_per_layer = np.array(head_tests["sig_per_layer"])
    return sig_per_layer >= (n_heads * threshold)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def make_figure(
    model_results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Generate fig5: two-panel attention entropy profiles.

    Left: Llama-3.1-8B, Right: R1-Distill-Llama-8B.
    Blue: conflict, Orange: control. Shaded where gap is significant.
    """
    set_paper_theme()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, result in zip(axes, model_results):
        profiles = result["profiles"]
        n_layers = len(profiles["conflict_mean"])
        layers = np.arange(n_layers)
        sig_layers = result["sig_layer_mask"]

        # Plot mean lines
        ax.plot(
            layers, profiles["conflict_mean"],
            color=COLORS["conflict"], label="Conflict (S2-like)",
        )
        ax.plot(
            layers, profiles["control_mean"],
            color=COLORS["control"], label="Control (S1-like)",
        )

        # SEM bands
        ax.fill_between(
            layers,
            profiles["conflict_mean"] - profiles["conflict_sem"],
            profiles["conflict_mean"] + profiles["conflict_sem"],
            color=COLORS["conflict"], alpha=0.15,
        )
        ax.fill_between(
            layers,
            profiles["control_mean"] - profiles["control_sem"],
            profiles["control_mean"] + profiles["control_sem"],
            color=COLORS["control"], alpha=0.15,
        )

        # Shade significant gap regions
        # Find contiguous runs of significant layers for cleaner shading
        for i in range(n_layers):
            if sig_layers[i]:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color=COLORS["gap_fill"])

        ax.set_xlabel("Layer")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_title(result["short_name"])
        ax.legend(loc="upper left", framealpha=0.8)

        # Annotate peak gap layer
        peak = result["peak_gap_info"]
        peak_l = peak["peak_layer"]
        ax.annotate(
            f"peak gap L{peak_l}",
            xy=(peak_l, profiles["conflict_mean"][peak_l]),
            xytext=(peak_l + 2, profiles["conflict_mean"][peak_l] + 0.01),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            fontsize=8, color="gray",
        )

    axes[0].set_ylabel("Mean Normalized Entropy")

    fig.suptitle(
        "Attention Entropy: Conflict vs Control Items",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig5_attention_entropy.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] fig5_attention_entropy -> {pdf_path}")

    return pdf_path


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------


def build_summary(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the attention_summary.json payload."""
    summary: dict[str, Any] = {"models": {}}

    for result in model_results:
        name = result["short_name"]
        ht = result["head_tests"]
        s2 = result["s2_heads"]
        pg = result["peak_gap_info"]
        prof = result["profiles"]

        # Key paper number: proportion of heads with significantly higher
        # entropy on conflict items (among all significant heads, how many
        # show conflict > control)
        prop_conflict_higher_of_sig = (
            ht["n_conflict_higher"] / ht["n_significant"]
            if ht["n_significant"] > 0 else 0.0
        )

        # Among S2-specialized heads (strong effect), proportion conflict_higher
        prop_s2_conflict = (
            s2["n_conflict_higher"] / s2["n_specialized"]
            if s2["n_specialized"] > 0 else 0.0
        )

        summary["models"][name] = {
            "model_id": result["model_id"],
            "n_layers": result["n_layers"],
            "n_heads": result["n_heads"],
            "n_items": prof["n_conflict"] + prof["n_control"],
            "n_conflict": prof["n_conflict"],
            "n_control": prof["n_control"],
            "head_level_tests": {
                "metric": ht["metric"],
                "fdr_method": ht["fdr_method"],
                "alpha": ht["alpha"],
                "n_tests": ht["n_total"],
                "n_significant": ht["n_significant"],
                "proportion_significant": round(ht["proportion_significant"], 4),
                "n_conflict_higher": ht["n_conflict_higher"],
                "n_control_higher": ht["n_control_higher"],
                "proportion_conflict_higher_of_significant": round(
                    prop_conflict_higher_of_sig, 4
                ),
            },
            "s2_specialized_heads": {
                "criteria": s2["criteria"],
                "n_specialized": s2["n_specialized"],
                "n_total": s2["n_total"],
                "proportion": round(s2["proportion"], 4),
                "n_conflict_higher": s2["n_conflict_higher"],
                "n_control_higher": s2["n_control_higher"],
                "proportion_conflict_higher": round(prop_s2_conflict, 4),
                "kv_group_n_specialized": s2["kv_group"]["n_specialized"],
                "kv_group_proportion": round(s2["kv_group"]["proportion"], 4),
            },
            "layer_profile": {
                "peak_gap_layer": pg["peak_layer"],
                "peak_gap_magnitude": round(pg["peak_gap"], 6),
                "peak_gap_signed": round(pg["peak_gap_signed"], 6),
                "robust_peak_layer": pg["robust_peak_layer"],
                "conflict_mean_by_layer": [round(v, 6) for v in prof["conflict_mean"]],
                "control_mean_by_layer": [round(v, 6) for v in prof["control_mean"]],
                "gap_by_layer": [round(v, 6) for v in prof["gap"]],
            },
        }

    # Top-level key numbers for the paper
    summary["paper_highlights"] = {}
    for result in model_results:
        name = result["short_name"]
        ht = result["head_tests"]
        s2 = result["s2_heads"]
        pg = result["peak_gap_info"]

        n_sig = ht["n_significant"]
        n_total = ht["n_total"]
        pct_sig = 100 * n_sig / n_total
        pct_conflict_higher = (
            100 * ht["n_conflict_higher"] / n_sig if n_sig > 0 else 0.0
        )

        summary["paper_highlights"][name] = {
            "headline": (
                f"{pct_sig:.1f}% of heads ({n_sig}/{n_total}) show significant "
                f"entropy differences (BH-FDR q<0.05); of those, "
                f"{pct_conflict_higher:.1f}% have higher entropy on conflict items"
            ),
            "s2_specialized": (
                f"{s2['n_specialized']} heads ({100 * s2['proportion']:.1f}%) meet "
                f"S2-specialization criteria (q<0.05 + |r_rb|>=0.3)"
            ),
            "peak_gap": (
                f"Peak entropy gap at layer {pg['peak_layer']} "
                f"(delta={pg['peak_gap_signed']:+.4f} normalized entropy)"
            ),
        }

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 72)
    print("ATTENTION ENTROPY ANALYSIS SUMMARY")
    print("=" * 72)

    for model_name, highlights in summary["paper_highlights"].items():
        print(f"\n--- {model_name} ---")
        print(f"  {highlights['headline']}")
        print(f"  {highlights['s2_specialized']}")
        print(f"  {highlights['peak_gap']}")

        model_data = summary["models"][model_name]
        ht = model_data["head_level_tests"]
        s2 = model_data["s2_specialized_heads"]
        lp = model_data["layer_profile"]

        print(f"\n  Head-level tests ({ht['metric']}, {ht['fdr_method']} q<{ht['alpha']}):")
        print(f"    Significant: {ht['n_significant']}/{ht['n_tests']} "
              f"({100 * ht['proportion_significant']:.1f}%)")
        print(f"    Conflict > Control: {ht['n_conflict_higher']}  |  "
              f"Control > Conflict: {ht['n_control_higher']}")

        print(f"\n  S2-specialized heads ({s2['criteria']}):")
        print(f"    Query heads: {s2['n_specialized']}/{s2['n_total']} "
              f"({100 * s2['proportion']:.1f}%)")
        print(f"    KV groups: {s2['kv_group_n_specialized']} "
              f"({100 * s2['kv_group_proportion']:.1f}%)")
        print(f"    Direction: {s2['n_conflict_higher']} conflict-higher, "
              f"{s2['n_control_higher']} control-higher")

        print(f"\n  Layer profile:")
        print(f"    Peak gap layer: {lp['peak_gap_layer']} "
              f"(delta={lp['peak_gap_signed']:+.6f})")
        print(f"    Robust peak: layer {lp['robust_peak_layer']}")

        # Show layer gap profile as a mini sparkline
        gap = np.array(lp["gap_by_layer"])
        max_gap = max(abs(gap.max()), abs(gap.min()))
        if max_gap > 0:
            bars = ""
            for g in gap:
                normalized = g / max_gap
                if normalized > 0.5:
                    bars += "+"
                elif normalized > 0.1:
                    bars += "."
                elif normalized < -0.5:
                    bars += "-"
                elif normalized < -0.1:
                    bars += ","
                else:
                    bars += " "
            print(f"    Gap profile: [{bars}]")
            print(f"    (+ = conflict>control, - = control>conflict)")

    # Comparative summary
    model_names = list(summary["models"].keys())
    if len(model_names) == 2:
        m1, m2 = model_names
        s1 = summary["models"][m1]["s2_specialized_heads"]
        s2 = summary["models"][m2]["s2_specialized_heads"]
        print(f"\n--- Comparative ---")
        print(f"  S2-specialized heads: {m1}={s1['n_specialized']} vs {m2}={s2['n_specialized']}")
        ratio = s2["proportion"] / s1["proportion"] if s1["proportion"] > 0 else float("inf")
        print(f"  Ratio (R1/Llama): {ratio:.2f}x")

        h1 = summary["models"][m1]["head_level_tests"]
        h2 = summary["models"][m2]["head_level_tests"]
        print(f"  Significant heads: {m1}={h1['n_significant']} vs {m2}={h2['n_significant']}")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze attention entropy results")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results_pod/attention"),
        help="Directory containing *_attention.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/attention"),
        help="Directory for summary JSON output",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for figure output",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="norm_entropy",
        help="Metric for layer profiles (default: norm_entropy)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    figure_dir = project_root / args.figure_dir

    # Model configs: (filename, short display name)
    model_files = [
        ("llama31_attention.json", "Llama-3.1-8B"),
        ("r1_distill_attention.json", "R1-Distill-Llama-8B"),
    ]

    print("Loading attention data...")
    model_results: list[dict[str, Any]] = []

    for filename, short_name in model_files:
        path = data_dir / filename
        if not path.exists():
            print(f"  [SKIP] {filename} not found at {path}", file=sys.stderr)
            continue

        data = load_attention_data(path)

        # Compute layer profiles from raw item data
        profiles = compute_layer_profiles(data, metric=args.metric)

        # Analyze pre-computed statistical tests
        head_tests = analyze_head_tests(data, metric=args.metric)

        # S2-specialized head analysis
        s2_heads = analyze_s2_heads(data)

        # Peak gap analysis
        peak_gap_info = find_peak_gap(profiles, head_tests)

        # Layer significance mask for figure shading
        sig_mask = identify_significant_layers(
            head_tests, n_heads=data["n_heads"], threshold=0.25,
        )

        model_results.append({
            "short_name": short_name,
            "model_id": data["model"],
            "n_layers": data["n_layers"],
            "n_heads": data["n_heads"],
            "profiles": profiles,
            "head_tests": head_tests,
            "s2_heads": s2_heads,
            "peak_gap_info": peak_gap_info,
            "sig_layer_mask": sig_mask,
        })

    if not model_results:
        print("No data loaded. Check --data-dir.", file=sys.stderr)
        sys.exit(1)

    # Generate figure
    print("\nGenerating figure...")
    make_figure(model_results, figure_dir)

    # Build and save summary
    print("\nBuilding summary...")
    summary = build_summary(model_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "attention_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] Summary -> {summary_path}")

    # Print human-readable summary
    print_summary(summary)


if __name__ == "__main__":
    main()
