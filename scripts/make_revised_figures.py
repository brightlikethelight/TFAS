#!/usr/bin/env python3
"""Generate ALL figures for the revised NeurIPS paper.

Figures:
    1. Steering dose-response (lure + correct, Llama only, single clean panel)
    2. Probe logit histogram (already exists — skip if present)
    3. Within-CoT trajectory (already exists — skip if present)
    4. Scale comparison 7B vs 32B (already exists — skip if present)
    5. Text baseline vs activation probe per-category AUC comparison (NEW)

Usage::

    python scripts/make_revised_figures.py
    python scripts/make_revised_figures.py --force   # regenerate all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

LLAMA_STEERING = PROJECT_ROOT / "results" / "causal" / "probe_steering_llama_l14.json"
TEXT_BASELINE = PROJECT_ROOT / "results" / "probes" / "text_baseline_probe.json"

# ---------------------------------------------------------------------------
# Publication theme — NeurIPS camera-ready compatible
# ---------------------------------------------------------------------------


def set_paper_theme() -> None:
    """Publication-grade rcParams: serif, no top/right spines, 300 DPI."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "pdf.fonttype": 42,  # TrueType for camera-ready
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, name: str) -> Path:
    """Save figure as PDF + PNG to figures/."""
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name} -> {pdf}")
    return pdf


# ============================================================================
# Figure 1: Steering dose-response — THE key figure
# ============================================================================

# Colorblind-friendly (Wong 2011)
C_LURE = "#D55E00"       # vermillion
C_CORRECT = "#009E73"    # bluish green
C_RANDOM = "#999999"     # gray
C_BASELINE = "#555555"   # dark gray


def make_figure1_steering(force: bool = False) -> None:
    """Single clean panel: Llama lure + correct rate vs alpha, with random band."""
    out_name = "fig_steering_correctness"
    if not force and (FIG_DIR / f"{out_name}.pdf").exists():
        print(f"  [SKIP] {out_name} exists (use --force to regenerate)")
        # Regenerate anyway since the spec says it needs updating
        pass

    with open(LLAMA_STEERING) as f:
        data = json.load(f)

    alphas_data = data["alphas"]
    random_data = data["random_controls"]
    alpha_vals = sorted(float(a) for a in alphas_data.keys())

    # Extract probe rates
    lure_rate = np.array([alphas_data[str(a)]["lure_rate"] * 100 for a in alpha_vals])
    correct_rate = np.array([alphas_data[str(a)]["correct_rate"] * 100 for a in alpha_vals])

    # Random controls
    rand_lure_mean = np.array([random_data[str(a)]["mean_lure_rate"] * 100 for a in alpha_vals])
    rand_lure_std = np.array([random_data[str(a)]["std"] * 100 for a in alpha_vals])
    rand_correct_mean = np.array(
        [random_data[str(a)]["mean_correct_rate"] * 100 for a in alpha_vals]
    )
    rand_correct_std = np.array(
        [random_data[str(a)]["std_correct_rate"] * 100 for a in alpha_vals]
    )

    # Build single-panel figure
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # --- Random control bands ---
    ax.fill_between(
        alpha_vals,
        rand_lure_mean - rand_lure_std,
        rand_lure_mean + rand_lure_std,
        color=C_LURE, alpha=0.10, zorder=1,
    )
    ax.fill_between(
        alpha_vals,
        rand_correct_mean - rand_correct_std,
        rand_correct_mean + rand_correct_std,
        color=C_CORRECT, alpha=0.10, zorder=1,
    )

    # Random mean lines (dashed, thin)
    ax.plot(alpha_vals, rand_lure_mean, color=C_LURE, linewidth=0.8,
            linestyle=":", alpha=0.5, zorder=2)
    ax.plot(alpha_vals, rand_correct_mean, color=C_CORRECT, linewidth=0.8,
            linestyle=":", alpha=0.5, zorder=2)

    # --- Probe direction lines ---
    ax.plot(
        alpha_vals, lure_rate,
        color=C_LURE, marker="o", markersize=6,
        markeredgecolor="white", markeredgewidth=0.8,
        label="Lure rate (probe)", linewidth=2.0, zorder=4,
    )
    ax.plot(
        alpha_vals, correct_rate,
        color=C_CORRECT, marker="s", markersize=6,
        markeredgecolor="white", markeredgewidth=0.8,
        label="Correct rate (probe)", linewidth=2.0, zorder=4,
    )

    # --- Random control label ---
    # Single label for the random bands
    ax.fill_between([], [], [], color=C_RANDOM, alpha=0.2,
                    label="Random direction ($\\pm$1 SD)")

    # --- Crossover annotation ---
    # Find where lure and correct cross
    cross_idx = None
    for i in range(len(alpha_vals) - 1):
        if lure_rate[i] >= correct_rate[i] and lure_rate[i + 1] < correct_rate[i + 1]:
            cross_idx = i
            break

    if cross_idx is not None:
        # Interpolate crossover alpha
        a1, a2 = alpha_vals[cross_idx], alpha_vals[cross_idx + 1]
        l1, l2 = lure_rate[cross_idx], lure_rate[cross_idx + 1]
        c1, c2 = correct_rate[cross_idx], correct_rate[cross_idx + 1]
        # linear interp
        denom = (c2 - c1) - (l2 - l1)
        if abs(denom) > 1e-6:
            cross_alpha = a1 + (l1 - c1) / denom * (a2 - a1)
            cross_val = l1 + (cross_alpha - a1) / (a2 - a1) * (l2 - l1)
            ax.axvline(cross_alpha, color="#AAAAAA", linestyle="--", linewidth=0.7, zorder=1)
            ax.annotate(
                f"crossover\n$\\alpha$ = {cross_alpha:.1f}",
                xy=(cross_alpha, cross_val),
                xytext=(cross_alpha + 1.2, cross_val + 8),
                fontsize=7.5, color="#666666", ha="left",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8),
            )

    # --- Key result annotation ---
    idx_neg5 = alpha_vals.index(-5.0)
    idx_pos5 = alpha_vals.index(5.0)
    delta_lure = lure_rate[idx_pos5] - lure_rate[idx_neg5]
    delta_correct = correct_rate[idx_pos5] - correct_rate[idx_neg5]

    ax.annotate(
        f"$\\alpha$=-5: {lure_rate[idx_neg5]:.0f}% lure",
        xy=(-5, lure_rate[idx_neg5]),
        xytext=(-4.5, lure_rate[idx_neg5] + 5),
        fontsize=7, color=C_LURE, fontweight="bold",
    )
    ax.annotate(
        f"$\\alpha$=+5: {correct_rate[idx_pos5]:.0f}% correct",
        xy=(5, correct_rate[idx_pos5]),
        xytext=(2.5, correct_rate[idx_pos5] + 5),
        fontsize=7, color=C_CORRECT, fontweight="bold",
    )

    # --- S1/S2 direction labels ---
    ax.annotate(
        "$\\leftarrow$ S1 (lure-biased)",
        xy=(-5.3, 22), fontsize=7.5, color="#666666", ha="left", va="bottom",
    )
    ax.annotate(
        "S2 (deliberative) $\\rightarrow$",
        xy=(5.3, 22), fontsize=7.5, color="#666666", ha="right", va="bottom",
    )

    # Axes
    ax.set_xlabel("Steering strength ($\\alpha$)")
    ax.set_ylabel("Rate (%)")
    ax.set_xlim(-5.8, 5.8)
    ax.set_ylim(20, 75)
    ax.set_xticks([-5, -3, -1, 0, 1, 3, 5])

    ax.set_title(
        "Causal Steering Dose-Response: Llama-3.1-8B Layer 14",
        fontsize=11, fontweight="bold", pad=10,
    )

    ax.legend(
        loc="upper center", frameon=True, framealpha=0.95,
        edgecolor="#CCCCCC", ncol=3, fontsize=7.5,
        bbox_to_anchor=(0.5, -0.15),
    )

    fig.tight_layout()
    _save(fig, out_name)


# ============================================================================
# Figure 5: Text baseline vs activation probe per-category AUC
# ============================================================================


def make_figure5_text_baseline() -> None:
    """Grouped bar chart: activation probe vs text baseline AUC per category."""

    # --- Load text baseline data ---
    with open(TEXT_BASELINE) as f:
        text_data = json.load(f)

    per_cat_text = text_data["results"]["per_category"]

    # --- Activation probe per-category AUC (from transfer matrix diagonal, L14 Llama) ---
    # These are within-category CVd AUC at the best layer (L14)
    # From transfer_matrix_l14_llama.json diagonal entries
    activation_aucs = {
        "anchoring": 1.000,
        "arithmetic": 1.000,
        "base_rate": 1.000,
        "conjunction": 1.000,
        "crt": 1.000,
        "framing": 1.000,
        "syllogism": 0.936,
    }

    # Also add the llama31_ALL_layer_aucs.json overall best for reference
    # Peak layer 14 all-categories AUC = 0.999
    activation_overall = 0.999

    # Text baseline per-category
    text_aucs = {}
    for cat, res in per_cat_text.items():
        if "skipped" in res and res["skipped"]:
            continue
        text_aucs[cat] = res["auc"]

    # Text baseline overall
    text_overall = text_data["results"]["all_categories"]["auc"]  # 0.840

    # Determine categories to show (present in both)
    categories = sorted(set(activation_aucs.keys()) & set(text_aucs.keys()))

    # Add extra categories from text baseline that have data
    # (availability, certainty_effect, loss_aversion, sunk_cost also exist in text but not activation)
    # Only show the 7 categories that have both probes
    # Order: vulnerable first, then immune
    vuln = ["base_rate", "conjunction", "syllogism"]
    immune = ["anchoring", "arithmetic", "crt", "framing"]
    extra = sorted(set(text_aucs.keys()) - set(vuln) - set(immune))
    ordered_cats = vuln + immune

    n_cats = len(ordered_cats)
    x = np.arange(n_cats)
    width = 0.35

    # Data arrays
    act_vals = [activation_aucs.get(c, np.nan) for c in ordered_cats]
    txt_vals = [text_aucs.get(c, np.nan) for c in ordered_cats]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # --- Bars ---
    C_ACT = "#0072B2"   # blue (Wong)
    C_TXT = "#E69F00"   # amber/orange (Wong)

    bars_act = ax.bar(
        x - width / 2, act_vals, width,
        label=f"Activation probe (L14, AUC$_{{all}}$={activation_overall:.3f})",
        color=C_ACT, edgecolor="white", linewidth=0.6, zorder=3,
    )
    bars_txt = ax.bar(
        x + width / 2, txt_vals, width,
        label=f"Text baseline (AUC$_{{all}}$={text_overall:.3f})",
        color=C_TXT, edgecolor="white", linewidth=0.6, zorder=3,
    )

    # --- Value labels ---
    for bars, vals in [(bars_act, act_vals), (bars_txt, txt_vals)]:
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            fontsize = 6.5 if v < 0.15 else 7
            y_pos = max(v + 0.02, 0.04)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{v:.3f}" if v < 0.1 else f"{v:.2f}",
                ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
                rotation=0 if n_cats <= 8 else 45,
            )

    # --- Highlight the syllogism gap ---
    syl_idx = ordered_cats.index("syllogism")
    syl_act = act_vals[syl_idx]
    syl_txt = txt_vals[syl_idx]
    ax.annotate(
        f"Activation: {syl_act:.3f}\nText: {syl_txt:.3f}\n$\\Delta$ = {syl_act - syl_txt:+.3f}",
        xy=(syl_idx, max(syl_act, syl_txt) + 0.03),
        xytext=(syl_idx + 1.8, 0.65),
        fontsize=7, fontweight="bold", color="#333333",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0,
                        connectionstyle="arc3,rad=-0.2"),
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="#CCCCCC", alpha=0.9),
    )

    # --- Chance line ---
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax.text(n_cats - 0.6, 0.515, "chance", fontsize=7, color="#999999", ha="right")

    # --- Vulnerable / Immune divider ---
    div_x = len(vuln) - 0.5
    ax.axvline(div_x, color="#AAAAAA", linestyle=":", linewidth=0.8, alpha=0.5, zorder=1)
    ax.text(div_x - 0.1, 1.05, "vulnerable", fontsize=7.5, color="#888888",
            ha="right", va="bottom", transform=ax.get_xaxis_transform())
    ax.text(div_x + 0.1, 1.05, "immune", fontsize=7.5, color="#888888",
            ha="left", va="bottom", transform=ax.get_xaxis_transform())

    # --- Axes ---
    display_labels = [c.replace("_", "\n") for c in ordered_cats]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=8)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax.set_title(
        "Activation Probe vs Text Baseline: Per-Category AUC",
        fontsize=11, fontweight="bold", pad=10,
    )

    ax.legend(
        loc="lower right", frameon=True, framealpha=0.95,
        edgecolor="#CCCCCC", fontsize=8,
    )

    fig.tight_layout()
    _save(fig, "fig_text_baseline_comparison")


# ============================================================================
# Verification: check that figures 2-4 exist
# ============================================================================


def verify_existing_figures() -> dict[str, bool]:
    """Check that figures 2-4 exist on disk."""
    expected = {
        "fig_probe_logit_histogram": "Figure 2: Probe logit histogram",
        "fig_within_cot_trajectory": "Figure 3: Within-CoT trajectory",
        "fig_scale_comparison": "Figure 4: Scale comparison (7B vs 32B)",
    }

    status = {}
    for name, desc in expected.items():
        pdf = FIG_DIR / f"{name}.pdf"
        png = FIG_DIR / f"{name}.png"
        exists = pdf.exists() and png.exists()
        status[name] = exists
        marker = "[OK]" if exists else "[MISSING]"
        print(f"  {marker} {desc}")
        if exists:
            pdf_size = pdf.stat().st_size
            print(f"         {pdf} ({pdf_size:,} bytes)")
    return status


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all revised NeurIPS paper figures.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate all figures even if they exist.",
    )
    args = parser.parse_args()

    set_paper_theme()

    print("=" * 65)
    print("Revised NeurIPS Paper Figures")
    print("=" * 65)

    # Figure 1: Steering dose-response (always regenerate — it needs updating)
    print("\n--- Figure 1: Steering dose-response (updated) ---")
    make_figure1_steering(force=True)

    # Figures 2-4: Verify existence
    print("\n--- Figures 2-4: Verification ---")
    status = verify_existing_figures()

    missing = [k for k, v in status.items() if not v]
    if missing:
        print(f"\n  WARNING: {len(missing)} figures missing. Run their scripts:")
        scripts = {
            "fig_probe_logit_histogram": "scripts/probe_logit_histogram.py (requires GPU/H5)",
            "fig_within_cot_trajectory": "scripts/make_within_cot_figure.py",
            "fig_scale_comparison": "scripts/make_scale_figure.py",
        }
        for name in missing:
            print(f"    python {scripts.get(name, '???')}")

    # Figure 5: Text baseline comparison (NEW)
    print("\n--- Figure 5: Text baseline vs activation probe (NEW) ---")
    make_figure5_text_baseline()

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Figure 1: fig_steering_correctness.pdf      [REGENERATED]")
    for name, desc in [
        ("fig_probe_logit_histogram", "Figure 2"),
        ("fig_within_cot_trajectory", "Figure 3"),
        ("fig_scale_comparison", "Figure 4"),
    ]:
        tag = "OK" if status.get(name, False) else "MISSING"
        print(f"  {desc}: {name}.pdf      [{tag}]")
    print(f"  Figure 5: fig_text_baseline_comparison.pdf   [NEW]")
    print(f"\n  All figures in: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
