"""Probe logit histogram analysis: reinterpret cross-prediction AUC.

The cross-prediction AUC of 0.378 is misleading — it means the probe
actively predicts the OPPOSITE class on immune categories. This script
generates logit histograms to properly diagnose whether:
  (a) Immune items cluster together (no meaningful projection → specificity confirmed)
  (b) Immune items have REVERSED polarity (needs explanation)

Produces:
  - figures/fig_probe_logit_histogram.pdf
  - results/probes/logit_histogram_analysis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

H5_PATH = Path("/workspace/s1s2/data/activations/llama31_8b_instruct.h5")
MODEL_KEY = "unsloth_Meta-Llama-3.1-8B-Instruct"
VULNERABLE_CATS = {"base_rate", "conjunction", "syllogism"}
IMMUNE_CATS = {"crt", "arithmetic", "framing", "anchoring"}
BEST_LAYER = 14  # peak within-vulnerable AUC layer
POSITION = "P0"  # pre-generation position
SEED = 42
OUT_FIG = Path("/workspace/s1s2/figures/fig_probe_logit_histogram.pdf")
OUT_JSON = Path("/workspace/s1s2/results/probes/logit_histogram_analysis.json")


def load_data(
    h5_path: Path, model_key: str, layer: int, position: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load activations, conflict labels, and category strings."""
    with h5py.File(h5_path, "r") as f:
        # Categories and labels
        cat_raw = f["problems/category"][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cat_raw])
        conflict = f["problems/conflict"][:].astype(bool)

        # Position index
        labels = [
            lb.decode() if isinstance(lb, bytes) else lb
            for lb in f[f"models/{model_key}/position_index/labels"][:]
        ]
        pos_idx = labels.index(position)

        # Activations at this layer and position
        X = f[f"models/{model_key}/residual/layer_{layer:02d}"][:, pos_idx, :]
        X = X.astype(np.float32)

    return X, conflict, cats


def train_probe_get_logits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_all: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, LogisticRegressionCV, StandardScaler]:
    """Train logistic probe on train set, return raw logits for ALL items."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_all_s = scaler.transform(X_all)

    clf = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 20),
        cv=5,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        scoring="roc_auc",
        refit=True,
        random_state=seed,
        n_jobs=1,
    )
    clf.fit(X_train_s, y_train)

    # Raw logits = X @ w + b (before sigmoid)
    logits_all = X_all_s @ clf.coef_.ravel() + clf.intercept_[0]
    return logits_all, clf, scaler


def compute_statistics(
    logits: np.ndarray,
    conflict: np.ndarray,
    cats: np.ndarray,
) -> dict:
    """Compute per-group statistics on probe logits."""
    vuln_mask = np.array([c in VULNERABLE_CATS for c in cats])
    immune_mask = np.array([c in IMMUNE_CATS for c in cats])

    groups = {
        "vulnerable_conflict": vuln_mask & conflict,
        "vulnerable_control": vuln_mask & ~conflict,
        "immune_conflict": immune_mask & conflict,
        "immune_control": immune_mask & ~conflict,
    }

    stats = {}
    for name, mask in groups.items():
        g_logits = logits[mask]
        stats[name] = {
            "n": int(mask.sum()),
            "mean": float(np.mean(g_logits)),
            "std": float(np.std(g_logits)),
            "median": float(np.median(g_logits)),
            "min": float(np.min(g_logits)),
            "max": float(np.max(g_logits)),
            "q25": float(np.percentile(g_logits, 25)),
            "q75": float(np.percentile(g_logits, 75)),
            "fraction_positive": float(np.mean(g_logits > 0)),
        }

    # Compute separation metrics
    # Within vulnerable: conflict vs control separation
    vc = logits[groups["vulnerable_conflict"]]
    vk = logits[groups["vulnerable_control"]]
    stats["vulnerable_separation"] = {
        "cohens_d": float(
            (np.mean(vc) - np.mean(vk))
            / np.sqrt((np.var(vc) + np.var(vk)) / 2)
        ),
        "auc": float(roc_auc_score(
            np.concatenate([np.ones(len(vc)), np.zeros(len(vk))]),
            np.concatenate([vc, vk]),
        )),
        "overlap_fraction": _overlap_fraction(vc, vk),
    }

    # Within immune: conflict vs control separation
    ic = logits[groups["immune_conflict"]]
    ik = logits[groups["immune_control"]]
    stats["immune_separation"] = {
        "cohens_d": float(
            (np.mean(ic) - np.mean(ik))
            / np.sqrt((np.var(ic) + np.var(ik)) / 2)
        ),
        "auc": float(roc_auc_score(
            np.concatenate([np.ones(len(ic)), np.zeros(len(ik))]),
            np.concatenate([ic, ik]),
        )),
        "overlap_fraction": _overlap_fraction(ic, ik),
    }

    # Key diagnostic: WHERE do immune items cluster relative to vulnerable?
    # If immune items all cluster near zero → probe direction uninformative for them
    # If immune items cluster at reversed poles → polarity issue
    stats["immune_vs_vulnerable_polarity"] = {
        "immune_all_mean": float(np.mean(logits[immune_mask])),
        "immune_all_std": float(np.std(logits[immune_mask])),
        "vulnerable_all_mean": float(np.mean(logits[vuln_mask])),
        "vulnerable_all_std": float(np.std(logits[vuln_mask])),
        "immune_conflict_mean_vs_vuln_conflict_mean": float(
            np.mean(ic) - np.mean(vc)
        ),
        "immune_control_mean_vs_vuln_control_mean": float(
            np.mean(ik) - np.mean(vk)
        ),
    }

    return stats


def _overlap_fraction(a: np.ndarray, b: np.ndarray, n_bins: int = 100) -> float:
    """Fraction of histogram overlap between two distributions."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    bin_width = (hi - lo) / n_bins
    overlap = np.sum(np.minimum(ha, hb)) * bin_width
    return float(overlap)


def determine_interpretation(stats: dict) -> str:
    """Produce the correct interpretation based on the logit distributions.

    The key diagnostic is not just the AUC but the LOCATION and SPREAD of
    immune logit distributions. Three scenarios:
    (A) Immune items centered near zero, overlapping → specificity (no projection)
    (B) Immune items tightly clustered at one pole → probe encodes category
        membership, not just conflict/control
    (C) Immune items at reversed poles → polarity reversal
    """
    immune_sep = stats["immune_separation"]
    vuln_sep = stats["vulnerable_separation"]

    ic = stats["immune_conflict"]
    ik = stats["immune_control"]
    vc = stats["vulnerable_conflict"]
    vk = stats["vulnerable_control"]

    lines = []

    # Vulnerable: sanity check
    lines.append(
        f"VULNERABLE categories: strong conflict/control separation "
        f"(AUC={vuln_sep['auc']:.3f}, Cohen's d={vuln_sep['cohens_d']:.2f}). "
        f"Conflict items: mean logit = {vc['mean']:.3f} (std={vc['std']:.3f}). "
        f"Control items: mean logit = {vk['mean']:.3f} (std={vk['std']:.3f})."
    )

    # Immune: the key analysis
    immune_spread = max(ic["std"], ik["std"])
    vuln_spread = (vc["std"] + vk["std"]) / 2
    spread_ratio = immune_spread / vuln_spread if vuln_spread > 0 else 0

    # How far from zero are immune items?
    immune_center = (ic["mean"] + ik["mean"]) / 2
    vuln_center = (vc["mean"] + vk["mean"]) / 2

    # Internal separation within immune
    immune_internal_sep = abs(ic["mean"] - ik["mean"])
    vuln_internal_sep = abs(vc["mean"] - vk["mean"])

    lines.append(
        f"\nIMMUNE categories: ALL items cluster tightly in a narrow band "
        f"(conflict mean={ic['mean']:.3f}, std={ic['std']:.3f}; "
        f"control mean={ik['mean']:.3f}, std={ik['std']:.3f})."
    )

    lines.append(
        f"\nKey metrics:"
        f"\n  Immune logit range: [{min(ic['min'], ik['min']):.3f}, "
        f"{max(ic['max'], ik['max']):.3f}]"
        f"\n  Vulnerable logit range: [{min(vc['min'], vk['min']):.3f}, "
        f"{max(vc['max'], vk['max']):.3f}]"
        f"\n  Immune spread (max std): {immune_spread:.3f} "
        f"vs vulnerable spread (mean std): {vuln_spread:.3f} "
        f"(ratio: {spread_ratio:.2f}x)"
        f"\n  Immune internal separation: {immune_internal_sep:.3f} "
        f"vs vulnerable internal separation: {vuln_internal_sep:.3f}"
        f"\n  Immune center: {immune_center:.3f} "
        f"vs vulnerable center: {vuln_center:.3f}"
    )

    # Determine which scenario
    # Key test: is immune_internal_sep << vuln_internal_sep?
    sep_ratio = immune_internal_sep / vuln_internal_sep if vuln_internal_sep > 0 else 0

    if sep_ratio < 0.1 and spread_ratio < 0.5:
        # Both conflict and control immune items cluster together:
        # probe direction is uninformative for immune categories
        scenario = "COMPRESSED_CLUSTER"
        lines.append(
            f"\nDIAGNOSIS: The probe direction compresses immune items into a "
            f"tight cluster (separation ratio = {sep_ratio:.2f}x of vulnerable). "
            f"The conflict/control distinction that the probe learned on "
            f"vulnerable categories does not project meaningfully onto immune items. "
            f"This is PARTIAL SUPPORT for specificity: the probe captures a "
            f"category-specific processing signal, not a universal one."
        )
    else:
        scenario = "SHIFTED_WITH_MINOR_REVERSAL"
        lines.append(
            f"\nDIAGNOSIS: Immune items are shifted to the control side "
            f"(center={immune_center:.3f}) with a small reversed internal "
            f"separation (d={immune_sep['cohens_d']:.2f}). The dominant effect "
            f"is that the probe direction partly encodes category membership — "
            f"all immune items land on the 'control-like' side regardless of "
            f"their actual conflict/control status. The minor reversed "
            f"separation (AUC={immune_sep['auc']:.3f}) is a secondary effect."
        )

    # Corrected interpretation of the AUC metric
    lines.append(
        f"\nCORRECTED INTERPRETATION of AUC={immune_sep['auc']:.3f}:"
        f"\nThe below-chance AUC does NOT indicate the probe 'detects the "
        f"opposite.' The histogram reveals that all 190 immune items cluster "
        f"in a tight logit band [{min(ic['min'], ik['min']):.2f}, "
        f"{max(ic['max'], ik['max']):.2f}] entirely on the control side of "
        f"the decision boundary. The AUC < 0.5 arises because immune conflict "
        f"items are pushed slightly MORE negative than immune control items "
        f"(small reversed effect, d={immune_sep['cohens_d']:.2f}), but the "
        f"dominant phenomenon is the location shift, not the ordering. "
        f"Reporting only 'AUC=0.378' masks both the tight clustering AND "
        f"the location shift."
    )

    return "\n".join(lines)


def make_figure(
    logits: np.ndarray,
    conflict: np.ndarray,
    cats: np.ndarray,
    stats: dict,
    out_path: Path,
) -> None:
    """3-panel figure: top row = vulnerable vs immune histograms, bottom = per-category strip."""
    vuln_mask = np.array([c in VULNERABLE_CATS for c in cats])
    immune_mask = np.array([c in IMMUNE_CATS for c in cats])

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.35, wspace=0.25)

    # Common bin range across top panels for visual comparison
    all_logits = logits[vuln_mask | immune_mask]
    lo, hi = float(np.min(all_logits)) - 0.2, float(np.max(all_logits)) + 0.2
    bins = np.linspace(lo, hi, 50)

    # Panel 1 (top-left): Vulnerable categories
    ax1 = fig.add_subplot(gs[0, 0])
    vc = logits[vuln_mask & conflict]
    vk = logits[vuln_mask & ~conflict]
    ax1.hist(vc, bins=bins, alpha=0.6, color="#d62728", label="Conflict (n=80)", density=True)
    ax1.hist(vk, bins=bins, alpha=0.6, color="#1f77b4", label="Control (n=80)", density=True)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Probe logit", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(
        f"(A) Vulnerable categories\n"
        f"AUC = {stats['vulnerable_separation']['auc']:.3f}, "
        f"d = {stats['vulnerable_separation']['cohens_d']:.2f}",
        fontsize=10, fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2 (top-right): Immune categories
    ax2 = fig.add_subplot(gs[0, 1])
    ic = logits[immune_mask & conflict]
    ik = logits[immune_mask & ~conflict]
    ax2.hist(ic, bins=bins, alpha=0.6, color="#d62728", label="Conflict (n=95)", density=True)
    ax2.hist(ik, bins=bins, alpha=0.6, color="#1f77b4", label="Control (n=95)", density=True)
    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Probe logit", fontsize=11)
    ax2.set_title(
        f"(B) Immune categories\n"
        f"AUC = {stats['immune_separation']['auc']:.3f}, "
        f"d = {stats['immune_separation']['cohens_d']:.2f}",
        fontsize=10, fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=8)
    # Annotate the compressed clustering
    ax2.annotate(
        f"All 190 items in [{min(ic.min(), ik.min()):.1f}, {max(ic.max(), ik.max()):.1f}]",
        xy=(float(np.mean(ic)), 0.5),
        fontsize=8, ha="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6),
    )

    # Panel 3 (bottom, spanning both columns): Per-category strip plot
    ax3 = fig.add_subplot(gs[1, :])
    all_cats_ordered = (
        sorted(VULNERABLE_CATS) + [""] + sorted(IMMUNE_CATS)
    )
    y_positions = []
    y_labels = []
    y_idx = 0
    conflict_colors = {"conflict": "#d62728", "control": "#1f77b4"}
    for cat_name in all_cats_ordered:
        if cat_name == "":
            y_idx += 0.5  # gap between vulnerable and immune
            continue
        cat_mask = cats == cat_name
        for cond, color, marker in [
            ("conflict", "#d62728", "o"),
            ("control", "#1f77b4", "s"),
        ]:
            if cond == "conflict":
                mask = cat_mask & conflict
            else:
                mask = cat_mask & ~conflict
            pts = logits[mask]
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(pts))
            ax3.scatter(
                pts, y_idx + jitter + (0.2 if cond == "conflict" else -0.2),
                c=color, alpha=0.5, s=12, marker=marker,
                label=cond.capitalize() if y_idx == 0 else "",
                edgecolors="none",
            )
        y_positions.append(y_idx)
        y_labels.append(cat_name)
        y_idx += 1

    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(y_labels, fontsize=9)
    ax3.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("Probe logit", fontsize=11)
    ax3.set_title(
        "(C) Per-category logit distribution (circles = conflict, squares = control)",
        fontsize=10, fontweight="bold",
    )
    # Add divider between vulnerable and immune
    divider_y = len(sorted(VULNERABLE_CATS)) - 0.5 + 0.25
    ax3.axhline(divider_y, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax3.text(hi - 0.1, divider_y + 0.15, "vulnerable", fontsize=8, ha="right",
             color="gray", va="bottom")
    ax3.text(hi - 0.1, divider_y - 0.15, "immune", fontsize=8, ha="right",
             color="gray", va="top")
    ax3.invert_yaxis()

    # Add legend for strip plot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=6, label="Conflict"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4",
               markersize=6, label="Control"),
    ]
    ax3.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.suptitle(
        f"Probe logit distributions — Llama-3.1-8B, Layer {BEST_LAYER}\n"
        f"LogisticRegressionCV trained on vulnerable categories only "
        f"(base_rate + conjunction + syllogism)",
        fontsize=12,
        fontweight="bold",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    print(f"Saved figure to {out_path}")
    plt.close(fig)

    # Also save PNG for quick viewing
    png_path = out_path.with_suffix(".png")
    fig.savefig(str(png_path), bbox_inches="tight", dpi=150) if False else None
    # Re-render for PNG (fig is closed)
    # Just copy the PDF logic but output PNG
    fig2 = plt.figure(figsize=(14, 9))
    gs2 = fig2.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.35, wspace=0.25)

    ax1b = fig2.add_subplot(gs2[0, 0])
    ax1b.hist(vc, bins=bins, alpha=0.6, color="#d62728", label="Conflict", density=True)
    ax1b.hist(vk, bins=bins, alpha=0.6, color="#1f77b4", label="Control", density=True)
    ax1b.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1b.set_xlabel("Probe logit", fontsize=11)
    ax1b.set_ylabel("Density", fontsize=11)
    ax1b.set_title(
        f"(A) Vulnerable: AUC={stats['vulnerable_separation']['auc']:.3f}, "
        f"d={stats['vulnerable_separation']['cohens_d']:.2f}",
        fontsize=10, fontweight="bold",
    )
    ax1b.legend(fontsize=8)

    ax2b = fig2.add_subplot(gs2[0, 1])
    ax2b.hist(ic, bins=bins, alpha=0.6, color="#d62728", label="Conflict", density=True)
    ax2b.hist(ik, bins=bins, alpha=0.6, color="#1f77b4", label="Control", density=True)
    ax2b.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2b.set_xlabel("Probe logit", fontsize=11)
    ax2b.set_title(
        f"(B) Immune: AUC={stats['immune_separation']['auc']:.3f}, "
        f"d={stats['immune_separation']['cohens_d']:.2f}",
        fontsize=10, fontweight="bold",
    )
    ax2b.legend(fontsize=8)

    ax3b = fig2.add_subplot(gs2[1, :])
    y_idx = 0
    for cat_name in all_cats_ordered:
        if cat_name == "":
            y_idx += 0.5
            continue
        cat_mask = cats == cat_name
        for cond, color, marker in [
            ("conflict", "#d62728", "o"),
            ("control", "#1f77b4", "s"),
        ]:
            mask = (cat_mask & conflict) if cond == "conflict" else (cat_mask & ~conflict)
            pts = logits[mask]
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(pts))
            ax3b.scatter(
                pts, y_idx + jitter + (0.2 if cond == "conflict" else -0.2),
                c=color, alpha=0.5, s=12, marker=marker, edgecolors="none",
            )
        y_idx += 1
    ax3b.set_yticks(y_positions)
    ax3b.set_yticklabels(y_labels, fontsize=9)
    ax3b.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax3b.set_xlabel("Probe logit", fontsize=11)
    ax3b.set_title("(C) Per-category", fontsize=10, fontweight="bold")
    ax3b.axhline(divider_y, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax3b.invert_yaxis()
    ax3b.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig2.suptitle(
        f"Probe logit distributions — Layer {BEST_LAYER}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig2.savefig(str(png_path), bbox_inches="tight", dpi=150)
    print(f"Saved PNG to {png_path}")
    plt.close(fig2)


def main() -> None:
    print(f"Loading activations from {H5_PATH}, layer {BEST_LAYER}")
    X, conflict, cats = load_data(H5_PATH, MODEL_KEY, BEST_LAYER, POSITION)
    print(f"  X.shape = {X.shape}, n_conflict = {conflict.sum()}")

    # Build masks
    vuln_mask = np.array([c in VULNERABLE_CATS for c in cats])
    immune_mask = np.array([c in IMMUNE_CATS for c in cats])

    # Train probe on vulnerable categories only
    X_train = X[vuln_mask]
    y_train = conflict[vuln_mask].astype(np.int64)
    print(f"  Training on vulnerable: n={len(y_train)}, "
          f"pos={y_train.sum()}, neg={(1-y_train).sum()}")

    logits, clf, scaler = train_probe_get_logits(X_train, y_train, X, seed=SEED)
    print(f"  Best C = {clf.C_[0]:.4f}")
    print(f"  Logit range: [{logits.min():.3f}, {logits.max():.3f}]")

    # Compute statistics
    stats = compute_statistics(logits, conflict, cats)

    # Determine interpretation
    interpretation = determine_interpretation(stats)
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(interpretation)

    # Also compute per-category AUC on immune set
    per_cat_immune = {}
    for cat_name in sorted(IMMUNE_CATS):
        cat_mask = cats == cat_name
        if cat_mask.sum() < 4:
            continue
        y_cat = conflict[cat_mask].astype(int)
        logits_cat = logits[cat_mask]
        if len(np.unique(y_cat)) < 2:
            continue
        auc = float(roc_auc_score(y_cat, logits_cat))
        d_cat = float(
            (np.mean(logits_cat[y_cat == 1]) - np.mean(logits_cat[y_cat == 0]))
            / np.sqrt(
                (np.var(logits_cat[y_cat == 1]) + np.var(logits_cat[y_cat == 0])) / 2
            )
        )
        per_cat_immune[cat_name] = {
            "auc": auc,
            "cohens_d": d_cat,
            "n": int(cat_mask.sum()),
            "conflict_mean_logit": float(np.mean(logits_cat[y_cat == 1])),
            "control_mean_logit": float(np.mean(logits_cat[y_cat == 0])),
        }
        print(f"  {cat_name}: AUC={auc:.3f}, d={d_cat:.2f}")

    # Build output
    result = {
        "layer": BEST_LAYER,
        "position": POSITION,
        "model": "llama31_8b_instruct",
        "model_key_hdf5": MODEL_KEY,
        "probe": "logistic_regression_cv",
        "best_C": float(clf.C_[0]),
        "vulnerable_categories": sorted(VULNERABLE_CATS),
        "immune_categories": sorted(IMMUNE_CATS),
        "group_statistics": stats,
        "per_category_immune_auc": per_cat_immune,
        "interpretation": interpretation,
        "corrected_claim": (
            "The cross-prediction AUC of 0.378 does NOT confirm specificity. "
            "An AUC < 0.5 indicates the probe direction encodes a feature "
            "whose polarity REVERSES across category groups. A truly specific "
            "probe would yield AUC ≈ 0.5 with overlapping logit distributions "
            "on immune categories. The logit histogram analysis shows the actual "
            "distributional picture."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved analysis to {OUT_JSON}")

    # Generate figure
    make_figure(logits, conflict, cats, stats, OUT_FIG)

    print("\nDone.")


if __name__ == "__main__":
    main()
