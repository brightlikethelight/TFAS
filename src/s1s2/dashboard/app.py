"""Gradio dashboard for interactive exploration of s1s2 analysis results.

Six tabs covering every workstream plus a hypothesis evaluation summary.
Each tab gracefully degrades when data is missing, showing an informative
message instead of crashing. A ``--synthetic`` mode generates plausible
demo data so the dashboard can be demoed without real pipeline outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Force non-interactive backend before any figure creation.
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_KEYS: list[str] = [
    "llama-3.1-8b-instruct",
    "gemma-2-9b-it",
    "r1-distill-llama-8b",
    "r1-distill-qwen-7b",
]

MODEL_LAYERS: dict[str, int] = {
    "llama-3.1-8b-instruct": 32,
    "gemma-2-9b-it": 42,
    "r1-distill-llama-8b": 32,
    "r1-distill-qwen-7b": 28,
}

MODEL_HEADS: dict[str, int] = {
    "llama-3.1-8b-instruct": 32,
    "gemma-2-9b-it": 16,
    "r1-distill-llama-8b": 32,
    "r1-distill-qwen-7b": 28,
}

TARGETS: list[str] = ["task_type", "correctness", "bias_susceptible"]
POSITIONS: list[str] = ["P0", "P2", "T50", "Tend"]
CATEGORIES: list[str] = [
    "crt",
    "base_rate",
    "syllogism",
    "anchoring",
    "framing",
    "conjunction",
    "arithmetic",
]

# Colorblind-safe palette matching src/s1s2/viz/theme.py
MODEL_COLORS: dict[str, str] = {
    "llama-3.1-8b-instruct": "#1f77b4",
    "gemma-2-9b-it": "#17becf",
    "r1-distill-llama-8b": "#ff7f0e",
    "r1-distill-qwen-7b": "#e377c2",
}

COLORS: dict[str, str] = {
    "s1": "#d62728",
    "s2": "#2ca02c",
    "significant": "#e377c2",
    "non_significant": "#c7c7c7",
    "falsified": "#8c564b",
    "baseline": "#7f7f7f",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None if it doesn't exist or is malformed."""
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _find_probe_results(
    results_dir: Path, model: str, target: str, position: str
) -> list[dict[str, Any]]:
    """Find all layer-level probe JSONs for a (model, target, position) combo.

    File naming convention: ``{model}_{target}_layer{NN}_{position}.json``
    """
    pattern = f"{model}_{target}_layer*_{position}.json"
    found = sorted(results_dir.glob(pattern))
    out = []
    for p in found:
        data = _load_json(p)
        if data is not None:
            out.append(data)
    return out


def _find_sae_results(results_dir: Path, model: str) -> list[dict[str, Any]]:
    """Find all SAE differential JSONs for a model."""
    # Try both flat naming and nested
    results: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob(f"{model}*.json")):
        data = _load_json(p)
        if data is not None:
            results.append(data)
    # Also check nested dirs
    nested = results_dir / model
    if nested.is_dir():
        for p in sorted(nested.glob("*.json")):
            data = _load_json(p)
            if data is not None:
                results.append(data)
    return results


def _find_attention_results(results_dir: Path, model: str) -> dict[str, Any] | None:
    """Find attention head classification results for a model."""
    # Try flat naming first
    for p in sorted(results_dir.glob(f"{model}*.json")):
        data = _load_json(p)
        if data is not None:
            return data
    # Nested
    nested = results_dir / model
    if nested.is_dir():
        for p in sorted(nested.glob("*.json")):
            data = _load_json(p)
            if data is not None:
                return data
    return None


def _find_geometry_results(results_dir: Path, model: str) -> list[dict[str, Any]]:
    """Find geometry (silhouette/separability) JSONs for a model."""
    results: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob(f"{model}*.json")):
        data = _load_json(p)
        if data is not None:
            results.append(data)
    nested = results_dir / model
    if nested.is_dir():
        for p in sorted(nested.glob("*.json")):
            data = _load_json(p)
            if data is not None:
                results.append(data)
    return results


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _synth_behavior(model: str) -> dict[str, Any]:
    """Generate plausible behavioral outcomes for demo mode."""
    rng = np.random.default_rng(hash(model) % 2**32)
    is_reasoning = "r1" in model or "distill" in model
    # Reasoning models: higher accuracy, lower lure rate
    base_acc = 0.72 if is_reasoning else 0.55
    base_lure = 0.15 if is_reasoning else 0.35
    out: dict[str, Any] = {"model": model, "categories": {}}
    for cat in CATEGORIES:
        cat_seed = rng.integers(0, 2**16)
        cat_rng = np.random.default_rng(cat_seed)
        jitter = cat_rng.uniform(-0.1, 0.1)
        acc_conflict = np.clip(base_acc + jitter - 0.15, 0.1, 0.95)
        acc_no_conflict = np.clip(base_acc + jitter + 0.10, 0.3, 0.99)
        lure_rate = np.clip(base_lure + cat_rng.uniform(-0.08, 0.08), 0.02, 0.60)
        out["categories"][cat] = {
            "accuracy_conflict": float(acc_conflict),
            "accuracy_no_conflict": float(acc_no_conflict),
            "lure_rate": float(lure_rate),
            "n_conflict": int(cat_rng.integers(20, 40)),
            "n_no_conflict": int(cat_rng.integers(20, 40)),
        }
    return out


def _synth_probe_results(
    model: str, target: str, position: str
) -> list[dict[str, Any]]:
    """Generate layer-wise probe results for demo mode."""
    n_layers = MODEL_LAYERS.get(model, 32)
    rng = np.random.default_rng(hash((model, target, position)) % 2**32)
    is_reasoning = "r1" in model or "distill" in model
    results = []
    for layer in range(n_layers):
        # S-curve: AUC rises in mid layers, drops at the end
        frac = layer / max(n_layers - 1, 1)
        base_auc = 0.5 + 0.35 * np.sin(np.pi * frac)
        if is_reasoning:
            base_auc += 0.05
        auc = float(np.clip(base_auc + rng.normal(0, 0.03), 0.45, 0.98))
        ci_half = float(rng.uniform(0.02, 0.06))
        selectivity = float(np.clip(auc - 0.5 - rng.uniform(0, 0.08), 0.0, 0.5))
        control_auc = float(np.clip(0.5 + rng.normal(0, 0.03), 0.40, 0.60))
        p_perm = float(np.clip(rng.exponential(0.05), 1e-6, 1.0))
        results.append({
            "model": model,
            "target": target,
            "layer": layer,
            "position": position,
            "n_samples": 400,
            "probes": {
                "logreg": {
                    "auc_mean": auc,
                    "auc_std": float(rng.uniform(0.01, 0.04)),
                    "selectivity": selectivity,
                    "control_auc_mean": control_auc,
                    "ci": [auc - ci_half, auc + ci_half],
                    "p_perm": p_perm,
                    "n_folds": 5,
                },
                "mass_mean": {
                    "auc_mean": float(np.clip(auc - 0.05 + rng.normal(0, 0.02), 0.4, 0.95)),
                    "auc_std": float(rng.uniform(0.02, 0.05)),
                    "selectivity": float(np.clip(selectivity - 0.02, 0.0, 0.5)),
                    "control_auc_mean": float(np.clip(0.5 + rng.normal(0, 0.03), 0.40, 0.60)),
                    "ci": [auc - ci_half - 0.03, auc + ci_half - 0.02],
                    "p_perm": float(np.clip(rng.exponential(0.08), 1e-6, 1.0)),
                    "n_folds": 5,
                },
            },
        })
    return results


def _synth_sae_results(model: str, layer: int) -> dict[str, Any]:
    """Generate SAE differential results for demo mode."""
    rng = np.random.default_rng(hash((model, layer)) % 2**32)
    n_features = 500
    log2fc = rng.normal(0, 1.2, size=n_features).astype(float)
    neg_log10_p = np.abs(log2fc) * rng.uniform(0.5, 3.0, size=n_features) + rng.exponential(0.3, size=n_features)
    significant = neg_log10_p > -np.log10(0.05)
    # Falsification: ~20% of significant features are spurious
    n_sig = int(significant.sum())
    falsified = np.zeros(n_features, dtype=bool)
    if n_sig > 0:
        falsified_idx = rng.choice(np.where(significant)[0], size=max(1, n_sig // 5), replace=False)
        falsified[falsified_idx] = True

    features = []
    for i in range(n_features):
        features.append({
            "feature_id": int(i),
            "log2fc": float(log2fc[i]),
            "neg_log10_p": float(neg_log10_p[i]),
            "significant": bool(significant[i]),
            "falsified": bool(falsified[i]),
        })
    return {
        "model": model,
        "layer": layer,
        "n_features_total": n_features,
        "n_significant": int(significant.sum()),
        "n_significant_after_falsification": int((significant & ~falsified).sum()),
        "n_falsified": int(falsified.sum()),
        "features": features,
    }


def _synth_attention_results(model: str) -> dict[str, Any]:
    """Generate attention head classification results for demo mode."""
    n_layers = MODEL_LAYERS.get(model, 32)
    n_heads = MODEL_HEADS.get(model, 32)
    rng = np.random.default_rng(hash(("attn", model)) % 2**32)
    # Classification: most heads are non-specialized
    labels = ["non-specialized", "S2-specialized", "S1-specialized", "mixed"]
    probs = [0.65, 0.15, 0.12, 0.08]
    classifications = []
    entropy_profile = []
    for layer_idx in range(n_layers):
        layer_entropy = float(
            2.0 + rng.normal(0, 0.3) - 0.3 * np.sin(np.pi * layer_idx / n_layers)
        )
        entropy_profile.append({"layer": layer_idx, "mean_entropy": layer_entropy})
        for head_idx in range(n_heads):
            label = rng.choice(labels, p=probs)
            classifications.append({
                "layer": layer_idx,
                "head": head_idx,
                "classification": str(label),
                "entropy_s1_mean": float(rng.uniform(1.5, 3.0)),
                "entropy_s2_mean": float(rng.uniform(1.5, 3.0)),
                "effect_size": float(rng.normal(0, 0.4)),
                "p_value": float(np.clip(rng.exponential(0.2), 1e-8, 1.0)),
            })
    return {
        "model": model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_classifications": classifications,
        "layer_entropy_profile": entropy_profile,
    }


def _synth_geometry_results(model: str) -> dict[str, Any]:
    """Generate geometry (silhouette + PCA) results for demo mode."""
    n_layers = MODEL_LAYERS.get(model, 32)
    rng = np.random.default_rng(hash(("geom", model)) % 2**32)
    is_reasoning = "r1" in model or "distill" in model
    layers = []
    for layer_idx in range(n_layers):
        frac = layer_idx / max(n_layers - 1, 1)
        # Mid-layer peak in silhouette
        base_sil = 0.02 + 0.12 * np.sin(np.pi * frac)
        if is_reasoning:
            base_sil += 0.03
        sil = float(np.clip(base_sil + rng.normal(0, 0.02), -0.05, 0.30))
        ci_half = float(rng.uniform(0.01, 0.04))
        p_perm = float(np.clip(rng.exponential(0.1), 1e-6, 1.0))
        layers.append({
            "layer": layer_idx,
            "silhouette": sil,
            "silhouette_ci": [sil - ci_half, sil + ci_half],
            "p_perm": p_perm,
        })
    # Synthetic PCA projections (2D scatter)
    n_points = 200
    pca_x = rng.normal(0, 1, size=n_points).tolist()
    pca_y = rng.normal(0, 1, size=n_points).tolist()
    labels = rng.choice(["conflict", "no-conflict"], size=n_points).tolist()
    # Shift conflict points slightly for visual separability
    for i in range(n_points):
        if labels[i] == "conflict":
            pca_x[i] += 0.5
    return {
        "model": model,
        "layer_silhouettes": layers,
        "pca_projection": {
            "layer": n_layers // 2,
            "pc1": pca_x,
            "pc2": pca_y,
            "labels": labels,
        },
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _apply_theme(fig: plt.Figure) -> None:
    """Apply a minimal version of the paper theme to a figure."""
    for ax in fig.get_axes():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)


def _no_data_fig(msg: str) -> plt.Figure:
    """Return a figure with a centered 'no data' message."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0.5, 0.5, msg,
        ha="center", va="center", fontsize=13, color="#666",
        transform=ax.transAxes,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    return fig


# -- Tab 1: Behavioral Overview --

def plot_behavioral(
    model: str,
    results_dir: Path | None,
    synthetic: bool,
) -> plt.Figure:
    """Per-category accuracy bars (conflict vs no-conflict) and lure rate."""
    if synthetic:
        data = _synth_behavior(model)
    else:
        if results_dir is None:
            return _no_data_fig("No results directory configured.")
        # Try loading from HDF5 behavioral data or a summary JSON
        summary_path = results_dir / "behavior" / f"{model}_summary.json"
        data = _load_json(summary_path)  # type: ignore[assignment]
        if data is None:
            return _no_data_fig(
                f"No behavioral results found for {model}.\n"
                "Run the extraction pipeline first, or use --synthetic mode."
            )

    cats = list(data.get("categories", {}).keys())
    if not cats:
        return _no_data_fig(f"No category data for {model}.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(cats))
    width = 0.35

    acc_conflict = [data["categories"][c]["accuracy_conflict"] for c in cats]
    acc_no_conflict = [data["categories"][c]["accuracy_no_conflict"] for c in cats]
    lure_rates = [data["categories"][c]["lure_rate"] for c in cats]

    ax1.bar(x - width / 2, acc_conflict, width, label="Conflict (S1 lure)", color=COLORS["s1"])
    ax1.bar(x + width / 2, acc_no_conflict, width, label="No conflict", color=COLORS["s2"])
    ax1.set_xlabel("Category")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Accuracy by Category - {model}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=45, ha="right")
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    ax2.bar(x, lure_rates, color=COLORS["s1"], alpha=0.8)
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Lure Rate")
    ax2.set_title(f"S1 Lure Rate - {model}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, rotation=45, ha="right")
    ax2.set_ylim(0, max(lure_rates) * 1.3 if lure_rates else 1.0)

    _apply_theme(fig)
    fig.tight_layout()
    return fig


# -- Tab 2: Probing Results --

def plot_probes(
    model: str,
    target: str,
    position: str,
    results_dir: Path | None,
    synthetic: bool,
) -> plt.Figure:
    """Layer-wise ROC-AUC curve with CI bands and selectivity overlay."""
    if synthetic:
        layer_results = _synth_probe_results(model, target, position)
    else:
        if results_dir is None:
            return _no_data_fig("No results directory configured.")
        probe_dir = results_dir / "probes"
        layer_results = _find_probe_results(probe_dir, model, target, position)
        if not layer_results:
            return _no_data_fig(
                f"No probe results for {model} / {target} / {position}.\n"
                "Run: python scripts/run_probes.py"
            )

    # Sort by layer
    layer_results.sort(key=lambda d: d.get("layer", 0))
    layers = [d["layer"] for d in layer_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Extract logreg results (primary probe)
    probe_key = "logreg"
    aucs, ci_lo, ci_hi, selectivities, control_aucs = [], [], [], [], []
    for d in layer_results:
        probe_data = (d.get("probes") or {}).get(probe_key)
        if probe_data is None:
            aucs.append(float("nan"))
            ci_lo.append(float("nan"))
            ci_hi.append(float("nan"))
            selectivities.append(float("nan"))
            control_aucs.append(float("nan"))
            continue
        aucs.append(probe_data.get("auc_mean", float("nan")))
        ci = probe_data.get("ci", [float("nan"), float("nan")])
        ci_lo.append(ci[0] if ci else float("nan"))
        ci_hi.append(ci[1] if ci else float("nan"))
        selectivities.append(probe_data.get("selectivity", float("nan")))
        control_aucs.append(probe_data.get("control_auc_mean", float("nan")))

    color = MODEL_COLORS.get(model, "#333")

    # AUC curve with CI band
    ax1.plot(layers, aucs, "o-", color=color, label=f"{probe_key} AUC", markersize=3)
    ax1.fill_between(layers, ci_lo, ci_hi, alpha=0.2, color=color)
    ax1.plot(layers, control_aucs, "--", color=COLORS["baseline"], label="Control (random labels)", linewidth=1)
    ax1.axhline(0.5, color="#aaa", linestyle=":", linewidth=0.8)
    ax1.set_ylabel("ROC-AUC")
    ax1.set_title(f"Probe: {model} / {target} / {position}")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(0.35, 1.02)

    # Selectivity (Hewitt-Liang)
    ax2.bar(layers, selectivities, color=color, alpha=0.7, width=0.8)
    ax2.axhline(0.05, color="red", linestyle="--", linewidth=1, label="5pp threshold")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Selectivity (AUC - Control AUC)")
    ax2.set_title("Hewitt-Liang Selectivity")
    ax2.legend(loc="upper right", fontsize=8)

    _apply_theme(fig)
    fig.tight_layout()
    return fig


# -- Tab 3: SAE Features --

def plot_sae(
    model: str,
    layer: int,
    results_dir: Path | None,
    synthetic: bool,
) -> tuple[plt.Figure, str]:
    """Volcano plot + feature count summary."""
    if synthetic:
        data = _synth_sae_results(model, layer)
    else:
        if results_dir is None:
            return _no_data_fig("No results directory configured."), "No data."
        sae_dir = results_dir / "sae"
        candidates = _find_sae_results(sae_dir, model)
        data = None
        for c in candidates:
            if c.get("layer") == layer:
                data = c
                break
        if data is None:
            return (
                _no_data_fig(
                    f"No SAE results for {model} layer {layer}.\n"
                    "Run: python scripts/run_sae.py"
                ),
                "No data.",
            )

    features = data.get("features", [])
    if not features:
        return _no_data_fig("No feature data found."), "No features."

    log2fc = np.array([f["log2fc"] for f in features])
    neg_log10_p = np.array([f["neg_log10_p"] for f in features])
    significant = np.array([f.get("significant", False) for f in features])
    falsified = np.array([f.get("falsified", False) for f in features])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Non-significant
    mask_ns = ~significant
    ax.scatter(
        log2fc[mask_ns], neg_log10_p[mask_ns],
        c=COLORS["non_significant"], s=8, alpha=0.5, label="Not significant",
    )
    # Significant, not falsified
    mask_sig = significant & ~falsified
    ax.scatter(
        log2fc[mask_sig], neg_log10_p[mask_sig],
        c=COLORS["significant"], s=15, alpha=0.8, label="Significant (survived falsification)",
    )
    # Falsified
    mask_f = falsified
    ax.scatter(
        log2fc[mask_f], neg_log10_p[mask_f],
        c=COLORS["falsified"], s=15, alpha=0.8, marker="x", label="Falsified (spurious)",
    )

    ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=0.8, label="p=0.05")
    ax.set_xlabel("log2 fold-change (S1 vs S2)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(f"SAE Feature Volcano: {model} Layer {layer}")
    ax.legend(loc="upper right", fontsize=8)

    _apply_theme(fig)
    fig.tight_layout()

    n_total = data.get("n_features_total", len(features))
    n_sig = data.get("n_significant", int(significant.sum()))
    n_after = data.get("n_significant_after_falsification", int((significant & ~falsified).sum()))
    n_falsified = data.get("n_falsified", int(falsified.sum()))
    summary = (
        f"**Features:** {n_total} total | "
        f"{n_sig} significant (BH-FDR q<0.05) | "
        f"{n_falsified} falsified (Ma et al.) | "
        f"{n_after} surviving"
    )
    return fig, summary


# -- Tab 4: Attention Entropy --

def plot_attention(
    model: str,
    results_dir: Path | None,
    synthetic: bool,
) -> tuple[plt.Figure, plt.Figure]:
    """Head classification heatmap + layer-level entropy profile."""
    if synthetic:
        data = _synth_attention_results(model)
    else:
        if results_dir is None:
            return _no_data_fig("No results directory configured."), _no_data_fig("")
        attn_dir = results_dir / "attention"
        data = _find_attention_results(attn_dir, model)
        if data is None:
            msg = (
                f"No attention results for {model}.\n"
                "Run: python scripts/run_attention.py"
            )
            return _no_data_fig(msg), _no_data_fig("")

    n_layers = data.get("n_layers", MODEL_LAYERS.get(model, 32))
    n_heads = data.get("n_heads", MODEL_HEADS.get(model, 32))
    classifications = data.get("head_classifications", [])
    entropy_profile = data.get("layer_entropy_profile", [])

    # -- Heatmap --
    class_map = {"non-specialized": 0, "S2-specialized": 1, "S1-specialized": 2, "mixed": 3}
    class_colors = ["#c7c7c7", "#2ca02c", "#d62728", "#ff7f0e"]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(class_colors)

    grid = np.zeros((n_layers, n_heads), dtype=int)
    for entry in classifications:
        lay = entry.get("layer", 0)
        hd = entry.get("head", 0)
        if 0 <= lay < n_layers and 0 <= hd < n_heads:
            grid[lay, hd] = class_map.get(entry.get("classification", "non-specialized"), 0)

    fig1, ax1 = plt.subplots(figsize=(max(8, n_heads * 0.35), max(6, n_layers * 0.25)))
    ax1.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_title(f"Head Classification: {model}")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=class_colors[i], label=label)
        for label, i in class_map.items()
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=2)
    fig1.tight_layout()

    # -- Entropy profile --
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    if entropy_profile:
        ep_layers = [e["layer"] for e in entropy_profile]
        ep_entropy = [e["mean_entropy"] for e in entropy_profile]
        color = MODEL_COLORS.get(model, "#333")
        ax2.plot(ep_layers, ep_entropy, "o-", color=color, markersize=3)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Mean Normalized Entropy")
        ax2.set_title(f"Layer Entropy Profile: {model}")
    else:
        ax2.text(0.5, 0.5, "No entropy profile data.", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="#666")
        ax2.axis("off")

    _apply_theme(fig2)
    fig2.tight_layout()
    return fig1, fig2


# -- Tab 5: Geometry --

def plot_geometry(
    model: str,
    results_dir: Path | None,
    synthetic: bool,
) -> tuple[plt.Figure, plt.Figure]:
    """Silhouette curve + PCA scatter."""
    if synthetic:
        data = _synth_geometry_results(model)
    else:
        if results_dir is None:
            return _no_data_fig("No results directory configured."), _no_data_fig("")
        geom_dir = results_dir / "geometry"
        results = _find_geometry_results(geom_dir, model)
        # Try to merge results into a single dict
        data: dict[str, Any] = {}  # type: ignore[no-redef]
        for r in results:
            data.update(r)
        if not data:
            msg = (
                f"No geometry results for {model}.\n"
                "Run: python scripts/run_geometry.py"
            )
            return _no_data_fig(msg), _no_data_fig("")

    layer_sils = data.get("layer_silhouettes", [])
    pca_data = data.get("pca_projection", {})

    # -- Silhouette curve --
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    if layer_sils:
        layers = [s["layer"] for s in layer_sils]
        sils = [s["silhouette"] for s in layer_sils]
        ci_lo = [s.get("silhouette_ci", [s["silhouette"], s["silhouette"]])[0] for s in layer_sils]
        ci_hi = [s.get("silhouette_ci", [s["silhouette"], s["silhouette"]])[1] for s in layer_sils]
        color = MODEL_COLORS.get(model, "#333")
        ax1.plot(layers, sils, "o-", color=color, markersize=3)
        ax1.fill_between(layers, ci_lo, ci_hi, alpha=0.2, color=color)
        ax1.axhline(0.0, color="#aaa", linestyle=":", linewidth=0.8)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Cosine Silhouette Score")
        ax1.set_title(f"Layer-wise Silhouette: {model}")
    else:
        ax1.text(0.5, 0.5, "No silhouette data.", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=12, color="#666")
        ax1.axis("off")
    _apply_theme(fig1)
    fig1.tight_layout()

    # -- PCA scatter --
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    if pca_data and "pc1" in pca_data and "pc2" in pca_data:
        pc1 = np.array(pca_data["pc1"])
        pc2 = np.array(pca_data["pc2"])
        pca_labels = pca_data.get("labels", ["unknown"] * len(pc1))
        pca_layer = pca_data.get("layer", "?")
        for label, color_key in [("conflict", "s1"), ("no-conflict", "s2")]:
            mask = np.array([lbl == label for lbl in pca_labels])
            if mask.any():
                ax2.scatter(pc1[mask], pc2[mask], c=COLORS[color_key], s=15,
                            alpha=0.6, label=label)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title(f"PCA Projection (Layer {pca_layer}): {model}")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No PCA projection data.", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="#666")
        ax2.axis("off")
    _apply_theme(fig2)
    fig2.tight_layout()
    return fig1, fig2


# -- Tab 6: Hypothesis Evaluation --

def build_hypothesis_table(
    results_dir: Path | None,
    synthetic: bool,
) -> str:
    """Summary table of H1-H6 verdicts as a Markdown string."""
    hypotheses = [
        ("H1", "Behavioral dual-process signature",
         "Conflict items elicit lower accuracy and higher lure rates than controls across all models"),
        ("H2", "Linear decodability of processing mode",
         "Logistic probes can decode conflict vs no-conflict from residual stream activations "
         "with AUC > 0.65 and selectivity > 5pp"),
        ("H3", "SAE features differentiate S1/S2",
         "Significant SAE features survive Ma et al. falsification and are not token-level artifacts"),
        ("H4", "Attention entropy tracks deliberation",
         "S2-specialized heads exist with significantly higher entropy on conflict trials"),
        ("H5", "Geometric separability",
         "Conflict vs no-conflict representations are geometrically separable (silhouette > 0) "
         "in mid-to-late layers"),
        ("H6", "Reasoning training amplifies the gradient",
         "R1-Distill models show stronger signatures across all workstreams compared to "
         "their non-reasoning counterparts"),
    ]

    if synthetic:
        verdicts = []
        demo_verdicts = ["PASS", "PASS", "INCONCLUSIVE", "PASS", "PASS", "INCONCLUSIVE"]
        for (hid, name, desc), verdict in zip(hypotheses, demo_verdicts, strict=False):
            verdicts.append((hid, name, desc, verdict))
    else:
        # In real mode, we'd need to aggregate across workstreams
        verdicts = []
        for hid, name, desc in hypotheses:
            verdicts.append((hid, name, desc, "NOT EVALUATED"))

    # Build markdown table
    lines = ["| Hypothesis | Description | Verdict |", "|---|---|---|"]
    for hid, name, desc, verdict in verdicts:
        if verdict == "PASS":
            badge = "PASS"
        elif verdict == "FAIL":
            badge = "FAIL"
        elif verdict == "INCONCLUSIVE":
            badge = "INCONCLUSIVE"
        else:
            badge = "NOT EVALUATED"
        lines.append(f"| **{hid}**: {name} | {desc} | {badge} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    results_dir: str | Path | None = None,
    activations_path: str | Path | None = None,
    synthetic: bool = False,
    title: str = "s1s2 Analysis Dashboard",
) -> gr.Blocks:
    """Build and return the Gradio Blocks app (does not launch it).

    Parameters
    ----------
    results_dir:
        Path to the ``results/`` directory. If None, defaults to ``results/``
        relative to the repo root.
    activations_path:
        Path to the HDF5 activation cache. Currently unused by the dashboard
        (all data is loaded from workstream result JSONs), but reserved for
        future direct-from-HDF5 tabs.
    synthetic:
        If True, generate synthetic demo data instead of loading from disk.
    title:
        Dashboard title displayed at the top of the page.
    """
    if results_dir is not None:
        results_dir = Path(results_dir)
    elif not synthetic:
        # Default: look for results/ relative to the s1s2 repo root
        _repo = Path(__file__).resolve().parent.parent.parent.parent
        candidate = _repo / "results"
        if candidate.is_dir():
            results_dir = candidate

    # -----------------------------------------------------------------------
    # Build the Gradio Blocks layout
    # -----------------------------------------------------------------------

    with gr.Blocks(title=title) as app:
        gr.Markdown(f"# {title}")
        mode_label = "synthetic demo data" if synthetic else "real results"
        gr.Markdown(f"*Mode: {mode_label}*")

        # -- Tab 1: Behavioral Overview --
        with gr.Tab("Behavioral Overview"):
            gr.Markdown("### Per-category accuracy and S1 lure rates")
            model_beh = gr.Dropdown(choices=MODEL_KEYS, value=MODEL_KEYS[0], label="Model")
            plot_beh = gr.Plot(label="Behavioral Results")

            def _update_behavioral(model: str) -> plt.Figure:
                fig = plot_behavioral(model, results_dir, synthetic)
                return fig

            model_beh.change(_update_behavioral, inputs=[model_beh], outputs=[plot_beh])
            app.load(_update_behavioral, inputs=[model_beh], outputs=[plot_beh])

        # -- Tab 2: Probing Results --
        with gr.Tab("Probing Results"):
            gr.Markdown("### Layer-wise ROC-AUC with CI bands and Hewitt-Liang selectivity")
            with gr.Row():
                model_probe = gr.Dropdown(choices=MODEL_KEYS, value=MODEL_KEYS[0], label="Model")
                target_probe = gr.Dropdown(choices=TARGETS, value=TARGETS[0], label="Target")
                pos_probe = gr.Dropdown(choices=POSITIONS, value=POSITIONS[0], label="Position")
            plot_probe = gr.Plot(label="Probe Results")

            def _update_probes(model: str, target: str, position: str) -> plt.Figure:
                fig = plot_probes(model, target, position, results_dir, synthetic)
                return fig

            for inp in [model_probe, target_probe, pos_probe]:
                inp.change(
                    _update_probes,
                    inputs=[model_probe, target_probe, pos_probe],
                    outputs=[plot_probe],
                )
            app.load(
                _update_probes,
                inputs=[model_probe, target_probe, pos_probe],
                outputs=[plot_probe],
            )

        # -- Tab 3: SAE Features --
        with gr.Tab("SAE Features"):
            gr.Markdown("### Volcano plot with Ma et al. falsification overlay")
            with gr.Row():
                model_sae = gr.Dropdown(choices=MODEL_KEYS, value=MODEL_KEYS[0], label="Model")
                layer_sae = gr.Slider(
                    minimum=0, maximum=41, step=1, value=16, label="Layer"
                )
            plot_sae_out = gr.Plot(label="Volcano Plot")
            summary_sae = gr.Markdown(label="Feature Summary")

            def _update_sae(model: str, layer: int) -> tuple[plt.Figure, str]:
                fig, summary = plot_sae(model, int(layer), results_dir, synthetic)
                return fig, summary

            for inp in [model_sae, layer_sae]:
                inp.change(
                    _update_sae,
                    inputs=[model_sae, layer_sae],
                    outputs=[plot_sae_out, summary_sae],
                )
            app.load(
                _update_sae,
                inputs=[model_sae, layer_sae],
                outputs=[plot_sae_out, summary_sae],
            )

        # -- Tab 4: Attention Entropy --
        with gr.Tab("Attention Entropy"):
            gr.Markdown("### Head classification heatmap and layer entropy profile")
            model_attn = gr.Dropdown(choices=MODEL_KEYS, value=MODEL_KEYS[0], label="Model")
            plot_attn_heatmap = gr.Plot(label="Head Classifications")
            plot_attn_profile = gr.Plot(label="Entropy Profile")

            def _update_attention(model: str) -> tuple[plt.Figure, plt.Figure]:
                fig_hm, fig_ep = plot_attention(model, results_dir, synthetic)
                return fig_hm, fig_ep

            model_attn.change(
                _update_attention,
                inputs=[model_attn],
                outputs=[plot_attn_heatmap, plot_attn_profile],
            )
            app.load(
                _update_attention,
                inputs=[model_attn],
                outputs=[plot_attn_heatmap, plot_attn_profile],
            )

        # -- Tab 5: Geometry --
        with gr.Tab("Geometry"):
            gr.Markdown("### Silhouette curves and PCA projections")
            model_geom = gr.Dropdown(choices=MODEL_KEYS, value=MODEL_KEYS[0], label="Model")
            plot_geom_sil = gr.Plot(label="Silhouette Score")
            plot_geom_pca = gr.Plot(label="PCA Projection")

            def _update_geometry(model: str) -> tuple[plt.Figure, plt.Figure]:
                fig_sil, fig_pca = plot_geometry(model, results_dir, synthetic)
                return fig_sil, fig_pca

            model_geom.change(
                _update_geometry,
                inputs=[model_geom],
                outputs=[plot_geom_sil, plot_geom_pca],
            )
            app.load(
                _update_geometry,
                inputs=[model_geom],
                outputs=[plot_geom_sil, plot_geom_pca],
            )

        # -- Tab 6: Hypothesis Evaluation --
        with gr.Tab("Hypothesis Evaluation"):
            gr.Markdown("### Summary of hypothesis verdicts (H1-H6)")
            gr.Markdown(
                value=build_hypothesis_table(results_dir, synthetic)
            )

    return app
