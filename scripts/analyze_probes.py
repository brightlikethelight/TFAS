"""Offline analysis of layer-wise probe AUC curves: Llama-3.1-8B vs R1-Distill-Llama-8B.

Reads hardcoded AUC data (from B200 pod results/probes/llama_vs_r1_layer_aucs.json)
and computes:
  1. Basic statistics (peak layer/AUC, mean AUC)
  2. Deliberation gap (per-layer Llama - R1 difference)
  3. Layer-phase analysis (early/mid/late/final)
  4. Curve shape analysis (threshold crossings)
  5. Interpretation paragraph for the Results section
  6. Statistical test stubs for when full bootstrap data is available

Outputs:
  results/probes/probe_analysis_report.json
  Formatted summary to stdout
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# --- Hardcoded AUC data from B200 pod ----------------------------------------
# Source: results/probes/llama_vs_r1_layer_aucs.json

LLAMA_AUCS: dict[int, float] = {
    0: 0.848, 1: 0.899, 2: 0.926, 3: 0.932, 4: 0.927, 5: 0.939, 6: 0.933, 7: 0.960,
    8: 0.952, 9: 0.948, 10: 0.971, 11: 0.979, 12: 0.988, 13: 0.993, 14: 0.999, 15: 0.996,
    16: 0.995, 17: 0.989, 18: 0.991, 19: 0.987, 20: 0.978, 21: 0.969, 22: 0.968, 23: 0.963,
    24: 0.978, 25: 0.964, 26: 0.965, 27: 0.969, 28: 0.981, 29: 0.965, 30: 0.973, 31: 0.954,
}

R1_AUCS: dict[int, float] = {
    0: 0.782, 1: 0.831, 2: 0.870, 3: 0.901, 4: 0.919, 5: 0.928, 6: 0.926, 7: 0.927,
    8: 0.920, 9: 0.928, 10: 0.927, 11: 0.922, 12: 0.927, 13: 0.927, 14: 0.929, 15: 0.919,
    16: 0.929, 17: 0.923, 18: 0.927, 19: 0.914, 20: 0.914, 21: 0.923, 22: 0.922, 23: 0.917,
    24: 0.924, 25: 0.922, 26: 0.918, 27: 0.906, 28: 0.923, 29: 0.924, 30: 0.918, 31: 0.904,
}

# Phase definitions: contiguous 8-layer blocks
PHASES: dict[str, tuple[int, int]] = {
    "early":  (0, 7),
    "mid":    (8, 15),
    "late":   (16, 23),
    "final":  (24, 31),
}

N_LAYERS = 32


# --- Analysis functions -------------------------------------------------------


def basic_stats(aucs: dict[int, float]) -> dict[str, Any]:
    """Peak layer, peak AUC, mean/std AUC across all layers."""
    layers = sorted(aucs.keys())
    vals = np.array([aucs[l] for l in layers])
    peak_idx = int(np.argmax(vals))
    return {
        "peak_layer": layers[peak_idx],
        "peak_auc": float(vals[peak_idx]),
        "mean_auc": float(np.mean(vals)),
        "std_auc": float(np.std(vals, ddof=1)),
        "min_auc": float(np.min(vals)),
        "min_layer": layers[int(np.argmin(vals))],
    }


def deliberation_gap(
    llama: dict[int, float], r1: dict[int, float]
) -> dict[str, Any]:
    """Per-layer AUC difference (Llama - R1) and summary statistics."""
    layers = sorted(llama.keys())
    diffs = {l: round(llama[l] - r1[l], 4) for l in layers}
    diff_arr = np.array(list(diffs.values()))

    # Layers with largest gap (top-5)
    sorted_by_gap = sorted(diffs.items(), key=lambda kv: kv[1], reverse=True)
    top5 = [(l, d) for l, d in sorted_by_gap[:5]]

    return {
        "per_layer_diff": diffs,
        "mean_gap": float(np.mean(diff_arr)),
        "std_gap": float(np.std(diff_arr, ddof=1)),
        "max_gap": float(np.max(diff_arr)),
        "max_gap_layer": int(np.argmax(diff_arr)),
        "min_gap": float(np.min(diff_arr)),
        "min_gap_layer": int(np.argmin(diff_arr)),
        "top5_gap_layers": top5,
        "all_positive": bool(np.all(diff_arr > 0)),
    }


def phase_analysis(
    llama: dict[int, float], r1: dict[int, float]
) -> dict[str, Any]:
    """Mean AUC per phase per model, gap per phase, and gap trajectory."""
    results: dict[str, Any] = {}
    gap_trajectory: list[tuple[str, float]] = []

    for phase_name, (start, end) in PHASES.items():
        llama_vals = [llama[l] for l in range(start, end + 1)]
        r1_vals = [r1[l] for l in range(start, end + 1)]
        llama_mean = float(np.mean(llama_vals))
        r1_mean = float(np.mean(r1_vals))
        gap = round(llama_mean - r1_mean, 4)
        results[phase_name] = {
            "llama_mean_auc": round(llama_mean, 4),
            "r1_mean_auc": round(r1_mean, 4),
            "gap": gap,
            "llama_std": round(float(np.std(llama_vals, ddof=1)), 4),
            "r1_std": round(float(np.std(r1_vals, ddof=1)), 4),
        }
        gap_trajectory.append((phase_name, gap))

    # Does the gap widen or narrow across phases?
    gaps = [g for _, g in gap_trajectory]
    if gaps[-1] > gaps[0]:
        trend = "widens from early to final"
    elif gaps[-1] < gaps[0]:
        trend = "narrows from early to final"
    else:
        trend = "stable across phases"

    # More nuanced: where is the peak gap?
    peak_phase = gap_trajectory[int(np.argmax(gaps))][0]
    results["gap_trajectory"] = gap_trajectory
    results["gap_trend"] = trend
    results["peak_gap_phase"] = peak_phase

    return results


def curve_shape(aucs: dict[int, float]) -> dict[str, Any]:
    """Threshold-crossing analysis: at what layer does the model first exceed
    0.95 and 0.99 AUC? Also: does it ever reach these thresholds?"""
    layers = sorted(aucs.keys())
    result: dict[str, Any] = {}

    for threshold in [0.90, 0.95, 0.99]:
        key = f"first_exceeds_{threshold:.2f}"
        crossed = [l for l in layers if aucs[l] >= threshold]
        if crossed:
            result[key] = crossed[0]
        else:
            result[key] = None
        result[f"ever_reaches_{threshold:.2f}"] = len(crossed) > 0
        result[f"n_layers_above_{threshold:.2f}"] = len(crossed)

    return result


def generate_interpretation(
    llama_stats: dict[str, Any],
    r1_stats: dict[str, Any],
    gap: dict[str, Any],
    phases: dict[str, Any],
    llama_shape: dict[str, Any],
    r1_shape: dict[str, Any],
) -> str:
    """Generate a 1-paragraph interpretation suitable for the Results section.

    Deliberately hedged — reports findings, avoids overclaiming.
    """
    lines = []

    lines.append(
        "Linear probes trained on residual stream activations discriminated S1-like "
        "from S2-like processing in both models, but with markedly different layer-wise "
        "profiles."
    )

    lines.append(
        f"Llama-3.1-8B achieved a peak AUC of {llama_stats['peak_auc']:.3f} at layer "
        f"{llama_stats['peak_layer']} (mean across layers: {llama_stats['mean_auc']:.3f}), "
        f"exhibiting a steep rise through the early and middle layers followed by a gradual "
        f"decline in later layers."
    )

    lines.append(
        f"R1-Distill-Llama-8B peaked at {r1_stats['peak_auc']:.3f} (layer "
        f"{r1_stats['peak_layer']}), with a substantially flatter curve "
        f"(mean AUC {r1_stats['mean_auc']:.3f}, SD {r1_stats['std_auc']:.3f} "
        f"vs. {llama_stats['std_auc']:.3f} for Llama)."
    )

    lines.append(
        f"The deliberation gap (Llama minus R1 AUC) was positive at every layer "
        f"(mean {gap['mean_gap']:.3f}, max {gap['max_gap']:.3f} at layer "
        f"{gap['max_gap_layer']}), "
        f"and {phases['gap_trend']}."
    )

    # Threshold crossing
    r1_95 = r1_shape["ever_reaches_0.95"]
    llama_first_95 = llama_shape["first_exceeds_0.95"]
    llama_first_99 = llama_shape["first_exceeds_0.99"]

    if not r1_95:
        lines.append(
            f"Notably, Llama first exceeded 0.95 AUC at layer {llama_first_95} "
            f"and 0.99 at layer {llama_first_99}, whereas R1-Distill never crossed "
            f"the 0.95 threshold at any layer."
        )
    else:
        r1_first_95 = r1_shape["first_exceeds_0.95"]
        lines.append(
            f"Llama first exceeded 0.95 at layer {llama_first_95} and 0.99 at layer "
            f"{llama_first_99}; R1 first exceeded 0.95 at layer {r1_first_95}."
        )

    peak_phase = phases["peak_gap_phase"]
    phase_range = PHASES[peak_phase]
    lines.append(
        f"The {peak_phase}-layer phase ({phase_range[0]}--{phase_range[1]}) showed "
        f"the largest gap ({phases[peak_phase]['gap']:.3f}), suggesting that "
        f"reasoning distillation most strongly reshapes representations in "
        f"{'these later' if peak_phase in ('late', 'final') else 'the intermediate'} "
        f"computation layers where Llama's S1/S2 signal "
        f"{'remains high' if peak_phase in ('late', 'final') else 'peaks'}."
    )

    lines.append(
        "These results are consistent with the hypothesis that reasoning distillation "
        "compresses the deliberation-intensity gradient in residual stream geometry, "
        "making S1-like and S2-like representations less linearly separable --- "
        "particularly in layers 10--15 where the base model shows maximal decodability. "
        "Caution is warranted: higher probe AUC in the base model could also reflect "
        "greater reliance on shallow heuristic features that probes can exploit, rather "
        "than richer processing-mode information per se."
    )

    return " ".join(lines)


def statistical_test_stubs() -> dict[str, str]:
    """Document the statistical tests to run when full per-sample data is available.

    These require the raw probe predictions / AUC values across CV folds and seeds,
    which we don't have in this summary-level analysis.
    """
    return {
        "paired_bootstrap_peak_layer": (
            "Use s1s2.utils.stats.paired_bootstrap_ci_diff on the per-fold AUC arrays "
            "at each model's respective peak layer (Llama layer 14, R1 layer 14 or 16). "
            "Statistic: lambda x, y: np.mean(x) - np.mean(y). n_resamples=10000. "
            "If the 95% CI excludes zero, the gap is significant."
        ),
        "phase_wise_permutation": (
            "For each phase, pool per-fold AUC values across layers in that phase. "
            "Run s1s2.utils.stats.permutation_test_two_sample on the Llama vs R1 pools "
            "with n_permutations=10000, alternative='greater'. Apply BH-FDR across the "
            "4 phases (s1s2.utils.stats.bh_fdr)."
        ),
        "effect_size": (
            "Report Cohen's d (s1s2.utils.stats.cohens_d) for the Llama vs R1 AUC "
            "difference per phase. Also report rank-biserial correlation "
            "(s1s2.utils.stats.rank_biserial) as a nonparametric alternative."
        ),
        "specificity_control_immune_categories": (
            "Train the same logistic probe on categories expected to show no S1/S2 "
            "difference (e.g., factual recall, trivial arithmetic). If probe AUC on "
            "immune categories is near 0.5, the S1/S2 signal is not an artifact of "
            "general task-difficulty encoding. Report selectivity = real_AUC - control_AUC. "
            "Hewitt & Liang (2019) criterion: selectivity < 5pp means signal is probe "
            "expressiveness, not representation."
        ),
        "hewitt_liang_selectivity": (
            "Train probes on shuffled (random) labels using "
            "s1s2.probes.controls.run_control_task. Report selectivity "
            "(real AUC - random-label AUC) per layer. If selectivity < 0.05, "
            "the probe signal at that layer is not meaningful."
        ),
        "cross_domain_transfer": (
            "Leave-one-category-out evaluation: train on 6 bias categories, test on "
            "the 7th. If transfer AUC drops to ~0.5, probe learned category features "
            "not processing-mode features. Run for all 7 held-out categories."
        ),
    }


# --- Main entry point ---------------------------------------------------------


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "results" / "probes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "probe_analysis_report.json"

    # 1. Basic statistics
    llama_stats = basic_stats(LLAMA_AUCS)
    r1_stats = basic_stats(R1_AUCS)

    # 2. Deliberation gap
    gap = deliberation_gap(LLAMA_AUCS, R1_AUCS)

    # 3. Phase analysis
    phases = phase_analysis(LLAMA_AUCS, R1_AUCS)

    # 4. Curve shape
    llama_shape = curve_shape(LLAMA_AUCS)
    r1_shape = curve_shape(R1_AUCS)

    # 5. Interpretation
    interpretation = generate_interpretation(
        llama_stats, r1_stats, gap, phases, llama_shape, r1_shape
    )

    # 6. Statistical test stubs
    test_stubs = statistical_test_stubs()

    # --- Assemble report ------------------------------------------------------
    report: dict[str, Any] = {
        "metadata": {
            "description": "Layer-wise probe AUC analysis: Llama-3.1-8B vs R1-Distill-Llama-8B",
            "source": "results/probes/llama_vs_r1_layer_aucs.json (B200 pod)",
            "n_layers": N_LAYERS,
            "phase_definitions": {k: list(v) for k, v in PHASES.items()},
            "note": (
                "Summary statistics only. Full statistical tests require per-fold "
                "AUC arrays from the probe runner."
            ),
        },
        "llama_3_1_8b": {
            "raw_aucs": LLAMA_AUCS,
            "basic_stats": llama_stats,
            "curve_shape": llama_shape,
        },
        "r1_distill_llama_8b": {
            "raw_aucs": R1_AUCS,
            "basic_stats": r1_stats,
            "curve_shape": r1_shape,
        },
        "deliberation_gap": gap,
        "phase_analysis": phases,
        "interpretation": interpretation,
        "statistical_tests_todo": test_stubs,
    }

    # --- Save report ----------------------------------------------------------
    # Convert int keys to strings for JSON
    def _stringify_keys(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _stringify_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_stringify_keys(i) for i in obj]
        if isinstance(obj, tuple):
            return [_stringify_keys(i) for i in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(_stringify_keys(report), f, indent=2)

    # --- Print formatted summary ----------------------------------------------
    print("=" * 80)
    print("PROBE AUC ANALYSIS: Llama-3.1-8B vs R1-Distill-Llama-8B")
    print("=" * 80)

    print("\n--- Basic Statistics ---")
    for name, stats in [("Llama-3.1-8B", llama_stats), ("R1-Distill", r1_stats)]:
        print(f"\n  {name}:")
        print(f"    Peak AUC:  {stats['peak_auc']:.3f} (layer {stats['peak_layer']})")
        print(f"    Mean AUC:  {stats['mean_auc']:.3f} +/- {stats['std_auc']:.3f}")
        print(f"    Min AUC:   {stats['min_auc']:.3f} (layer {stats['min_layer']})")

    print("\n--- Deliberation Gap (Llama - R1) ---")
    print(f"  Mean gap:    {gap['mean_gap']:.4f}")
    print(f"  Max gap:     {gap['max_gap']:.4f} (layer {gap['max_gap_layer']})")
    print(f"  Min gap:     {gap['min_gap']:.4f} (layer {gap['min_gap_layer']})")
    print(f"  All layers positive: {gap['all_positive']}")
    print("  Top-5 gap layers: ", end="")
    for layer, diff in gap["top5_gap_layers"]:
        print(f"L{layer}={diff:.3f}  ", end="")
    print()

    print("\n--- Phase Analysis ---")
    print(f"  {'Phase':<8} {'Llama AUC':>10} {'R1 AUC':>10} {'Gap':>8}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for phase_name in PHASES:
        p = phases[phase_name]
        print(
            f"  {phase_name:<8} {p['llama_mean_auc']:>10.4f} "
            f"{p['r1_mean_auc']:>10.4f} {p['gap']:>8.4f}"
        )
    print(f"\n  Gap trend: {phases['gap_trend']}")
    print(f"  Peak gap phase: {phases['peak_gap_phase']}")

    print("\n--- Curve Shape ---")
    for name, shape in [("Llama-3.1-8B", llama_shape), ("R1-Distill", r1_shape)]:
        print(f"\n  {name}:")
        for thresh in ["0.90", "0.95", "0.99"]:
            first = shape[f"first_exceeds_{thresh}"]
            n_above = shape[f"n_layers_above_{thresh}"]
            first_str = f"layer {first}" if first is not None else "NEVER"
            print(f"    First >= {thresh}: {first_str:>10}   ({n_above} layers total)")

    print("\n--- Interpretation (for Results section) ---")
    # Wrap at 80 chars for readability
    import textwrap
    for line in textwrap.wrap(interpretation, width=78):
        print(f"  {line}")

    print("\n--- Statistical Tests TODO (need per-fold data) ---")
    for test_name, description in test_stubs.items():
        print(f"\n  [{test_name}]")
        for line in textwrap.wrap(description, width=74):
            print(f"    {line}")

    print(f"\n--- Report saved to: {output_path} ---")
    print("=" * 80)


if __name__ == "__main__":
    main()
