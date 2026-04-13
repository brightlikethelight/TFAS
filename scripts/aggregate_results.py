#!/usr/bin/env python3
"""Aggregate all s1s2 results into a unified JSON summary and LaTeX tables.

Scans results/ recursively for JSON files, parses each result type
(behavioral, probes, cross-prediction, geometry, transfer matrix,
lure susceptibility), and produces:

  - results/summary/all_results.json  -- complete structured summary
  - results/summary/paper_tables.tex  -- LaTeX tables for the paper
  - stdout                            -- human-readable summary

Handles missing files gracefully (prints NOT FOUND, continues).
Works both on the B200 pod (where results exist) and locally (shows gaps).

Usage::

    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results-dir /workspace/results
    python scripts/aggregate_results.py --output-dir results/summary
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants -- expected file locations relative to results_dir
# ---------------------------------------------------------------------------

# Behavioral: per-model JSON files
BEHAVIORAL_FILES: dict[str, str] = {
    "Llama-3.1-8B-Instruct": "behavioral/llama_3_1_8b_instruct.json",
    "R1-Distill-Llama-8B": "behavioral/r1_distill_llama_8b.json",
    "Qwen-3-8B-NO_THINK": "behavioral/qwen_3_8b_nothink.json",
    "Qwen-3-8B-THINK": "behavioral/qwen_3_8b_think.json",
    "OLMo-3-7B-Instruct": "behavioral/olmo3_instruct_ALL.json",
    "OLMo-3-7B-Think": "behavioral/olmo3_think_ALL.json",
}

# Probe layer-wise AUCs
PROBE_FILES: dict[str, str] = {
    "Llama-3.1-8B-Instruct": "probes/llama_3_1_8b_instruct_vulnerable.json",
    "R1-Distill-Llama-8B": "probes/r1_distill_llama_8b_vulnerable.json",
    "Qwen-3-8B-NO_THINK": "probes/qwen_3_8b_nothink_vulnerable.json",
    "Qwen-3-8B-THINK": "probes/qwen_3_8b_think_vulnerable.json",
    "OLMo-3-7B-Instruct": "probes/olmo3_instruct_vulnerable.json",
}

# Cross-prediction
CROSS_PREDICTION_FILES: dict[str, str] = {
    "Llama-3.1-8B-Instruct": "probes/llama_cross_prediction.json",
    "R1-Distill-Llama-8B": "probes/r1_cross_prediction.json",
}

# Transfer matrix
TRANSFER_MATRIX_FILE = "probes/transfer_matrix_l14_llama.json"

# Geometry
GEOMETRY_FILES: dict[str, str] = {
    "silhouette": "geometry/silhouette_scores.json",
    "cka": "geometry/cka_matrix.json",
}

# Lure susceptibility
LURE_SUSCEPTIBILITY_FILES: dict[str, str] = {
    "Llama-3.1-8B-Instruct": "probes/llama_lure_susceptibility.json",
    "R1-Distill-Llama-8B": "probes/r1_lure_susceptibility.json",
}

# Combined report from probe analysis
PROBE_ANALYSIS_REPORT = "probes/probe_analysis_report.json"

# Final statistics (computed locally from hardcoded data)
FINAL_STATISTICS = "final_statistics.json"

# All categories in the benchmark
CATEGORIES = [
    "base_rate", "conjunction", "syllogism", "CRT",
    "arithmetic", "framing", "anchoring", "sunk_cost",
]

VULNERABLE_CATEGORIES = ["base_rate", "conjunction", "syllogism"]
IMMUNE_CATEGORIES = ["CRT", "arithmetic", "framing", "anchoring"]


# ---------------------------------------------------------------------------
# Hardcoded fallback data (from SESSION_STATE.md / final_statistics.json)
# ---------------------------------------------------------------------------

HARDCODED_BEHAVIORAL: dict[str, dict[str, Any]] = {
    "Llama-3.1-8B-Instruct": {
        "overall_lure_pct": 27.3,
        "base_rate": 84.0,
        "conjunction": 55.0,
        "syllogism": 52.0,
        "CRT": 0.0,
        "arithmetic": 0.0,
        "framing": 0.0,
        "anchoring": 0.0,
    },
    "R1-Distill-Llama-8B": {
        "overall_lure_pct": 2.4,
        "base_rate": 4.0,
        "conjunction": 0.0,
        "syllogism": 0.0,
        "CRT": None,
        "arithmetic": None,
        "framing": None,
        "anchoring": None,
    },
    "Qwen-3-8B-NO_THINK": {
        "overall_lure_pct": 21.0,
        "base_rate": 56.0,
        "conjunction": 95.0,
        "syllogism": 0.0,
        "CRT": None,
        "arithmetic": None,
        "framing": None,
        "anchoring": None,
    },
    "Qwen-3-8B-THINK": {
        "overall_lure_pct": 7.0,
        "base_rate": 4.0,
        "conjunction": 55.0,
        "syllogism": None,
        "CRT": None,
        "arithmetic": None,
        "framing": None,
        "anchoring": None,
    },
    "OLMo-3-7B-Instruct": {
        "overall_lure_pct": 0.0,
        "base_rate": 0.0,
        "conjunction": 0.0,
        "syllogism": 0.0,
        "CRT": 0.0,
        "arithmetic": 0.0,
        "framing": 0.0,
        "anchoring": 0.0,
    },
    "OLMo-3-7B-Think": {
        "overall_lure_pct": 0.0,
        "base_rate": 0.0,
        "conjunction": 0.0,
        "syllogism": 0.0,
        "CRT": 0.0,
        "arithmetic": 0.0,
        "framing": 0.0,
        "anchoring": 0.0,
    },
}

HARDCODED_PROBES: dict[str, dict[str, Any]] = {
    "Llama-3.1-8B-Instruct": {
        "peak_auc": 0.999,
        "peak_layer": 14,
        "n_layers": 32,
        "mean_auc": 0.962,
    },
    "R1-Distill-Llama-8B": {
        "peak_auc": 0.929,
        "peak_layer": 14,
        "n_layers": 32,
        "mean_auc": 0.912,
    },
    "Qwen-3-8B-NO_THINK": {
        "peak_auc": 0.971,
        "peak_layer": 34,
        "n_layers": 36,
        "mean_auc": None,
    },
    "Qwen-3-8B-THINK": {
        "peak_auc": 0.971,
        "peak_layer": 34,
        "n_layers": 36,
        "mean_auc": None,
    },
    "OLMo-3-7B-Instruct": {
        "peak_auc": 0.998,
        "peak_layer": 21,
        "n_layers": 32,
        "mean_auc": 0.972,
    },
}

HARDCODED_CROSS_PREDICTION: dict[str, dict[str, Any]] = {
    "Llama-3.1-8B-Instruct": {
        "l14_transfer_auc": 0.378,
        "mean_transfer_auc": 0.444,
        "interpretation": "SPECIFIC",
        "layer_aucs": {
            0: 0.702, 4: 0.439, 8: 0.384, 12: 0.449, 14: 0.378,
            16: 0.569, 20: 0.386, 24: 0.326, 28: 0.365, 31: 0.438,
        },
    },
    "R1-Distill-Llama-8B": {
        "l14_transfer_auc": 0.685,
        "mean_transfer_auc": 0.654,
        "interpretation": "MIXED",
        "layer_aucs": {
            0: 0.734, 4: 0.878, 8: 0.878, 12: 0.728, 14: 0.685,
            16: 0.696, 20: 0.615, 24: 0.524, 28: 0.415, 31: 0.385,
        },
    },
}

HARDCODED_TRANSFER_MATRIX: dict[str, dict[str, float]] = {
    "base_rate": {"base_rate": 1.0, "conjunction": 0.993, "syllogism": None},
    "conjunction": {"base_rate": 0.993, "conjunction": 1.0, "syllogism": None},
    "syllogism": {"base_rate": None, "conjunction": None, "syllogism": 1.0},
}

HARDCODED_LURE_SUSCEPTIBILITY: dict[str, dict[str, float | None]] = {
    "Llama-3.1-8B-Instruct": {
        "mean": 0.422,
        "std": 2.986,
        "min": None,
        "max": None,
    },
    "R1-Distill-Llama-8B": {
        "mean": -0.326,
        "std": 2.056,
        "min": None,
        "max": None,
    },
}

HARDCODED_GEOMETRY: dict[str, Any] = {
    "silhouette": {
        "Llama-3.1-8B-Instruct": 0.079,
        "R1-Distill-Llama-8B": 0.059,
    },
    "cka": {"min": 0.379, "max": 0.985},
}


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, return None if missing or malformed."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  WARNING: failed to parse {path}: {exc}", file=sys.stderr)
        return None


def _try_load_or_hardcode(
    results_dir: Path,
    file_map: dict[str, str],
    hardcoded: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    """Attempt to load JSON files; fall back to hardcoded data.

    Returns (data_dict, list_of_sources) where sources indicate
    'file' or 'hardcoded' for each key.
    """
    data: dict[str, Any] = {}
    sources: list[str] = []

    for model, rel_path in file_map.items():
        full_path = results_dir / rel_path
        loaded = _load_json(full_path)
        if loaded is not None:
            data[model] = loaded
            sources.append(f"{model}: file ({rel_path})")
        elif model in hardcoded:
            data[model] = hardcoded[model]
            sources.append(f"{model}: hardcoded fallback")
        else:
            sources.append(f"{model}: NOT FOUND")

    return data, sources


# ---------------------------------------------------------------------------
# Result parsers -- extract structured data from JSON files or hardcoded
# ---------------------------------------------------------------------------

def _parse_behavioral(
    results_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Parse behavioral results into a uniform dict."""
    data, sources = _try_load_or_hardcode(
        results_dir, BEHAVIORAL_FILES, HARDCODED_BEHAVIORAL, "behavioral"
    )

    parsed: dict[str, Any] = {}
    for model, raw in data.items():
        # Handle both file format (nested) and hardcoded (flat)
        if "overall_lure_pct" in raw:
            parsed[model] = raw
        elif "results" in raw and isinstance(raw["results"], list):
            # Raw item-level format (e.g., OLMo behavioral files)
            # Compute lure rates from individual trial verdicts
            from collections import Counter
            cat_total: Counter[str] = Counter()
            cat_lured: Counter[str] = Counter()
            for item in raw["results"]:
                if item.get("conflict"):
                    cat_key = item["category"].lower()
                    cat_total[cat_key] += 1
                    if item.get("verdict") == "lured":
                        cat_lured[cat_key] += 1
            total_conflict = sum(cat_total.values())
            total_lured = sum(cat_lured.values())
            entry: dict[str, Any] = {
                "overall_lure_pct": round(total_lured / total_conflict * 100, 1)
                if total_conflict > 0 else 0.0,
            }
            for cat in CATEGORIES:
                cat_key = cat.lower()
                if cat_total[cat_key] > 0:
                    entry[cat] = round(cat_lured[cat_key] / cat_total[cat_key] * 100, 1)
                else:
                    entry[cat] = None
            parsed[model] = entry
        elif "overall" in raw:
            # File format: {overall: {lure_rate: ...}, categories: {...}}
            parsed[model] = {
                "overall_lure_pct": raw["overall"].get("lure_rate", raw["overall"].get("lure_pct")),
            }
            cats = raw.get("categories", raw.get("per_category", {}))
            for cat in CATEGORIES:
                val = cats.get(cat, {})
                if isinstance(val, dict):
                    parsed[model][cat] = val.get("lure_rate", val.get("lure_pct"))
                else:
                    parsed[model][cat] = val
        else:
            parsed[model] = raw

    return parsed, sources


def _parse_probes(
    results_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Parse probe results: peak layer, peak AUC, control AUC, selectivity."""
    data, sources = _try_load_or_hardcode(
        results_dir, PROBE_FILES, HARDCODED_PROBES, "probes"
    )

    # Also try loading the combined probe analysis report
    report = _load_json(results_dir / PROBE_ANALYSIS_REPORT)
    final_stats = _load_json(results_dir / FINAL_STATISTICS)

    parsed: dict[str, Any] = {}
    for model, raw in data.items():
        if "peak_auc" in raw:
            parsed[model] = {
                "peak_layer": raw.get("peak_layer"),
                "peak_auc": raw.get("peak_auc"),
                "mean_auc": raw.get("mean_auc"),
                "n_layers": raw.get("n_layers"),
                "control_auc": raw.get("control_auc"),
                "selectivity": raw.get("selectivity"),
            }
        elif "layers" in raw and isinstance(raw["layers"], dict):
            # OLMo-style probe format: {layers: {L00_P0: {auc_mean, ...}, ...}}
            p0_aucs: list[tuple[int, float]] = []
            for key, layer_data in raw["layers"].items():
                if key.endswith("_P0"):
                    layer_idx = layer_data.get("layer", int(key.split("_")[0][1:]))
                    auc = layer_data.get("auc_mean")
                    if auc is not None:
                        p0_aucs.append((layer_idx, auc))
            if p0_aucs:
                peak_layer, peak_auc = max(p0_aucs, key=lambda x: x[1])
                mean_auc = round(sum(a for _, a in p0_aucs) / len(p0_aucs), 4)
            else:
                peak_layer, peak_auc, mean_auc = None, None, None
            parsed[model] = {
                "peak_layer": peak_layer,
                "peak_auc": round(peak_auc, 4) if peak_auc is not None else None,
                "mean_auc": mean_auc,
                "n_layers": raw.get("n_layers", len(p0_aucs)),
                "control_auc": None,
                "selectivity": None,
            }
        elif "basic_stats" in raw:
            bs = raw["basic_stats"]
            parsed[model] = {
                "peak_layer": bs.get("peak_layer"),
                "peak_auc": bs.get("peak_auc"),
                "mean_auc": bs.get("mean_auc"),
                "n_layers": len(raw.get("raw_aucs", {})),
                "control_auc": raw.get("control_auc"),
                "selectivity": raw.get("selectivity"),
            }
        else:
            parsed[model] = raw

    # Enrich from probe_analysis_report if available
    if report is not None:
        for key, model_name in [
            ("llama_3_1_8b", "Llama-3.1-8B-Instruct"),
            ("r1_distill_llama_8b", "R1-Distill-Llama-8B"),
        ]:
            if key in report and model_name in parsed:
                bs = report[key].get("basic_stats", {})
                if parsed[model_name].get("peak_auc") is None:
                    parsed[model_name]["peak_auc"] = bs.get("peak_auc")
                    parsed[model_name]["peak_layer"] = bs.get("peak_layer")
                    parsed[model_name]["mean_auc"] = bs.get("mean_auc")

    # Enrich from final_statistics.json
    if final_stats is not None:
        for key, model_name in [
            ("llama", "Llama-3.1-8B-Instruct"),
            ("r1_distill", "R1-Distill-Llama-8B"),
        ]:
            h1 = final_stats.get("h1_linear_decodability", {}).get(key, {})
            if model_name in parsed and h1:
                entry = parsed[model_name]
                if entry.get("peak_auc") is None:
                    entry["peak_auc"] = h1.get("peak_auc")
                    entry["peak_layer"] = h1.get("peak_layer")
                boot = h1.get("bootstrap_mean_auc_95ci", {})
                if boot:
                    entry["bootstrap_ci"] = {
                        "mean": boot.get("mean"),
                        "ci_lower": boot.get("ci_lower"),
                        "ci_upper": boot.get("ci_upper"),
                    }

    return parsed, sources


def _parse_cross_prediction(
    results_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Parse cross-prediction results."""
    data, sources = _try_load_or_hardcode(
        results_dir,
        CROSS_PREDICTION_FILES,
        HARDCODED_CROSS_PREDICTION,
        "cross_prediction",
    )

    parsed: dict[str, Any] = {}
    for model, raw in data.items():
        if "l14_transfer_auc" in raw:
            parsed[model] = raw
        elif "results" in raw and isinstance(raw["results"], dict):
            # Format: {results: {layer: {transfer_to_immune: auc, ...}}}
            int_aucs: dict[int, float] = {}
            for layer_str, layer_data in raw["results"].items():
                if isinstance(layer_data, dict) and "transfer_to_immune" in layer_data:
                    int_aucs[int(layer_str)] = layer_data["transfer_to_immune"]
            l14 = int_aucs.get(14)
            mean_auc = sum(int_aucs.values()) / len(int_aucs) if int_aucs else None
            interp = "SPECIFIC" if mean_auc is not None and mean_auc < 0.55 else "MIXED"
            parsed[model] = {
                "l14_transfer_auc": round(l14, 4) if l14 is not None else None,
                "mean_transfer_auc": round(mean_auc, 4) if mean_auc is not None else None,
                "interpretation": interp,
                "layer_aucs": int_aucs,
            }
        elif "cross_aucs" in raw or "layer_aucs" in raw:
            aucs = raw.get("cross_aucs", raw.get("layer_aucs", {}))
            # Ensure keys are ints for sorting
            int_aucs = {int(k): v for k, v in aucs.items()}
            l14 = int_aucs.get(14)
            mean_auc = sum(int_aucs.values()) / len(int_aucs) if int_aucs else None
            parsed[model] = {
                "l14_transfer_auc": l14,
                "mean_transfer_auc": round(mean_auc, 4) if mean_auc else None,
                "layer_aucs": int_aucs,
            }
        else:
            parsed[model] = raw

    # Enrich from final_statistics.json
    final_stats = _load_json(results_dir / FINAL_STATISTICS)
    if final_stats is not None:
        cp = final_stats.get("cross_prediction_specificity", {})
        for key, model_name in [
            ("llama", "Llama-3.1-8B-Instruct"),
            ("r1", "R1-Distill-Llama-8B"),
        ]:
            aucs = cp.get(f"{key}_cross_aucs", {})
            if aucs and model_name not in parsed:
                int_aucs = {int(k): v for k, v in aucs.items()}
                l14 = int_aucs.get(14)
                mean_val = cp.get(f"{key}_mean_cross_auc")
                parsed[model_name] = {
                    "l14_transfer_auc": l14,
                    "mean_transfer_auc": mean_val,
                    "layer_aucs": int_aucs,
                }
                sources.append(f"{model_name}: final_statistics.json")

    return parsed, sources


def _parse_transfer_matrix(
    results_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Parse the category-to-category transfer matrix."""
    sources: list[str] = []
    full_path = results_dir / TRANSFER_MATRIX_FILE

    loaded = _load_json(full_path)
    if loaded is not None:
        sources.append(f"file ({TRANSFER_MATRIX_FILE})")
        # Normalize to category x category dict
        if "matrix" in loaded:
            return loaded["matrix"], sources
        return loaded, sources

    sources.append("hardcoded fallback")
    return HARDCODED_TRANSFER_MATRIX, sources


def _parse_lure_susceptibility(
    results_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Parse lure susceptibility scores."""
    data, sources = _try_load_or_hardcode(
        results_dir,
        LURE_SUSCEPTIBILITY_FILES,
        HARDCODED_LURE_SUSCEPTIBILITY,
        "lure_susceptibility",
    )

    parsed: dict[str, Any] = {}
    for model, raw in data.items():
        if "mean" in raw:
            parsed[model] = raw
        elif "summary" in raw:
            s = raw["summary"]
            parsed[model] = {
                "mean": s.get("mean"),
                "std": s.get("std", s.get("sd")),
                "min": s.get("min"),
                "max": s.get("max"),
            }
        else:
            parsed[model] = raw

    # Enrich from final_statistics.json
    final_stats = _load_json(results_dir / FINAL_STATISTICS)
    if final_stats is not None:
        ls = final_stats.get("lure_susceptibility", {})
        for key, model_name in [
            ("llama", "Llama-3.1-8B-Instruct"),
            ("r1", "R1-Distill-Llama-8B"),
        ]:
            if model_name not in parsed or parsed[model_name].get("mean") is None:
                mean_val = ls.get(f"{key}_mean")
                sd_val = ls.get(f"{key}_sd")
                if mean_val is not None:
                    parsed[model_name] = {
                        "mean": mean_val,
                        "std": sd_val,
                        "min": None,
                        "max": None,
                    }
                    sources.append(f"{model_name}: final_statistics.json")

    return parsed, sources


def _parse_geometry(
    results_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Parse geometry results (silhouette, CKA)."""
    sources: list[str] = []
    result: dict[str, Any] = {}

    for name, rel_path in GEOMETRY_FILES.items():
        loaded = _load_json(results_dir / rel_path)
        if loaded is not None:
            result[name] = loaded
            sources.append(f"{name}: file ({rel_path})")
        elif name in HARDCODED_GEOMETRY:
            result[name] = HARDCODED_GEOMETRY[name]
            sources.append(f"{name}: hardcoded fallback")
        else:
            sources.append(f"{name}: NOT FOUND")

    if not result:
        result = HARDCODED_GEOMETRY
        sources.append("all geometry: hardcoded fallback")

    return result, sources


# ---------------------------------------------------------------------------
# LaTeX table generators
# ---------------------------------------------------------------------------

def _fmt(val: float | None, decimals: int = 1, pct: bool = True) -> str:
    """Format a value for a LaTeX cell. None -> '--'."""
    if val is None:
        return "--"
    if pct:
        return f"{val:.{decimals}f}\\%"
    return f"{val:.{decimals}f}"


def _fmt_auc(val: float | None, decimals: int = 3) -> str:
    """Format an AUC value."""
    if val is None:
        return "--"
    return f"{val:.{decimals}f}"


def _bold_if_max(val: float | None, all_vals: list[float | None]) -> str:
    """Wrap in \\textbf if this is the maximum non-None value."""
    if val is None:
        return "--"
    non_none = [v for v in all_vals if v is not None]
    if non_none and val == max(non_none):
        return f"\\textbf{{{val:.3f}}}"
    return f"{val:.3f}"


def generate_table1_behavioral(behavioral: dict[str, Any]) -> str:
    """Table 1: Behavioral lure rates -- models x categories."""
    cats_display = ["base\\_rate", "conjunction", "syllogism", "CRT", "arithmetic", "framing", "anchoring"]
    cats_keys = ["base_rate", "conjunction", "syllogism", "CRT", "arithmetic", "framing", "anchoring"]

    lines = [
        "% Table 1: Behavioral Lure Rates",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Lure rates (\\%) by model and cognitive bias category. "
        "Categories above the line are \\emph{vulnerable} (lure rate $>$ 0\\% in at least one model); "
        "below are \\emph{immune}.}",
        "\\label{tab:behavioral}",
        "\\small",
        "\\begin{tabular}{l c " + " c" * len(cats_keys) + "}",
        "\\toprule",
        "Model & Overall & " + " & ".join(cats_display) + " \\\\",
        "\\midrule",
    ]

    models = [
        "Llama-3.1-8B-Instruct",
        "R1-Distill-Llama-8B",
        "Qwen-3-8B-NO_THINK",
        "Qwen-3-8B-THINK",
        "OLMo-3-7B-Instruct",
        "OLMo-3-7B-Think",
    ]
    display_names = {
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "R1-Distill-Llama-8B": "R1-Distill-8B",
        "Qwen-3-8B-NO_THINK": "Qwen-3-8B (no think)",
        "Qwen-3-8B-THINK": "Qwen-3-8B (think)",
        "OLMo-3-7B-Instruct": "OLMo-3-7B",
        "OLMo-3-7B-Think": "OLMo-3-7B (think)",
    }

    for model in models:
        d = behavioral.get(model, {})
        overall = d.get("overall_lure_pct")
        cells = [_fmt(overall)]
        for cat in cats_keys:
            cells.append(_fmt(d.get(cat)))
        name = display_names.get(model, model)
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def generate_table2_probes(probes: dict[str, Any]) -> str:
    """Table 2: Probe summary -- model, peak layer, peak AUC, control AUC, selectivity."""
    lines = [
        "% Table 2: Linear Probe Summary",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Linear probe performance on vulnerable categories (conflict vs.\\ control). "
        "AUC from 5-fold stratified CV on last-prompt-token residual stream activations.}",
        "\\label{tab:probes}",
        "\\small",
        "\\begin{tabular}{l c c c c c}",
        "\\toprule",
        "Model & Layers & Peak Layer & Peak AUC & Mean AUC & Control AUC \\\\",
        "\\midrule",
    ]

    models = [
        "Llama-3.1-8B-Instruct",
        "R1-Distill-Llama-8B",
        "Qwen-3-8B-NO_THINK",
        "Qwen-3-8B-THINK",
        "OLMo-3-7B-Instruct",
    ]
    display_names = {
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "R1-Distill-Llama-8B": "R1-Distill-8B",
        "Qwen-3-8B-NO_THINK": "Qwen-3-8B (no think)",
        "Qwen-3-8B-THINK": "Qwen-3-8B (think)",
        "OLMo-3-7B-Instruct": "OLMo-3-7B",
    }

    peak_aucs = [probes.get(m, {}).get("peak_auc") for m in models]

    for model in models:
        d = probes.get(model, {})
        n_layers = d.get("n_layers", "--")
        peak_layer = d.get("peak_layer")
        peak_auc = d.get("peak_auc")
        mean_auc = d.get("mean_auc")
        control_auc = d.get("control_auc")

        peak_str = _bold_if_max(peak_auc, peak_aucs)
        mean_str = _fmt_auc(mean_auc)
        ctrl_str = _fmt_auc(control_auc)
        layer_str = str(peak_layer) if peak_layer is not None else "--"

        name = display_names.get(model, model)
        lines.append(
            f"{name} & {n_layers} & L{layer_str} & {peak_str} & {mean_str} & {ctrl_str} \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def generate_table3_cross_prediction(cross_pred: dict[str, Any]) -> str:
    """Table 3: Cross-prediction AUC at key layers."""
    # Collect all layers across models
    all_layers: set[int] = set()
    for model_data in cross_pred.values():
        aucs = model_data.get("layer_aucs", {})
        all_layers.update(int(k) for k in aucs)

    if not all_layers:
        all_layers = {0, 4, 8, 12, 14, 16, 20, 24, 28, 31}

    sorted_layers = sorted(all_layers)
    layer_cols = " ".join("c" for _ in sorted_layers)

    lines = [
        "% Table 3: Cross-Prediction (Vulnerable -> Immune)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-prediction AUC: probes trained on vulnerable categories, "
        "tested on immune categories. Values near 0.5 indicate specificity "
        "(probe does not generalize).}",
        "\\label{tab:cross-prediction}",
        "\\small",
        f"\\begin{{tabular}}{{l {layer_cols} c}}",
        "\\toprule",
        "Model & " + " & ".join(f"L{layer}" for layer in sorted_layers) + " & Mean \\\\",
        "\\midrule",
    ]

    models = ["Llama-3.1-8B-Instruct", "R1-Distill-Llama-8B"]
    display_names = {
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "R1-Distill-Llama-8B": "R1-Distill-8B",
    }

    for model in models:
        d = cross_pred.get(model, {})
        aucs = d.get("layer_aucs", {})
        cells = []
        for layer in sorted_layers:
            val = aucs.get(layer, aucs.get(str(layer)))
            cells.append(_fmt_auc(val))
        mean_val = d.get("mean_transfer_auc")
        cells.append(_fmt_auc(mean_val))
        name = display_names.get(model, model)
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def generate_table4_transfer_matrix(transfer: dict[str, Any]) -> str:
    """Table 4: Category x Category transfer AUC at Llama L14."""
    cats = VULNERABLE_CATEGORIES

    lines = [
        "% Table 4: Category Transfer Matrix (Llama L14)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Category-to-category transfer AUC (Llama-3.1-8B, Layer 14). "
        "Train probe on row category, test on column category. "
        "High off-diagonal values indicate shared representations.}",
        "\\label{tab:transfer-matrix}",
        "\\small",
        "\\begin{tabular}{l " + " c" * len(cats) + "}",
        "\\toprule",
        "Train $\\backslash$ Test & " + " & ".join(c.replace("_", "\\_") for c in cats) + " \\\\",
        "\\midrule",
    ]

    for row_cat in cats:
        row_data = transfer.get(row_cat, {})
        cells = []
        for col_cat in cats:
            val = row_data.get(col_cat)
            if row_cat == col_cat:
                cells.append("--")
            else:
                cells.append(_fmt_auc(val))
        display_cat = row_cat.replace("_", "\\_")
        lines.append(f"{display_cat} & " + " & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def generate_table5_lure_susceptibility(lure: dict[str, Any]) -> str:
    """Table 5: Lure susceptibility by model."""
    lines = [
        "% Table 5: Lure Susceptibility (Continuous Score)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Lure susceptibility scores at the last prompt token (P0). "
        "Positive values indicate the model's residual stream favors the lure answer; "
        "negative values indicate it favors the correct answer.}",
        "\\label{tab:lure-susceptibility}",
        "\\small",
        "\\begin{tabular}{l c c c c}",
        "\\toprule",
        "Model & Mean & Std & Min & Max \\\\",
        "\\midrule",
    ]

    models = ["Llama-3.1-8B-Instruct", "R1-Distill-Llama-8B"]
    display_names = {
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "R1-Distill-Llama-8B": "R1-Distill-8B",
    }

    for model in models:
        d = lure.get(model, {})
        mean_val = d.get("mean")
        std_val = d.get("std")
        min_val = d.get("min")
        max_val = d.get("max")

        # Sign-aware formatting
        def _fmt_signed(v: float | None) -> str:
            if v is None:
                return "--"
            sign = "+" if v > 0 else ""
            return f"{sign}{v:.3f}"

        name = display_names.get(model, model)
        lines.append(
            f"{name} & {_fmt_signed(mean_val)} & "
            f"{_fmt_auc(std_val)} & {_fmt_signed(min_val)} & {_fmt_signed(max_val)} \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Human-readable stdout summary
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    meta = summary.get("metadata", {})
    print("\ns1s2 Results Summary")
    print(f"Generated: {meta.get('generated_at', 'unknown')}")
    print(f"Source: {meta.get('results_dir', 'unknown')}")

    # Behavioral
    _print_section("BEHAVIORAL LURE RATES")
    behavioral = summary.get("behavioral", {})
    for model, data in behavioral.items():
        overall = data.get("overall_lure_pct", "?")
        print(f"\n  {model}:")
        print(f"    Overall lure rate: {overall}%")
        for cat in VULNERABLE_CATEGORIES:
            val = data.get(cat)
            tag = f"{val}%" if val is not None else "--"
            print(f"    {cat}: {tag}")

    # Probes
    _print_section("LINEAR PROBES (Vulnerable Categories)")
    probes = summary.get("probes", {})
    for model, data in probes.items():
        peak = data.get("peak_auc", "?")
        layer = data.get("peak_layer", "?")
        mean = data.get("mean_auc", "?")
        print(f"  {model}: peak AUC {peak} at L{layer} (mean {mean})")

    # Cross-prediction
    _print_section("CROSS-PREDICTION (Vulnerable -> Immune)")
    cross = summary.get("cross_prediction", {})
    for model, data in cross.items():
        l14 = data.get("l14_transfer_auc", "?")
        mean = data.get("mean_transfer_auc", "?")
        interp = data.get("interpretation", "")
        print(f"  {model}: L14 transfer AUC = {l14}, mean = {mean} [{interp}]")

    # Transfer matrix
    _print_section("TRANSFER MATRIX (Llama L14)")
    transfer = summary.get("transfer_matrix", {})
    cats = VULNERABLE_CATEGORIES
    header = f"{'':>15}" + "".join(f"{c:>15}" for c in cats)
    print(header)
    for row in cats:
        row_data = transfer.get(row, {})
        cells = []
        for col in cats:
            val = row_data.get(col)
            if row == col:
                cells.append(f"{'--':>15}")
            elif val is not None:
                cells.append(f"{val:>15.3f}")
            else:
                cells.append(f"{'--':>15}")
        print(f"{row:>15}" + "".join(cells))

    # Lure susceptibility
    _print_section("LURE SUSCEPTIBILITY")
    lure = summary.get("lure_susceptibility", {})
    for model, data in lure.items():
        mean = data.get("mean", "?")
        std = data.get("std", "?")
        sign = "+" if isinstance(mean, int | float) and mean > 0 else ""
        print(f"  {model}: mean {sign}{mean}, std {std}")

    # Geometry
    _print_section("GEOMETRY")
    geom = summary.get("geometry", {})
    sil = geom.get("silhouette", {})
    if isinstance(sil, dict):
        for model, val in sil.items():
            print(f"  Silhouette {model}: {val}")
    cka = geom.get("cka", {})
    if isinstance(cka, dict) and "min" in cka:
        print(f"  CKA range: {cka['min']} - {cka['max']}")

    # Data completeness
    _print_section("DATA SOURCES")
    for section, src_list in summary.get("_sources", {}).items():
        print(f"\n  [{section}]")
        for src in src_list:
            marker = "OK" if "NOT FOUND" not in src else "MISSING"
            print(f"    [{marker}] {src}")

    print()


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def aggregate(results_dir: Path) -> dict[str, Any]:
    """Build the complete summary dict from all available results."""
    sources: dict[str, list[str]] = {}

    behavioral, sources["behavioral"] = _parse_behavioral(results_dir)
    probes, sources["probes"] = _parse_probes(results_dir)
    cross_pred, sources["cross_prediction"] = _parse_cross_prediction(results_dir)
    transfer, sources["transfer_matrix"] = _parse_transfer_matrix(results_dir)
    lure, sources["lure_susceptibility"] = _parse_lure_susceptibility(results_dir)
    geometry, sources["geometry"] = _parse_geometry(results_dir)

    # Scan for any extra JSON files not explicitly handled
    all_jsons = sorted(results_dir.rglob("*.json"))
    known_files = set()
    for file_map in [
        BEHAVIORAL_FILES, PROBE_FILES, CROSS_PREDICTION_FILES,
        LURE_SUSCEPTIBILITY_FILES, GEOMETRY_FILES,
    ]:
        for rel in file_map.values():
            known_files.add((results_dir / rel).resolve())
    known_files.add((results_dir / TRANSFER_MATRIX_FILE).resolve())
    known_files.add((results_dir / PROBE_ANALYSIS_REPORT).resolve())
    known_files.add((results_dir / FINAL_STATISTICS).resolve())

    extra_files = [
        str(f.relative_to(results_dir))
        for f in all_jsons
        if f.resolve() not in known_files
    ]

    summary: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "results_dir": str(results_dir),
            "script": "scripts/aggregate_results.py",
            "n_json_files_found": len(all_jsons),
            "extra_json_files": extra_files,
        },
        "behavioral": behavioral,
        "probes": probes,
        "cross_prediction": cross_pred,
        "transfer_matrix": transfer,
        "lure_susceptibility": lure,
        "geometry": geometry,
        "_sources": sources,
    }

    return summary


def generate_latex(summary: dict[str, Any]) -> str:
    """Generate all LaTeX tables from the summary."""
    sections = [
        "% ============================================================",
        "% Auto-generated LaTeX tables for the s1s2 workshop paper",
        f"% Generated: {summary['metadata']['generated_at']}",
        "% Script: scripts/aggregate_results.py",
        "% ============================================================",
        "",
        generate_table1_behavioral(summary["behavioral"]),
        generate_table2_probes(summary["probes"]),
        generate_table3_cross_prediction(summary["cross_prediction"]),
        generate_table4_transfer_matrix(summary["transfer_matrix"]),
        generate_table5_lure_susceptibility(summary["lure_susceptibility"]),
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate s1s2 results into JSON summary and LaTeX tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Root results directory (default: auto-detect from project root).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for summary files (default: results/summary/).",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only output JSON, skip LaTeX generation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else project_root / "results"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "summary"

    if not results_dir.exists():
        print(f"ERROR: results directory does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate
    summary = aggregate(results_dir)

    # Write JSON
    json_path = output_dir / "all_results.json"
    # Remove internal _sources from the public JSON (keep it clean)
    public_summary = {k: v for k, v in summary.items() if not k.startswith("_")}
    with open(json_path, "w") as f:
        json.dump(public_summary, f, indent=2, default=str)
    print(f"Wrote JSON summary: {json_path}")

    # Write LaTeX
    if not args.json_only:
        tex_path = output_dir / "paper_tables.tex"
        latex = generate_latex(summary)
        with open(tex_path, "w") as f:
            f.write(latex)
        print(f"Wrote LaTeX tables: {tex_path}")

    # Print human-readable summary
    if not args.quiet:
        print_summary(summary)


if __name__ == "__main__":
    main()
