"""Orchestration for the attention entropy analysis workstream.

This module is the high-level entry point. It loads precomputed per-head
attention metrics from the activation cache (written by the extraction
workstream into ``/models/{key}/attention/{metric}`` per the data contract)
and drives:

- Per-head differential Mann-Whitney U tests via :mod:`.heads`
- Layer-level aggregation via :mod:`.layers`
- Temporal entropy trajectories via :mod:`.trajectories`
- Cross-model matched-pair comparison
- Result serialization to ``results/attention/{model_key}/``

We never compute metrics from raw attention matrices during analysis — that
work happens online in the extraction pipeline (see ``s1s2.extract.hooks``)
because the full (L, H, T, T) tensor is too large to materialize offline for
long generation traces. A small utility
:func:`compute_metrics_from_attention_pattern` is provided for downstream
callers that happen to have a raw 1-D attention distribution and want the
exact same definitions.

Results layout (relative to ``output_dir``)::

    results/attention/
      {model_key}/
        head_classifications.json
        layer_summary.json
        differential_tests.parquet    # long-format, all per-head tests
        trajectories.json
        figures/                      # populated by viz.py (gitignored)
      cross_model/
        {modelA}__vs__{modelB}.json
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Float

from s1s2.utils import io as ioh
from s1s2.utils.logging import get_logger
from s1s2.utils.stats import gini_coefficient

__all__ = [
    "GEMMA_FAMILY",
    "METRIC_DIRECTIONS",
    "METRIC_NAMES",
    "AttentionConfig",
    "ModelAttentionData",
    "ModelAttentionReport",
    "analyze_all_models",
    "analyze_model",
    "bh_fdr_joint",
    "compare_matched_pair",
    "compute_metrics_from_attention_pattern",
    "load_model_attention_data",
    "save_model_report",
]

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Metric definitions                                                          #
# --------------------------------------------------------------------------- #

METRIC_NAMES: tuple[str, ...] = (
    "entropy",
    "entropy_normalized",
    "gini",
    "max_attn",
    "focus_5",
)

# Direction convention: what does "more S2-like" look like?
#
# - entropy / entropy_normalized: HIGHER entropy = broader distribution =
#   more exploration = S2-like.
# - gini / max_attn / focus_5: HIGHER = more concentrated = more heuristic =
#   S1-like, so "S2-like means LOWER".
#
# We encode this as +1 (higher=S2) or -1 (lower=S2) so the consensus-head
# logic can use a single sign convention across metrics.
METRIC_DIRECTIONS: dict[str, int] = {
    "entropy": +1,
    "entropy_normalized": +1,
    "gini": -1,
    "max_attn": -1,
    "focus_5": -1,
}

# The Gemma-2 family alternates full vs sliding-window layers. We need to
# know which model family to apply the separation in.
GEMMA_FAMILY = "gemma-2"


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Knobs for an attention-analysis run.

    Mirrors ``configs/attention.yaml``. Kept as a plain dataclass so this
    module is independent of Hydra and can be unit tested.
    """

    activations_path: str
    output_dir: str = "results/attention"
    seed: int = 0

    metrics: tuple[str, ...] = METRIC_NAMES
    positions: tuple[str, ...] = ("P0", "P2", "T50", "Tend")
    fdr_q: float = 0.05
    effect_size_threshold: float = 0.3
    multi_metric_consensus: int = 3

    report_kv_group_aggregate: bool = True
    gemma_separate_window_layers: bool = True

    # matched-pair cross-model comparison. The key/value are model_key strings.
    matched_pairs: tuple[tuple[str, str], ...] = (
        ("llama-3.1-8b-instruct", "r1-distill-llama-8b"),
    )

    def validate(self) -> None:
        bad = [m for m in self.metrics if m not in METRIC_NAMES]
        if bad:
            raise ValueError(f"unknown metrics {bad!r}; allowed {METRIC_NAMES}")
        if not 0 < self.fdr_q < 1:
            raise ValueError(f"fdr_q must be in (0, 1); got {self.fdr_q}")
        if self.multi_metric_consensus < 1 or self.multi_metric_consensus > len(self.metrics):
            raise ValueError(
                f"multi_metric_consensus must be in [1, {len(self.metrics)}]; "
                f"got {self.multi_metric_consensus}"
            )


# --------------------------------------------------------------------------- #
# Data containers                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class ModelAttentionData:
    """In-memory view of one model's attention metrics.

    Attributes
    ----------
    model_key : HDF5 model key (``{hf_id with _ for /}``).
    model_config_key : human-readable key from ``configs/models.yaml``
        (e.g. ``"llama-3.1-8b-instruct"``).
    family : "llama" | "gemma" | "qwen2"
    n_layers, n_heads, n_kv_heads : architecture ints
    position_labels : list of position labels on disk
    position_indices : indices into ``position_labels`` for the requested
        analysis positions. Invalid positions are dropped.
    metrics : dict of metric_name -> (n_problems, n_layers, n_heads, n_selected_positions)
    conflict : (n_problems,) bool, True = S1-eliciting problem
    is_reasoning_model : whether the model is a reasoning model (determines
        which positions are valid).
    """

    model_key: str
    model_config_key: str
    family: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    is_reasoning_model: bool
    position_labels: list[str]
    selected_positions: list[str]
    metrics: dict[str, np.ndarray]
    conflict: np.ndarray
    sliding_window: int | None = None

    @property
    def group_size(self) -> int:
        """Number of query heads sharing one KV head (GQA group size)."""
        return self.n_heads // self.n_kv_heads

    @property
    def n_problems(self) -> int:
        return int(self.conflict.shape[0])

    @property
    def n_selected_positions(self) -> int:
        return len(self.selected_positions)


@dataclass
class ModelAttentionReport:
    """Fully-analyzed attention results for a single model."""

    model_key: str
    model_config_key: str
    family: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    is_reasoning_model: bool
    selected_positions: list[str]
    n_problems: int
    n_conflict: int
    n_no_conflict: int
    head_classifications: list[dict[str, Any]]
    kv_group_classifications: list[dict[str, Any]] | None
    layer_summary: list[dict[str, Any]]
    trajectories: dict[str, Any]
    gemma_layer_types: dict[str, list[int]] | None
    config: dict[str, Any]
    elapsed_s: float
    # Kept in-memory only. The long-format dataframe goes to parquet separately.
    differential_tests_df: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict WITHOUT the parquet-destined frame."""
        out: dict[str, Any] = {
            "model_key": self.model_key,
            "model_config_key": self.model_config_key,
            "family": self.family,
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "n_kv_heads": int(self.n_kv_heads),
            "is_reasoning_model": bool(self.is_reasoning_model),
            "selected_positions": list(self.selected_positions),
            "n_problems": int(self.n_problems),
            "n_conflict": int(self.n_conflict),
            "n_no_conflict": int(self.n_no_conflict),
            "head_classifications": self.head_classifications,
            "kv_group_classifications": self.kv_group_classifications,
            "layer_summary": self.layer_summary,
            "trajectories": self.trajectories,
            "gemma_layer_types": self.gemma_layer_types,
            "config": self.config,
            "elapsed_s": float(self.elapsed_s),
        }
        return out


# --------------------------------------------------------------------------- #
# Fallback: compute metrics from a raw 1-D attention distribution             #
# --------------------------------------------------------------------------- #


@beartype
def compute_metrics_from_attention_pattern(
    pattern: Float[np.ndarray, "t"],
) -> dict[str, float]:
    """Compute the 5 analysis metrics from a single 1-D attention row.

    This exists as a sanity helper for downstream callers that happen to have
    a raw attention vector and want to compute the same metrics we read from
    HDF5. The primary pipeline does NOT call this — metrics are precomputed
    online during extraction (see ``s1s2.extract.hooks._row_metrics``).

    The definitions here MUST match the extraction pipeline. We reuse
    :func:`s1s2.utils.stats.gini_coefficient` so the definition lives in one
    place.

    Returns
    -------
    dict with keys: entropy, entropy_normalized, gini, max_attn, focus_5
    """
    pat = np.clip(pattern.astype(np.float64, copy=False), 0.0, None)
    s = pat.sum()
    if s <= 0 or pat.size == 0:
        return dict.fromkeys(METRIC_NAMES, 0.0)
    probs = pat / s
    t = probs.size
    safe = np.clip(probs, 1e-12, 1.0)
    entropy_bits = float(-np.sum(safe * np.log2(safe)))
    max_entropy = float(np.log2(t)) if t > 1 else 1.0
    entropy_normalized = float(entropy_bits / max_entropy) if max_entropy > 0 else 0.0
    gini = gini_coefficient(probs.astype(np.float64))
    max_attn = float(np.max(probs))
    k = min(5, t)
    # Use -sort trick to get top-k without full sort
    top_k = np.partition(probs, -k)[-k:]
    focus_5 = float(np.sum(top_k))
    return {
        "entropy": entropy_bits,
        "entropy_normalized": entropy_normalized,
        "gini": gini,
        "max_attn": max_attn,
        "focus_5": focus_5,
    }


# --------------------------------------------------------------------------- #
# BH-FDR across an arbitrary 1-D vector                                       #
# --------------------------------------------------------------------------- #


@beartype
def bh_fdr_joint(
    pvalues: Float[np.ndarray, "n"], q: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Thin wrapper around :func:`s1s2.utils.stats.bh_fdr`.

    Kept local so callers of this module don't have to import from utils.
    """
    from s1s2.utils.stats import bh_fdr

    return bh_fdr(pvalues, q=q)


# --------------------------------------------------------------------------- #
# HDF5 loader                                                                 #
# --------------------------------------------------------------------------- #


@beartype
def _resolve_model_config(
    model_config_key: str, models_yaml_path: str | Path | None = None
) -> dict[str, Any]:
    """Look up a model's entry in ``configs/models.yaml``.

    We do this without Hydra so loaders are importable without a Hydra
    context. Returns a dict with at least: hdf5_key, family, n_layers, n_heads,
    n_kv_heads, is_reasoning, and optionally sliding_window.
    """
    import yaml  # type: ignore[import-untyped]

    if models_yaml_path is None:
        # Default: look next to this source file in the configs/ dir.
        here = Path(__file__).resolve()
        candidate = here.parents[3] / "configs" / "models.yaml"
        models_yaml_path = candidate
    with open(models_yaml_path) as fh:
        data = yaml.safe_load(fh)
    entry = data.get("models", {}).get(model_config_key)
    if entry is None:
        raise KeyError(
            f"model_config_key={model_config_key!r} not in models.yaml "
            f"at {models_yaml_path}"
        )
    return entry


@beartype
def load_model_attention_data(
    f: h5py.File,
    model_config_key: str,
    selected_positions: tuple[str, ...] | list[str],
    metrics_to_load: tuple[str, ...] | list[str] = METRIC_NAMES,
    models_yaml_path: str | Path | None = None,
) -> ModelAttentionData:
    """Load one model's attention metrics + conflict labels from HDF5.

    Parameters
    ----------
    f : open HDF5 handle (from ``ioh.open_activations``)
    model_config_key : e.g. ``"llama-3.1-8b-instruct"``
    selected_positions : subset of position labels to analyze. Invalid
        positions for the model (e.g. T0 on a non-reasoning model) are
        silently dropped.
    metrics_to_load : which of the 5 metrics to load into memory.
    models_yaml_path : optional override for the models.yaml location
        (testing).
    """
    cfg = _resolve_model_config(model_config_key, models_yaml_path=models_yaml_path)
    model_key = cfg["hdf5_key"]
    if model_key not in f["/models"]:
        raise KeyError(
            f"model {model_key!r} not in activation file; available: "
            f"{list(f['/models'].keys())}"
        )
    labels = ioh.position_labels(f, model_key)
    valid = ioh.position_valid(f, model_key)

    # Keep only positions that are (a) requested, (b) present on disk, and
    # (c) valid for at least one problem (so that non-reasoning models don't
    # report garbage for T-positions).
    kept_positions: list[str] = []
    kept_indices: list[int] = []
    for pos in selected_positions:
        if pos not in labels:
            logger.debug("position %s not in labels for %s; skipping", pos, model_key)
            continue
        idx = labels.index(pos)
        if not bool(valid[:, idx].any()):
            logger.debug("position %s marked invalid for %s; skipping", pos, model_key)
            continue
        kept_positions.append(pos)
        kept_indices.append(idx)

    if not kept_positions:
        raise ValueError(
            f"no requested positions are valid for {model_key} "
            f"(requested={list(selected_positions)}, have={labels})"
        )

    conflict = f["/problems/conflict"][:].astype(bool)

    metrics: dict[str, np.ndarray] = {}
    for m in metrics_to_load:
        arr = ioh.get_attention_metric(f, model_key, m)
        # Shape: (n_problems, n_layers, n_heads, n_positions)
        if arr.ndim != 4:
            raise ValueError(
                f"attention metric {m!r} for {model_key} has unexpected "
                f"shape {arr.shape}"
            )
        metrics[m] = arr[..., kept_indices].astype(np.float32, copy=False)

    meta_attrs = ioh.model_metadata(f, model_key)
    n_layers = int(meta_attrs.get("n_layers", cfg["n_layers"]))
    n_heads = int(meta_attrs.get("n_heads", cfg["n_heads"]))
    n_kv_heads = int(meta_attrs.get("n_kv_heads", cfg["n_kv_heads"]))
    is_reasoning = bool(meta_attrs.get("is_reasoning_model", cfg.get("is_reasoning", False)))

    return ModelAttentionData(
        model_key=model_key,
        model_config_key=model_config_key,
        family=str(cfg.get("family", "unknown")),
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        is_reasoning_model=is_reasoning,
        position_labels=labels,
        selected_positions=kept_positions,
        metrics=metrics,
        conflict=conflict,
        sliding_window=(
            int(cfg["sliding_window"]) if "sliding_window" in cfg else None
        ),
    )


# --------------------------------------------------------------------------- #
# Gemma-2 layer partition                                                     #
# --------------------------------------------------------------------------- #


@beartype
def gemma_layer_partition(n_layers: int) -> dict[str, list[int]]:
    """Return ``{"global": even_layers, "sliding_window": odd_layers}``.

    Gemma-2 alternates: layer 0 is global, layer 1 is sliding-window, etc.
    Callers that want to analyze Gemma must split on this mask — pooling
    odd and even layers conflates fundamentally different attention kernels.
    """
    evens = list(range(0, n_layers, 2))
    odds = list(range(1, n_layers, 2))
    return {"global": evens, "sliding_window": odds}


@beartype
def is_gemma_family(family: str) -> bool:
    return family.lower().startswith("gemma")


# --------------------------------------------------------------------------- #
# Main per-model entry point                                                  #
# --------------------------------------------------------------------------- #


@beartype
def analyze_model(
    data: ModelAttentionData, config: AttentionConfig
) -> ModelAttentionReport:
    """Run the full per-model analysis pipeline.

    This function is pure: it takes a :class:`ModelAttentionData` and a
    :class:`AttentionConfig` and returns a :class:`ModelAttentionReport`. No
    IO side-effects. The CLI layer handles persistence.
    """
    from s1s2.attention.heads import (
        classify_heads,
        head_classifications_to_records,
        kv_group_classify,
        run_all_head_differential_tests,
    )
    from s1s2.attention.layers import layer_summary
    from s1s2.attention.trajectories import compute_trajectories

    t0 = time.time()
    config.validate()

    # --- Per-head tests across all (layer, head, position, metric) -----
    df = run_all_head_differential_tests(
        data,
        metrics=tuple(config.metrics),
    )

    # Apply BH-FDR across ALL tests for this model (heads x metrics x positions)
    df = _apply_bh_in_place(df, q=config.fdr_q)

    # --- Consensus head classification ---------------------------------
    head_classifs = classify_heads(
        df,
        n_layers=data.n_layers,
        n_heads=data.n_heads,
        min_significant=config.multi_metric_consensus,
        entropy_effect_threshold=config.effect_size_threshold,
    )
    head_records = head_classifications_to_records(head_classifs)

    # --- KV-group classification ---------------------------------------
    kv_records: list[dict[str, Any]] | None = None
    if config.report_kv_group_aggregate and data.group_size > 1:
        kv_classifs = kv_group_classify(
            data,
            metrics=tuple(config.metrics),
            q=config.fdr_q,
            min_significant=config.multi_metric_consensus,
            entropy_effect_threshold=config.effect_size_threshold,
        )
        kv_records = [asdict(c) for c in kv_classifs]

    # --- Layer aggregation ---------------------------------------------
    layer_recs = layer_summary(
        data,
        head_classifs=head_classifs,
        metric="entropy",
    )

    # --- Temporal trajectories -----------------------------------------
    traj = compute_trajectories(
        data,
        metric="entropy",
    )

    # --- Gemma layer partition (documented, not enforced: pooling is
    #     done inside layer/head analyses already, but we report which
    #     layers are which so downstream viz can respect it) -------------
    gemma_partition: dict[str, list[int]] | None = None
    if config.gemma_separate_window_layers and is_gemma_family(data.family):
        gemma_partition = gemma_layer_partition(data.n_layers)

    elapsed = float(time.time() - t0)

    report = ModelAttentionReport(
        model_key=data.model_key,
        model_config_key=data.model_config_key,
        family=data.family,
        n_layers=data.n_layers,
        n_heads=data.n_heads,
        n_kv_heads=data.n_kv_heads,
        is_reasoning_model=data.is_reasoning_model,
        selected_positions=list(data.selected_positions),
        n_problems=int(data.n_problems),
        n_conflict=int(data.conflict.sum()),
        n_no_conflict=int((~data.conflict).sum()),
        head_classifications=head_records,
        kv_group_classifications=kv_records,
        layer_summary=layer_recs,
        trajectories=traj,
        gemma_layer_types=gemma_partition,
        config={
            "metrics": list(config.metrics),
            "positions": list(data.selected_positions),
            "fdr_q": float(config.fdr_q),
            "effect_size_threshold": float(config.effect_size_threshold),
            "multi_metric_consensus": int(config.multi_metric_consensus),
            "report_kv_group_aggregate": bool(config.report_kv_group_aggregate),
            "gemma_separate_window_layers": bool(config.gemma_separate_window_layers),
            "seed": int(config.seed),
        },
        elapsed_s=elapsed,
        differential_tests_df=df,
    )
    logger.info(
        "analyze_model(%s) done: %d heads, %d layers, %d positions, %d problems, %.2fs",
        data.model_key,
        data.n_layers * data.n_heads,
        data.n_layers,
        data.n_selected_positions,
        data.n_problems,
        elapsed,
    )
    return report


def _apply_bh_in_place(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """Attach ``q_value`` and ``significant`` columns to the differential-test df.

    BH-FDR is applied JOINTLY across all rows (heads x metrics x positions).
    This is the correct correction granularity for exploratory head hunting.
    """
    if df.empty:
        df["q_value"] = np.array([], dtype=np.float64)
        df["significant"] = np.array([], dtype=bool)
        return df
    pvals = df["p_value"].to_numpy(dtype=np.float64)
    rejected, qvals = bh_fdr_joint(pvals, q=q)
    df = df.copy()
    df["q_value"] = qvals
    df["significant"] = rejected
    return df


# --------------------------------------------------------------------------- #
# Multi-model driver                                                           #
# --------------------------------------------------------------------------- #


@beartype
def analyze_all_models(
    activations_path: str | Path,
    model_config_keys: list[str],
    config: AttentionConfig,
) -> dict[str, ModelAttentionReport]:
    """Analyze multiple models from a single activation file."""
    out: dict[str, ModelAttentionReport] = {}
    with ioh.open_activations(activations_path) as f:
        for key in model_config_keys:
            logger.info("loading %s", key)
            data = load_model_attention_data(
                f,
                model_config_key=key,
                selected_positions=config.positions,
                metrics_to_load=config.metrics,
            )
            out[key] = analyze_model(data, config)
    return out


# --------------------------------------------------------------------------- #
# Matched-pair cross-model comparison                                         #
# --------------------------------------------------------------------------- #


@beartype
def compare_matched_pair(
    data_a: ModelAttentionData,
    data_b: ModelAttentionData,
    metric: str = "entropy",
    position: str | None = None,
) -> dict[str, Any]:
    """Compare S1-S2 differentials between two architecture-matched models.

    For each (layer, head), compute::

        delta_a = mean(metric[conflict]) - mean(metric[~conflict])  # for model A
        delta_b = mean(metric[conflict]) - mean(metric[~conflict])  # for model B

    Returns a dict with per-head arrays so downstream viz
    (:func:`s1s2.attention.viz.plot_cross_model_scatter`) can scatter delta_a
    vs delta_b and highlight heads that flipped after distillation.

    Requires matching architectures (same ``n_layers`` and ``n_heads``).
    """
    if (data_a.n_layers, data_a.n_heads) != (data_b.n_layers, data_b.n_heads):
        raise ValueError(
            f"matched-pair comparison requires same architecture; "
            f"{data_a.model_key}={data_a.n_layers}x{data_a.n_heads} vs "
            f"{data_b.model_key}={data_b.n_layers}x{data_b.n_heads}"
        )
    if metric not in data_a.metrics or metric not in data_b.metrics:
        raise KeyError(f"metric {metric!r} not loaded for one of the models")

    # Pick a position that's valid for both. If caller passes None, prefer P0
    # (always valid), else the first position valid for both.
    common = [p for p in data_a.selected_positions if p in data_b.selected_positions]
    if position is None:
        position = "P0" if "P0" in common else (common[0] if common else None)
    if position is None or position not in common:
        raise ValueError(
            f"no common valid position between {data_a.model_key} and "
            f"{data_b.model_key}; wanted {position}, common={common}"
        )
    idx_a = data_a.selected_positions.index(position)
    idx_b = data_b.selected_positions.index(position)

    m_a = data_a.metrics[metric][..., idx_a]  # (n_problems, L, H)
    m_b = data_b.metrics[metric][..., idx_b]

    conflict_a = data_a.conflict
    conflict_b = data_b.conflict

    delta_a = (
        m_a[conflict_a].mean(axis=0) - m_a[~conflict_a].mean(axis=0)
    )  # (L, H)
    delta_b = (
        m_b[conflict_b].mean(axis=0) - m_b[~conflict_b].mean(axis=0)
    )

    return {
        "model_a": data_a.model_key,
        "model_b": data_b.model_key,
        "metric": metric,
        "position": position,
        "n_layers": int(data_a.n_layers),
        "n_heads": int(data_a.n_heads),
        "delta_a": delta_a.astype(np.float32).tolist(),
        "delta_b": delta_b.astype(np.float32).tolist(),
    }


# --------------------------------------------------------------------------- #
# Serialization                                                               #
# --------------------------------------------------------------------------- #


@beartype
def save_model_report(
    report: ModelAttentionReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write ``report`` to ``{output_dir}/{model_key}/``.

    Writes:
    - ``head_classifications.json``
    - ``layer_summary.json``
    - ``trajectories.json``
    - ``differential_tests.parquet``
    - ``report.json`` (the full asdict-style report minus the df)

    Returns a dict of ``{name: Path}`` for the files written.
    """
    out_dir = Path(output_dir) / report.model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    p = out_dir / "head_classifications.json"
    with p.open("w") as fh:
        json.dump(report.head_classifications, fh, indent=2)
    paths["head_classifications"] = p

    p = out_dir / "layer_summary.json"
    with p.open("w") as fh:
        json.dump(report.layer_summary, fh, indent=2)
    paths["layer_summary"] = p

    p = out_dir / "trajectories.json"
    with p.open("w") as fh:
        json.dump(report.trajectories, fh, indent=2)
    paths["trajectories"] = p

    p = out_dir / "report.json"
    with p.open("w") as fh:
        json.dump(report.to_json(), fh, indent=2)
    paths["report"] = p

    if report.differential_tests_df is not None and not report.differential_tests_df.empty:
        p = out_dir / "differential_tests.parquet"
        try:
            report.differential_tests_df.to_parquet(p, index=False)
        except (ImportError, ValueError) as exc:
            logger.warning("parquet write failed (%s); falling back to CSV", exc)
            p = out_dir / "differential_tests.csv"
            report.differential_tests_df.to_csv(p, index=False)
        paths["differential_tests"] = p

    if report.kv_group_classifications is not None:
        p = out_dir / "kv_group_classifications.json"
        with p.open("w") as fh:
            json.dump(report.kv_group_classifications, fh, indent=2)
        paths["kv_group_classifications"] = p

    return paths
