"""Tests for the attention entropy workstream.

Covers:
- :func:`differential_test_vector` returning a sane dict on a planted signal
- :func:`run_head_differential_tests_for_metric` vectorisation correctness
- BH-FDR application across many tests via :func:`bh_fdr_joint`
- Head consensus classification (S1 / S2 / mixed / non-specialized)
- GQA non-independence: KV-group median pooling
- Gemma-2 sliding-window layer separation
- ``compute_metrics_from_attention_pattern`` round-trip with the
  extraction-side ``_row_metrics`` definition
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# OpenMP guard for macOS — torch + numpy can crash without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Allow tests to import the package without `pip install -e .`.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from s1s2.attention.core import (  # noqa: E402
    METRIC_NAMES,
    ModelAttentionData,
    _apply_bh_in_place,
    bh_fdr_joint,
    compute_metrics_from_attention_pattern,
    gemma_layer_partition,
    is_gemma_family,
)
from s1s2.attention.heads import (  # noqa: E402
    classify_heads,
    differential_test_vector,
    head_classifications_to_records,
    kv_group_classify,
    kv_group_median_pool,
    run_all_head_differential_tests,
    run_head_differential_tests_for_metric,
)
from s1s2.attention.layers import layer_metric_array, layer_summary  # noqa: E402
from s1s2.attention.trajectories import (  # noqa: E402
    compute_trajectories,
    select_t_positions,
    trajectory_features,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _build_synth_data(
    *,
    n_problems: int = 40,
    n_layers: int = 4,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    n_positions: int = 2,
    family: str = "llama",
    is_reasoning: bool = False,
    seed: int = 0,
    plant_layer: int | None = 1,
    plant_head: int | None = 0,
    plant_strength: float = 0.6,
) -> ModelAttentionData:
    """Build a :class:`ModelAttentionData` with an optional planted signal.

    By default plants a +0.6 mean shift in entropy on (layer 1, head 0)
    for conflict items so the per-head differential test will pick it up.
    """
    rng = np.random.default_rng(seed)
    conflict = np.zeros(n_problems, dtype=bool)
    conflict[: n_problems // 2] = True

    shape = (n_problems, n_layers, n_heads, n_positions)
    base_entropy = rng.uniform(0.5, 2.5, size=shape).astype(np.float32)
    if plant_layer is not None and plant_head is not None:
        base_entropy[conflict, plant_layer, plant_head, :] += plant_strength

    entropy_norm = np.clip(base_entropy / np.log2(32.0), 0.0, 1.0).astype(np.float32)
    gini = np.clip(1.0 - entropy_norm, 0.0, 1.0).astype(np.float32)
    max_attn = np.clip(0.9 * gini + 0.05, 0.0, 1.0).astype(np.float32)
    focus_5 = np.clip(0.95 * gini + 0.05, 0.0, 1.0).astype(np.float32)

    return ModelAttentionData(
        model_key="synth",
        model_config_key="synth",
        family=family,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        is_reasoning_model=is_reasoning,
        position_labels=["P0", "P2"],
        selected_positions=["P0", "P2"][:n_positions],
        metrics={
            "entropy": base_entropy,
            "entropy_normalized": entropy_norm,
            "gini": gini,
            "max_attn": max_attn,
            "focus_5": focus_5,
        },
        conflict=conflict,
    )


# --------------------------------------------------------------------------- #
# compute_metrics_from_attention_pattern                                       #
# --------------------------------------------------------------------------- #


class TestComputeMetricsFromPattern:
    def test_uniform_distribution(self):
        pat = np.ones(8) / 8
        m = compute_metrics_from_attention_pattern(pat)
        assert m["entropy"] == pytest.approx(3.0, abs=1e-6)  # log2(8)
        assert m["entropy_normalized"] == pytest.approx(1.0, abs=1e-6)
        assert m["max_attn"] == pytest.approx(1 / 8)

    def test_point_mass(self):
        pat = np.zeros(10)
        pat[3] = 1.0
        m = compute_metrics_from_attention_pattern(pat)
        assert m["entropy"] == pytest.approx(0.0, abs=1e-6)
        assert m["max_attn"] == pytest.approx(1.0)

    def test_zero_pattern(self):
        pat = np.zeros(10)
        m = compute_metrics_from_attention_pattern(pat)
        assert m["entropy"] == 0.0
        assert m["max_attn"] == 0.0

    def test_unnormalized_input_normalises(self):
        pat = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_metrics_from_attention_pattern(pat)
        # Should produce non-negative entropy ≤ log2(4) = 2
        assert 0.0 <= m["entropy"] <= 2.0
        assert m["max_attn"] == pytest.approx(0.4, abs=1e-6)


# --------------------------------------------------------------------------- #
# differential_test_vector                                                     #
# --------------------------------------------------------------------------- #


class TestDifferentialTestVector:
    def test_obvious_signal_detected(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(size=80)
        conflict = np.zeros(80, dtype=bool)
        conflict[:40] = True
        vals[:40] += 2.0  # plant a strong signal
        out = differential_test_vector(vals.astype(np.float32), conflict)
        assert out["p_value"] < 0.001
        assert out["effect_size_rb"] > 0.5  # strong positive
        assert out["mean_conflict"] > out["mean_noconflict"]

    def test_null_signal_not_significant(self):
        rng = np.random.default_rng(1)
        vals = rng.normal(size=80).astype(np.float32)
        conflict = np.zeros(80, dtype=bool)
        conflict[:40] = True
        out = differential_test_vector(vals, conflict)
        assert out["p_value"] > 0.01
        assert abs(out["effect_size_rb"]) < 0.5

    def test_empty_group_returns_nan_dict(self):
        vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        conflict = np.zeros(3, dtype=bool)  # all False -> empty conflict group
        out = differential_test_vector(vals, conflict)
        assert out["p_value"] == 1.0
        assert out["n_conflict"] == 0
        assert out["n_noconflict"] == 3

    def test_constant_values(self):
        vals = np.full(20, 2.5, dtype=np.float32)
        conflict = np.array([True] * 10 + [False] * 10)
        out = differential_test_vector(vals, conflict)
        # All identical -> degenerate, returns p=1 and effect_size=0
        assert out["p_value"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# run_head_differential_tests_for_metric                                       #
# --------------------------------------------------------------------------- #


class TestRunHeadDifferentialTests:
    def test_planted_head_significant(self):
        data = _build_synth_data(n_problems=80, plant_layer=1, plant_head=0)
        df = run_head_differential_tests_for_metric(
            data.metrics["entropy"],
            data.conflict,
            positions=list(data.selected_positions),
            metric_name="entropy",
            group_size=data.group_size,
        )
        # The planted head (layer=1, head=0) should be significant before FDR.
        target = df[(df["layer"] == 1) & (df["head"] == 0)]
        assert len(target) > 0
        assert (target["p_value"] < 0.05).any()

    def test_non_planted_heads_mostly_null(self):
        data = _build_synth_data(n_problems=80, plant_layer=None, plant_head=None)
        df = run_head_differential_tests_for_metric(
            data.metrics["entropy"],
            data.conflict,
            positions=list(data.selected_positions),
            metric_name="entropy",
            group_size=data.group_size,
        )
        # Under H0 most p-values should not be tiny
        assert (df["p_value"] > 0.05).mean() > 0.5

    def test_all_metrics_run(self):
        data = _build_synth_data(n_problems=60)
        df = run_all_head_differential_tests(data, metrics=METRIC_NAMES)
        assert set(df["metric"]) == set(METRIC_NAMES)
        assert len(df) == data.n_layers * data.n_heads * data.n_selected_positions * len(
            METRIC_NAMES
        )

    def test_kv_group_assignment(self):
        """With 4 heads and 2 KV heads, group_size=2 -> heads 0,1 -> group 0."""
        data = _build_synth_data(n_problems=40, n_heads=4, n_kv_heads=2)
        df = run_head_differential_tests_for_metric(
            data.metrics["entropy"],
            data.conflict,
            positions=list(data.selected_positions),
            metric_name="entropy",
            group_size=data.group_size,
        )
        head0 = df[(df["layer"] == 0) & (df["head"] == 0)].iloc[0]
        head1 = df[(df["layer"] == 0) & (df["head"] == 1)].iloc[0]
        head2 = df[(df["layer"] == 0) & (df["head"] == 2)].iloc[0]
        assert head0["kv_group"] == 0
        assert head1["kv_group"] == 0  # same group as head0
        assert head2["kv_group"] == 1


# --------------------------------------------------------------------------- #
# BH-FDR joint application                                                     #
# --------------------------------------------------------------------------- #


class TestBHJoint:
    def test_bh_fdr_joint_runs(self):
        pvals = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
        rejected, qvals = bh_fdr_joint(pvals, q=0.05)
        assert rejected[0]
        assert not rejected[-1]
        assert qvals.shape == pvals.shape

    def test_apply_bh_in_place_attaches_columns(self):
        df = pd.DataFrame(
            {
                "layer": [0, 0, 1, 1],
                "head": [0, 1, 0, 1],
                "position": ["P0"] * 4,
                "metric": ["entropy"] * 4,
                "p_value": [0.001, 0.01, 0.5, 0.9],
                "u_statistic": [0.0] * 4,
                "effect_size_rb": [0.5, 0.4, 0.0, 0.0],
                "cohens_d": [0.0] * 4,
                "mean_conflict": [0.0] * 4,
                "mean_noconflict": [0.0] * 4,
                "median_conflict": [0.0] * 4,
                "median_noconflict": [0.0] * 4,
                "n_conflict": [10.0] * 4,
                "n_noconflict": [10.0] * 4,
                "kv_group": [0, 0, 0, 0],
                "direction": np.array([1, 1, 0, 0], dtype=np.int8),
            }
        )
        out = _apply_bh_in_place(df, q=0.05)
        assert "q_value" in out.columns
        assert "significant" in out.columns
        # First row's tiny p should be flagged
        assert bool(out.iloc[0]["significant"]) is True

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame(columns=["p_value"])
        out = _apply_bh_in_place(df, q=0.05)
        assert "q_value" in out.columns
        assert "significant" in out.columns


# --------------------------------------------------------------------------- #
# classify_heads                                                               #
# --------------------------------------------------------------------------- #


class TestClassifyHeads:
    def test_planted_head_classified(self):
        """Boost entropy on (layer 1, head 0) on every metric so that all
        five metrics flag it; classification should land at S2_specialized."""
        n_problems = 100
        n_layers = 3
        n_heads = 4
        n_positions = 2
        rng = np.random.default_rng(0)
        conflict = np.zeros(n_problems, dtype=bool)
        conflict[: n_problems // 2] = True
        shape = (n_problems, n_layers, n_heads, n_positions)
        # Build a metric where conflict items have a HUGE shift on (1,0)
        # for entropy/entropy_normalized (S2 direction = +1) and a HUGE
        # negative shift on gini/max_attn/focus_5 (S2 direction = -1).
        ent = rng.uniform(0.5, 2.5, size=shape).astype(np.float32)
        ent[conflict, 1, 0, :] += 1.5
        ent_norm = ent / np.log2(32.0)
        gini = 1.0 - np.clip(ent_norm, 0.0, 1.0)
        max_attn = 0.5 * gini + 0.1
        focus_5 = 0.5 * gini + 0.1

        data = ModelAttentionData(
            model_key="m",
            model_config_key="m",
            family="llama",
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_heads,  # group_size=1 for this test
            is_reasoning_model=False,
            position_labels=["P0", "P2"],
            selected_positions=["P0", "P2"],
            metrics={
                "entropy": ent.astype(np.float32),
                "entropy_normalized": ent_norm.astype(np.float32),
                "gini": gini.astype(np.float32),
                "max_attn": max_attn.astype(np.float32),
                "focus_5": focus_5.astype(np.float32),
            },
            conflict=conflict,
        )
        df = run_all_head_differential_tests(data, metrics=METRIC_NAMES)
        df = _apply_bh_in_place(df, q=0.05)
        classifs = classify_heads(
            df,
            n_layers=n_layers,
            n_heads=n_heads,
            min_significant=3,
            entropy_effect_threshold=0.3,
        )
        target = next(c for c in classifs if c.layer == 1 and c.head == 0)
        assert target.classification == "S2_specialized"
        assert target.n_significant_metrics >= 3

    def test_non_specialized_head(self):
        """Pure noise -> classification is non_specialized."""
        data = _build_synth_data(
            n_problems=60, plant_layer=None, plant_head=None, n_kv_heads=4
        )
        df = run_all_head_differential_tests(data, metrics=METRIC_NAMES)
        df = _apply_bh_in_place(df, q=0.05)
        classifs = classify_heads(
            df,
            n_layers=data.n_layers,
            n_heads=data.n_heads,
            min_significant=3,
            entropy_effect_threshold=0.3,
        )
        # The vast majority should be non_specialized.
        non_spec = sum(1 for c in classifs if c.classification == "non_specialized")
        assert non_spec >= len(classifs) * 0.7

    def test_records_serialization(self):
        data = _build_synth_data(n_problems=30, n_kv_heads=2)
        df = run_all_head_differential_tests(data, metrics=METRIC_NAMES)
        df = _apply_bh_in_place(df, q=0.05)
        classifs = classify_heads(
            df, n_layers=data.n_layers, n_heads=data.n_heads
        )
        records = head_classifications_to_records(classifs)
        # Records are dicts -> JSON-friendly
        import json
        s = json.dumps(records)
        assert isinstance(s, str)


# --------------------------------------------------------------------------- #
# GQA: KV-group median pooling                                                 #
# --------------------------------------------------------------------------- #


class TestKVGroupPooling:
    def test_median_pool_shape(self):
        """4 query heads / 2 KV groups -> result has 2 grouped 'heads'."""
        rng = np.random.default_rng(0)
        arr = rng.normal(size=(10, 3, 4, 2)).astype(np.float32)
        pooled = kv_group_median_pool(arr, group_size=2)
        assert pooled.shape == (10, 3, 2, 2)

    def test_median_pool_values(self):
        # Heads in same KV group: median should land in the middle.
        arr = np.array(
            [[[[1.0], [3.0], [10.0], [20.0]]]], dtype=np.float32
        )  # shape (1, 1, 4, 1)
        pooled = kv_group_median_pool(arr, group_size=2)
        assert pooled.shape == (1, 1, 2, 1)
        # Group 0: median([1, 3]) = 2; group 1: median([10, 20]) = 15
        assert pooled[0, 0, 0, 0] == pytest.approx(2.0)
        assert pooled[0, 0, 1, 0] == pytest.approx(15.0)

    def test_pool_size_one_is_identity(self):
        arr = np.random.randn(2, 3, 4, 1).astype(np.float32)
        pooled = kv_group_median_pool(arr, group_size=1)
        np.testing.assert_array_equal(pooled, arr)

    def test_indivisible_raises(self):
        arr = np.zeros((2, 1, 5, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="not divisible"):
            kv_group_median_pool(arr, group_size=2)

    def test_kv_group_classify_collapses_to_n_kv_heads(self):
        """4 query heads + 2 KV heads -> classification returns 2 entries per layer."""
        data = _build_synth_data(
            n_problems=60, n_heads=4, n_kv_heads=2, plant_layer=1, plant_head=0
        )
        classifs = kv_group_classify(data, metrics=METRIC_NAMES, q=0.05)
        # One classification per (layer, kv-group). Total = n_layers * n_kv_heads
        assert len(classifs) == data.n_layers * data.n_kv_heads
        # The "head" field on a kv_group_classify result is the KV-group index
        kv_indices = sorted({c.head for c in classifs})
        assert kv_indices == list(range(data.n_kv_heads))


# --------------------------------------------------------------------------- #
# Gemma sliding-window separation                                              #
# --------------------------------------------------------------------------- #


class TestGemmaSlidingWindow:
    def test_layer_partition_alternates(self):
        part = gemma_layer_partition(n_layers=6)
        assert part["global"] == [0, 2, 4]
        assert part["sliding_window"] == [1, 3, 5]

    def test_layer_partition_odd_total(self):
        part = gemma_layer_partition(n_layers=5)
        assert part["global"] == [0, 2, 4]
        assert part["sliding_window"] == [1, 3]

    def test_gemma_family_detection(self):
        assert is_gemma_family("gemma-2") is True
        assert is_gemma_family("gemma") is True
        assert is_gemma_family("llama") is False
        assert is_gemma_family("Gemma-2-9b") is True

    def test_layer_summary_marks_gemma_layer_types(self):
        """Layer summary on a gemma model has layer_type 'global'/'sliding_window'."""
        data = _build_synth_data(
            n_problems=40, n_layers=4, n_heads=2, n_kv_heads=2,
            family="gemma-2", plant_layer=None, plant_head=None,
        )
        df = run_all_head_differential_tests(data, metrics=METRIC_NAMES)
        df = _apply_bh_in_place(df, q=0.05)
        classifs = classify_heads(
            df, n_layers=data.n_layers, n_heads=data.n_heads
        )
        summary = layer_summary(data, head_classifs=classifs, metric="entropy")
        # Even layers -> global; odd layers -> sliding_window
        assert summary[0]["layer_type"] == "global"
        assert summary[1]["layer_type"] == "sliding_window"
        assert summary[2]["layer_type"] == "global"
        assert summary[3]["layer_type"] == "sliding_window"

    def test_layer_summary_standard_for_llama(self):
        data = _build_synth_data(
            n_problems=40, family="llama",
            plant_layer=None, plant_head=None,
        )
        df = run_all_head_differential_tests(data, metrics=METRIC_NAMES)
        df = _apply_bh_in_place(df, q=0.05)
        classifs = classify_heads(
            df, n_layers=data.n_layers, n_heads=data.n_heads
        )
        summary = layer_summary(data, head_classifs=classifs, metric="entropy")
        for s in summary:
            assert s["layer_type"] == "standard"


# --------------------------------------------------------------------------- #
# Layer aggregation                                                            #
# --------------------------------------------------------------------------- #


class TestLayerAggregation:
    def test_layer_metric_array_default(self):
        data = _build_synth_data(n_problems=20)
        arr = layer_metric_array(data, metric="entropy")
        assert arr.shape == (data.n_problems, data.n_layers, data.n_heads)

    def test_layer_metric_array_specific_position(self):
        data = _build_synth_data(n_problems=20, n_positions=2)
        arr = layer_metric_array(data, metric="entropy", position="P0")
        assert arr.shape == (data.n_problems, data.n_layers, data.n_heads)

    def test_layer_metric_array_unknown_metric(self):
        data = _build_synth_data(n_problems=20)
        with pytest.raises(KeyError, match="metric"):
            layer_metric_array(data, metric="ghost")

    def test_layer_metric_array_unknown_position(self):
        data = _build_synth_data(n_problems=20, n_positions=2)
        with pytest.raises(KeyError, match="position"):
            layer_metric_array(data, metric="entropy", position="Tend")


# --------------------------------------------------------------------------- #
# Trajectories                                                                 #
# --------------------------------------------------------------------------- #


class TestTrajectories:
    def test_select_t_positions_subset(self):
        kept, idx = select_t_positions(["P0", "T0", "T50", "Tend"])
        assert kept == ["T0", "T50", "Tend"]
        assert idx == [1, 2, 3]

    def test_select_t_positions_none(self):
        kept, idx = select_t_positions(["P0", "P2"])
        assert kept == []
        assert idx == []

    def test_trajectory_features_slope(self):
        # Increasing trajectory -> positive slope.
        traj = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        feats = trajectory_features(traj)
        assert feats["slope"][0] > 0
        assert feats["slope"][1] < 0

    def test_compute_trajectories_unavailable_with_p_only(self):
        data = _build_synth_data(n_problems=20, n_positions=2)
        out = compute_trajectories(data, metric="entropy")
        assert out["available"] is False

    def test_compute_trajectories_with_t_positions(self):
        rng = np.random.default_rng(0)
        n_problems, n_layers, n_heads = 20, 2, 2
        n_positions = 5  # P0, P2, T0, T25, T50
        positions = ["P0", "P2", "T0", "T25", "T50"]
        shape = (n_problems, n_layers, n_heads, n_positions)
        ent = rng.uniform(size=shape).astype(np.float32)
        gini = (1.0 - ent).astype(np.float32)
        data = ModelAttentionData(
            model_key="m",
            model_config_key="m",
            family="llama",
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            is_reasoning_model=True,
            position_labels=positions,
            selected_positions=positions,
            metrics={
                "entropy": ent,
                "entropy_normalized": ent,
                "gini": gini,
                "max_attn": gini,
                "focus_5": gini,
            },
            conflict=np.array([True, False] * (n_problems // 2)),
        )
        out = compute_trajectories(data, metric="entropy")
        assert out["available"] is True
        assert out["t_positions"] == ["T0", "T25", "T50"]
        assert out["n_t"] == 3
        assert "per_head" in out
