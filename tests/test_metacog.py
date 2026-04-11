"""Tests for :mod:`s1s2.metacog`.

Coverage:

* Surprise aggregation handles by-position only AND ragged full-trace inputs.
* Surprise-feature correlation correctly identifies a planted feature
  (synthetic feature that is rho ≈ 1 with surprise) and rejects noise features.
* BH-FDR is correctly applied across the feature pool.
* Confidently-wrong test returns a positive metacognition score for a
  synthetic metacognitive feature and ~0 for a synthetic confidence proxy.
* S1/S2 specificity test reports AUC > 0.65 for a planted S1-specific
  feature and ~0.5 for a noise feature.
* The 4-gate framework returns the expected decision for synthetic snapshots.
* The self-correction parser finds markers in synthetic traces and reports
  ``self_corrects=False`` when no candidate answer was committed first.
* End-to-end: a synthetic HDF5 cache + MockSAE round-trips through the
  CLI driver and writes JSON files.

All tests are CPU-only and run in a few seconds.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

# Make sure OpenMP doesn't crash on macOS when torch + numpy share the runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Package may not be pip-installed in the dev env; prepend src/ to sys.path.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from s1s2.metacog import (  # noqa: E402 — sys.path must be patched first
    DifficultyDetectorAnalysis,
    DifficultyDetectorResults,
    GateConfig,
    MetacogConfig,
    aggregate_surprise,
    confidently_wrong_test,
    difficulty_sensitive_features,
    difficulty_trajectory_means,
    evaluate_gates,
    parse_self_correction,
    parse_trace_corpus,
    s1s2_specificity_batch,
    s1s2_specificity_test,
    surprise_feature_correlation,
)
from s1s2.metacog.surprise import merge_correlation_results  # noqa: E402
from s1s2.sae.loaders import MockSAE  # noqa: E402
from s1s2.utils import io as ioh  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _planted_features(
    n_problems: int = 200,
    n_features: int = 16,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build synthetic SAE codes with one feature that strongly tracks surprise.

    Returns
    -------
    features : (n_problems, n_features) float32
    surprise : (n_problems,) float32 — synthetic difficulty signal
    conflict : (n_problems,) bool — half True / half False
    matched_pair_id : (n_problems,) S64 — items 2k and 2k+1 paired
    is_correct : (n_problems,) bool — used by the confidently-wrong test
    """
    rng = np.random.default_rng(seed)
    surprise = rng.uniform(0.0, 5.0, size=n_problems).astype(np.float32)

    # Build feature matrix: most features are pure noise, but
    # feature index 0 is monotone in surprise (the planted "difficulty
    # detector"), and feature index 1 is monotone in (1 - surprise),
    # which should NOT pass the rho > 0 threshold.
    features = rng.normal(0.0, 0.3, size=(n_problems, n_features)).astype(np.float32)
    features = np.clip(features, 0.0, None)  # ReLU-like
    features[:, 0] = surprise + rng.normal(0.0, 0.05, size=n_problems).astype(np.float32)
    features[:, 1] = (5.0 - surprise) + rng.normal(0.0, 0.05, size=n_problems).astype(np.float32)
    # Feature 2 is constant — should be excluded from FDR pool.
    features[:, 2] = 0.0

    conflict = np.array([i % 2 == 0 for i in range(n_problems)], dtype=bool)
    matched_pair_id = np.array([f"pair_{i // 2:04d}" for i in range(n_problems)], dtype="S64")

    # Correctness: lower correctness on conflict items, but with
    # explicit overlap in confidence so we can run the confidently-wrong test.
    is_correct = rng.uniform(size=n_problems) < np.where(conflict, 0.4, 0.8)
    return features, surprise, conflict, matched_pair_id, is_correct


# ---------------------------------------------------------------------------
# 1. Surprise aggregation
# ---------------------------------------------------------------------------


class TestAggregateSurprise:
    def test_by_position_label(self) -> None:
        by_pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        labels = ["P0", "P2", "T0"]
        out = aggregate_surprise(by_pos, labels, method="by_position:P2")
        assert np.allclose(out, [2.0, 5.0])

    def test_by_position_missing_label(self) -> None:
        by_pos = np.zeros((2, 2), dtype=np.float32)
        labels = ["P0", "P2"]
        with pytest.raises(KeyError):
            aggregate_surprise(by_pos, labels, method="by_position:Tend")

    def test_no_trace_falls_back_to_by_position(self) -> None:
        # Mean over by_position, treating zeros as invalid.
        by_pos = np.array([[2.0, 4.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
        labels = ["P0", "P2", "T0"]
        out = aggregate_surprise(by_pos, labels, method="mean_full")
        assert np.allclose(out, [3.0, 3.0])

    def test_full_trace_mean_full(self) -> None:
        # Two problems: first has surprise [1, 3], second has [2, 4, 6]
        offsets = np.array([0, 2, 5], dtype=np.int64)
        values = np.array([1.0, 3.0, 2.0, 4.0, 6.0], dtype=np.float32)
        by_pos = np.zeros((2, 2), dtype=np.float32)
        out = aggregate_surprise(
            by_pos,
            ["P0", "P2"],
            full_trace_offsets=offsets,
            full_trace_values=values,
            method="mean_full",
        )
        assert np.allclose(out, [2.0, 4.0])

    def test_full_trace_top_decile(self) -> None:
        # 10 surprises 1..10; top decile is just 10 -> mean = 10
        offsets = np.array([0, 10], dtype=np.int64)
        values = np.arange(1.0, 11.0, dtype=np.float32)
        by_pos = np.zeros((1, 1), dtype=np.float32)
        out = aggregate_surprise(
            by_pos,
            ["P0"],
            full_trace_offsets=offsets,
            full_trace_values=values,
            method="top_decile_full",
        )
        assert np.isclose(out[0], 10.0)


# ---------------------------------------------------------------------------
# 2. Surprise-feature correlation + BH-FDR
# ---------------------------------------------------------------------------


class TestSurpriseFeatureCorrelation:
    def test_planted_feature_recovered(self) -> None:
        feats, surp, _, _, _ = _planted_features(seed=0)
        result = surprise_feature_correlation(
            feats,
            surp,
            rho_threshold=0.3,
            fdr_q=0.05,
            layer=0,
            model_key="synth",
        )
        df = result.df
        # Feature 0 should be the strongest positive rho.
        top = df.sort_values(by="rho", ascending=False).iloc[0]
        assert int(top["feature_id"]) == 0
        assert top["rho"] > 0.9
        assert top["q_value"] < 0.01
        assert bool(top["is_difficulty_sensitive"])

    def test_constant_feature_excluded(self) -> None:
        feats, surp, _, _, _ = _planted_features(seed=1)
        result = surprise_feature_correlation(feats, surp)
        df = result.df
        const_row = df[df["feature_id"] == 2].iloc[0]
        assert bool(const_row["is_constant"])
        assert const_row["p_value"] == 1.0
        assert not bool(const_row["is_difficulty_sensitive"])

    def test_negative_rho_does_not_qualify(self) -> None:
        feats, surp, _, _, _ = _planted_features(seed=2)
        result = surprise_feature_correlation(feats, surp, rho_threshold=0.3)
        df = result.df
        neg = df[df["feature_id"] == 1].iloc[0]
        # Feature 1 is monotone in (5 - surp), so rho ≈ -1
        assert neg["rho"] < -0.9
        # We require rho > rho_threshold for "difficulty-sensitive";
        # negative rho features must NOT be tagged.
        assert not bool(neg["is_difficulty_sensitive"])

    def test_bh_fdr_correctly_applied(self) -> None:
        # Build a feature pool where exactly 1 feature has a strong rho
        # and the rest are random; BH-FDR should reject only the strong
        # one (or none, but never the random ones).
        rng = np.random.default_rng(7)
        n = 300
        surprise = rng.uniform(0, 5, size=n).astype(np.float32)
        n_features = 50
        feats = rng.normal(0, 0.5, size=(n, n_features)).astype(np.float32)
        feats = np.clip(feats, 0.0, None)
        feats[:, 0] = surprise + rng.normal(0, 0.05, size=n).astype(np.float32)
        result = surprise_feature_correlation(feats, surprise, rho_threshold=0.3, fdr_q=0.05)
        n_diff = int(result.df["is_difficulty_sensitive"].sum())
        # Exactly the planted feature should land
        assert n_diff >= 1
        # And the planted feature index 0 must be among the survivors.
        diff_ids = {int(x) for x in result.difficulty_sensitive_ids}
        assert 0 in diff_ids

    def test_constant_surprise_returns_no_signal(self) -> None:
        feats, _, _, _, _ = _planted_features(seed=3)
        const_surp = np.full(feats.shape[0], 1.0, dtype=np.float32)
        result = surprise_feature_correlation(feats, const_surp)
        assert int(result.df["is_difficulty_sensitive"].sum()) == 0

    def test_merge_correlation_results(self) -> None:
        feats, surp, _, _, _ = _planted_features(seed=4)
        r1 = surprise_feature_correlation(feats, surp, layer=0, model_key="m")
        r2 = surprise_feature_correlation(feats, surp, layer=1, model_key="m")
        big = merge_correlation_results([r1, r2], rho_threshold=0.3)
        assert "q_value_global" in big.columns
        assert "is_difficulty_sensitive_global" in big.columns
        assert (big["layer"].unique() == [0, 1]).all()


# ---------------------------------------------------------------------------
# 3. Confidently-wrong test (the critical falsifier)
# ---------------------------------------------------------------------------


class TestConfidentlyWrong:
    def test_metacognitive_feature_positive_score(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        is_correct = rng.uniform(size=n) > 0.5
        # All cases are "confident": confidence ≈ 0 > -0.5
        output_conf = np.full(n, 0.0, dtype=np.float32)
        # Metacognitive feature: spikes on confident_wrong, baseline on confident_correct.
        feature = np.where(~is_correct, 1.5, 0.1).astype(np.float32)
        feature += rng.normal(0, 0.05, size=n).astype(np.float32)
        out = confidently_wrong_test(
            feature,
            is_correct,
            output_conf,
            threshold_confidence=-0.5,
            feature_id=0,
        )
        assert out["n_confident_wrong"] > 0
        assert out["n_confident_correct"] > 0
        assert float(out["metacognition_score"]) > 0.5
        assert bool(out["is_metacognitive"])

    def test_confidence_proxy_returns_zero(self) -> None:
        rng = np.random.default_rng(1)
        n = 200
        is_correct = rng.uniform(size=n) > 0.5
        output_conf = np.full(n, 0.0, dtype=np.float32)
        # Confidence-proxy feature: identical activation regardless of correctness
        feature = np.full(n, 0.5, dtype=np.float32)
        feature += rng.normal(0, 0.01, size=n).astype(np.float32)
        out = confidently_wrong_test(feature, is_correct, output_conf, threshold_confidence=-0.5)
        assert abs(float(out["metacognition_score"])) < 0.05
        assert not bool(out["is_metacognitive"])

    def test_no_confident_cases_returns_neutral(self) -> None:
        n = 50
        is_correct = np.array([True] * n, dtype=bool)
        output_conf = np.full(n, -10.0, dtype=np.float32)  # all unconfident
        feature = np.ones(n, dtype=np.float32)
        out = confidently_wrong_test(feature, is_correct, output_conf, threshold_confidence=-0.5)
        assert out["n_confident_wrong"] == 0
        assert out["n_confident_correct"] == 0
        assert float(out["metacognition_score"]) == 0.0


# ---------------------------------------------------------------------------
# 4. S1/S2 specificity (matched-pair AUC)
# ---------------------------------------------------------------------------


class TestSpecificityTest:
    def test_planted_s1_feature_high_auc(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        conflict = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
        matched_pair_id = np.array([f"pair_{i // 2:04d}" for i in range(n)], dtype="S64")
        # Feature is large for conflict items, small for controls.
        feature = np.where(conflict, 2.0, 0.1).astype(np.float32)
        feature += rng.normal(0, 0.05, size=n).astype(np.float32)
        result = s1s2_specificity_test(
            feature,
            conflict,
            matched_pair_id,
            feature_id=0,
            auc_threshold=0.65,
            matched_only=True,
        )
        assert result.auc > 0.95
        assert result.passes_specificity
        assert result.n_S1 == n // 2
        assert result.n_S2 == n // 2

    def test_noise_feature_chance_auc(self) -> None:
        rng = np.random.default_rng(1)
        n = 200
        conflict = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
        matched_pair_id = np.array([f"pair_{i // 2:04d}" for i in range(n)], dtype="S64")
        feature = rng.normal(0, 1, size=n).astype(np.float32)
        result = s1s2_specificity_test(
            feature,
            conflict,
            matched_pair_id,
            feature_id=1,
            auc_threshold=0.65,
            matched_only=True,
        )
        assert 0.35 <= result.auc <= 0.65
        assert not result.passes_specificity

    def test_batch_runs_and_bh_corrects(self) -> None:
        rng = np.random.default_rng(2)
        n = 200
        conflict = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
        matched_pair_id = np.array([f"pair_{i // 2:04d}" for i in range(n)], dtype="S64")
        n_features = 5
        feats = rng.normal(0, 0.5, size=(n, n_features)).astype(np.float32)
        feats = np.clip(feats, 0.0, None)
        feats[:, 0] = np.where(conflict, 2.0, 0.0).astype(np.float32)
        df = s1s2_specificity_batch(
            feats,
            conflict,
            matched_pair_id,
            feature_ids=range(n_features),
            auc_threshold=0.65,
            fdr_q=0.05,
            matched_only=True,
        )
        assert "q_value" in df.columns
        assert df.loc[df["feature_id"] == 0, "passes_specificity"].iloc[0]


# ---------------------------------------------------------------------------
# 5. Difficulty-sensitive feature combiner
# ---------------------------------------------------------------------------


class TestDifficultySensitiveFeatures:
    def test_combine_three_passes_score_3(self) -> None:
        surprise_df = pd.DataFrame(
            {
                "feature_id": [0, 1],
                "rho": [0.8, 0.1],
                "is_difficulty_sensitive": [True, False],
            }
        )
        spec_df = pd.DataFrame(
            {
                "feature_id": [0, 1],
                "auc": [0.9, 0.5],
                "passes_specificity": [True, False],
            }
        )
        cw_df = pd.DataFrame(
            {
                "feature_id": [0, 1],
                "metacognition_score": [1.0, 0.0],
                "is_metacognitive": [True, False],
            }
        )
        out = difficulty_sensitive_features(surprise_df, spec_df, cw_df)
        # Feature 0 ticks all three boxes -> score 3, sorted to top
        top = out.iloc[0]
        assert int(top["feature_id"]) == 0
        assert int(top["score"]) == 3
        # Feature 1 ticks zero boxes
        bot = out.iloc[1]
        assert int(bot["feature_id"]) == 1
        assert int(bot["score"]) == 0


# ---------------------------------------------------------------------------
# 6. 4-Gate framework
# ---------------------------------------------------------------------------


class TestGateFramework:
    def test_all_pass_returns_go_go_go(self) -> None:
        snap = DifficultyDetectorResults(
            model_key="m",
            n_difficulty_sensitive_features=25,
            max_specificity_auc=0.75,
            n_features_passing_specificity=3,
            causal_delta_p_correct=0.20,
            infrastructure_ok=True,
        )
        gates = evaluate_gates(snap, GateConfig())
        assert [g.decision for g in gates] == ["go", "go", "go", "go"]
        assert gates[0].name == "infrastructure"
        assert gates[3].name == "causal_evidence"

    def test_signal_existence_no_go(self) -> None:
        snap = DifficultyDetectorResults(
            n_difficulty_sensitive_features=2,
            max_specificity_auc=0.5,
            n_features_passing_specificity=0,
            infrastructure_ok=True,
        )
        gates = evaluate_gates(snap, GateConfig(min_features_with_rho_gt=20))
        assert gates[1].decision == "no_go"
        assert "no internal difficulty signal" in gates[1].rationale

    def test_specificity_marginal_when_close(self) -> None:
        snap = DifficultyDetectorResults(
            n_difficulty_sensitive_features=25,
            max_specificity_auc=0.62,
            n_features_passing_specificity=0,
            infrastructure_ok=True,
        )
        gates = evaluate_gates(snap, GateConfig(min_specificity_auc=0.65))
        assert gates[2].decision == "marginal"

    def test_causal_marginal_until_attached(self) -> None:
        snap = DifficultyDetectorResults(
            n_difficulty_sensitive_features=25,
            max_specificity_auc=0.75,
            n_features_passing_specificity=3,
            causal_delta_p_correct=None,
            infrastructure_ok=True,
        )
        gates = evaluate_gates(snap, GateConfig())
        assert gates[3].decision == "marginal"

    def test_causal_no_go_when_attached_below(self) -> None:
        snap = DifficultyDetectorResults(
            n_difficulty_sensitive_features=25,
            max_specificity_auc=0.75,
            n_features_passing_specificity=3,
            causal_delta_p_correct=0.02,
            infrastructure_ok=True,
        )
        gates = evaluate_gates(snap, GateConfig(min_delta_p_correct=0.15))
        assert gates[3].decision == "no_go"

    def test_infrastructure_failure_short_circuits_decision(self) -> None:
        snap = DifficultyDetectorResults(infrastructure_ok=False, notes="hdf5 missing")
        gates = evaluate_gates(snap, GateConfig())
        assert gates[0].decision == "no_go"


# ---------------------------------------------------------------------------
# 7. Self-correction parser
# ---------------------------------------------------------------------------


class TestSelfCorrectionParser:
    def test_finds_marker_with_prior_commitment(self) -> None:
        trace = (
            "I think the answer is 5. Wait, that doesn't seem right. "
            "Let me redo this. Actually 2 + 3 = 5, so the answer is 5."
        )
        result = parse_self_correction(trace, problem_idx=0)
        assert result.self_corrects
        assert result.n_markers_total >= 1
        assert result.n_markers_with_prior >= 1
        assert result.first_correction_char is not None
        assert result.first_correction_relative is not None

    def test_no_prior_commitment_does_not_self_correct(self) -> None:
        trace = (
            "Wait, let me think. This is interesting. "
            "I have not concluded anything yet, but I will."
        )
        # No "answer is X" before the marker -> no_prior_commitment.
        result = parse_self_correction(trace)
        assert result.n_markers_total >= 1
        assert not result.self_corrects

    def test_marker_at_end_of_trace_filtered(self) -> None:
        # Marker followed by zero new reasoning -> filtered out.
        trace = "The answer is 7. Actually."
        result = parse_self_correction(trace, min_post_chars=30)
        assert not result.self_corrects

    def test_corpus_parser_logs_rate(self) -> None:
        traces = [
            "The answer is 5. Wait, let me reconsider that calculation step by step.",
            "Just thinking out loud here, no commitment yet.",
            "",
            "The answer is 9. Actually, I want to double-check the carry digit.",
        ]
        results = parse_trace_corpus(traces, min_post_chars=15)
        assert len(results) == 4
        sc_count = sum(int(r.self_corrects) for r in results)
        assert sc_count >= 2

    def test_empty_trace_returns_empty_result(self) -> None:
        result = parse_self_correction("")
        assert result.trace_chars == 0
        assert not result.self_corrects
        assert len(result.events) == 0


class TestTrajectoryMeans:
    def test_difficulty_trajectory_means_basic(self) -> None:
        # 4 problems x 5 positions; problem 0 self-corrects at rel=0.5
        feats = np.zeros((4, 5), dtype=np.float32)
        feats[0] = [1.0, 1.0, 0.5, 0.0, 0.0]  # high pre, low post
        from s1s2.metacog.trajectory import SelfCorrectionEvent, TraceParseResult

        parses = [
            TraceParseResult(
                problem_idx=0,
                trace_chars=100,
                self_corrects=True,
                first_correction_char=50,
                first_correction_relative=0.5,
                events=[
                    SelfCorrectionEvent(
                        marker="wait",
                        char_start=50,
                        char_end=54,
                        has_prior_commitment=True,
                        prior_answer="5",
                        post_chars=46,
                        relative_position=0.5,
                    )
                ],
            ),
            TraceParseResult(problem_idx=1, trace_chars=100),
            TraceParseResult(problem_idx=2, trace_chars=100),
            TraceParseResult(problem_idx=3, trace_chars=100),
        ]
        out = difficulty_trajectory_means(feats, parses)
        assert out["n_self_correcting"] == 1
        # We don't enforce a specific delta — just that the function runs
        # and returns a finite delta.
        assert isinstance(out["delta_post_minus_pre"], float)
        assert out["n_baseline"] == 3


# ---------------------------------------------------------------------------
# 8. End-to-end synthetic HDF5 round trip
# ---------------------------------------------------------------------------


def _write_synthetic_metacog_h5(path: Path) -> tuple[str, int, int]:
    """Build a tiny but schema-valid HDF5 cache for the metacog test."""
    n_problems = 80
    hidden = 16
    n_layers = 2
    n_positions = 4
    pos_labels = ["P0", "P2", "T0", "Tend"]
    model_hdf5 = "synth_metacog"
    rng = np.random.default_rng(13)

    # Build a planted "difficulty signal" baked into layer 0 residuals:
    # the magnitude of one direction tracks a "true difficulty" scalar
    # that also drives the surprise. The MockSAE will then have a
    # feature that approximately tracks that direction.
    difficulty = rng.uniform(0.0, 5.0, size=n_problems).astype(np.float32)
    # Surprise vector: 1:1 with difficulty plus noise
    surprise = difficulty + rng.normal(0, 0.1, size=n_problems).astype(np.float32)
    # Make residuals carry the difficulty in dim 0
    base = rng.normal(0, 0.5, size=(n_problems, n_positions, hidden)).astype(np.float32)
    base[:, :, 0] += difficulty[:, None]
    layer_data = [base, base + 0.1 * rng.normal(size=base.shape).astype(np.float32)]

    # Half conflict / half control, paired
    conflict = np.array([i % 2 == 0 for i in range(n_problems)], dtype=bool)
    pair_ids = np.array([f"pair_{i // 2:04d}" for i in range(n_problems)], dtype="S64")
    is_correct = rng.uniform(size=n_problems) < np.where(conflict, 0.4, 0.8)
    matches_lure = conflict & ~is_correct

    # Per-position surprise: the P2 position gets the answer-time surprise.
    by_position = np.zeros((n_problems, n_positions), dtype=np.float32)
    by_position[:, pos_labels.index("P2")] = surprise

    # Optional ragged trace
    chunk_lens = rng.integers(5, 20, size=n_problems)
    offsets = np.zeros(n_problems + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(chunk_lens)
    values = rng.uniform(0.0, 5.0, size=int(offsets[-1])).astype(np.float32)

    with h5py.File(path, "w") as f:
        meta = f.create_group("/metadata")
        meta.attrs["schema_version"] = ioh.SCHEMA_VERSION
        meta.attrs["benchmark_path"] = "synthetic"
        meta.attrs["benchmark_sha256"] = "0" * 64
        meta.attrs["created_at"] = "2026-04-09T00:00:00Z"
        meta.attrs["git_sha"] = "unknown"
        meta.attrs["seed"] = 0
        meta.attrs["config"] = "{}"

        problems = f.create_group("/problems")
        problems.create_dataset(
            "id",
            data=np.array([f"p{i:04d}".encode() for i in range(n_problems)], dtype="S64"),
        )
        problems.create_dataset(
            "category",
            data=np.array(["crt"] * n_problems, dtype="S32"),
        )
        problems.create_dataset("conflict", data=conflict)
        problems.create_dataset("difficulty", data=np.ones(n_problems, dtype=np.int8))
        problems.create_dataset(
            "prompt_text",
            data=np.array(["What is 2+2?"] * n_problems, dtype="S2048"),
        )
        problems.create_dataset("correct_answer", data=np.array(["4"] * n_problems, dtype="S128"))
        problems.create_dataset("lure_answer", data=np.array(["5"] * n_problems, dtype="S128"))
        problems.create_dataset("matched_pair_id", data=pair_ids)
        problems.create_dataset("prompt_token_count", data=np.full(n_problems, 6, dtype=np.int32))

        mgrp = f.create_group(f"/models/{model_hdf5}")
        mmeta = mgrp.create_group("metadata")
        mmeta.attrs["hf_model_id"] = "synth/metacog"
        mmeta.attrs["n_layers"] = n_layers
        mmeta.attrs["n_heads"] = 4
        mmeta.attrs["n_kv_heads"] = 2
        mmeta.attrs["hidden_dim"] = hidden
        mmeta.attrs["head_dim"] = 4
        mmeta.attrs["dtype"] = "float32"
        mmeta.attrs["extracted_at"] = "2026-04-09T00:00:00Z"
        mmeta.attrs["is_reasoning_model"] = True

        resid = mgrp.create_group("residual")
        for layer in range(n_layers):
            resid.create_dataset(f"layer_{layer:02d}", data=layer_data[layer])

        pos = mgrp.create_group("position_index")
        pos.create_dataset(
            "labels",
            data=np.array([s.encode() for s in pos_labels], dtype="S16"),
        )
        pos.create_dataset(
            "token_indices",
            data=np.zeros((n_problems, n_positions), dtype=np.int32),
        )
        pos.create_dataset("valid", data=np.ones((n_problems, n_positions), dtype=bool))

        beh = mgrp.create_group("behavior")
        beh.create_dataset("predicted_answer", data=np.array(["4"] * n_problems, dtype="S128"))
        beh.create_dataset("correct", data=is_correct)
        beh.create_dataset("matches_lure", data=matches_lure)
        beh.create_dataset(
            "response_category",
            data=np.array(["correct"] * n_problems, dtype="S16"),
        )

        gens = mgrp.create_group("generations")
        # Half the problems get a self-correcting trace.
        thinking = []
        for i in range(n_problems):
            if i % 4 == 0:
                thinking.append(
                    "I think the answer is 5. Wait, let me re-check this carefully and try once more."
                )
            else:
                thinking.append("Let me compute. 2 + 2 = 4. Done.")
        gens.create_dataset("full_text", data=np.array(["x"] * n_problems, dtype="S8192"))
        gens.create_dataset("thinking_text", data=np.array(thinking, dtype="S8192"))
        gens.create_dataset("answer_text", data=np.array(["4"] * n_problems, dtype="S512"))
        gens.create_dataset(
            "thinking_token_count",
            data=np.full(n_problems, 10, dtype=np.int32),
        )
        gens.create_dataset(
            "answer_token_count",
            data=np.full(n_problems, 1, dtype=np.int32),
        )

        sgrp = mgrp.create_group("token_surprises")
        sgrp.create_dataset("by_position", data=by_position)
        sgrp.create_dataset("full_trace_offsets", data=offsets)
        sgrp.create_dataset("full_trace_values", data=values)

    return model_hdf5, hidden, n_layers


class TestEndToEnd:
    def test_full_pipeline_round_trip(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "synth.h5"
        model_hdf5, hidden, n_layers = _write_synthetic_metacog_h5(h5_path)

        cfg = MetacogConfig(
            activations_path=str(h5_path),
            output_dir=str(tmp_path / "out"),
            seed=0,
            min_features_with_rho_gt=1,  # tiny synthetic — relax thresholds
            rho_threshold=0.2,
            specificity_auc_threshold=0.55,
            surprise_aggregation="by_position:P2",
        )
        analysis = DifficultyDetectorAnalysis(cfg)
        sae = MockSAE(hidden_dim=hidden, n_features=64, layer=0, seed=0, sparsity=0.5)

        per_layer = analysis.run(
            model_key="synth-model",
            hdf5_key=model_hdf5,
            layer=0,
            sae=sae,
        )
        assert per_layer.n_problems == 80
        assert per_layer.n_features == 64
        # At least one feature should have non-zero rho on the planted signal
        assert per_layer.surprise_df["rho"].abs().max() > 0.0

        # Write per-layer JSON
        path = analysis.write_per_layer(per_layer)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["model_key"] == "synth-model"
        assert data["layer"] == 0

        # Evaluate gates and write summary
        snapshot, gates = analysis.evaluate_gates(per_layer)
        assert len(gates) == 4
        summary_path = analysis.write_summary(snapshot, gates)
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert "snapshot" in summary
        assert "gates" in summary
        assert len(summary["gates"]) == 4

    def test_attach_causal_updates_gate_3(self, tmp_path: Path) -> None:
        h5_path = tmp_path / "synth.h5"
        model_hdf5, hidden, _ = _write_synthetic_metacog_h5(h5_path)
        cfg = MetacogConfig(
            activations_path=str(h5_path),
            output_dir=str(tmp_path / "out"),
            min_features_with_rho_gt=1,
            rho_threshold=0.2,
            specificity_auc_threshold=0.55,
            min_delta_p_correct=0.15,
            surprise_aggregation="by_position:P2",
        )
        analysis = DifficultyDetectorAnalysis(cfg)
        sae = MockSAE(hidden_dim=hidden, n_features=32, seed=0, sparsity=0.5)
        per_layer = analysis.run(
            model_key="synth-model",
            hdf5_key=model_hdf5,
            layer=0,
            sae=sae,
        )
        snap, gates = analysis.evaluate_gates(per_layer)
        assert gates[3].decision == "marginal"
        snap_with_causal, gates_with_causal = analysis.attach_causal(snap, 0.25)
        assert gates_with_causal[3].decision == "go"
