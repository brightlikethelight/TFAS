"""Tests for :mod:`s1s2.utils` (stats, seeding, IO).

The utils module is the foundation under every workstream so we lean
heavy on edge cases here. Anything that goes wrong in :func:`bh_fdr`
or :func:`bootstrap_ci` quietly poisons all five workstreams' results.

These tests use no GPU, no transformers, and no SAEs. They run in
under a couple of seconds on CPU.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# OpenMP guard for macOS — torch + numpy can crash without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Allow tests to import the package without `pip install -e .`.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from s1s2.utils import io as ioh  # noqa: E402
from s1s2.utils.seed import set_global_seed  # noqa: E402
from s1s2.utils.stats import (  # noqa: E402
    bh_fdr,
    bootstrap_ci,
    cohens_d,
    gini_coefficient,
    paired_bootstrap_ci_diff,
    permutation_test_two_sample,
    rank_biserial,
    shannon_entropy_bits,
)

# --------------------------------------------------------------------------- #
# BH-FDR                                                                       #
# --------------------------------------------------------------------------- #


class TestBHFDR:
    def test_matches_statsmodels_reference(self):
        """Compare against statsmodels (or a hand-computed reference)."""
        pvals = np.array([0.001, 0.01, 0.02, 0.03, 0.04, 0.1, 0.5, 0.9])
        rejected, qvals = bh_fdr(pvals, q=0.05)
        # Hand-checked: q[i] = min over j>=i of n/(j+1) * p_sorted[j]
        # n=8: p_sorted = [.001, .01, .02, .03, .04, .1, .5, .9]
        # raw = [.008, .04, .0533, .06, .064, .1333, .5714, .9]
        # monotone from right: [.008, .04, .0533, .06, .064, .1333, .5714, .9]
        try:
            from statsmodels.stats.multitest import multipletests

            ref_rejected, ref_qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            np.testing.assert_allclose(qvals, ref_qvals, atol=1e-12)
            np.testing.assert_array_equal(rejected, ref_rejected)
        except ImportError:
            # Fall back to hand-checked values
            assert qvals[0] == pytest.approx(0.008, abs=1e-3)
            assert rejected[0] is np.True_ or rejected[0] is True or rejected[0]
            assert not rejected[7]

    def test_all_significant(self):
        """All extremely small p-values: everything rejected."""
        pvals = np.array([1e-10, 1e-9, 1e-8])
        rejected, qvals = bh_fdr(pvals, q=0.05)
        assert rejected.all()
        assert (qvals < 0.05).all()

    def test_none_significant(self):
        """All large p-values: nothing rejected."""
        pvals = np.array([0.5, 0.8, 0.9, 0.99])
        rejected, qvals = bh_fdr(pvals, q=0.05)
        assert not rejected.any()

    def test_single_pvalue(self):
        rejected, qvals = bh_fdr(np.array([0.01]), q=0.05)
        assert rejected[0]
        assert qvals[0] == pytest.approx(0.01)

    def test_qvalues_clipped(self):
        """q-values are clipped to [0, 1] (raw BH can exceed 1)."""
        pvals = np.array([0.9, 0.95, 0.99])
        _, qvals = bh_fdr(pvals, q=0.05)
        assert (qvals <= 1.0).all()
        assert (qvals >= 0.0).all()


# --------------------------------------------------------------------------- #
# Permutation test                                                             #
# --------------------------------------------------------------------------- #


class TestPermutationTest:
    def test_rejects_null_on_obvious_signal(self):
        """Two well-separated samples => p-value is small."""
        rng = np.random.default_rng(0)
        x = rng.normal(loc=2.0, scale=1.0, size=100)
        y = rng.normal(loc=-2.0, scale=1.0, size=100)
        obs, p = permutation_test_two_sample(x, y, n_permutations=500, seed=0)
        assert obs > 3.0  # mean(x) - mean(y) ≈ 4
        assert p < 0.05

    def test_fails_to_reject_on_random_data(self):
        """Same-distribution samples => p-value is moderate."""
        rng = np.random.default_rng(1)
        x = rng.normal(size=80)
        y = rng.normal(size=80)
        _, p = permutation_test_two_sample(x, y, n_permutations=500, seed=0)
        # Not strict — under H0 the p-value is uniform on [0, 1]. Just
        # check that it's nowhere near zero (i.e. ≥ 0.05 with 90% prob).
        assert p > 0.01

    def test_one_sided_alternatives(self):
        rng = np.random.default_rng(2)
        x = rng.normal(loc=1.0, size=100)
        y = rng.normal(loc=0.0, size=100)
        _, p_greater = permutation_test_two_sample(
            x, y, n_permutations=500, seed=0, alternative="greater"
        )
        _, p_less = permutation_test_two_sample(
            x, y, n_permutations=500, seed=0, alternative="less"
        )
        assert p_greater < 0.05
        assert p_less > 0.5  # opposite tail

    def test_north_correction_no_zero_pvalues(self):
        """The +1 correction guarantees the smallest possible p-value > 0."""
        rng = np.random.default_rng(3)
        x = rng.normal(loc=10.0, size=50)
        y = rng.normal(loc=-10.0, size=50)
        _, p = permutation_test_two_sample(x, y, n_permutations=100, seed=0)
        # Min p with the +1 correction is 1/(n_perms+1) = 1/101 ≈ 0.0099
        assert p >= 1.0 / 101.0
        assert p > 0.0

    def test_invalid_alternative_raises(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=10)
        y = rng.normal(size=10)
        with pytest.raises(ValueError, match="alternative"):
            permutation_test_two_sample(
                x, y, n_permutations=10, alternative="bogus"
            )


# --------------------------------------------------------------------------- #
# Bootstrap CI                                                                 #
# --------------------------------------------------------------------------- #


class TestBootstrapCI:
    def test_ci_contains_true_mean(self):
        """For a known distribution, CI should contain the true mean."""
        rng = np.random.default_rng(0)
        true_mean = 5.0
        data = rng.normal(loc=true_mean, scale=1.0, size=200)
        point, lo, hi = bootstrap_ci(
            data, statistic=lambda v: float(np.mean(v)), n_resamples=500, seed=0
        )
        assert lo < true_mean < hi
        assert abs(point - np.mean(data)) < 1e-10

    def test_ci_shrinks_with_more_data(self):
        """CI width decreases as sample size grows."""
        rng = np.random.default_rng(1)
        small = rng.normal(size=20)
        large = rng.normal(size=2000)

        _, lo_s, hi_s = bootstrap_ci(
            small, statistic=lambda v: float(np.mean(v)), n_resamples=500, seed=0
        )
        _, lo_l, hi_l = bootstrap_ci(
            large, statistic=lambda v: float(np.mean(v)), n_resamples=500, seed=0
        )
        assert (hi_l - lo_l) < (hi_s - lo_s)

    def test_paired_ci_requires_equal_lengths(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="equal lengths"):
            paired_bootstrap_ci_diff(
                x, y, statistic=lambda a, b: float(a.mean() - b.mean()),
                n_resamples=10,
            )

    def test_paired_ci_runs(self):
        rng = np.random.default_rng(2)
        x = rng.normal(loc=1.0, size=50)
        y = rng.normal(loc=0.0, size=50)
        point, lo, hi = paired_bootstrap_ci_diff(
            x, y, statistic=lambda a, b: float(a.mean() - b.mean()),
            n_resamples=200, seed=0,
        )
        assert lo < point < hi
        assert point > 0  # x has higher mean


# --------------------------------------------------------------------------- #
# Effect sizes                                                                 #
# --------------------------------------------------------------------------- #


class TestEffectSizes:
    def test_cohens_d_separated(self):
        rng = np.random.default_rng(0)
        x = rng.normal(loc=2.0, size=100)
        y = rng.normal(loc=0.0, size=100)
        d = cohens_d(x, y)
        # d ≈ (mu_x - mu_y) / sigma ≈ 2.0
        assert d == pytest.approx(2.0, abs=0.3)

    def test_cohens_d_zero_variance(self):
        x = np.ones(10)
        y = np.ones(10)
        d = cohens_d(x, y)
        assert d == 0.0

    def test_rank_biserial_range(self):
        rng = np.random.default_rng(1)
        x = rng.normal(loc=2.0, size=50)
        y = rng.normal(loc=0.0, size=50)
        rb = rank_biserial(x, y)
        assert -1.0 <= rb <= 1.0
        assert rb > 0.5  # x stochastically larger

    def test_rank_biserial_symmetry(self):
        """rank_biserial(x, y) == -rank_biserial(y, x)."""
        rng = np.random.default_rng(2)
        x = rng.normal(loc=1.0, size=30)
        y = rng.normal(loc=0.0, size=30)
        assert rank_biserial(x, y) == pytest.approx(-rank_biserial(y, x), abs=1e-9)

    def test_gini_uniform_is_zero(self):
        """Gini of equal values is 0."""
        v = np.ones(50)
        assert gini_coefficient(v) == pytest.approx(0.0, abs=1e-9)

    def test_gini_max_inequality(self):
        """One value, all others zero — Gini approaches (n-1)/n."""
        v = np.zeros(100)
        v[0] = 1.0
        g = gini_coefficient(v)
        assert 0.95 < g <= 1.0

    def test_gini_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            gini_coefficient(np.array([1.0, -1.0]))

    def test_gini_empty(self):
        assert gini_coefficient(np.array([])) == 0.0


# --------------------------------------------------------------------------- #
# Shannon entropy                                                              #
# --------------------------------------------------------------------------- #


class TestShannonEntropy:
    def test_uniform_max_entropy(self):
        probs = np.ones(8) / 8
        h = shannon_entropy_bits(probs)
        assert h == pytest.approx(3.0, abs=1e-9)  # log2(8) = 3

    def test_point_mass_zero_entropy(self):
        probs = np.zeros(4)
        probs[0] = 1.0
        h = shannon_entropy_bits(probs)
        assert h == pytest.approx(0.0, abs=1e-6)

    def test_batch_mode(self):
        probs = np.array(
            [
                [0.5, 0.5],
                [1.0, 0.0],
                [0.25, 0.75],
            ]
        )
        h = shannon_entropy_bits(probs)
        assert h.shape == (3,)
        assert h[0] == pytest.approx(1.0, abs=1e-6)
        assert h[1] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Seed determinism                                                             #
# --------------------------------------------------------------------------- #


class TestSeedDeterminism:
    def test_set_global_seed_numpy(self):
        set_global_seed(42, deterministic_torch=False)
        a = np.random.rand(5)
        set_global_seed(42, deterministic_torch=False)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_set_global_seed_random(self):
        set_global_seed(42, deterministic_torch=False)
        a = [random.random() for _ in range(5)]
        set_global_seed(42, deterministic_torch=False)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_set_global_seed_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        set_global_seed(42, deterministic_torch=False)
        a = torch.randn(5)
        set_global_seed(42, deterministic_torch=False)
        b = torch.randn(5)
        torch.testing.assert_close(a, b)

    def test_different_seeds_differ(self):
        set_global_seed(1, deterministic_torch=False)
        a = np.random.rand(5)
        set_global_seed(2, deterministic_torch=False)
        b = np.random.rand(5)
        assert not np.allclose(a, b)


# --------------------------------------------------------------------------- #
# IO round-trip                                                                #
# --------------------------------------------------------------------------- #


class TestIORoundTrip:
    """Round-trip the write_* helpers in :mod:`s1s2.utils.io` against
    read_* / get_* / load_* readers, on a tiny synthetic schema-conformant
    file we build inline (no fixture, just to keep this self-contained).
    """

    def _build(self, tmp_path: Path) -> Path:
        path = tmp_path / "rt.h5"
        n_problems = 4
        n_layers = 2
        n_positions = 3
        hidden = 6
        n_heads = 2
        with h5py.File(path, "w") as f:
            ioh.write_run_metadata(
                f,
                benchmark_path="rt-bench",
                benchmark_sha256="0" * 64,
                created_at="2026-04-09T00:00:00+00:00",
                git_sha="abc123",
                seed=7,
                config_json='{"k":1}',
            )
            ioh.write_problem_metadata(
                f,
                ids=[f"p{i}" for i in range(n_problems)],
                categories=["crt"] * n_problems,
                conflict=np.array([True, False, True, False], dtype=bool),
                difficulty=np.array([1, 2, 3, 4], dtype=np.int8),
                prompt_text=[f"prompt {i}" for i in range(n_problems)],
                correct_answer=["A", "B", "C", "D"],
                lure_answer=["L1", "", "L3", ""],
                matched_pair_id=["p0", "p0", "p1", "p1"],
                prompt_token_count=np.full(n_problems, 10, dtype=np.int32),
            )
            ioh.write_model_metadata(
                f,
                "rt_model",
                hf_model_id="rt/model",
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                hidden_dim=hidden,
                head_dim=hidden // n_heads,
                dtype="float32",
                extracted_at="2026-04-09T00:00:01+00:00",
                is_reasoning_model=False,
            )
            for layer in range(n_layers):
                data = (
                    np.arange(n_problems * n_positions * hidden, dtype=np.float32)
                    .reshape(n_problems, n_positions, hidden)
                    + layer * 1000.0
                )
                ioh.write_residual_layer(f, "rt_model", layer=layer, data=data)
            ioh.write_position_index(
                f,
                "rt_model",
                labels=["P0", "P2", "T0"],
                token_indices=np.zeros((n_problems, n_positions), dtype=np.int32),
                valid=np.array(
                    [[True, True, False]] * n_problems, dtype=bool
                ),
            )
            shape = (n_problems, n_layers, n_heads, n_positions)
            ioh.write_attention_metrics(
                f,
                "rt_model",
                {
                    "entropy": np.full(shape, 1.5, dtype=np.float32),
                    "entropy_normalized": np.full(shape, 0.5, dtype=np.float32),
                    "gini": np.full(shape, 0.3, dtype=np.float32),
                    "max_attn": np.full(shape, 0.4, dtype=np.float32),
                    "focus_5": np.full(shape, 0.7, dtype=np.float32),
                    "effective_rank": np.full(shape, 2.8, dtype=np.float32),
                },
            )
            ioh.write_token_surprises(
                f,
                "rt_model",
                by_position=np.full((n_problems, n_positions), 1.1, dtype=np.float32),
            )
            ioh.write_generations(
                f,
                "rt_model",
                full_text=[f"gen {i}" for i in range(n_problems)],
                thinking_text=[""] * n_problems,
                answer_text=["A", "B", "C", "D"],
                thinking_token_count=np.zeros(n_problems, dtype=np.int32),
                answer_token_count=np.full(n_problems, 1, dtype=np.int32),
            )
            ioh.write_behavior(
                f,
                "rt_model",
                predicted_answer=["A", "B", "C", "D"],
                correct=np.array([True, True, False, False]),
                matches_lure=np.array([False, False, True, False]),
                response_category=["correct", "correct", "lure", "other_wrong"],
            )
        return path

    def test_round_trip_preserves_schema(self, tmp_path: Path):
        path = self._build(tmp_path)
        with ioh.open_activations(path) as f:
            assert f["/metadata"].attrs["schema_version"] == ioh.SCHEMA_VERSION
            meta = ioh.run_metadata(f)
            assert meta["benchmark_path"] == "rt-bench"
            assert meta["seed"] == 7

            assert ioh.n_problems(f) == 4
            assert ioh.list_models(f) == ["rt_model"]
            mmeta = ioh.model_metadata(f, "rt_model")
            assert mmeta["hf_model_id"] == "rt/model"
            assert mmeta["n_layers"] == 2

    def test_round_trip_problem_metadata(self, tmp_path: Path):
        path = self._build(tmp_path)
        with ioh.open_activations(path) as f:
            pm = ioh.load_problem_metadata(f)
        assert list(pm["id"]) == ["p0", "p1", "p2", "p3"]
        assert list(pm["category"]) == ["crt"] * 4
        np.testing.assert_array_equal(pm["conflict"], [True, False, True, False])
        np.testing.assert_array_equal(pm["difficulty"], [1, 2, 3, 4])

    def test_round_trip_residual(self, tmp_path: Path):
        path = self._build(tmp_path)
        with ioh.open_activations(path) as f:
            arr = ioh.get_residual(f, "rt_model", layer=1)
            # The +1000*layer offset means the (0,0,0) entry of layer 1 should be 1000.0
            assert arr[0, 0, 0] == pytest.approx(1000.0, abs=1e-3)
            arr_p0 = ioh.get_residual(f, "rt_model", layer=0, position="P0")
            assert arr_p0.shape == (4, 6)

    def test_round_trip_attention_metric(self, tmp_path: Path):
        path = self._build(tmp_path)
        with ioh.open_activations(path) as f:
            ent = ioh.get_attention_metric(f, "rt_model", "entropy")
            assert ent.shape == (4, 2, 2, 3)
            assert (ent == 1.5).all()

    def test_round_trip_behavior(self, tmp_path: Path):
        path = self._build(tmp_path)
        with ioh.open_activations(path) as f:
            beh = ioh.get_behavior(f, "rt_model")
        assert list(beh["response_category"]) == [
            "correct", "correct", "lure", "other_wrong"
        ]
        np.testing.assert_array_equal(beh["correct"], [True, True, False, False])

    def test_round_trip_generations(self, tmp_path: Path):
        path = self._build(tmp_path)
        with ioh.open_activations(path) as f:
            gens = ioh.get_generations(f, "rt_model")
        assert list(gens["full_text"]) == ["gen 0", "gen 1", "gen 2", "gen 3"]
        assert list(gens["answer_text"]) == ["A", "B", "C", "D"]

    def test_invalid_position_raises(self, tmp_path: Path):
        path = self._build(tmp_path)
        with ioh.open_activations(path) as f, pytest.raises(KeyError):
            ioh.get_residual(f, "rt_model", layer=0, position="Tswitch")

    def test_invalid_metric_raises(self, tmp_path: Path):
        path = self._build(tmp_path)
        with ioh.open_activations(path) as f, pytest.raises(ValueError):
            ioh.get_attention_metric(f, "rt_model", "bogus")
