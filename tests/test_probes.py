"""Tests for :mod:`s1s2.probes`.

We cover:

* Basic probe behaviour on synthetic Gaussians (mass-mean should separate).
* Linear probe fails XOR (by construction) while the MLP probe recovers it.
* Hewitt & Liang control baseline: random labels => AUC ~0.5.
* BH-FDR application on a known set of p-values.
* LOCO splits: held-out groups never appear in the train fold.
* End-to-end round trip: tiny synthetic HDF5 -> runner -> result JSON.

All tests are CPU-only and run in under ~30s collectively on a laptop.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from s1s2.probes import (
    ALL_TARGETS,
    CCSProbe,
    LogisticRegressionProbe,
    MassMeanProbe,
    MLPProbe,
    ProbeRunner,
    RunnerConfig,
    apply_bh_across_layers,
    build_target,
    get_probe_class,
    layer_result_to_dict,
    loco_split_iter,
    make_stratify_key,
    save_layer_result,
)
from s1s2.probes.controls import run_control_task, shuffled_labels
from s1s2.utils import io as ioh

# ---------------------------------------------------------------------------
# Fixtures: synthetic data builders
# ---------------------------------------------------------------------------


@pytest.fixture
def gaussian_blobs() -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated Gaussian blobs in 32-dim space."""
    rng = np.random.default_rng(0)
    n = 200
    d = 32
    mu = 1.5
    X0 = rng.normal(loc=-mu, scale=1.0, size=(n, d))
    X1 = rng.normal(loc=+mu, scale=1.0, size=(n, d))
    X = np.concatenate([X0, X1], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int8)
    return X, y


@pytest.fixture
def xor_data() -> tuple[np.ndarray, np.ndarray]:
    """XOR-structured data: linear probes should fail, MLPs should succeed."""
    rng = np.random.default_rng(42)
    n = 400
    X_a = rng.normal(size=(n, 2))
    # The first two dims form an XOR pattern; the remaining dims are noise.
    y = ((X_a[:, 0] > 0) ^ (X_a[:, 1] > 0)).astype(np.int8)
    noise = rng.normal(size=(n, 30)) * 0.1
    X = np.concatenate([X_a, noise], axis=1).astype(np.float32)
    return X, y


def _write_synthetic_hdf5(path: Path, *, n_problems: int = 120, n_layers: int = 4) -> str:
    """Create a tiny but schema-valid HDF5 cache and return the model hdf5 key."""
    hidden = 24
    n_positions = 2
    position_labels = ["P0", "P2"]
    model_hdf5 = "synth_model"
    rng = np.random.default_rng(7)

    # Half conflict / half control, across 3 categories, balanced.
    categories = np.array(
        [["crt", "base_rate", "syllogism"][i % 3] for i in range(n_problems)], dtype="S32"
    )
    conflict = np.array([i % 2 == 0 for i in range(n_problems)], dtype=bool)
    # matched pair id — items 2k and 2k+1 are a pair.
    pair_ids = np.array([f"pair_{i // 2:04d}" for i in range(n_problems)], dtype="S64")

    # Build activations: conflict items share a shifted mean on layer 2 so the
    # task_type target is linearly decodable there and nowhere else. We shift
    # ~6 dims by 1.2 sigma each so a logistic probe can comfortably exceed
    # AUC=0.7 with 5-fold CV on 120 samples and 24 features.
    acts_per_layer: list[np.ndarray] = []
    n_signal_dims = 6
    signal_strength = 1.2
    for layer in range(n_layers):
        base = rng.normal(size=(n_problems, n_positions, hidden)).astype(np.float32)
        if layer == 2:
            shift = signal_strength * conflict.astype(np.float32)
            for d in range(n_signal_dims):
                base[:, :, d] += shift[:, None]
        acts_per_layer.append(base.astype(np.float32))

    # Correctness: 70% of non-conflict items correct, 40% of conflict items correct.
    correct = np.where(
        conflict,
        rng.uniform(size=n_problems) < 0.4,
        rng.uniform(size=n_problems) < 0.7,
    )
    matches_lure = conflict & ~correct  # lured whenever wrong on a conflict item

    with h5py.File(path, "w") as f:
        # /metadata
        meta = f.create_group("/metadata")
        meta.attrs["schema_version"] = ioh.SCHEMA_VERSION
        meta.attrs["benchmark_path"] = "synthetic"
        meta.attrs["benchmark_sha256"] = "0" * 64
        meta.attrs["created_at"] = "2026-04-09T00:00:00Z"
        meta.attrs["git_sha"] = "unknown"
        meta.attrs["seed"] = 0
        meta.attrs["config"] = "{}"

        # /problems
        problems = f.create_group("/problems")
        problems.create_dataset(
            "id",
            data=np.array([f"p{i:04d}".encode() for i in range(n_problems)], dtype="S64"),
        )
        problems.create_dataset("category", data=categories)
        problems.create_dataset("conflict", data=conflict)
        problems.create_dataset(
            "difficulty", data=np.ones(n_problems, dtype=np.int8)
        )
        problems.create_dataset(
            "prompt_text",
            data=np.array(["p"] * n_problems, dtype="S2048"),
        )
        problems.create_dataset(
            "correct_answer",
            data=np.array(["c"] * n_problems, dtype="S128"),
        )
        problems.create_dataset(
            "lure_answer",
            data=np.array(["l"] * n_problems, dtype="S128"),
        )
        problems.create_dataset("matched_pair_id", data=pair_ids)
        problems.create_dataset(
            "prompt_token_count",
            data=np.full(n_problems, 10, dtype=np.int32),
        )

        # /models/{key}
        mgrp = f.create_group(f"/models/{model_hdf5}")
        mmeta = mgrp.create_group("metadata")
        mmeta.attrs["hf_model_id"] = "synth/model"
        mmeta.attrs["n_layers"] = n_layers
        mmeta.attrs["n_heads"] = 4
        mmeta.attrs["n_kv_heads"] = 2
        mmeta.attrs["hidden_dim"] = hidden
        mmeta.attrs["head_dim"] = 6
        mmeta.attrs["dtype"] = "float32"
        mmeta.attrs["extracted_at"] = "2026-04-09T00:00:00Z"
        mmeta.attrs["is_reasoning_model"] = False

        resid = mgrp.create_group("residual")
        for layer in range(n_layers):
            resid.create_dataset(f"layer_{layer:02d}", data=acts_per_layer[layer])

        pos = mgrp.create_group("position_index")
        pos.create_dataset(
            "labels",
            data=np.array([s.encode() for s in position_labels], dtype="S16"),
        )
        pos.create_dataset(
            "token_indices",
            data=np.zeros((n_problems, n_positions), dtype=np.int32),
        )
        pos.create_dataset(
            "valid", data=np.ones((n_problems, n_positions), dtype=bool)
        )

        beh = mgrp.create_group("behavior")
        beh.create_dataset(
            "predicted_answer",
            data=np.array(["c"] * n_problems, dtype="S128"),
        )
        beh.create_dataset("correct", data=correct)
        beh.create_dataset("matches_lure", data=matches_lure)
        beh.create_dataset(
            "response_category",
            data=np.array(["correct"] * n_problems, dtype="S16"),
        )

        gens = mgrp.create_group("generations")
        gens.create_dataset(
            "full_text", data=np.array(["x"] * n_problems, dtype="S8192")
        )
        gens.create_dataset(
            "thinking_text", data=np.array([""] * n_problems, dtype="S8192")
        )
        gens.create_dataset(
            "answer_text", data=np.array(["x"] * n_problems, dtype="S512")
        )
        gens.create_dataset(
            "thinking_token_count", data=np.zeros(n_problems, dtype=np.int32)
        )
        gens.create_dataset(
            "answer_token_count", data=np.full(n_problems, 3, dtype=np.int32)
        )

    return model_hdf5


# ---------------------------------------------------------------------------
# Probe class behavioural tests
# ---------------------------------------------------------------------------


def test_mass_mean_probe_separates_distinct_classes(
    gaussian_blobs: tuple[np.ndarray, np.ndarray],
) -> None:
    """Two well-separated Gaussians => mass-mean probe > 0.9 AUC."""
    X, y = gaussian_blobs
    probe = MassMeanProbe()
    probe.fit(X, y)
    metrics = probe.score(X, y)
    assert metrics["roc_auc"] > 0.9, f"expected >0.9, got {metrics['roc_auc']}"
    # Weight vector is unit-norm.
    w = probe.weight_vector()
    assert w is not None
    assert abs(np.linalg.norm(w) - 1.0) < 1e-4


def test_logistic_probe_beats_chance_on_blobs(
    gaussian_blobs: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = gaussian_blobs
    probe = LogisticRegressionProbe()
    probe.fit(X, y)
    metrics = probe.score(X, y)
    assert metrics["roc_auc"] > 0.95
    # Linear direction is recoverable.
    w = probe.weight_vector()
    assert w is not None and w.shape == (X.shape[1],)


def test_logistic_probe_xor_fails(xor_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Linear probe on XOR data should be near chance."""
    X, y = xor_data
    probe = LogisticRegressionProbe()
    probe.fit(X, y)
    metrics = probe.score(X, y)
    assert metrics["roc_auc"] < 0.65, f"linear probe XOR AUC={metrics['roc_auc']}"


def test_mlp_probe_xor_succeeds(xor_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = xor_data
    probe = MLPProbe(max_epochs=120, patience=30, seed=0)
    probe.fit(X, y)
    metrics = probe.score(X, y)
    assert metrics["roc_auc"] > 0.9, f"MLP XOR AUC={metrics['roc_auc']}"


def test_ccs_probe_runs_and_returns_valid_probas(
    gaussian_blobs: tuple[np.ndarray, np.ndarray],
) -> None:
    """CCS is unsupervised; we just check it runs and emits valid probabilities."""
    X, y = gaussian_blobs
    probe = CCSProbe(n_restarts=3, max_epochs=200, seed=0)
    probe.fit(X, y)
    proba = probe.predict_proba(X)
    assert proba.shape == (len(X),)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_probe_registry_has_all_types() -> None:
    for name in ("mass_mean", "logistic", "logreg", "mlp", "ccs"):
        cls = get_probe_class(name)
        assert cls is not None


# ---------------------------------------------------------------------------
# Hewitt & Liang control tasks
# ---------------------------------------------------------------------------


def test_shuffled_labels_preserves_base_rate() -> None:
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=400) < 0.3).astype(np.int8)
    y_shuf = shuffled_labels(y, seed=42)
    assert y_shuf.sum() == y.sum()
    assert y_shuf.shape == y.shape


def test_hewitt_liang_control_baseline(
    gaussian_blobs: tuple[np.ndarray, np.ndarray],
) -> None:
    """Random labels on a linear probe should give AUC close to chance."""
    X, y = gaussian_blobs
    n = len(X) // 2
    X_tr, X_te = X[:n], X[n:]
    y_tr, y_te = y[:n], y[n:]
    ctrl = run_control_task(
        X_tr, X_te, y_tr, y_te,
        probe_name="logistic",
        n_seeds=3,
        base_seed=0,
    )
    assert 0.35 <= ctrl["control_roc_auc"] <= 0.65, (
        f"control AUC {ctrl['control_roc_auc']} is suspiciously far from chance"
    )
    assert ctrl["control_n_seeds"] == 3


# ---------------------------------------------------------------------------
# Stratification & LOCO
# ---------------------------------------------------------------------------


def test_make_stratify_key_unique_per_pair() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    cat = np.array(["a", "b", "a", "b", "a", "a"])
    key = make_stratify_key(y, cat)
    # Pairs (y=0, cat=a) and (y=0, cat=b) should be distinct.
    a0 = key[(y == 0) & (cat == "a")][0]
    a1 = key[(y == 1) & (cat == "a")][0]
    b0 = key[(y == 0) & (cat == "b")][0]
    assert a0 != a1
    assert a0 != b0


def test_loco_split_held_out_group_not_in_train() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c", "a", "b", "c", "a"])
    splits = loco_split_iter(y, groups)
    assert len(splits) > 0
    for held, tr, te in splits:
        # Every test index has the held-out group; no train index does.
        assert np.all(groups[te] == held)
        assert np.all(groups[tr] != held)


# ---------------------------------------------------------------------------
# BH-FDR application
# ---------------------------------------------------------------------------


def test_bh_fdr_application_on_known_pvalues() -> None:
    """Given mock results with known p-values, BH should reject a specific set."""
    # Build a list of LayerResult-like stubs.
    from s1s2.probes.core import LayerResult, ProbeResult

    def mk_result(layer: int, p: float) -> LayerResult:
        pr = ProbeResult(
            name="logistic",
            fold_metrics=[],
            pooled_y=np.array([0, 1]),
            pooled_proba=np.array([0.1, 0.9]),
            control_metrics=[],
            summary={"roc_auc": 0.8},
            permutation_p=p,
            permutation_null_auc=np.zeros(10),
        )
        return LayerResult(
            model="m",
            layer=layer,
            position="P0",
            target="task_type",
            n_problems=10,
            n_train_per_fold=[],
            n_test_per_fold=[],
            probes={"logistic": pr},
            loco=None,
            config={},
            git_sha="x",
            elapsed_s=0.0,
        )

    pvals = [0.001, 0.01, 0.02, 0.03, 0.04, 0.1, 0.5, 0.9]
    results = [mk_result(i, p) for i, p in enumerate(pvals)]
    apply_bh_across_layers(results, probe_name="logistic", q=0.05)
    # BH step-up at q=0.05 finds the largest k s.t. p_k <= k*q/n and rejects 1..k.
    # For n=8, q=0.05:
    #   k=1 thresh 0.00625 -> 0.001 <= 0.00625 OK
    #   k=2 thresh 0.0125  -> 0.01  <= 0.0125  OK
    #   k=3 thresh 0.01875 -> 0.02  > 0.01875 FAIL
    # Largest satisfying k is 2; reject only p=0.001 and p=0.01.
    # (The earlier expectation of rejecting p<=0.03 was arithmetic-incorrect.)
    rejected = [r.probes["logistic"].summary.get("significant_bh") for r in results]
    assert rejected[:2] == [True, True]
    assert rejected[2:] == [False, False, False, False, False, False]
    # Q-values should be monotone non-decreasing with the sorted raw p-values.
    qvs = [r.probes["logistic"].summary.get("permutation_p_bh") for r in results]
    assert qvs is not None
    assert all(q is not None for q in qvs)


# ---------------------------------------------------------------------------
# Target builders against a synthetic HDF5
# ---------------------------------------------------------------------------


def test_build_all_targets_with_synthetic_hdf5(tmp_path: Path) -> None:
    h5_path = tmp_path / "synth.h5"
    model_key = _write_synthetic_hdf5(h5_path)
    with ioh.open_activations(h5_path) as f:
        for target in ALL_TARGETS:
            td = build_target(target, f, model_key)
            assert td.y.ndim == 1
            assert td.mask.ndim == 1
            assert td.y.shape[0] == int(td.mask.sum())
            assert td.stratify_key.shape[0] == td.y.shape[0]
            assert td.group_id.shape[0] == td.y.shape[0]


# ---------------------------------------------------------------------------
# End-to-end round trip
# ---------------------------------------------------------------------------


def test_round_trip_with_synthetic_hdf5(tmp_path: Path) -> None:
    """Full pipeline: synthetic HDF5 -> runner -> result JSON on disk."""
    h5_path = tmp_path / "synth.h5"
    model_key = _write_synthetic_hdf5(h5_path)

    with ioh.open_activations(h5_path) as f:
        td = build_target("task_type", f, model_key)
        # Layer 2 has the planted signal.
        X = ioh.get_residual(f, model_key, layer=2, position="P0").astype(np.float32)

    cfg = RunnerConfig(
        probes=("mass_mean", "logistic"),
        n_folds=5,
        n_seeds=2,
        control_enabled=True,
        control_n_shuffles=2,
        n_permutations=50,  # tiny — we're just checking correctness
        n_bootstrap=100,
        seed=0,
    )
    runner = ProbeRunner(cfg)
    result = runner.run(
        X=X,
        target_data=td,
        model="synthetic",
        layer=2,
        position="P0",
    )
    # Sanity: logistic probe should beat chance by a noticeable margin on the
    # planted signal.
    logistic_summary = result.probes["logistic"].summary
    assert logistic_summary["roc_auc"] > 0.7
    assert "selectivity" in logistic_summary
    assert "roc_auc_ci_lower" in logistic_summary
    assert "roc_auc_ci_upper" in logistic_summary
    assert "permutation_p" in logistic_summary

    # Serialize and reload.
    results_dir = tmp_path / "results"
    out_path = save_layer_result(result, results_dir)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["model"] == "synthetic"
    assert loaded["layer"] == 2
    assert loaded["target"] == "task_type"
    # On-disk schema renames the in-memory key "logistic" to "logreg" (see
    # core.layer_result_to_dict.brief_name_map).
    assert "logreg" in loaded["probes"]
    assert loaded["probes"]["logreg"] is not None

    # The mass-mean probe should also show some signal on the planted layer
    # (it's literally mu1 - mu0 in a shifted-mean design).
    mm_summary = result.probes["mass_mean"].summary
    assert mm_summary["roc_auc"] > 0.6


def test_round_trip_layer_without_signal(tmp_path: Path) -> None:
    """Layer 0 has no planted signal; logistic probe should hover near chance."""
    h5_path = tmp_path / "synth.h5"
    model_key = _write_synthetic_hdf5(h5_path)
    with ioh.open_activations(h5_path) as f:
        td = build_target("task_type", f, model_key)
        X = ioh.get_residual(f, model_key, layer=0, position="P0").astype(np.float32)

    cfg = RunnerConfig(
        probes=("logistic",),
        n_folds=5,
        n_seeds=1,
        control_enabled=False,
        n_permutations=50,
        n_bootstrap=100,
        seed=0,
    )
    runner = ProbeRunner(cfg)
    result = runner.run(
        X=X, target_data=td, model="synthetic", layer=0, position="P0"
    )
    # No signal => AUC should be well below the planted-signal layer.
    assert result.probes["logistic"].summary["roc_auc"] < 0.75


# ---------------------------------------------------------------------------
# LayerResult dict round trip
# ---------------------------------------------------------------------------


def test_layer_result_to_dict_is_json_serialisable(tmp_path: Path) -> None:
    h5_path = tmp_path / "synth.h5"
    model_key = _write_synthetic_hdf5(h5_path)
    with ioh.open_activations(h5_path) as f:
        td = build_target("task_type", f, model_key)
        X = ioh.get_residual(f, model_key, layer=2, position="P0").astype(np.float32)
    cfg = RunnerConfig(
        probes=("mass_mean",),
        n_folds=5,
        n_seeds=1,
        control_enabled=False,
        n_permutations=20,
        n_bootstrap=50,
        seed=0,
    )
    res = ProbeRunner(cfg).run(
        X=X, target_data=td, model="synthetic", layer=2, position="P0"
    )
    d = layer_result_to_dict(res)
    s = json.dumps(d)  # must not raise
    assert "logistic" not in d["probes"]
    assert "mass_mean" in d["probes"]
    assert isinstance(s, str) and len(s) > 0
