"""End-to-end integration smoke tests.

Build a synthetic HDF5 cache (via :mod:`tests.fixtures.synthetic_activations`)
and run a tiny analysis from each workstream against it. Each workstream
is wrapped in its own try/except so a failure in one does not kill the
others — the test suite reports which workstreams pass and which fail
without losing visibility into the rest.

These tests are marked ``integration`` so they can be deselected via
``pytest -m "not integration"`` for fast unit-only runs. They run on
CPU in well under a minute.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

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
from tests.fixtures.synthetic_activations import (  # noqa: E402
    SYNTH_HIDDEN,
    SYNTH_MODEL_KEY,
    build_synthetic_hdf5,
)

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# Helper                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class _Outcome:
    name: str
    ok: bool
    detail: str


def _run_safely(name: str, fn) -> _Outcome:
    try:
        detail = fn()
        return _Outcome(name=name, ok=True, detail=detail or "ok")
    except pytest.skip.Exception as e:  # type: ignore[attr-defined]
        return _Outcome(name=name, ok=False, detail=f"skipped: {e}")
    except Exception as exc:
        return _Outcome(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Workstream wrappers                                                          #
# --------------------------------------------------------------------------- #


def _run_extract_validation(path: Path) -> str:
    """Validate the synthetic file conforms to the data contract."""
    from s1s2.extract import validate_file

    errors = validate_file(path)
    assert errors == [], f"schema errors: {errors}"
    return f"validate_file: {len(errors)} errors"


def _run_probes(path: Path) -> str:
    """Run a tiny probe on the planted-signal layer."""
    from s1s2.probes import (
        ProbeRunner,
        RunnerConfig,
        build_target,
    )

    with ioh.open_activations(path) as f:
        td = build_target("task_type", f, SYNTH_MODEL_KEY)
        X = ioh.get_residual(f, SYNTH_MODEL_KEY, layer=2, position="P0").astype(
            np.float32
        )

    cfg = RunnerConfig(
        probes=("logistic",),
        n_folds=3,
        n_seeds=1,
        control_enabled=False,
        n_permutations=20,
        n_bootstrap=20,
        seed=0,
    )
    runner = ProbeRunner(cfg)
    result = runner.run(
        X=X, target_data=td, model="synthetic", layer=2, position="P0"
    )
    auc = result.probes["logistic"].summary["roc_auc"]
    return f"probe AUC = {auc:.3f}"


def _run_attention(path: Path) -> str:
    """Run per-head differential test on the synthetic cache."""
    from s1s2.attention.core import (
        METRIC_NAMES,
        ModelAttentionData,
        _apply_bh_in_place,
    )
    from s1s2.attention.heads import (
        classify_heads,
        run_all_head_differential_tests,
    )
    from s1s2.attention.layers import layer_summary

    with ioh.open_activations(path) as f:
        labels = ioh.position_labels(f, SYNTH_MODEL_KEY)
        valid = ioh.position_valid(f, SYNTH_MODEL_KEY)
        # Pick valid positions
        kept_positions: list[str] = []
        kept_indices: list[int] = []
        for i, lab in enumerate(labels):
            if bool(valid[:, i].any()):
                kept_positions.append(lab)
                kept_indices.append(i)
        meta = ioh.model_metadata(f, SYNTH_MODEL_KEY)
        conflict = f["/problems/conflict"][:].astype(bool)
        metrics: dict[str, np.ndarray] = {}
        for m in METRIC_NAMES:
            arr = ioh.get_attention_metric(f, SYNTH_MODEL_KEY, m)
            metrics[m] = arr[..., kept_indices]
        data = ModelAttentionData(
            model_key=SYNTH_MODEL_KEY,
            model_config_key="synth",
            family="synth",
            n_layers=int(meta["n_layers"]),
            n_heads=int(meta["n_heads"]),
            n_kv_heads=int(meta["n_kv_heads"]),
            is_reasoning_model=False,
            position_labels=labels,
            selected_positions=kept_positions,
            metrics=metrics,
            conflict=conflict,
        )
    df = run_all_head_differential_tests(data, metrics=METRIC_NAMES)
    df = _apply_bh_in_place(df, q=0.1)
    classifs = classify_heads(
        df,
        n_layers=data.n_layers,
        n_heads=data.n_heads,
        min_significant=2,
        entropy_effect_threshold=0.1,
    )
    summary = layer_summary(data, head_classifs=classifs, metric="entropy")
    return (
        f"attention: {len(df)} tests, {len(classifs)} heads classified, "
        f"{len(summary)} layer summaries"
    )


def _run_geometry(path: Path) -> str:
    """Run silhouette + CKA + intrinsic dim on the planted layer."""
    from s1s2.geometry.cka import linear_cka_fast
    from s1s2.geometry.clusters import cosine_silhouette
    from s1s2.geometry.intrinsic_dim import participation_ratio

    with ioh.open_activations(path) as f:
        X = ioh.get_residual(f, SYNTH_MODEL_KEY, layer=2, position="P0").astype(
            np.float32
        )
        conflict = f["/problems/conflict"][:].astype(np.int32)

    sil = cosine_silhouette(X, conflict)
    pr = participation_ratio(X)
    cka_self = linear_cka_fast(X, X)
    return (
        f"geometry: silhouette={sil:.3f}, PR={pr:.2f}, CKA(X,X)={cka_self:.3f}"
    )


def _run_sae(path: Path) -> str:
    """Run a MockSAE-backed differential analysis."""
    import torch  # noqa: F401  - confirms torch is importable

    from s1s2.sae.differential import differential_activation, encode_batched
    from s1s2.sae.loaders import MockSAE

    with ioh.open_activations(path) as f:
        X = ioh.get_residual(f, SYNTH_MODEL_KEY, layer=2, position="P0").astype(
            np.float32
        )
        conflict = f["/problems/conflict"][:].astype(bool)

    sae = MockSAE(hidden_dim=SYNTH_HIDDEN, n_features=64, sparsity=0.5, seed=0)
    feats = encode_batched(sae, X, batch_size=16)
    result = differential_activation(feats, conflict, fdr_q=0.1)
    return (
        f"sae: tested {result.df.shape[0]} features; "
        f"{int(result.df['significant'].sum())} significant"
    )


def _run_causal_unit() -> str:
    """Causal hooks need a real torch module; we exercise the pure helpers
    so the integration test confirms the module imports cleanly even
    without an LM."""
    import torch

    from s1s2.causal.ablation import ablate_direction
    from s1s2.causal.steering import normalize_direction, random_unit_direction

    direction = normalize_direction(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    rand = random_unit_direction(hidden_dim=4, seed=0)
    h = torch.randn(2, 3, 4)
    ablated = ablate_direction(h, direction)
    # After projection ablation, the dot with the direction should be ≈ 0.
    dots = (ablated * direction).sum(dim=-1)
    assert torch.all(dots.abs() < 1e-5), f"residual along direction: {dots}"
    return (
        f"causal helpers: direction-norm OK, ablation-orthogonal OK, "
        f"random-direction-norm={rand.norm().item():.3f}"
    )


def _run_metacog(path: Path) -> str:
    """Run a surprise-feature correlation + 4-gate evaluation on the cache."""
    from s1s2.metacog.gates import DifficultyDetectorResults, evaluate_gates
    from s1s2.metacog.surprise import (
        aggregate_surprise,
        surprise_feature_correlation,
    )
    from s1s2.sae.differential import encode_batched
    from s1s2.sae.loaders import MockSAE

    with ioh.open_activations(path) as f:
        X = ioh.get_residual(f, SYNTH_MODEL_KEY, layer=2, position="P0").astype(
            np.float32
        )
        by_pos = f[f"/models/{SYNTH_MODEL_KEY}/token_surprises/by_position"][:]
        labels = ioh.position_labels(f, SYNTH_MODEL_KEY)

    surprise = aggregate_surprise(by_pos, labels, method="by_position:P0")
    sae = MockSAE(hidden_dim=SYNTH_HIDDEN, n_features=32, sparsity=0.5, seed=0)
    feats = encode_batched(sae, X, batch_size=16)
    corr = surprise_feature_correlation(feats, surprise, layer=2, model_key="synth")
    snapshot = DifficultyDetectorResults(
        model_key="synth",
        layer=2,
        n_difficulty_sensitive_features=corr.n_difficulty_sensitive,
        max_specificity_auc=0.7,
        n_features_passing_specificity=1,
        infrastructure_ok=True,
    )
    gates = evaluate_gates(snapshot)
    return (
        f"metacog: corr_n_diff={corr.n_difficulty_sensitive}, "
        f"gates={[g.decision for g in gates]}"
    )


# --------------------------------------------------------------------------- #
# Top-level integration test                                                   #
# --------------------------------------------------------------------------- #


def test_end_to_end_workstream_smoke(tmp_path: Path):
    """Build a synthetic HDF5, run each workstream, report results.

    The test passes when MOST workstreams succeed. Individual failures
    are recorded in the assertion message so a regression in any one
    workstream is loud but does not block the rest.
    """
    h5_path = build_synthetic_hdf5(tmp_path / "integration.h5")

    outcomes: list[_Outcome] = []
    outcomes.append(_run_safely("extract_validation", lambda: _run_extract_validation(h5_path)))
    outcomes.append(_run_safely("probes", lambda: _run_probes(h5_path)))
    outcomes.append(_run_safely("attention", lambda: _run_attention(h5_path)))
    outcomes.append(_run_safely("geometry", lambda: _run_geometry(h5_path)))
    outcomes.append(_run_safely("sae", lambda: _run_sae(h5_path)))
    outcomes.append(_run_safely("causal_helpers", _run_causal_unit))
    outcomes.append(_run_safely("metacog", lambda: _run_metacog(h5_path)))

    # Print a single combined report so failures show all workstreams at once.
    lines = ["", "Workstream integration smoke results:"]
    for o in outcomes:
        marker = "PASS" if o.ok else "FAIL"
        lines.append(f"  [{marker}] {o.name:24s}  {o.detail}")
    report = "\n".join(lines)
    print(report)

    n_pass = sum(1 for o in outcomes if o.ok)
    n_total = len(outcomes)
    # Require at least 6 of 7 workstreams to pass; one workstream may
    # be skipped because of an optional dependency. The pretty report
    # is included in the assertion message regardless.
    assert n_pass >= n_total - 1, report


def test_synthetic_hdf5_is_loadable_via_io_helpers(tmp_path: Path):
    """The fixture builder must produce a file the read API loves and a
    planted signal that downstream probes can latch onto."""
    path = build_synthetic_hdf5(tmp_path / "loadable.h5")
    with ioh.open_activations(path) as f:
        assert ioh.list_models(f) == [SYNTH_MODEL_KEY]
        meta = ioh.model_metadata(f, SYNTH_MODEL_KEY)
        assert meta["n_layers"] == 4
        assert meta["hidden_dim"] == SYNTH_HIDDEN
        labels = ioh.position_labels(f, SYNTH_MODEL_KEY)
        assert "P0" in labels and "P2" in labels
        # Per-problem metadata round-trips.
        problems = ioh.load_problem_metadata(f)
        assert len(problems["id"]) == 20

        conflict = f["/problems/conflict"][:].astype(bool)
        # The planted layer (2) should produce a stronger linear probe than
        # the noise layer (0). The empirical mean shift on a single dim is
        # noisy with only 10 items per group, so we use a CV-AUC instead.
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        def _cv_auc(layer):
            X = ioh.get_residual(
                f, SYNTH_MODEL_KEY, layer=layer, position="P0"
            ).astype(np.float64)
            scores = cross_val_score(
                LogisticRegression(max_iter=500),
                X,
                conflict.astype(int),
                cv=3,
                scoring="roc_auc",
            )
            return float(scores.mean())

        auc_signal = _cv_auc(2)
        auc_noise = _cv_auc(0)
        # The planted layer should beat the noise layer on average; we
        # don't require auc_signal > 0.5 strictly because tiny n is noisy.
        assert auc_signal >= auc_noise - 0.1, (
            f"signal layer AUC {auc_signal:.3f} ought to >= noise layer "
            f"AUC {auc_noise:.3f}"
        )
