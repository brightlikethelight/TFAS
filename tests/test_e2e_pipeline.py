"""End-to-end integration test: tiny model -> extraction -> all workstreams.

Unlike the existing smoke test (which uses synthetic HDF5 data), this test
drives the REAL pipeline: load a tiny model, generate responses, extract
activations into HDF5, then feed that HDF5 through probes, SAE, attention,
and geometry analysis. Results are meaningless (random weights) but the
plumbing must not crash.

Uses ``sshleifer/tiny-gpt2`` (2-layer GPT-2, hidden_size=2) on CPU. The
model is cached locally after first download.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

# macOS OpenMP guard
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Allow import without pip install -e .
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from s1s2.extract import (  # noqa: E402 — sys.path must be patched first
    ActivationWriter,
    ExtractionConfig,
    GenerationConfig,
    ModelSpec,
    RunMetadata,
    SingleModelExtractor,
    build_problem_metadata_from_items,
    validate_file,
)
from s1s2.utils.io import (  # noqa: E402
    get_attention_metric,
    get_residual,
    open_activations,
    position_labels,
    position_valid,
)
from s1s2.utils.types import ALL_POSITIONS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TINY_MODEL_ID = "sshleifer/tiny-gpt2"
TINY_HDF5_KEY = "sshleifer_tiny-gpt2"
BENCHMARK_PATH = _REPO / "data" / "benchmark" / "benchmark.jsonl"

# We select 5 items: 2 conflict + 3 no-conflict from different categories.
# Indices are chosen to span crt, base_rate, syllogism.
SELECTED_IDS = [
    "crt_ratio_coffee_pastry_conflict",        # conflict, crt
    "crt_ratio_brush_paint_conflict",          # conflict, crt
    "crt_ratio_coffee_pastry_control",         # no-conflict, crt
]
# We'll add 2 more from different categories dynamically.


def _load_benchmark_items(n_conflict: int = 2, n_control: int = 3) -> list[SimpleNamespace]:
    """Load a curated subset from the real benchmark JSONL.

    Picks items from different categories for coverage.
    """
    if not BENCHMARK_PATH.exists():
        pytest.skip(f"benchmark file not found at {BENCHMARK_PATH}")

    all_items: list[dict] = []
    with BENCHMARK_PATH.open() as f:
        for line in f:
            all_items.append(json.loads(line))

    conflict_items: list[dict] = []
    control_items: list[dict] = []
    seen_cats_conflict: set[str] = set()
    seen_cats_control: set[str] = set()

    for item in all_items:
        cat = item["category"]
        if item["conflict"] and cat not in seen_cats_conflict and len(conflict_items) < n_conflict:
            conflict_items.append(item)
            seen_cats_conflict.add(cat)
        elif not item["conflict"] and cat not in seen_cats_control and len(control_items) < n_control:
            control_items.append(item)
            seen_cats_control.add(cat)
        if len(conflict_items) >= n_conflict and len(control_items) >= n_control:
            break

    items = conflict_items + control_items
    assert len(items) == n_conflict + n_control, (
        f"expected {n_conflict + n_control} items, got {len(items)}"
    )
    return [SimpleNamespace(**d) for d in items]


# ---------------------------------------------------------------------------
# Shared fixture: run extraction once, reuse HDF5 across test functions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def e2e_hdf5(tmp_path_factory) -> Path:
    """Run extraction with tiny-gpt2 on 5 benchmark items. Returns HDF5 path."""
    try:
        import torch
        from transformers import (  # noqa: F401 — availability check
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError:
        pytest.skip("transformers/torch not available")

    items = _load_benchmark_items(n_conflict=2, n_control=3)

    spec = ModelSpec(
        key="tiny-gpt2",
        hdf5_key=TINY_HDF5_KEY,
        hf_id=TINY_MODEL_ID,
        family="gpt2",
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        hidden_dim=2,
        head_dim=1,
        is_reasoning=False,
    )
    gen_cfg = GenerationConfig(
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        seed=42,
    )
    extr_cfg = ExtractionConfig(
        dtype="float16",
        attn_implementation="eager",
        log_every=1,
    )

    extractor = SingleModelExtractor(
        spec=spec,
        generation_cfg=gen_cfg,
        extraction_cfg=extr_cfg,
        device="cpu",
        torch_dtype=torch.float32,
    )
    extractor.load()

    hidden_dim = extractor.model.config.hidden_size
    n_layers = spec.n_layers
    n_heads = spec.n_heads

    out_dir = tmp_path_factory.mktemp("e2e")
    out_path = out_dir / "e2e_activations.h5"

    with ActivationWriter(out_path) as writer:
        writer.write_run_metadata(
            RunMetadata.build(
                benchmark_path=str(BENCHMARK_PATH),
                seed=42,
                config_json=json.dumps({"test": "e2e_pipeline"}),
            )
        )

        prompt_counts = []
        for item in items:
            ids = extractor.tokenizer(item.prompt, return_tensors="pt").input_ids
            prompt_counts.append(int(ids.shape[1]))

        writer.write_problems(build_problem_metadata_from_items(items, prompt_counts))

        writer.create_model_group(
            model_key=spec.hdf5_key,
            hf_model_id=spec.hf_id,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=spec.n_kv_heads,
            hidden_dim=hidden_dim,
            head_dim=spec.head_dim,
            dtype="float16",
            is_reasoning_model=False,
        )

        extractor.run(
            items=items,
            writer=writer,
            effective_n_layers=n_layers,
            effective_n_heads=n_heads,
            effective_hidden_dim=hidden_dim,
        )

    extractor.unload()
    return out_path


@pytest.fixture(scope="module")
def e2e_items() -> list[SimpleNamespace]:
    """The benchmark items used for extraction."""
    return _load_benchmark_items(n_conflict=2, n_control=3)


# ---------------------------------------------------------------------------
# Test 1: HDF5 schema conformance
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
class TestE2ESchemaConformance:
    """Verify the extracted HDF5 conforms to data_contract.md."""

    def test_validate_file_passes(self, e2e_hdf5: Path):
        errors = validate_file(e2e_hdf5)
        assert errors == [], f"schema validation errors: {errors}"

    def test_metadata_and_problems(self, e2e_hdf5: Path):
        with open_activations(e2e_hdf5) as f:
            assert f["/metadata"].attrs["schema_version"] == 1
            assert f["/problems/id"].shape[0] == 5
            # Categories are real benchmark categories
            cats = [c.decode("utf-8") for c in f["/problems/category"][:]]
            assert all(
                c in ("crt", "base_rate", "syllogism", "anchoring", "framing",
                       "conjunction", "arithmetic")
                for c in cats
            )

    def test_residual_shapes(self, e2e_hdf5: Path):
        with open_activations(e2e_hdf5) as f:
            n_pos = len(ALL_POSITIONS)
            resid = get_residual(f, TINY_HDF5_KEY, layer=0)
            assert resid.shape[0] == 5
            assert resid.shape[1] == n_pos
            # hidden_dim is whatever tiny-gpt2 has (2)
            assert resid.shape[2] > 0
            assert resid.dtype == np.float16

    def test_attention_shapes(self, e2e_hdf5: Path):
        with open_activations(e2e_hdf5) as f:
            ent = get_attention_metric(f, TINY_HDF5_KEY, "entropy")
            assert ent.ndim == 4
            assert ent.shape[0] == 5  # n_problems
            assert ent.shape[3] == len(ALL_POSITIONS)
            assert np.all(np.isfinite(ent))

    def test_position_labels(self, e2e_hdf5: Path):
        with open_activations(e2e_hdf5) as f:
            labels = position_labels(f, TINY_HDF5_KEY)
            assert labels == list(ALL_POSITIONS)

    def test_behavior_fields(self, e2e_hdf5: Path):
        with open_activations(e2e_hdf5) as f:
            cats = f[f"/models/{TINY_HDF5_KEY}/behavior/response_category"][:]
            assert cats.shape == (5,)
            for c in cats:
                assert c.decode("utf-8") in ("correct", "lure", "other_wrong", "refusal")

    def test_generations_written(self, e2e_hdf5: Path):
        with open_activations(e2e_hdf5) as f:
            full = f[f"/models/{TINY_HDF5_KEY}/generations/full_text"][:]
            assert full.shape == (5,)
            # Each generation should be non-empty (tiny-gpt2 always produces something)
            for g in full:
                text = g.decode("utf-8") if isinstance(g, bytes) else str(g)
                assert len(text) > 0

    def test_surprises_written(self, e2e_hdf5: Path):
        with open_activations(e2e_hdf5) as f:
            by_pos = f[f"/models/{TINY_HDF5_KEY}/token_surprises/by_position"][:]
            assert by_pos.shape == (5, len(ALL_POSITIONS))
            offsets = f[f"/models/{TINY_HDF5_KEY}/token_surprises/full_trace_offsets"][:]
            assert offsets.shape == (6,)  # n_problems + 1
            assert offsets[0] == 0


# ---------------------------------------------------------------------------
# Test 2: Probing pipeline
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
class TestE2EProbes:
    """Run probes on the extracted activations."""

    def test_probe_runner_completes(self, e2e_hdf5: Path):
        """Train logistic probes at all layers, position P0, target task_type.

        With 5 items the probes will have near-chance AUC but must not crash.
        We use logistic only (not mass_mean) because mass_mean produces NaN
        when a 2-fold split leaves one fold with a single class and the class-1
        centroid is computed from an empty slice. Logistic regression handles
        this gracefully internally.
        """
        from s1s2.probes.core import ProbeRunner, RunnerConfig
        from s1s2.probes.targets import build_target

        with open_activations(e2e_hdf5) as f:
            td = build_target("task_type", f, TINY_HDF5_KEY)
            n_layers = int(f[f"/models/{TINY_HDF5_KEY}/metadata"].attrs["n_layers"])

            # Minimal config: 2 folds (we only have 5 items), 1 seed, no LOCO.
            # Use logistic only. The inner LogisticRegressionCV also needs
            # cv=2 because with 2 outer folds each training set has ~3 items,
            # which is too small for the default inner cv=5.
            config = RunnerConfig(
                probes=("logistic",),
                n_folds=2,
                n_seeds=1,
                control_enabled=False,
                n_permutations=10,
                n_bootstrap=10,
                run_loco=False,
                seed=0,
                probe_kwargs={"logistic": {"cv": 2}},
            )
            runner = ProbeRunner(config)

            results = []
            for layer in range(n_layers):
                X = get_residual(f, TINY_HDF5_KEY, layer=layer, position="P0")
                X = X.astype(np.float32, copy=False)
                result = runner.run(
                    X=X,
                    target_data=td,
                    model=TINY_HDF5_KEY,
                    layer=layer,
                    position="P0",
                )
                results.append(result)

        assert len(results) == n_layers
        for r in results:
            assert "logistic" in r.probes
            auc = r.probes["logistic"].summary.get("roc_auc", None)
            assert auc is not None
            assert 0.0 <= auc <= 1.0

    def test_probe_result_serializes(self, e2e_hdf5: Path, tmp_path: Path):
        """Verify probe results can be saved as JSON."""
        from s1s2.probes.core import ProbeRunner, RunnerConfig, save_layer_result
        from s1s2.probes.targets import build_target

        with open_activations(e2e_hdf5) as f:
            td = build_target("task_type", f, TINY_HDF5_KEY)
            config = RunnerConfig(
                probes=("logistic",),
                n_folds=2,
                n_seeds=1,
                control_enabled=False,
                n_permutations=5,
                n_bootstrap=5,
                run_loco=False,
                seed=0,
                probe_kwargs={"logistic": {"cv": 2}},
            )
            runner = ProbeRunner(config)
            X = get_residual(f, TINY_HDF5_KEY, layer=0, position="P0")
            X = X.astype(np.float32, copy=False)
            result = runner.run(X=X, target_data=td, model=TINY_HDF5_KEY, layer=0, position="P0")

        out_dir = tmp_path / "probe_results"
        path = save_layer_result(result, out_dir)
        assert path.exists()
        with path.open() as fh:
            data = json.load(fh)
        assert data["model"] == TINY_HDF5_KEY
        assert data["target"] == "task_type"
        assert "probes" in data


# ---------------------------------------------------------------------------
# Test 3: SAE analysis with MockSAE
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
class TestE2ESAE:
    """Run SAE differential analysis using MockSAE."""

    def test_mock_sae_differential(self, e2e_hdf5: Path):
        """Encode activations with MockSAE, run differential analysis.

        With random weights + 5 items this will find 0 significant features,
        but the pipeline must run without error.
        """
        from s1s2.sae.differential import differential_activation, encode_batched
        from s1s2.sae.loaders import MockSAE, reconstruction_report

        with open_activations(e2e_hdf5) as f:
            hidden_dim = int(f[f"/models/{TINY_HDF5_KEY}/metadata"].attrs["hidden_dim"])
            X = get_residual(f, TINY_HDF5_KEY, layer=0, position="P0")
            X = X.astype(np.float32, copy=False)
            conflict = f["/problems/conflict"][:].astype(bool)

        # MockSAE with n_features > hidden_dim for exact reconstruction
        sae = MockSAE(hidden_dim=hidden_dim, n_features=max(64, hidden_dim * 2), layer=0, seed=0)

        # Reconstruction fidelity check
        report = reconstruction_report(sae, X, min_explained_variance=0.01, n_samples=min(5, X.shape[0]))
        assert report.n_samples <= X.shape[0]
        # With MockSAE and n_features >= hidden_dim, reconstruction should be decent
        assert report.explained_variance > -1.0  # just check it's a number

        # Encode and run differential
        feature_acts = encode_batched(sae, X)
        assert feature_acts.shape == (5, sae.n_features)

        result = differential_activation(
            feature_acts, conflict, fdr_q=0.05, subset_label="all"
        )
        assert result.df.shape[0] == sae.n_features
        assert "p_value" in result.df.columns
        assert "significant" in result.df.columns
        assert result.n_S1 + result.n_S2 == 5


# ---------------------------------------------------------------------------
# Test 4: Attention entropy analysis
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
class TestE2EAttention:
    """Run attention analysis on the cached metrics."""

    def test_attention_differential_manual(self, e2e_hdf5: Path):
        """Run per-head differential tests manually, bypassing models.yaml.

        The attention/core.py ``analyze_model`` function requires a
        models.yaml entry, which the tiny model doesn't have. Instead we
        directly build a ModelAttentionData and run the heads module.
        """
        from s1s2.attention.core import ModelAttentionData
        from s1s2.attention.heads import run_all_head_differential_tests

        with open_activations(e2e_hdf5) as f:
            labels = position_labels(f, TINY_HDF5_KEY)
            valid = position_valid(f, TINY_HDF5_KEY)
            n_layers = int(f[f"/models/{TINY_HDF5_KEY}/metadata"].attrs["n_layers"])
            n_heads = int(f[f"/models/{TINY_HDF5_KEY}/metadata"].attrs["n_heads"])
            n_kv_heads = int(f[f"/models/{TINY_HDF5_KEY}/metadata"].attrs["n_kv_heads"])
            conflict = f["/problems/conflict"][:].astype(bool)

            # Load metrics only for valid positions (P0, P2 for non-reasoning)
            selected = []
            selected_idx = []
            for pos in ("P0", "P2"):
                if pos in labels:
                    idx = labels.index(pos)
                    if valid[:, idx].any():
                        selected.append(pos)
                        selected_idx.append(idx)

            metrics: dict[str, np.ndarray] = {}
            for name in ("entropy", "entropy_normalized", "gini", "max_attn", "focus_5"):
                arr = get_attention_metric(f, TINY_HDF5_KEY, name)
                metrics[name] = arr[..., selected_idx].astype(np.float32, copy=False)

        data = ModelAttentionData(
            model_key=TINY_HDF5_KEY,
            model_config_key="tiny-gpt2-test",
            family="gpt2",
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            is_reasoning_model=False,
            position_labels=labels,
            selected_positions=selected,
            metrics=metrics,
            conflict=conflict,
        )

        df = run_all_head_differential_tests(data, metrics=("entropy", "gini"))
        assert len(df) > 0, "expected at least one test row"
        assert "p_value" in df.columns
        assert "effect_size_rb" in df.columns


# ---------------------------------------------------------------------------
# Test 5: Geometry analysis
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
class TestE2EGeometry:
    """Run geometry analysis on the extracted activations."""

    def test_silhouette_and_separability(self, e2e_hdf5: Path):
        """Compute cosine silhouette + linear separability on P0 activations.

        With 5 items in 2-dim space and random weights, results are degenerate
        but the code path must execute without error.
        """
        from s1s2.geometry.clusters import (
            calinski_harabasz,
            cosine_silhouette,
            davies_bouldin,
        )
        from s1s2.geometry.separability import linear_separability_with_d_gg_n_fix

        with open_activations(e2e_hdf5) as f:
            X = get_residual(f, TINY_HDF5_KEY, layer=0, position="P0")
            X = X.astype(np.float32, copy=False)
            conflict = f["/problems/conflict"][:].astype(bool)

        labels = conflict.astype(np.int64)

        # Silhouette needs >= 3 samples and 2 classes with >= 2 members each.
        # With 2 conflict + 3 control, this should work.
        sil = cosine_silhouette(X, labels)
        assert -1.0 <= sil <= 1.0

        ch = calinski_harabasz(X, labels)
        assert ch >= 0.0

        db = davies_bouldin(X, labels)
        assert db >= 0.0 or db == float("inf")

        # Separability: need >= 4 samples. With 5 items and 2-dim space, PCA
        # will clamp to 1 dimension. Use small n_shuffles for speed.
        sep = linear_separability_with_d_gg_n_fix(
            X, labels, pca_dim=1, n_shuffles=5, n_folds=2, seed=0
        )
        assert 0.0 <= sep.pca_cv_accuracy <= 1.0
        assert sep.n_samples == 5
        assert sep.pca_dim >= 1


# ---------------------------------------------------------------------------
# Test 6: Full pipeline summary (smoke)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
class TestE2EPipelineSummary:
    """Verify the full chain produced usable outputs."""

    def test_all_hdf5_groups_present(self, e2e_hdf5: Path):
        with h5py.File(e2e_hdf5, "r") as f:
            assert "/metadata" in f
            assert "/problems" in f
            assert f"/models/{TINY_HDF5_KEY}" in f
            assert f"/models/{TINY_HDF5_KEY}/residual" in f
            assert f"/models/{TINY_HDF5_KEY}/attention" in f
            assert f"/models/{TINY_HDF5_KEY}/position_index" in f
            assert f"/models/{TINY_HDF5_KEY}/token_surprises" in f
            assert f"/models/{TINY_HDF5_KEY}/generations" in f
            assert f"/models/{TINY_HDF5_KEY}/behavior" in f

    def test_all_attention_metrics_present(self, e2e_hdf5: Path):
        with h5py.File(e2e_hdf5, "r") as f:
            attn_grp = f[f"/models/{TINY_HDF5_KEY}/attention"]
            for name in ("entropy", "entropy_normalized", "gini",
                         "max_attn", "focus_5", "effective_rank"):
                assert name in attn_grp, f"missing attention metric: {name}"

    def test_residual_layers_match_model(self, e2e_hdf5: Path):
        with h5py.File(e2e_hdf5, "r") as f:
            n_layers = int(f[f"/models/{TINY_HDF5_KEY}/metadata"].attrs["n_layers"])
            resid_grp = f[f"/models/{TINY_HDF5_KEY}/residual"]
            for layer in range(n_layers):
                key = f"layer_{layer:02d}"
                assert key in resid_grp, f"missing residual layer: {key}"
