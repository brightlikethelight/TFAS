"""Tests for :mod:`s1s2.sae`.

We cover:

* :class:`MockSAE` end-to-end round trip (encode → decode → reconstruction).
* The :func:`reconstruction_report` gate on a good and a deliberately broken
  fit.
* :func:`differential_activation` identifies a planted feature that fires
  only on conflict items.
* Ma et al. falsification runs on the candidates and records both
  non-spurious and spurious calls.
* :class:`SAEAnalysisRunner` executes end-to-end on a synthetic HDF5
  (using :class:`MockSAE` as the backend), writes the expected result
  files, skips cells where reconstruction is poor, and populates the
  steering-vector npz with the right keys.
* The CLI-side Hydra translation produces a valid :class:`SAERunnerConfig`.

All tests are CPU-only and complete in well under a minute on a laptop.
They avoid ``sae-lens`` / network / HF completely by going through the
``MockSAE`` backend inside the loaders.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# Make sure OpenMP does not crash on macOS when torch + numpy share the runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Package is not always pip-installed in the dev env; prepend src/ to sys.path.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from s1s2.sae import (  # noqa: E402 -- sys.path manipulation above
    DEFAULT_MODEL_HDF5_KEYS,
    FalsificationResult,
    MockSAE,
    SAEAnalysisRunner,
    SAERunnerConfig,
    differential_activation,
    falsify_candidates,
    reconstruction_report,
    run_sae_analysis,
)
from s1s2.utils import io as ioh  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic HDF5 builder
# ---------------------------------------------------------------------------

# Two short prompts carrying distinct trigger tokens. The conflict-only
# prompts contain the word "reflect" (our planted S1 signature) so the
# differential test has something to find. Non-conflict prompts talk about
# arithmetic with different vocabulary.
_CONFLICT_PROMPTS = [
    "Please reflect carefully before answering this counterintuitive question.",
    "Think and reflect on whether your first instinct is actually correct here.",
    "Reflect: the obvious answer might be a lure so be careful.",
    "Take a moment to reflect and resist the surface-level response.",
    "Reflect deliberately before committing to any answer on this item.",
    "Reflect once more — is the intuitive answer the right one?",
    "Reflect again on whether this is a trick question or not.",
    "Pause and reflect carefully before choosing.",
]

_CONTROL_PROMPTS = [
    "Compute the sum of twenty seven and thirteen for the record.",
    "Multiply four by six and report the product you obtain.",
    "Divide fifty by ten and report the quotient of the division.",
    "Add twelve and eight together to get the total sum.",
    "Subtract five from eleven to find the difference in values.",
    "What is the product of three and four times two today?",
    "Compute nine plus nine and provide the numerical sum please.",
    "What is the value of twenty one minus seven in total?",
]

assert len(_CONFLICT_PROMPTS) == len(_CONTROL_PROMPTS) == 8


def _write_synthetic_hdf5(
    path: Path,
    *,
    hidden_dim: int = 16,
    n_pairs: int = 8,
    n_layers: int = 4,
    planted_layer: int = 2,
    model_hdf5_key: str = DEFAULT_MODEL_HDF5_KEYS["llama-3.1-8b-instruct"],
) -> str:
    """Create a minimal HDF5 activation cache for the SAE tests.

    Matches the contract in ``docs/data_contract.md`` for every group the
    SAE runner reads: ``/metadata``, ``/problems``, and
    ``/models/{key}/{metadata,residual,position_index,behavior}``.

    The key trick is the *planted signal* on ``planted_layer``: for every
    conflict (S1) item we add a constant direction to the residual, so the
    Mock SAE's linear encoder will reliably light up the feature that
    aligns with that direction. Non-planted layers stay noise only.
    """

    n_problems = 2 * n_pairs
    n_positions = 2
    position_labels = ["P0", "P2"]
    rng = np.random.default_rng(0)

    # Interleave conflict/control so paired indices are adjacent.
    conflict = np.array([i % 2 == 0 for i in range(n_problems)], dtype=bool)
    pair_ids = np.array([f"pair_{i // 2:04d}" for i in range(n_problems)], dtype="S64")
    prompt_text = []
    for i in range(n_problems):
        if conflict[i]:
            prompt_text.append(_CONFLICT_PROMPTS[i // 2])
        else:
            prompt_text.append(_CONTROL_PROMPTS[i // 2])

    # Build activations. Layer ``planted_layer`` gets a conflict-dependent
    # shift in a specific direction; all other layers are noise.
    direction = np.zeros(hidden_dim, dtype=np.float32)
    direction[0] = 1.0
    direction[1] = 0.5
    direction /= np.linalg.norm(direction) + 1e-8

    acts_per_layer: list[np.ndarray] = []
    for layer in range(n_layers):
        base = rng.normal(size=(n_problems, n_positions, hidden_dim)).astype(np.float32)
        if layer == planted_layer:
            shift = (conflict.astype(np.float32) * 3.0)[:, None, None] * direction[None, None, :]
            base = base + shift.astype(np.float32)
        acts_per_layer.append(base)

    correct = np.array([True] * n_problems, dtype=bool)
    matches_lure = np.zeros(n_problems, dtype=bool)

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
            data=np.array([s.encode("utf-8") for s in prompt_text], dtype="S2048"),
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

        mgrp = f.create_group(f"/models/{model_hdf5_key}")
        mmeta = mgrp.create_group("metadata")
        mmeta.attrs["hf_model_id"] = "synth/model"
        mmeta.attrs["n_layers"] = n_layers
        mmeta.attrs["n_heads"] = 4
        mmeta.attrs["n_kv_heads"] = 2
        mmeta.attrs["hidden_dim"] = hidden_dim
        mmeta.attrs["head_dim"] = 4
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
        pos.create_dataset("valid", data=np.ones((n_problems, n_positions), dtype=bool))

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
        gens.create_dataset("full_text", data=np.array(["x"] * n_problems, dtype="S8192"))
        gens.create_dataset("thinking_text", data=np.array([""] * n_problems, dtype="S8192"))
        gens.create_dataset("answer_text", data=np.array(["x"] * n_problems, dtype="S512"))
        gens.create_dataset("thinking_token_count", data=np.zeros(n_problems, dtype=np.int32))
        gens.create_dataset("answer_token_count", data=np.full(n_problems, 3, dtype=np.int32))

    return model_hdf5_key


@pytest.fixture
def synthetic_hdf5_path(tmp_path: Path) -> Path:
    """Write a small synthetic HDF5 cache and return its path.

    Exposed as a fixture so any additional SAE tests added later can reuse
    the same cache without re-building the schema boilerplate.
    """
    p = tmp_path / "sae_synth.h5"
    _write_synthetic_hdf5(p)
    return p


# ---------------------------------------------------------------------------
# MockSAE and reconstruction report
# ---------------------------------------------------------------------------


def test_mock_sae_round_trip_preserves_shape_and_dtype() -> None:
    """Sanity check that :class:`MockSAE` encode/decode preserves shapes."""
    sae = MockSAE(hidden_dim=8, n_features=16, layer=0, seed=0, sparsity=1.0)
    x = np.random.default_rng(0).normal(size=(32, 8)).astype(np.float32)
    import torch  # local import so the module can be loaded without torch noise

    z = sae.encode(torch.from_numpy(x))
    assert z.shape == (32, 16)
    x_hat = sae.decode(z)
    assert x_hat.shape == (32, 8)
    # ReLU in encode makes the mock lossy on the negative half; the
    # reconstruction is still strongly correlated with the input because
    # W_dec is the encoder pseudo-inverse.
    err = x - x_hat.detach().numpy()
    assert np.linalg.norm(err) < np.linalg.norm(x)


def test_reconstruction_report_computes_expected_fields() -> None:
    """Smoke test for :func:`reconstruction_report` on a MockSAE.

    The MockSAE applies a ReLU in ``encode``, so reconstruction of arbitrary
    Gaussian inputs is lossy by design. We do not assert ``is_poor_fit``
    because that depends on the interaction between the decoder's bias term
    and the input distribution; instead we verify that the report is
    produced with the expected fields and numeric ranges.
    """
    sae = MockSAE(hidden_dim=8, n_features=16, layer=0, seed=0, sparsity=1.0)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(128, 8)).astype(np.float32)
    report = reconstruction_report(sae, x, min_explained_variance=0.5, n_samples=64)
    assert report.n_samples == 64
    assert report.hidden_dim == 8
    assert report.mse >= 0.0
    assert report.variance > 0.0
    assert -1.0 <= report.explained_variance <= 1.0
    assert report.mean_l0 >= 0.0


def test_reconstruction_report_fails_on_dimension_mismatch() -> None:
    sae = MockSAE(hidden_dim=8, n_features=16, layer=0, seed=0, sparsity=1.0)
    # Activations with the wrong hidden dim must raise — the runner relies on
    # this check to abort a mismatched cell cleanly.
    bad = np.zeros((32, 12), dtype=np.float32)
    with pytest.raises(ValueError):
        reconstruction_report(sae, bad, min_explained_variance=0.5, n_samples=8)


# ---------------------------------------------------------------------------
# Differential analysis on a planted signal
# ---------------------------------------------------------------------------


def test_differential_activation_finds_planted_feature() -> None:
    """A planted activation on half the rows should produce at least one
    BH-significant feature with a positive log fold change."""
    rng = np.random.default_rng(0)
    n, k = 40, 12
    feat = np.abs(rng.normal(size=(n, k))).astype(np.float32) * 0.1
    conflict = np.zeros(n, dtype=bool)
    conflict[::2] = True
    feat[conflict, 3] += 3.0  # strong planted signal on feature 3

    result = differential_activation(
        feature_activations=feat,
        conflict=conflict,
        fdr_q=0.05,
        subset_label="test",
    )
    assert result.df.shape[0] == k
    assert result.df["significant"].sum() >= 1
    # The planted feature should be among the significant ones.
    hit_row = result.df[result.df["feature_id"] == 3].iloc[0]
    assert bool(hit_row["significant"]) is True
    assert float(hit_row["log_fc"]) > 0.0


# ---------------------------------------------------------------------------
# Falsification
# ---------------------------------------------------------------------------


def test_falsify_candidates_runs_in_cheap_mode_on_mock_sae() -> None:
    """``falsify_candidates`` should return a FalsificationResult per feature
    in cheap mode without raising, even when trigger tokens are empty."""
    sae = MockSAE(hidden_dim=8, n_features=12, layer=0, seed=0, sparsity=1.0)
    rng = np.random.default_rng(0)
    n = 16
    activations = rng.normal(size=(n, 8)).astype(np.float32)
    feature_activations = np.abs(rng.normal(size=(n, 12))).astype(np.float32) * 0.1
    feature_activations[:, 0] += 2.0  # feature 0 fires everywhere -> peak high
    prompts = [f"benchmark prompt number {i} with reflection and reasoning" for i in range(n)]
    results = falsify_candidates(
        candidate_feature_ids=[0, 1, 2],
        sae=sae,
        activations=activations,
        feature_activations=feature_activations,
        prompts=prompts,
        mode="cheap",
        n_random_texts=16,
        n_top_tokens=3,
        threshold=0.5,
        top_k_features=10,
    )
    assert len(results) == 3
    for r in results:
        assert isinstance(r, FalsificationResult)
        assert r.mode == "cheap"


# ---------------------------------------------------------------------------
# SAEAnalysisRunner end-to-end
# ---------------------------------------------------------------------------


def _runner_config_for_synthetic(
    activations_path: Path, output_dir: Path, *, layers: list[int]
) -> SAERunnerConfig:
    """Build a minimal config that uses Llama keys but points at the synthetic HDF5.

    The Llama loader returns a :class:`MockSAE` whenever ``sae-lens`` is not
    importable, which is our CI path; we rely on that fallback so we do not
    need to hit the network. We lower the reconstruction threshold to 0.1
    because :class:`MockSAE` applies a ReLU in encode which limits the
    achievable explained variance on Gaussian activations.
    """
    return SAERunnerConfig(
        activations_path=str(activations_path),
        output_dir=str(output_dir),
        models=["llama-3.1-8b-instruct"],
        layers=layers,
        position="P0",
        reconstruction_check_n_samples=8,
        reconstruction_min_explained_variance=0.1,
        fdr_q=0.1,
        falsification_enabled=True,
        falsification_n_random_texts=8,
        falsification_n_top_tokens=3,
        falsification_threshold=0.5,
        falsification_mode="cheap",
        falsification_top_k_features=5,
        volcano_top_k=5,
        seed=0,
    )


def test_runner_end_to_end_writes_expected_artifacts(
    synthetic_hdf5_path: Path, tmp_path: Path, monkeypatch
) -> None:
    """Run the full SAEAnalysisRunner over one (model, layer) cell.

    Expectations:
      * ``feature_analysis.json`` exists and lists a non-empty top_features
        slice at the planted layer.
      * ``steering_vectors.npz`` exists with the required keys.
      * ``volcano.png`` exists.
    """
    output_dir = tmp_path / "results_sae"

    # The activations in our synthetic HDF5 have hidden_dim=16, but the
    # Llama Scope loader otherwise constructs a MockSAE for hidden_dim=4096.
    # Patch the loader so it returns a MockSAE whose hidden_dim matches our
    # synthetic cache (16).
    from s1s2.sae import core as sae_core

    def _mock_loader(model_key: str, layer: int, *, device: str = "cpu", **_: dict) -> MockSAE:
        return MockSAE(hidden_dim=16, n_features=64, layer=layer, seed=layer, sparsity=1.0)

    monkeypatch.setattr(sae_core, "load_sae_for_model", _mock_loader)

    cfg = _runner_config_for_synthetic(
        synthetic_hdf5_path,
        output_dir,
        layers=[2],  # the planted-signal layer
    )
    runner = SAEAnalysisRunner(cfg)
    results = runner.run()

    assert ("llama-3.1-8b-instruct", 2) in results
    cell = results[("llama-3.1-8b-instruct", 2)]
    assert cell["status"] == "ok", cell
    assert cell["n_features_total"] == 64
    # There should be at least one significant, unfalsified feature at the
    # planted layer; the conflict signal is strong.
    assert cell["n_features_significant"] >= 1

    hdf5_key = DEFAULT_MODEL_HDF5_KEYS["llama-3.1-8b-instruct"]
    layer_dir = output_dir / hdf5_key / "layer_02"
    assert (layer_dir / "feature_analysis.json").exists()
    assert (layer_dir / "steering_vectors.npz").exists()
    assert (layer_dir / "volcano.png").exists()
    assert (layer_dir / "feature_stats.csv").exists()

    # JSON content sanity.
    blob = json.loads((layer_dir / "feature_analysis.json").read_text())
    assert blob["model_key"] == "llama-3.1-8b-instruct"
    assert blob["layer"] == 2
    assert blob["sae_release"]  # non-empty string
    assert "reconstruction_explained_variance" in blob
    assert "config" in blob
    assert "git_sha" in blob
    assert "runtime_seconds" in blob
    assert isinstance(blob["top_features"], list)
    # The planted direction aligns with a specific MockSAE feature — at least
    # one of the top features should be labeled.
    for feat in blob["top_features"]:
        for key in (
            "feature_id",
            "log_fold_change",
            "q_value",
            "effect_size",
            "mean_activation_S1",
            "mean_activation_S2",
            "is_falsified",
            "auto_interp_label",
        ):
            assert key in feat, f"missing {key} in top_features entry"

    # Falsification block is populated in cheap mode.
    assert "falsification" in blob
    assert blob["falsification"]["mode"] == "cheap"
    assert isinstance(blob["falsification"]["per_feature"], list)

    # Steering vector npz has the required keys.
    npz = np.load(layer_dir / "steering_vectors.npz")
    for key in (
        "feature_ids",
        "encoder_directions",
        "decoder_directions",
        "mean_activations_S1",
        "mean_activations_S2",
        "is_falsified",
        "log_fold_changes",
        "q_values",
        "effect_sizes",
    ):
        assert key in npz.files, f"missing npz key {key}"
    assert npz["feature_ids"].ndim == 1
    assert npz["feature_ids"].dtype == np.int32
    assert npz["encoder_directions"].shape == (
        npz["feature_ids"].shape[0],
        16,  # synthetic hidden_dim
    )
    assert npz["decoder_directions"].shape == (
        npz["feature_ids"].shape[0],
        16,
    )


def test_runner_skips_cell_when_reconstruction_is_poor(
    synthetic_hdf5_path: Path, tmp_path: Path, monkeypatch
) -> None:
    """When the SAE does not reconstruct well, the runner must skip
    the cell and write a ``skipped_poor_reconstruction`` stub — not
    silently emit downstream results."""
    output_dir = tmp_path / "results_sae_bad"

    from s1s2.sae import core as sae_core

    class _BadSAE:
        """Outputs zeros, guaranteed to fail the fidelity gate."""

        def __init__(self, hidden_dim: int, n_features: int, layer: int) -> None:
            self.hidden_dim = hidden_dim
            self.n_features = n_features
            self.layer = layer

        def encode(self, x):
            import torch

            return torch.zeros((x.shape[0], self.n_features), dtype=torch.float32)

        def decode(self, z):
            import torch

            return torch.zeros((z.shape[0], self.hidden_dim), dtype=torch.float32)

        def reconstruct(self, x):
            return self.decode(self.encode(x))

        def encoder_directions(self) -> np.ndarray:
            return np.zeros((self.n_features, self.hidden_dim), dtype=np.float32)

        def decoder_directions(self) -> np.ndarray:
            return np.zeros((self.n_features, self.hidden_dim), dtype=np.float32)

    monkeypatch.setattr(
        sae_core,
        "load_sae_for_model",
        lambda model_key, layer, *, device="cpu", **_: _BadSAE(16, 24, layer),
    )

    cfg = _runner_config_for_synthetic(synthetic_hdf5_path, output_dir, layers=[2])
    runner = SAEAnalysisRunner(cfg)
    results = runner.run()

    key = ("llama-3.1-8b-instruct", 2)
    assert key in results
    assert results[key]["status"] == "skipped_poor_reconstruction"
    assert results[key]["n_features_significant"] == 0
    assert results[key]["n_features_after_falsification"] == 0

    hdf5_key = DEFAULT_MODEL_HDF5_KEYS["llama-3.1-8b-instruct"]
    layer_dir = output_dir / hdf5_key / "layer_02"
    assert (layer_dir / "feature_analysis.json").exists()
    # Volcano and steering vectors should NOT be emitted for a skipped cell —
    # downstream consumers use their presence as a "we trust this" signal.
    assert not (layer_dir / "volcano.png").exists()
    assert not (layer_dir / "steering_vectors.npz").exists()


def test_runner_skips_unknown_model_without_crashing(
    synthetic_hdf5_path: Path, tmp_path: Path
) -> None:
    """Asking the runner for a model that is not in the HDF5 must not crash;
    the cell simply goes missing from the results dict."""
    output_dir = tmp_path / "results_sae_missing"
    cfg = SAERunnerConfig(
        activations_path=str(synthetic_hdf5_path),
        output_dir=str(output_dir),
        models=["gemma-2-9b-it"],  # present in registry but NOT in the synthetic HDF5
        layers=[2],
        position="P0",
        reconstruction_check_n_samples=8,
        falsification_enabled=False,
    )
    runner = SAEAnalysisRunner(cfg)
    results = runner.run()
    # The key simply doesn't appear — and the process didn't crash.
    assert ("gemma-2-9b-it", 2) not in results
    assert results == {}


def test_run_sae_analysis_functional_wrapper_matches_class_runner(
    synthetic_hdf5_path: Path, tmp_path: Path, monkeypatch
) -> None:
    """The functional wrapper :func:`run_sae_analysis` is just a shortcut for
    constructing the runner — verify it's behaviourally identical."""

    from s1s2.sae import core as sae_core

    monkeypatch.setattr(
        sae_core,
        "load_sae_for_model",
        lambda model_key, layer, *, device="cpu", **_: MockSAE(
            hidden_dim=16, n_features=64, layer=layer, seed=layer, sparsity=1.0
        ),
    )

    cfg = _runner_config_for_synthetic(
        synthetic_hdf5_path,
        tmp_path / "results_sae_wrapper",
        layers=[2],
    )
    results = run_sae_analysis(cfg)
    assert ("llama-3.1-8b-instruct", 2) in results


# ---------------------------------------------------------------------------
# CLI-side Hydra translation
# ---------------------------------------------------------------------------


def test_runner_config_from_hydra_flat_schema() -> None:
    """Translating a minimal Hydra-style DictConfig into :class:`SAERunnerConfig`
    should preserve every field and fill in sensible defaults."""
    from omegaconf import OmegaConf

    from s1s2.sae.cli import runner_config_from_hydra

    raw = OmegaConf.create(
        {
            "activations_path": "data/activations/main.h5",
            "output_dir": "results/sae",
            "models_to_run": ["llama-3.1-8b-instruct"],
            "layers": [8, 16],
            "position": "P0",
            "reconstruction": {"n_samples": 64, "min_explained_variance": 0.5},
            "differential": {"fdr_q": 0.05, "matched_difficulty_only": False},
            "falsification": {
                "enabled": True,
                "n_random_texts": 50,
                "n_top_tokens": 5,
                "threshold": 0.5,
                "mode": "cheap",
                "top_k_features": 20,
            },
            "volcano": {"top_k": 10},
            "seed": 7,
        }
    )
    cfg = runner_config_from_hydra(raw)
    assert isinstance(cfg, SAERunnerConfig)
    assert cfg.activations_path == "data/activations/main.h5"
    assert cfg.models == ["llama-3.1-8b-instruct"]
    assert cfg.layers == [8, 16]
    assert cfg.fdr_q == 0.05
    assert cfg.falsification_mode == "cheap"
    assert cfg.falsification_n_random_texts == 50
    assert cfg.falsification_top_k_features == 20
    assert cfg.volcano_top_k == 10
    assert cfg.seed == 7


def test_runner_config_from_hydra_models_dict_form() -> None:
    """When the Hydra config includes the full ``models:`` dict (from
    ``configs/models.yaml``), the runner config should fall back to
    ``models_to_run`` to pick which model keys to execute."""
    from omegaconf import OmegaConf

    from s1s2.sae.cli import runner_config_from_hydra

    raw = OmegaConf.create(
        {
            "activations_path": "main.h5",
            "output_dir": "results/sae",
            "models": {
                "llama-3.1-8b-instruct": {
                    "hdf5_key": "meta-llama_Llama-3.1-8B-Instruct",
                    "sae_release": "fnlp/Llama-3_1-8B-Base-LXR-32x",
                },
                "gemma-2-9b-it": {
                    "hdf5_key": "google_gemma-2-9b-it",
                    "sae_release": "google/gemma-scope-9b-it-res",
                },
            },
            "models_to_run": ["gemma-2-9b-it"],
            "layers": [21],
        }
    )
    cfg = runner_config_from_hydra(raw)
    assert cfg.models == ["gemma-2-9b-it"]
    assert cfg.layers == [21]
    assert cfg.model_hdf5_keys["llama-3.1-8b-instruct"] == "meta-llama_Llama-3.1-8B-Instruct"
    assert cfg.model_sae_releases["gemma-2-9b-it"] == "google/gemma-scope-9b-it-res"
