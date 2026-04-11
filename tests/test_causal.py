"""Tests for :mod:`s1s2.causal`.

Covers:

* Direction normalisation and random unit direction sampling.
* :class:`SteeringHook` registers, mutates the residual stream, and
  removes cleanly.
* :class:`AblationHook` projection ablation on toy tensors matches the
  pure-function :func:`ablate_direction`.
* Random direction control gives results measurably different from
  feature steering (the whole point of the control).
* :func:`build_curve` and :func:`fit_curve` produce sane shapes.
* :func:`score_capability` against a synthesised 20-item fixture.
* End-to-end :class:`CausalExperimentRunner` round trip with a mocked
  model + tokenizer + score function, verifying dose-response curve,
  ablation, capability comparison, and JSON serialisation.

All tests are CPU-only and should collectively run in <30s.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from s1s2.benchmark.loader import BenchmarkItem
from s1s2.causal import (
    AblationHook,
    CapabilityEvalConfig,
    CapabilityItem,
    CausalExperimentRunner,
    CausalRunnerConfig,
    FeatureSpec,
    RandomControlConfig,
    SteeringHook,
    ablate_direction,
    aggregate_p_correct,
    build_curve,
    compare_capability,
    fit_curve,
    is_canonical_s2_signature,
    load_capability_jsonl,
    load_feature_specs,
    normalize_direction,
    plot_ablation_bars,
    plot_dose_response,
    plot_feature_summary_bars,
    random_unit_direction,
    run_causal,
    save_capability_jsonl,
    save_cell_result,
    score_capability,
)

# ---------------------------------------------------------------------------
# Mock model / tokenizer / score function
# ---------------------------------------------------------------------------


class _MockDecoderLayer(torch.nn.Module):
    """Tiny decoder layer that returns a ``(hidden, aux)`` tuple.

    Mimics the HuggingFace Llama / Gemma / Qwen decoder-layer convention
    so we can attach the real steering hook to ``model.model.layers[i]``
    and verify the return tuple is handled correctly.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # Identity-ish linear layer so we can confirm the hook runs.
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        torch.nn.init.eye_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.proj(x)
        # Second tuple member is a dummy attention cache; some HF models
        # return ``(hidden, present_kv)`` or ``(hidden, attn_weights)``.
        aux = torch.zeros(1)
        return (out, aux)


class _MockInner(torch.nn.Module):
    """Inner ``.model`` that holds a ``layers`` ModuleList."""

    def __init__(self, n_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_MockDecoderLayer(hidden_dim) for _ in range(n_layers)])


class _MockModel(torch.nn.Module):
    """A toy causal-LM-like module. Has ``.model.layers[i]`` for hooks.

    The forward pass runs each decoder layer in turn and returns the
    final hidden state, so the SteeringHook's modifications actually
    propagate to the output. This lets us verify that alpha=0 vs alpha=3
    produce different outputs.
    """

    def __init__(self, n_layers: int = 3, hidden_dim: int = 16) -> None:
        super().__init__()
        self.model = _MockInner(n_layers=n_layers, hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.model.layers:
            h, _ = layer(h)
        return h


@dataclass
class _MockTokenizer:
    """Placeholder tokenizer — our score_fn never touches it."""

    pad_token_id: int = 0
    eos_token_id: int = 0


def _make_bench_items(n_pairs: int = 4) -> list[BenchmarkItem]:
    """Synthesise ``2 * n_pairs`` items (half conflict, half control)."""
    items: list[BenchmarkItem] = []
    for i in range(n_pairs):
        items.append(
            BenchmarkItem(
                id=f"p{i:03d}_conflict",
                category="crt",
                subcategory="test",
                conflict=True,
                difficulty=2,
                prompt=f"conflict prompt {i}",
                system_prompt=None,
                correct_answer="A",
                lure_answer="B",
                answer_pattern=r"\bA\b",
                lure_pattern=r"\bB\b",
                matched_pair_id=f"pair_{i:03d}",
                source="template",
                provenance_note="synthetic for tests",
            )
        )
        items.append(
            BenchmarkItem(
                id=f"p{i:03d}_control",
                category="crt",
                subcategory="test",
                conflict=False,
                difficulty=2,
                prompt=f"no-conflict prompt {i}",
                system_prompt=None,
                correct_answer="A",
                lure_answer="",
                answer_pattern=r"\bA\b",
                lure_pattern="",
                matched_pair_id=f"pair_{i:03d}",
                source="template",
                provenance_note="synthetic for tests",
            )
        )
    return items


def _steering_aware_score_fn(
    target_direction: torch.Tensor,
    *,
    threshold: float = 0.0,
) -> callable:
    """Return a score_fn that reads model state + active hooks.

    We use a simple trick: the score_fn runs a forward pass on a constant
    input, projects the output onto ``target_direction``, and returns True
    if the projection exceeds ``threshold``. Because the steering hook
    adds ``alpha * target_direction`` to the residual stream, this
    projection is monotonically increasing in alpha *exactly* for the
    feature direction and roughly zero for a random direction — which
    is the hallmark of a real S2-like feature the test is checking for.

    This keeps the test self-contained: no real model, no real tokenizer,
    but the runner exercise is end-to-end.
    """

    def score_fn(model, tokenizer, item) -> bool:
        # Constant probe input — we only care about the hook effect.
        x = torch.zeros(1, 1, model.hidden_dim, dtype=torch.float32)
        out = model(x)
        projection = float((out[0, 0] * target_direction).sum().item())
        # For conflict items, use the actual threshold. For no-conflict
        # items, always return True so no_conflict P(correct) is ~1 and
        # the intervention leaves it alone. This matches the canonical
        # "no effect on easy items" S2 signature.
        if not item.conflict:
            return True
        return projection >= threshold

    return score_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model() -> _MockModel:
    torch.manual_seed(0)
    return _MockModel(n_layers=3, hidden_dim=16)


@pytest.fixture
def mock_tokenizer() -> _MockTokenizer:
    return _MockTokenizer()


@pytest.fixture
def feature_direction() -> torch.Tensor:
    torch.manual_seed(1)
    v = torch.randn(16)
    return v / v.norm()


@pytest.fixture
def bench_items() -> list[BenchmarkItem]:
    return _make_bench_items(n_pairs=6)


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------


def test_normalize_direction_returns_unit_norm() -> None:
    v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    d = normalize_direction(v)
    assert d.shape == (3,)
    assert abs(float(d.norm().item()) - 1.0) < 1e-6


def test_normalize_direction_rejects_zero() -> None:
    with pytest.raises(ValueError):
        normalize_direction(np.zeros(8, dtype=np.float32))


def test_random_unit_direction_is_unit_norm_and_reproducible() -> None:
    v1 = random_unit_direction(32, seed=7)
    v2 = random_unit_direction(32, seed=7)
    v3 = random_unit_direction(32, seed=8)
    assert v1.shape == (32,)
    assert abs(float(v1.norm().item()) - 1.0) < 1e-5
    torch.testing.assert_close(v1, v2)
    assert not torch.allclose(v1, v3)


# ---------------------------------------------------------------------------
# SteeringHook behaviour
# ---------------------------------------------------------------------------


def test_steering_hook_registers_and_removes_cleanly(
    mock_model: _MockModel, feature_direction: torch.Tensor
) -> None:
    """Attaching the hook and exiting the context must leave the model pristine."""
    before = len(mock_model.model.layers[1]._forward_hooks)
    sh = SteeringHook(mock_model, layer=1, direction=feature_direction, alpha=2.0)
    assert not sh.is_attached
    with sh:
        assert sh.is_attached
        during = len(mock_model.model.layers[1]._forward_hooks)
        assert during == before + 1
        # Run a forward pass so the hook's call counter ticks.
        mock_model(torch.zeros(1, 1, mock_model.hidden_dim))
        assert sh.call_count == 1
    assert not sh.is_attached
    after = len(mock_model.model.layers[1]._forward_hooks)
    assert after == before


def test_steering_hook_alpha_zero_is_baseline(
    mock_model: _MockModel, feature_direction: torch.Tensor
) -> None:
    """alpha=0 under the hook must match the no-hook forward output exactly."""
    x = torch.zeros(1, 1, mock_model.hidden_dim)
    base = mock_model(x).clone()
    with SteeringHook(mock_model, layer=1, direction=feature_direction, alpha=0.0):
        hooked = mock_model(x).clone()
    torch.testing.assert_close(base, hooked)


def test_steering_hook_nonzero_alpha_changes_output(
    mock_model: _MockModel, feature_direction: torch.Tensor
) -> None:
    """alpha != 0 should produce an output measurably different from baseline."""
    x = torch.zeros(1, 1, mock_model.hidden_dim)
    base = mock_model(x).clone()
    with SteeringHook(mock_model, layer=1, direction=feature_direction, alpha=3.0):
        hooked = mock_model(x).clone()
    diff = (hooked - base).norm().item()
    # The hook added 3 * d at layer 1; with identity projection layers
    # this propagates to the output as 3 * d as well.
    assert diff > 0.5


def test_steering_hook_invalid_layer_raises(
    mock_model: _MockModel, feature_direction: torch.Tensor
) -> None:
    with (
        pytest.raises(IndexError),
        SteeringHook(mock_model, layer=99, direction=feature_direction, alpha=1.0),
    ):
        mock_model(torch.zeros(1, 1, mock_model.hidden_dim))


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def test_ablate_direction_removes_projection() -> None:
    """After ablation, the residual must be orthogonal to the ablated direction."""
    rng = np.random.default_rng(0)
    h = torch.from_numpy(rng.normal(size=(3, 4, 16)).astype(np.float32))
    d = torch.from_numpy(rng.normal(size=16).astype(np.float32))
    d = d / d.norm()
    h_ab = ablate_direction(h, d)
    dot = (h_ab * d).sum(dim=-1)
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-5)
    # Orthogonal residual should be preserved (not entirely zeroed).
    assert h_ab.norm().item() > 0.0


def test_ablation_hook_zeros_the_directional_component(
    mock_model: _MockModel, feature_direction: torch.Tensor
) -> None:
    """Under the ablation hook, the output should have no component along d."""
    x = torch.randn(1, 2, mock_model.hidden_dim)
    with AblationHook(mock_model, layer=2, direction=feature_direction):
        out = mock_model(x)
    # Last layer is layer 2; its output has been ablated by the hook. Any
    # subsequent layer would re-introduce the direction, but layer 2 is
    # the last one in our mock.
    proj = (out[0, :, :] * feature_direction).sum(dim=-1)
    torch.testing.assert_close(proj, torch.zeros_like(proj), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Random direction control vs feature direction
# ---------------------------------------------------------------------------


def test_random_control_output_differs_from_feature_steering(
    mock_model: _MockModel, feature_direction: torch.Tensor
) -> None:
    """Steering with feature direction must differ from steering with a random one."""
    x = torch.zeros(1, 1, mock_model.hidden_dim)
    with SteeringHook(mock_model, layer=1, direction=feature_direction, alpha=2.0):
        feat_out = mock_model(x).clone()
    rnd = random_unit_direction(mock_model.hidden_dim, seed=123)
    with SteeringHook(mock_model, layer=1, direction=rnd, alpha=2.0):
        rnd_out = mock_model(x).clone()
    delta = (feat_out - rnd_out).norm().item()
    assert delta > 1e-3, f"feature vs random should differ, got delta={delta}"


# ---------------------------------------------------------------------------
# Dose-response aggregation
# ---------------------------------------------------------------------------


def test_aggregate_p_correct_bounds() -> None:
    vec = np.array([1, 0, 1, 1, 0, 1, 1], dtype=np.float32)
    p, lo, hi = aggregate_p_correct(vec, n_bootstrap=200, seed=0)
    assert 0.0 <= lo <= p <= hi <= 1.0
    # Float32 -> float64 roundtrip loses ~1e-8 precision, accept that.
    assert abs(p - float(vec.astype(np.float64).mean())) < 1e-6


def test_build_curve_shape_and_fit() -> None:
    alphas = [-1.0, 0.0, 1.0]
    # A fake "real S2" pattern: conflict P(correct) rises, no-conflict flat,
    # random control flat.
    conflict = {
        -1.0: np.zeros(10, dtype=np.float32),
        0.0: np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
        1.0: np.ones(10, dtype=np.float32),
    }
    no_conflict = {
        -1.0: np.ones(10, dtype=np.float32),
        0.0: np.ones(10, dtype=np.float32),
        1.0: np.ones(10, dtype=np.float32),
    }
    random_c = {
        a: {0: np.zeros(10, dtype=np.float32) + 0.4, 1: np.zeros(10, dtype=np.float32) + 0.4}
        for a in alphas
    }
    curve = build_curve(
        model="mock",
        layer=0,
        feature_id=0,
        alphas=alphas,
        conflict_correct_by_alpha=conflict,
        no_conflict_correct_by_alpha=no_conflict,
        random_correct_by_alpha_seed=random_c,
        n_bootstrap=100,
        seed=0,
    )
    assert curve.model == "mock"
    assert len(curve.points) == 9  # 3 alphas x 3 groups
    assert curve.fit["conflict"]["slope"] > 0
    assert abs(curve.fit["no_conflict"]["slope"]) < 1e-6
    assert abs(curve.fit["random_control"]["slope"]) < 1e-6
    assert curve.fit["selectivity_vs_random"] > 0.3
    # Canonical signature recogniser: this should pass the test by construction.
    assert is_canonical_s2_signature(curve)


def test_fit_curve_handles_empty_group() -> None:
    from s1s2.causal.dose_response import DoseResponsePoint

    pts = [
        DoseResponsePoint(
            alpha=0.0, group="conflict", n=5, p_correct=0.6, ci_lower=0.4, ci_upper=0.8
        )
    ]
    fit = fit_curve(pts)
    assert "conflict" in fit
    # With only one point, slope/intercept should be zero but not raise.
    assert fit["conflict"]["slope"] == 0.0


# ---------------------------------------------------------------------------
# Capability evaluation
# ---------------------------------------------------------------------------


def _synth_capability_items(n: int = 20) -> list[CapabilityItem]:
    """Build a tiny multiple-choice dataset."""
    items: list[CapabilityItem] = []
    for i in range(n):
        items.append(
            CapabilityItem(
                id=f"cap_{i:03d}",
                question=f"Q{i}?",
                choices=(f"choice_a_{i}", f"choice_b_{i}", f"choice_c_{i}", f"choice_d_{i}"),
                correct_index=i % 4,
            )
        )
    return items


def test_capability_jsonl_round_trip(tmp_path: Path) -> None:
    items = _synth_capability_items(5)
    path = tmp_path / "mmlu_subset.jsonl"
    save_capability_jsonl(path, items)
    loaded = load_capability_jsonl(path)
    assert len(loaded) == 5
    assert loaded[0].id == "cap_000"
    assert loaded[0].correct_index == 0
    assert len(loaded[0].choices) == 4


def test_score_capability_with_mocked_lm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Score a tiny fixture with a monkey-patched log-likelihood function."""
    items = _synth_capability_items(10)

    # Monkey-patch the loglikelihood scorer to always return the correct choice.
    import s1s2.causal.capability as cap

    def fake_ll(model, tokenizer, question: str, choice: str, device: str = "cpu") -> float:
        # Reward choices that end with the index of the correct answer for that item.
        idx = int(question[1:-1])  # "Q3?" -> 3
        target = idx % 4
        letter = "abcd"[target]
        return 1.0 if f"choice_{letter}_" in choice else 0.0

    monkeypatch.setattr(cap, "_score_choice_loglikelihood", fake_ll)

    class _Dummy:
        pass

    result = score_capability(_Dummy(), _Dummy(), items, benchmark_name="mmlu-toy")
    assert result.accuracy == 1.0
    assert result.n_items == 10


def test_compare_capability_detects_large_drop() -> None:
    from s1s2.causal.capability import CapabilityResult

    baseline = CapabilityResult(
        benchmark="mmlu",
        n_items=10,
        accuracy=0.8,
        per_item_correct=[True] * 8 + [False] * 2,
    )
    intervention = CapabilityResult(
        benchmark="mmlu",
        n_items=10,
        accuracy=0.7,
        per_item_correct=[True] * 7 + [False] * 3,
    )
    cmp = compare_capability(baseline, intervention, max_acceptable_drop_pp=5.0)
    assert cmp.delta_pp == pytest.approx(-10.0)
    assert cmp.exceeded_max_drop is True


# ---------------------------------------------------------------------------
# Feature spec loading
# ---------------------------------------------------------------------------


def test_load_feature_specs_empty_dir_returns_empty(tmp_path: Path) -> None:
    specs = load_feature_specs(tmp_path / "nope", model_key="mock")
    assert specs == []


def test_load_feature_specs_from_json(tmp_path: Path) -> None:
    """Write a fake SAE-results JSON and verify loading."""
    payload = {
        "model": "llama-3.1-8b-instruct",
        "layer": 16,
        "top_features": [
            {
                "feature_id": 42,
                "effect_size": 0.8,
                "q_value": 1e-5,
                "direction": np.random.default_rng(0).normal(size=16).tolist(),
            },
            {
                "feature_id": 99,
                "effect_size": 0.3,
                "q_value": 1e-3,
                "direction": np.random.default_rng(1).normal(size=16).tolist(),
            },
        ],
    }
    results_dir = tmp_path / "sae_results"
    results_dir.mkdir()
    (results_dir / "llama-3.1-8b-instruct_layer16_features.json").write_text(json.dumps(payload))
    specs = load_feature_specs(
        results_dir,
        model_key="llama-3.1-8b-instruct",
        top_per_layer=3,
        hidden_dim=16,
    )
    assert len(specs) == 2
    assert specs[0].feature_id == 42
    assert specs[0].layer == 16
    assert abs(np.linalg.norm(specs[0].direction) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# End-to-end: runner with a mock model
# ---------------------------------------------------------------------------


def test_causal_runner_round_trip(
    tmp_path: Path,
    mock_model: _MockModel,
    mock_tokenizer: _MockTokenizer,
    feature_direction: torch.Tensor,
    bench_items: list[BenchmarkItem],
) -> None:
    """Full pipeline on a mock model: runner -> curve -> ablation -> JSON."""
    feature = FeatureSpec(
        model_key="mock",
        layer=1,
        feature_id=0,
        direction=feature_direction.numpy().copy(),
    )

    cfg = CausalRunnerConfig(
        alphas=(-1.0, 0.0, 1.0),
        top_features_per_layer=1,
        random_control=RandomControlConfig(n_directions=3, seed=1),
        capability_eval=None,
        seed=0,
        n_bootstrap=50,
    )
    runner = CausalExperimentRunner(cfg)
    score_fn = _steering_aware_score_fn(feature_direction)

    cell = runner.run_one(
        model=mock_model,
        tokenizer=mock_tokenizer,
        feature=feature,
        benchmark=bench_items,
        score_fn=score_fn,
    )

    # Curve should have 3 alphas x 3 groups = 9 points.
    assert len(cell.curve.points) == 9

    # Canonical S2 signature: alpha=+1 should push conflict P(correct)
    # up vs alpha=-1. Our score_fn projects onto feature direction and
    # returns True iff >= 0 -> monotonic in alpha.
    conflict_points = {p.alpha: p for p in cell.curve.points if p.group == "conflict"}
    assert conflict_points[1.0].p_correct >= conflict_points[-1.0].p_correct
    # Random-control slope should be near zero (random dirs projected onto
    # the target direction are roughly zero in expectation).
    slope_random = cell.curve.fit["random_control"]["slope"]
    slope_conflict = cell.curve.fit["conflict"]["slope"]
    assert slope_conflict > slope_random

    # Ablation result exists.
    assert cell.ablation is not None

    # Serialize and reload.
    out_path = save_cell_result(cell, tmp_path / "causal")
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["model"] == "mock"
    assert loaded["layer"] == 1
    assert loaded["feature_id"] == 0
    assert "curve" in loaded
    assert len(loaded["curve"]["points"]) == 9
    assert loaded["ablation"] is not None


def test_causal_runner_with_capability_eval(
    tmp_path: Path,
    mock_model: _MockModel,
    mock_tokenizer: _MockTokenizer,
    feature_direction: torch.Tensor,
    bench_items: list[BenchmarkItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capability evaluation should be invoked and the drop recorded."""
    mmlu_path = tmp_path / "mmlu.jsonl"
    save_capability_jsonl(mmlu_path, _synth_capability_items(8))

    # Patch the loglikelihood scorer so capability score doesn't need a real LM.
    import s1s2.causal.capability as cap

    def fake_ll(model, tokenizer, question: str, choice: str, device: str = "cpu") -> float:
        idx = int(question[1:-1])
        target = idx % 4
        letter = "abcd"[target]
        return 1.0 if f"choice_{letter}_" in choice else 0.0

    monkeypatch.setattr(cap, "_score_choice_loglikelihood", fake_ll)

    feature = FeatureSpec(
        model_key="mock",
        layer=1,
        feature_id=0,
        direction=feature_direction.numpy().copy(),
    )
    cfg = CausalRunnerConfig(
        alphas=(0.0, 1.0),
        top_features_per_layer=1,
        random_control=RandomControlConfig(n_directions=2, seed=1),
        capability_eval=CapabilityEvalConfig(
            mmlu_subset_path=str(mmlu_path),
            hellaswag_subset_path=None,
            n_examples_per_eval=8,
            max_acceptable_drop_pp=2.0,
        ),
        seed=0,
        n_bootstrap=50,
    )
    runner = CausalExperimentRunner(cfg)
    score_fn = _steering_aware_score_fn(feature_direction)

    cell = runner.run_one(
        model=mock_model,
        tokenizer=mock_tokenizer,
        feature=feature,
        benchmark=bench_items,
        score_fn=score_fn,
    )
    assert len(cell.capability) == 1
    assert cell.capability[0].benchmark == "mmlu"
    # With our deterministic scorer, baseline == intervention == 1.0 so
    # the drop is 0.
    assert cell.capability[0].delta_pp == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Visualisation smoke tests
# ---------------------------------------------------------------------------


def test_plot_dose_response_and_ablation_write_files(
    tmp_path: Path,
    mock_model: _MockModel,
    mock_tokenizer: _MockTokenizer,
    feature_direction: torch.Tensor,
    bench_items: list[BenchmarkItem],
) -> None:
    feature = FeatureSpec(
        model_key="mock",
        layer=1,
        feature_id=0,
        direction=feature_direction.numpy().copy(),
    )
    cfg = CausalRunnerConfig(
        alphas=(-1.0, 0.0, 1.0),
        top_features_per_layer=1,
        random_control=RandomControlConfig(n_directions=2, seed=1),
        capability_eval=None,
        seed=0,
        n_bootstrap=50,
    )
    runner = CausalExperimentRunner(cfg)
    score_fn = _steering_aware_score_fn(feature_direction)
    cell = runner.run_one(
        model=mock_model,
        tokenizer=mock_tokenizer,
        feature=feature,
        benchmark=bench_items,
        score_fn=score_fn,
    )

    dose_path = plot_dose_response(cell.curve, output_path=tmp_path / "dose.pdf")
    assert dose_path.exists() and dose_path.stat().st_size > 0

    if cell.ablation is not None:
        abl_path = plot_ablation_bars(cell, output_path=tmp_path / "abl.pdf")
        assert abl_path.exists() and abl_path.stat().st_size > 0

    summary_path = plot_feature_summary_bars([cell], output_path=tmp_path / "summary.pdf")
    assert summary_path.exists() and summary_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# End-to-end run_causal via Hydra-free entrypoint with monkey-patched loader
# ---------------------------------------------------------------------------


def test_run_causal_with_mocked_loader(
    tmp_path: Path,
    mock_model: _MockModel,
    mock_tokenizer: _MockTokenizer,
    feature_direction: torch.Tensor,
) -> None:
    """Drive run_causal end-to-end with an in-memory OmegaConf and fake loader."""
    from omegaconf import OmegaConf

    # Write a fake SAE-results JSON so load_feature_specs has something to chew on.
    sae_dir = tmp_path / "sae"
    sae_dir.mkdir()
    direction_list = feature_direction.numpy().tolist()
    sae_payload = {
        "model": "mock-llama",
        "layer": 1,
        "top_features": [
            {
                "feature_id": 7,
                "effect_size": 0.9,
                "q_value": 1e-6,
                "direction": direction_list,
            }
        ],
    }
    (sae_dir / "mock-llama_layer01_features.json").write_text(json.dumps(sae_payload))

    # Write a tiny benchmark.
    bench_path = tmp_path / "benchmark.jsonl"
    with bench_path.open("w") as fh:
        for it in _make_bench_items(n_pairs=3):
            fh.write(
                json.dumps(
                    {
                        "id": it.id,
                        "category": it.category,
                        "subcategory": it.subcategory,
                        "conflict": it.conflict,
                        "difficulty": it.difficulty,
                        "prompt": it.prompt,
                        "system_prompt": it.system_prompt,
                        "correct_answer": it.correct_answer,
                        "lure_answer": it.lure_answer,
                        "answer_pattern": it.answer_pattern,
                        "lure_pattern": it.lure_pattern,
                        "matched_pair_id": it.matched_pair_id,
                        "source": it.source,
                        "provenance_note": it.provenance_note,
                        "paraphrases": list(it.paraphrases),
                    }
                )
                + "\n"
            )

    output_dir = tmp_path / "causal_out"
    cfg = OmegaConf.create(
        {
            "sae_results_dir": str(sae_dir),
            "benchmark_path": str(bench_path),
            "output_dir": str(output_dir),
            "seed": 0,
            "models": ["mock-llama"],
            "models_hf": {"mock-llama": "mock-llama"},
            "device": "cpu",
            "alphas": [-1.0, 0.0, 1.0],
            "top_features_per_layer": 1,
            "random_control": {"n_directions": 2, "seed": 1},
            "capability_eval": None,
            "max_new_tokens": 16,
            "n_bootstrap": 50,
        }
    )

    def fake_loader(hf_id: str, device: str):
        return mock_model, mock_tokenizer

    score_fn = _steering_aware_score_fn(feature_direction)
    written = run_causal(cfg, model_loader=fake_loader, score_fn=score_fn)
    assert len(written) >= 1
    # Result JSON should exist.
    result_files = list(output_dir.glob("*.json"))
    assert result_files, f"no result JSON written, got {list(output_dir.iterdir())}"
    loaded = json.loads(result_files[0].read_text())
    assert loaded["model"] == "mock-llama"
    assert loaded["feature_id"] == 7
