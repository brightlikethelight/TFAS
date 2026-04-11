"""Tests for the results aggregation and report generation system.

Creates synthetic result JSONs matching each workstream's expected format,
then verifies:
- ResultsAggregator.from_directory loads them correctly
- report.to_markdown() produces valid Markdown
- Missing workstreams show "not available", don't crash
- Hypothesis evaluation computes correct verdicts from synthetic data
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Synthetic result builders
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _make_probe_result(
    model: str,
    target: str,
    layer: int,
    position: str,
    auc: float,
    selectivity_pp: float,
) -> dict:
    """Minimal probe result matching the LayerResult -> ProbeResult -> to_json schema."""
    return {
        "model": model,
        "layer": layer,
        "position": position,
        "target": target,
        "n_problems": 284,
        "probes": {
            "l2_logistic": {
                "summary": {
                    "mean_auc": auc,
                    "std_auc": 0.02,
                },
                "fold_metrics": [{"auc": auc}] * 5,
                "control_metrics": [{"auc": auc - selectivity_pp / 100}] * 5,
            },
        },
        "loco": None,
        "config": {"n_folds": 5},
        "git_sha": "abc1234",
        "elapsed_s": 1.0,
    }


def _make_sae_result(
    model_key: str,
    layer: int,
    n_significant: int,
    n_after_falsification: int,
    top_features: list[dict] | None = None,
) -> dict:
    if top_features is None:
        top_features = [
            {
                "feature_id": i,
                "log_fc": 1.5 - i * 0.2,
                "q_value": 0.001 * (i + 1),
                "effect_size": 0.5 - i * 0.05,
                "auto_interp_label": f"feature_{i}_desc",
            }
            for i in range(min(5, n_after_falsification))
        ]
    return {
        "model_key": model_key,
        "layer": layer,
        "n_features_total": 16384,
        "n_features_significant": n_significant,
        "n_features_after_falsification": n_after_falsification,
        "top_features": top_features,
        "status": "completed",
        "reconstruction_explained_variance": 0.85,
        "config": {},
        "git_sha": "abc1234",
        "runtime_seconds": 60.0,
    }


def _make_attention_result(
    model_key: str,
    n_heads: int,
    n_s2_specialized: int,
    n_layers: int = 32,
) -> dict:
    classifications = []
    for i in range(n_heads):
        cls = "s2_specialized" if i < n_s2_specialized else "unspecialized"
        classifications.append(
            {
                "layer": i % n_layers,
                "head": i // n_layers,
                "classification": cls,
                "n_significant_metrics": 4 if cls == "s2_specialized" else 0,
            }
        )
    return {
        "model_key": model_key,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_heads // 4,
        "head_classifications": classifications,
        "kv_group_classifications": classifications[: n_heads // 4],
        "config": {},
        "elapsed_s": 30.0,
    }


def _make_geometry_result(
    layer: int,
    silhouette_score: float,
    p_value: float,
    intrinsic_dim: float = 25.0,
) -> dict:
    return {
        "layer": layer,
        "silhouette": {
            "score": silhouette_score,
            "p_value": p_value,
            "ci_lower": silhouette_score - 0.01,
            "ci_upper": silhouette_score + 0.01,
        },
        "separability": {
            "accuracy": 0.65,
            "p_value": 0.01,
        },
        "intrinsic_dim_two_nn": intrinsic_dim,
        "participation_ratio": 30.0,
        "n_samples": 284,
        "n_features": 50,
    }


def _make_causal_result(
    model: str,
    layer: int,
    feature_id: int,
    best_delta_pp: float,
    random_delta_pp: float,
) -> dict:
    return {
        "model": model,
        "layer": layer,
        "feature_id": feature_id,
        "curve": {
            "best_alpha": 3.0,
            "best_delta_conflict_pp": best_delta_pp,
            "random_mean_delta_conflict_pp": random_delta_pp,
            "alphas": [0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
        },
        "ablation": None,
        "canonical_s2": True,
        "capability": [
            {
                "benchmark": "mmlu",
                "baseline_accuracy": 0.65,
                "intervention_accuracy": 0.63,
                "delta_pp": -2.0,
                "exceeded_max_drop": False,
            },
        ],
        "config": {},
        "elapsed_s": 120.0,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_results(tmp_path: Path) -> Path:
    """Build a full synthetic results tree with all workstreams."""
    root = tmp_path / "results"

    # Probes: 2 models x 2 targets x multiple layers
    for model in ["llama-3.1-8b-instruct", "r1-distill-llama-8b"]:
        for target in ["task_type", "correctness"]:
            for layer in [8, 16, 24]:
                auc = 0.55 + layer * 0.005
                sel = 3.0 + layer * 0.2
                if "r1" in model:
                    auc += 0.05
                    sel += 2.0
                data = _make_probe_result(model, target, layer, "P0", auc, sel)
                fname = f"{model}_{target}_layer{layer:02d}_P0.json"
                _write_json(root / "probes" / fname, data)

    # SAE
    for model in ["llama-3.1-8b-instruct", "r1-distill-llama-8b"]:
        for layer in [16, 24]:
            data = _make_sae_result(model, layer, n_significant=20, n_after_falsification=8)
            _write_json(root / "sae" / model / f"layer_{layer:02d}" / "feature_analysis.json", data)

    # Attention
    for model in ["llama-3.1-8b-instruct", "r1-distill-llama-8b"]:
        n_heads = 32
        n_s2 = 3 if "r1" in model else 1
        data = _make_attention_result(model, n_heads, n_s2)
        _write_json(root / "attention" / model / "head_classifications.json", data)

    # Geometry
    for model in ["llama-3.1-8b-instruct", "r1-distill-llama-8b"]:
        for layer in [8, 16, 24]:
            sil = 0.02 + layer * 0.001
            p = 0.001 if layer >= 16 else 0.2
            data = _make_geometry_result(layer, sil, p)
            _write_json(root / "geometry" / model / f"layer_{layer:02d}" / "geometry.json", data)

    # Causal
    for model in ["llama-3.1-8b-instruct", "r1-distill-llama-8b"]:
        data = _make_causal_result(model, 16, 1234, best_delta_pp=18.5, random_delta_pp=1.2)
        _write_json(
            root / "causal" / model / "layer_16_feature_1234" / "intervention_results.json",
            data,
        )

    return root


@pytest.fixture
def empty_results(tmp_path: Path) -> Path:
    """Results dir with subdirectories but no JSON files."""
    root = tmp_path / "results"
    for ws in ["probes", "sae", "attention", "geometry", "causal"]:
        (root / ws).mkdir(parents=True)
    return root


@pytest.fixture
def partial_results(tmp_path: Path) -> Path:
    """Only probes have data; everything else is empty."""
    root = tmp_path / "results"
    for ws in ["probes", "sae", "attention", "geometry", "causal"]:
        (root / ws).mkdir(parents=True)

    # Only probes
    for model in ["llama-3.1-8b-instruct", "r1-distill-llama-8b"]:
        for layer in [16]:
            data = _make_probe_result(model, "task_type", layer, "P0", 0.72, 12.0)
            fname = f"{model}_task_type_layer{layer:02d}_P0.json"
            _write_json(root / "probes" / fname, data)

    return root


# ---------------------------------------------------------------------------
# Tests: loading
# ---------------------------------------------------------------------------


class TestResultsAggregator:
    def test_from_directory_loads_all_workstreams(self, populated_results: Path) -> None:
        from s1s2.report import ResultsAggregator

        agg = ResultsAggregator.from_directory(populated_results)
        report = agg.aggregate()

        assert len(report.models) == 2
        assert "llama-3.1-8b-instruct" in report.models
        assert "r1-distill-llama-8b" in report.models

        # Probes loaded
        assert len(report.probes) == 2
        assert "task_type" in report.probes["llama-3.1-8b-instruct"]

        # SAE loaded
        assert len(report.sae) == 2

        # Attention loaded
        assert len(report.attention) == 2

        # Geometry loaded
        assert len(report.geometry) == 2

        # Causal loaded
        assert len(report.causal) == 2

    def test_empty_results_no_crash(self, empty_results: Path) -> None:
        from s1s2.report import ResultsAggregator

        agg = ResultsAggregator.from_directory(empty_results)
        report = agg.aggregate()

        assert report.models == []
        assert report.probes == {}
        assert report.sae == {}

    def test_partial_results_loads_available(self, partial_results: Path) -> None:
        from s1s2.report import ResultsAggregator

        agg = ResultsAggregator.from_directory(partial_results)
        report = agg.aggregate()

        assert len(report.probes) == 2
        assert report.sae == {}
        assert report.attention == {}


# ---------------------------------------------------------------------------
# Tests: Markdown output
# ---------------------------------------------------------------------------


class TestMarkdownOutput:
    def test_populated_markdown_is_valid(self, populated_results: Path) -> None:
        from s1s2.report import generate_report

        report = generate_report(populated_results)
        md = report.to_markdown()

        assert isinstance(md, str)
        assert len(md) > 100
        # Has all major sections.
        assert "## 1. Executive Summary" in md
        assert "## 2. Behavioral Results" in md
        assert "## 3. Probing Results" in md
        assert "## 4. SAE Results" in md
        assert "## 5. Attention Results" in md
        assert "## 6. Geometry Results" in md
        assert "## 7. Causal Results" in md
        assert "## 8. Hypothesis Evaluation" in md

    def test_empty_markdown_shows_not_available(self, empty_results: Path) -> None:
        from s1s2.report import generate_report

        report = generate_report(empty_results)
        md = report.to_markdown()

        assert "not yet available" in md.lower()
        assert "## 1. Executive Summary" in md

    def test_partial_markdown_has_probes_but_not_sae(self, partial_results: Path) -> None:
        from s1s2.report import generate_report

        report = generate_report(partial_results)
        md = report.to_markdown()

        # Probing results should be present.
        assert "Peak Layer AUC" in md or "peak_auc" in md.lower() or "0.720" in md
        # SAE should say not available.
        assert "SAE results not yet available" in md


# ---------------------------------------------------------------------------
# Tests: JSON output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_round_trips(self, populated_results: Path) -> None:
        from s1s2.report import generate_report

        report = generate_report(populated_results)
        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert "timestamp" in parsed
        assert "git_sha" in parsed
        assert "models" in parsed
        assert len(parsed["models"]) == 2
        assert "hypotheses" in parsed

    def test_json_has_all_sections(self, populated_results: Path) -> None:
        from s1s2.report import generate_report

        report = generate_report(populated_results)
        parsed = json.loads(report.to_json())

        for section in ["behavioral", "probes", "sae", "attention", "geometry", "causal"]:
            assert section in parsed


# ---------------------------------------------------------------------------
# Tests: hypothesis evaluation
# ---------------------------------------------------------------------------


class TestHypothesisEvaluation:
    def test_h1_pass_with_two_models(self, populated_results: Path) -> None:
        from s1s2.report import generate_report

        report = generate_report(populated_results)
        h1 = report.hypotheses["H1"]

        # With layer 24 giving AUC = 0.55 + 24*0.005 = 0.67 (+ 0.05 for R1)
        # and selectivity = 3.0 + 24*0.2 = 7.8 (+ 2.0 for R1)
        # Both models should pass: AUC > 0.6 and selectivity > 5pp
        assert h1["verdict"] == "PASS"
        assert len(h1["passing_models"]) >= 2

    def test_h1_fail_low_auc(self, tmp_path: Path) -> None:
        from s1s2.report import generate_report

        root = tmp_path / "results"
        # Probes with low AUC
        for model in ["model_a", "model_b"]:
            data = _make_probe_result(model, "task_type", 16, "P0", auc=0.52, selectivity_pp=2.0)
            _write_json(root / "probes" / f"{model}_task_type_layer16_P0.json", data)

        report = generate_report(root)
        assert report.hypotheses["H1"]["verdict"] == "FAIL"

    def test_h2_inconclusive_without_pair(self, tmp_path: Path) -> None:
        from s1s2.report import generate_report

        root = tmp_path / "results"
        data = _make_probe_result("gemma-2-9b-it", "task_type", 16, "P0", 0.72, 12.0)
        _write_json(root / "probes" / "gemma-2-9b-it_task_type_layer16_P0.json", data)

        report = generate_report(root)
        assert report.hypotheses["H2"]["verdict"] == "INCONCLUSIVE"

    def test_h3_pass_with_enough_features(self, populated_results: Path) -> None:
        from s1s2.report import generate_report

        report = generate_report(populated_results)
        h3 = report.hypotheses["H3"]
        # We set n_after_falsification=8 >= 5 threshold.
        assert h3["verdict"] == "PASS"

    def test_h3_fail_too_few_features(self, tmp_path: Path) -> None:
        from s1s2.report import generate_report

        root = tmp_path / "results"
        data = _make_sae_result("model_a", 16, n_significant=3, n_after_falsification=2)
        _write_json(root / "sae" / "model_a" / "layer_16" / "feature_analysis.json", data)

        report = generate_report(root)
        assert report.hypotheses["H3"]["verdict"] == "FAIL"

    def test_h4_pass_with_strong_steering(self, populated_results: Path) -> None:
        from s1s2.report import generate_report

        report = generate_report(populated_results)
        h4 = report.hypotheses["H4"]
        # best_delta_pp=18.5 > 15, random=1.2 < 3
        assert h4["verdict"] == "PASS"

    def test_h4_fail_weak_steering(self, tmp_path: Path) -> None:
        from s1s2.report import generate_report

        root = tmp_path / "results"
        data = _make_causal_result("model_a", 16, 100, best_delta_pp=5.0, random_delta_pp=4.0)
        _write_json(
            root / "causal" / "model_a" / "layer_16_feature_0100" / "intervention_results.json",
            data,
        )

        report = generate_report(root)
        assert report.hypotheses["H4"]["verdict"] == "FAIL"

    def test_h5_pass_enough_specialized_heads(self, tmp_path: Path) -> None:
        from s1s2.report import generate_report

        root = tmp_path / "results"
        # 10 out of 32 heads are S2-specialized = 31.25% > 5%
        data = _make_attention_result("model_a", n_heads=32, n_s2_specialized=10)
        _write_json(root / "attention" / "model_a" / "head_classifications.json", data)

        report = generate_report(root)
        assert report.hypotheses["H5"]["verdict"] == "PASS"

    def test_h5_fail_no_specialized_heads(self, tmp_path: Path) -> None:
        from s1s2.report import generate_report

        root = tmp_path / "results"
        data = _make_attention_result("model_a", n_heads=32, n_s2_specialized=0)
        _write_json(root / "attention" / "model_a" / "head_classifications.json", data)

        report = generate_report(root)
        assert report.hypotheses["H5"]["verdict"] == "FAIL"

    def test_h6_pass_two_models(self, populated_results: Path) -> None:
        from s1s2.report import generate_report

        report = generate_report(populated_results)
        h6 = report.hypotheses["H6"]
        # Layer 16 and 24 have p=0.001 < 0.05 and silhouette > 0.
        assert h6["verdict"] == "PASS"

    def test_h6_fail_high_p(self, tmp_path: Path) -> None:
        from s1s2.report import generate_report

        root = tmp_path / "results"
        for model in ["model_a", "model_b"]:
            data = _make_geometry_result(16, silhouette_score=0.01, p_value=0.5)
            _write_json(root / "geometry" / model / "layer_16" / "geometry.json", data)

        report = generate_report(root)
        assert report.hypotheses["H6"]["verdict"] == "FAIL"

    def test_all_inconclusive_when_empty(self, empty_results: Path) -> None:
        from s1s2.report import generate_report

        report = generate_report(empty_results)
        for hid in ["H1", "H2", "H3", "H4", "H5", "H6"]:
            assert report.hypotheses[hid]["verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Tests: save method
# ---------------------------------------------------------------------------


class TestSaveMethod:
    def test_save_creates_both_files(self, populated_results: Path, tmp_path: Path) -> None:
        from s1s2.report import generate_report

        report = generate_report(populated_results)
        out_dir = tmp_path / "output"
        report.save(out_dir)

        assert (out_dir / "report.json").is_file()
        assert (out_dir / "report.md").is_file()

        # JSON is valid
        parsed = json.loads((out_dir / "report.json").read_text())
        assert "hypotheses" in parsed

        # Markdown has content
        md = (out_dir / "report.md").read_text()
        assert "Executive Summary" in md


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_markdown_to_stdout(
        self, populated_results: Path, capsys: pytest.CaptureFixture
    ) -> None:
        from scripts.generate_report import main

        main(["--results-dir", str(populated_results)])
        captured = capsys.readouterr()
        assert "Executive Summary" in captured.out

    def test_cli_json_only(self, populated_results: Path, capsys: pytest.CaptureFixture) -> None:
        from scripts.generate_report import main

        main(["--results-dir", str(populated_results), "--json-only"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "hypotheses" in parsed

    def test_cli_output_to_file(self, populated_results: Path, tmp_path: Path) -> None:
        from scripts.generate_report import main

        out_file = tmp_path / "my_report.md"
        main(["--results-dir", str(populated_results), "--output", str(out_file)])
        assert out_file.is_file()
        assert "Executive Summary" in out_file.read_text()

    def test_cli_output_dir(self, populated_results: Path, tmp_path: Path) -> None:
        from scripts.generate_report import main

        out_dir = tmp_path / "report_output"
        main(["--results-dir", str(populated_results), "--output-dir", str(out_dir)])
        assert (out_dir / "report.json").is_file()
        assert (out_dir / "report.md").is_file()
