"""Smoke tests for the unified paper-figure pipeline.

These tests synthesize tiny JSON payloads matching each workstream's
expected schema and run the corresponding figure generator. They do
NOT exercise any real model data — the goal is to catch wiring bugs
(missing imports, wrong kwargs, path collisions) before they hit a
real run, and to verify the sweep is graceful when inputs are missing.

Every test:

* uses ``tmp_path`` so the filesystem is clean between runs
* forces a non-interactive matplotlib backend
* asserts either a file was produced OR the sweep reported a clean skip
  (never a crash)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Non-interactive backend must be set before matplotlib.pyplot imports.
import matplotlib
import pytest

matplotlib.use("Agg")

# Make sure OpenMP + friends don't break on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Allow tests to run without pip install.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from s1s2.viz.figure1_benchmark import make_figure_1_benchmark  # noqa: E402
from s1s2.viz.figure2_probes import make_figure_2_probes  # noqa: E402
from s1s2.viz.figure3_sae import make_figure_3_sae  # noqa: E402
from s1s2.viz.figure4_causal import make_figure_4_causal  # noqa: E402
from s1s2.viz.figure5_attention import make_figure_5_attention  # noqa: E402
from s1s2.viz.figure6_geometry import make_figure_6_geometry  # noqa: E402
from s1s2.viz.paper_figures import (  # noqa: E402
    PaperFiguresConfig,
    make_paper_figures,
)

# --------------------------------------------------------------------------- #
# Synthetic payload builders                                                   #
# --------------------------------------------------------------------------- #


def _write_benchmark_jsonl(path: Path) -> Path:
    """Write a tiny, schema-valid benchmark JSONL to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [
        {
            "id": "crt_01__conflict",
            "category": "crt",
            "subcategory": "ratio",
            "conflict": True,
            "difficulty": 2,
            "prompt": "A widget and a gadget cost $1.10. The widget costs $1.00 more than the gadget. How much is the gadget?",
            "system_prompt": None,
            "correct_answer": "5",
            "lure_answer": "10",
            "answer_pattern": r"\b5\b",
            "lure_pattern": r"\b10\b",
            "matched_pair_id": "crt_01_pair",
            "source": "template",
            "provenance_note": "synthetic",
            "paraphrases": [],
        },
        {
            "id": "crt_01__control",
            "category": "crt",
            "subcategory": "ratio",
            "conflict": False,
            "difficulty": 2,
            "prompt": "A widget costs $0.05 and a gadget costs $1.05. How much is the widget?",
            "system_prompt": None,
            "correct_answer": "5",
            "lure_answer": "",
            "answer_pattern": r"\b5\b",
            "lure_pattern": "",
            "matched_pair_id": "crt_01_pair",
            "source": "template",
            "provenance_note": "synthetic",
            "paraphrases": [],
        },
        {
            "id": "base_rate_01__conflict",
            "category": "base_rate",
            "subcategory": "medical",
            "conflict": True,
            "difficulty": 3,
            "prompt": "Base-rate lure prompt.",
            "system_prompt": None,
            "correct_answer": "A",
            "lure_answer": "B",
            "answer_pattern": "A",
            "lure_pattern": "B",
            "matched_pair_id": "base_rate_01_pair",
            "source": "template",
            "provenance_note": "synthetic",
            "paraphrases": [],
        },
        {
            "id": "base_rate_01__control",
            "category": "base_rate",
            "subcategory": "medical",
            "conflict": False,
            "difficulty": 3,
            "prompt": "Base-rate matched control prompt.",
            "system_prompt": None,
            "correct_answer": "A",
            "lure_answer": "",
            "answer_pattern": "A",
            "lure_pattern": "",
            "matched_pair_id": "base_rate_01_pair",
            "source": "template",
            "provenance_note": "synthetic",
            "paraphrases": [],
        },
    ]
    with path.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it))
            fh.write("\n")
    return path


def _write_synthetic_probe_results(root: Path) -> Path:
    """Drop 1 model x 3 layers of fake probe results matching the loader schema.

    The loader (``s1s2.viz.probe_plots.load_results_tree``) walks
    ``root/{model}/{target}/layer_NN_pos_{pos}.json``.
    """
    model = "llama-3.1-8b-instruct"
    target = "task_type"
    pos = "P0"
    out_dir = root / model / target
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer, auc in [(0, 0.52), (8, 0.73), (16, 0.88)]:
        payload = {
            "model": model,
            "target": target,
            "position": pos,
            "layer": layer,
            "probes": {
                "logistic": {
                    "summary": {
                        "roc_auc": auc,
                        "roc_auc_ci_lower": auc - 0.03,
                        "roc_auc_ci_upper": auc + 0.03,
                        "selectivity": 0.08,
                        "permutation_null_auc_95": 0.55,
                    }
                }
            },
        }
        with (out_dir / f"layer_{layer:02d}_pos_{pos}.json").open("w") as fh:
            json.dump(payload, fh)
    return root


def _write_synthetic_sae_results(root: Path) -> Path:
    """Drop a fake SAE differential result with a ``features`` table."""
    out_dir = root
    out_dir.mkdir(parents=True, exist_ok=True)
    features = [
        {
            "feature_id": i,
            "log_fc": (i - 10) * 0.15,
            "q_value": 0.5 / (i + 1),
            "is_falsified": (i % 5 == 0),
        }
        for i in range(20)
    ]
    payload = {
        "model": "llama-3.1-8b-instruct",
        "layer": 16,
        "features": features,
    }
    path = out_dir / "llama-3.1-8b-instruct_layer16_differential.json"
    with path.open("w") as fh:
        json.dump(payload, fh)
    return root


def _write_synthetic_causal_results(root: Path) -> Path:
    """Drop fake per-cell causal JSONs matching :class:`CausalCellResult`."""
    out_dir = root
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, bump in [
        ("llama-3.1-8b-instruct", 0.10),
        ("r1-distill-llama-8b", 0.20),
    ]:
        points = []
        for alpha in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            for group, base in [
                ("conflict", 0.45),
                ("no_conflict", 0.72),
                ("random_control", 0.45),
            ]:
                if group == "conflict":
                    p_corr = base + max(0.0, alpha) * bump
                elif group == "random_control":
                    p_corr = base + max(0.0, alpha) * 0.005
                else:
                    p_corr = base
                points.append(
                    {
                        "alpha": alpha,
                        "group": group,
                        "n": 20,
                        "p_correct": p_corr,
                        "ci_lower": p_corr - 0.05,
                        "ci_upper": p_corr + 0.05,
                        "per_seed": [],
                    }
                )
        payload = {
            "model": model,
            "layer": 16,
            "feature_id": 1234,
            "curve": {
                "model": model,
                "layer": 16,
                "feature_id": 1234,
                "alphas": [-3.0, -1.0, 0.0, 1.0, 3.0],
                "points": points,
                "fit": {
                    "conflict": {"slope": bump},
                    "random_control": {"slope": 0.005},
                },
            },
            "ablation": {
                "baseline_p_correct_conflict": 0.45,
                "ablated_p_correct_conflict": 0.45 - bump * 2,
                "baseline_p_correct_no_conflict": 0.72,
                "ablated_p_correct_no_conflict": 0.70,
            },
            "canonical_s2": True,
            "capability": [],
            "config": {},
            "elapsed_s": 1.0,
        }
        fname = f"{model}_layer16_feature001234.json"
        with (out_dir / fname).open("w") as fh:
            json.dump(payload, fh)
    return root


def _write_synthetic_attention_results(root: Path) -> Path:
    """Drop per-model attention layer-summary JSONs."""
    for model in ("llama-3.1-8b-instruct", "r1-distill-llama-8b"):
        mdir = root / model
        mdir.mkdir(parents=True, exist_ok=True)
        layer_summary = []
        for layer in range(8):
            delta = 0.05 * (layer % 4 - 1)
            layer_summary.append(
                {
                    "layer": layer,
                    "layer_type": "standard",
                    "metric": "entropy",
                    "position": "P0",
                    "mean": 1.2 + delta,
                    "max": 1.8,
                    "spread": 0.2,
                    "mean_conflict": 1.2 + delta,
                    "mean_noconflict": 1.2,
                    "delta": delta,
                    "wilcoxon_statistic": 10.0,
                    "wilcoxon_p": 0.03 if abs(delta) > 0.04 else 0.2,
                    "S2_head_count": 0,
                    "S1_head_count": 0,
                    "mixed_head_count": 0,
                    "non_specialized_head_count": 8,
                }
            )
        payload = {"model": model, "layer_summary": layer_summary}
        with (mdir / "layer_summary.json").open("w") as fh:
            json.dump(payload, fh)
    return root


def _write_synthetic_geometry_results(root: Path) -> Path:
    """Drop per-model silhouette + PCA JSONs."""
    for model in ("llama-3.1-8b-instruct", "r1-distill-llama-8b"):
        mdir = root / model
        mdir.mkdir(parents=True, exist_ok=True)
        layers = list(range(8))
        sil = [0.02 + 0.06 * i for i in layers]
        payload = {
            "model": model,
            "layers": layers,
            "silhouette": sil,
            "ci_lower": [s - 0.02 for s in sil],
            "ci_upper": [s + 0.02 for s in sil],
        }
        with (mdir / "silhouette.json").open("w") as fh:
            json.dump(payload, fh)
    # PCA sample with labels (2-class).
    pca_payload = {
        "model": "llama-3.1-8b-instruct",
        "layer": 16,
        "pca_projection": [[0.1, 0.2], [-0.1, -0.2], [0.3, 0.1], [-0.3, -0.1]],
        "labels": [1, 0, 1, 0],
        "explained_variance": [0.45, 0.12],
    }
    with (root / "llama-3.1-8b-instruct_pca.json").open("w") as fh:
        json.dump(pca_payload, fh)
    return root


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_results_dir(tmp_path: Path) -> Path:
    """Build a fully populated results tree covering every workstream."""
    root = tmp_path / "results"
    _write_synthetic_probe_results(root / "probes")
    _write_synthetic_sae_results(root / "sae")
    _write_synthetic_causal_results(root / "causal")
    _write_synthetic_attention_results(root / "attention")
    _write_synthetic_geometry_results(root / "geometry")
    return root


@pytest.fixture
def synthetic_benchmark_path(tmp_path: Path) -> Path:
    """Write and return the path to a tiny benchmark JSONL."""
    return _write_benchmark_jsonl(tmp_path / "benchmark" / "benchmark.jsonl")


@pytest.fixture
def cfg_full(
    tmp_path: Path,
    synthetic_results_dir: Path,
    synthetic_benchmark_path: Path,
) -> PaperFiguresConfig:
    return PaperFiguresConfig(
        results_dir=synthetic_results_dir,
        output_dir=tmp_path / "figures",
        benchmark_path=synthetic_benchmark_path,
        format="pdf",
    )


# --------------------------------------------------------------------------- #
# Config & import sanity                                                       #
# --------------------------------------------------------------------------- #


def test_paper_figures_config_instantiation(tmp_path: Path) -> None:
    cfg = PaperFiguresConfig(
        results_dir=tmp_path / "results",
        output_dir=tmp_path / "figures",
        benchmark_path=tmp_path / "bench.jsonl",
    )
    # Path coercion in __post_init__.
    assert isinstance(cfg.results_dir, Path)
    assert isinstance(cfg.output_dir, Path)
    assert isinstance(cfg.benchmark_path, Path)
    assert cfg.format == "pdf"
    assert cfg.dpi == 300
    assert len(cfg.include) == 6


def test_all_figure_generators_importable() -> None:
    """Catch import-time errors in any of the per-figure modules."""
    for fn in (
        make_figure_1_benchmark,
        make_figure_2_probes,
        make_figure_3_sae,
        make_figure_4_causal,
        make_figure_5_attention,
        make_figure_6_geometry,
    ):
        assert callable(fn)


# --------------------------------------------------------------------------- #
# Figure 1 — benchmark schematic (works from JSONL only)                       #
# --------------------------------------------------------------------------- #


def test_figure_1_renders_from_synthetic_benchmark(
    tmp_path: Path,
    synthetic_benchmark_path: Path,
) -> None:
    cfg = PaperFiguresConfig(
        results_dir=tmp_path / "results",
        output_dir=tmp_path / "figures",
        benchmark_path=synthetic_benchmark_path,
    )
    out = make_figure_1_benchmark(cfg)
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.suffix == ".pdf"


def test_figure_1_raises_on_missing_benchmark(tmp_path: Path) -> None:
    cfg = PaperFiguresConfig(
        results_dir=tmp_path / "results",
        output_dir=tmp_path / "figures",
        benchmark_path=tmp_path / "does_not_exist.jsonl",
    )
    with pytest.raises(FileNotFoundError):
        make_figure_1_benchmark(cfg)


# --------------------------------------------------------------------------- #
# Full sweep smoke tests                                                       #
# --------------------------------------------------------------------------- #


def test_make_paper_figures_empty_results_gracefully_skips(tmp_path: Path) -> None:
    """With no results at all, the sweep should return an empty dict, not crash."""
    cfg = PaperFiguresConfig(
        results_dir=tmp_path / "nonexistent_results",
        output_dir=tmp_path / "figures",
        benchmark_path=tmp_path / "nonexistent.jsonl",
    )
    results = make_paper_figures(cfg)
    assert isinstance(results, dict)
    # Everything is missing; nothing should come back successful.
    assert results == {}
    # But the output directory was still created.
    assert cfg.output_dir.exists()


def test_make_paper_figures_partial_tree_continues_past_failures(
    tmp_path: Path,
    synthetic_benchmark_path: Path,
) -> None:
    """With only the benchmark path valid, Figure 1 should succeed and others skip."""
    cfg = PaperFiguresConfig(
        results_dir=tmp_path / "empty_results",
        output_dir=tmp_path / "figures",
        benchmark_path=synthetic_benchmark_path,
    )
    results = make_paper_figures(cfg)
    assert "figure_1_benchmark" in results
    assert results["figure_1_benchmark"].exists()
    # No other workstream has results; they should all be absent.
    for name in (
        "figure_2_probes",
        "figure_3_sae",
        "figure_4_causal",
        "figure_5_attention",
        "figure_6_geometry",
    ):
        assert name not in results


def test_make_paper_figures_full_sweep_on_synthetic_tree(
    cfg_full: PaperFiguresConfig,
) -> None:
    """Run all six figures on a fully populated synthetic tree.

    Each figure should produce a file; we don't assert on content
    (that's the atomic plot-function's responsibility) — only that the
    pipeline stitches everything together without crashing and each
    generator returns a path that actually exists on disk.
    """
    results = make_paper_figures(cfg_full)
    expected = {
        "figure_1_benchmark",
        "figure_2_probes",
        "figure_3_sae",
        "figure_4_causal",
        "figure_5_attention",
        "figure_6_geometry",
    }
    missing = expected - set(results.keys())
    assert not missing, (
        f"expected all six figures to render, but missing: {sorted(missing)}; "
        f"got: {sorted(results.keys())}"
    )
    for name, path in results.items():
        assert path.exists(), f"{name} reported path {path} but file does not exist"
        assert path.stat().st_size > 0


def test_make_paper_figures_include_subset(
    cfg_full: PaperFiguresConfig,
) -> None:
    """Only the requested figures should be rendered."""
    cfg_full.include = ["figure_1_benchmark", "figure_3_sae"]
    results = make_paper_figures(cfg_full)
    assert set(results.keys()) <= {"figure_1_benchmark", "figure_3_sae"}
    # Figure 1 should definitely be in (benchmark path is valid).
    assert "figure_1_benchmark" in results


def test_make_paper_figures_unknown_figure_name_is_skipped(
    cfg_full: PaperFiguresConfig,
) -> None:
    cfg_full.include = ["figure_1_benchmark", "not_a_real_figure"]
    results = make_paper_figures(cfg_full)
    assert "figure_1_benchmark" in results
    assert "not_a_real_figure" not in results


def test_make_paper_figures_png_format(
    cfg_full: PaperFiguresConfig,
) -> None:
    cfg_full.format = "png"
    cfg_full.include = ["figure_1_benchmark"]
    results = make_paper_figures(cfg_full)
    assert "figure_1_benchmark" in results
    assert results["figure_1_benchmark"].suffix == ".png"
