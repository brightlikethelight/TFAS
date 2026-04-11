"""Tests for the s1s2 dashboard module.

These tests verify that the dashboard can be created in synthetic mode
without crashing, and that each plotting function returns valid figures.
They do NOT launch a Gradio server.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Allow import without pip install -e .
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

gradio = pytest.importorskip("gradio", reason="gradio not installed")

import matplotlib.pyplot as plt  # noqa: E402

from s1s2.dashboard.app import (  # noqa: E402
    MODEL_KEYS,
    POSITIONS,
    TARGETS,
    build_hypothesis_table,
    create_app,
    plot_attention,
    plot_behavioral,
    plot_geometry,
    plot_probes,
    plot_sae,
)


class TestCreateApp:
    """Verify the app factory produces a valid Gradio Blocks instance."""

    def test_synthetic_mode(self) -> None:
        app = create_app(synthetic=True)
        assert app is not None
        assert isinstance(app, gradio.Blocks)

    def test_real_mode_no_crash(self) -> None:
        """Real mode with no results should still construct the app."""
        app = create_app(synthetic=False, results_dir="/nonexistent")
        assert app is not None


class TestBehavioralTab:
    """Synthetic behavioral plots produce valid figures."""

    @pytest.mark.parametrize("model", MODEL_KEYS)
    def test_plot_returns_figure(self, model: str) -> None:
        fig = plot_behavioral(model, results_dir=None, synthetic=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestProbesTab:
    """Synthetic probe plots produce valid figures."""

    def test_default_produces_figure(self) -> None:
        fig = plot_probes(MODEL_KEYS[0], TARGETS[0], POSITIONS[0], None, synthetic=True)
        assert isinstance(fig, plt.Figure)
        n_axes = len(fig.get_axes())
        assert n_axes == 2, f"expected 2 subplots, got {n_axes}"
        plt.close(fig)

    @pytest.mark.parametrize("model", MODEL_KEYS)
    def test_all_models(self, model: str) -> None:
        fig = plot_probes(model, "task_type", "P0", None, synthetic=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSAETab:
    """Synthetic SAE plots produce valid figures and summary text."""

    def test_volcano_returns_figure_and_summary(self) -> None:
        fig, summary = plot_sae(MODEL_KEYS[0], 16, None, synthetic=True)
        assert isinstance(fig, plt.Figure)
        assert isinstance(summary, str)
        assert "Features" in summary
        plt.close(fig)


class TestAttentionTab:
    """Synthetic attention plots produce valid figures."""

    def test_heatmap_and_profile(self) -> None:
        fig_hm, fig_ep = plot_attention(MODEL_KEYS[0], None, synthetic=True)
        assert isinstance(fig_hm, plt.Figure)
        assert isinstance(fig_ep, plt.Figure)
        plt.close(fig_hm)
        plt.close(fig_ep)


class TestGeometryTab:
    """Synthetic geometry plots produce valid figures."""

    def test_silhouette_and_pca(self) -> None:
        fig_sil, fig_pca = plot_geometry(MODEL_KEYS[0], None, synthetic=True)
        assert isinstance(fig_sil, plt.Figure)
        assert isinstance(fig_pca, plt.Figure)
        plt.close(fig_sil)
        plt.close(fig_pca)


class TestHypothesisTable:
    """Hypothesis evaluation table renders in both modes."""

    def test_synthetic_table(self) -> None:
        table = build_hypothesis_table(None, synthetic=True)
        assert isinstance(table, str)
        assert "H1" in table
        assert "H6" in table
        assert "PASS" in table

    def test_real_table_not_evaluated(self) -> None:
        table = build_hypothesis_table(None, synthetic=False)
        assert "NOT EVALUATED" in table


class TestNoDataFallback:
    """Real mode with missing data shows informative messages, not crashes."""

    def test_probes_missing_data(self, tmp_path: Path) -> None:
        fig = plot_probes("llama-3.1-8b-instruct", "task_type", "P0", tmp_path, synthetic=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_behavioral_missing_data(self, tmp_path: Path) -> None:
        fig = plot_behavioral("llama-3.1-8b-instruct", tmp_path, synthetic=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_sae_missing_data(self, tmp_path: Path) -> None:
        fig, summary = plot_sae("llama-3.1-8b-instruct", 16, tmp_path, synthetic=False)
        assert isinstance(fig, plt.Figure)
        assert "No" in summary or "data" in summary.lower()
        plt.close(fig)

    def test_attention_missing_data(self, tmp_path: Path) -> None:
        fig_hm, fig_ep = plot_attention("llama-3.1-8b-instruct", tmp_path, synthetic=False)
        assert isinstance(fig_hm, plt.Figure)
        plt.close(fig_hm)
        plt.close(fig_ep)

    def test_geometry_missing_data(self, tmp_path: Path) -> None:
        fig_sil, fig_pca = plot_geometry("llama-3.1-8b-instruct", tmp_path, synthetic=False)
        assert isinstance(fig_sil, plt.Figure)
        plt.close(fig_sil)
        plt.close(fig_pca)
