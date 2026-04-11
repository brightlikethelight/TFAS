"""Tests for s1s2.utils.wandb_utils — no W&B account required.

Every test verifies that the public API degrades gracefully when wandb
is unavailable (import fails) or when no run is active. We mock
``wandb`` at the module level so no real network calls or logins happen.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers to simulate "wandb not installed" vs "wandb installed"
# ---------------------------------------------------------------------------


def _reload_with_wandb_absent():
    """Reload wandb_utils with wandb absent (ImportError on import)."""
    # Remove any cached wandb module so the try/except re-fires.
    saved = sys.modules.pop("wandb", None)
    # Patch builtins so `import wandb` raises ImportError inside the module.
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("mocked: wandb not installed")
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=_fake_import):
        # Also remove the cached module itself so importlib re-executes.
        sys.modules.pop("s1s2.utils.wandb_utils", None)
        mod = importlib.import_module("s1s2.utils.wandb_utils")

    # Restore the real wandb module reference if it existed.
    if saved is not None:
        sys.modules["wandb"] = saved
    return mod


def _reload_with_mock_wandb():
    """Reload wandb_utils with a mock wandb module (installed, but no real backend)."""
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.run = None  # no active run initially
    fake_wandb.init = mock.MagicMock(return_value=mock.MagicMock())
    fake_wandb.log = mock.MagicMock()
    fake_wandb.finish = mock.MagicMock()
    fake_wandb.Artifact = mock.MagicMock()
    fake_wandb.Image = mock.MagicMock()
    fake_wandb.log_artifact = mock.MagicMock()

    sys.modules["wandb"] = fake_wandb
    sys.modules.pop("s1s2.utils.wandb_utils", None)
    mod = importlib.import_module("s1s2.utils.wandb_utils")
    return mod, fake_wandb


# ---------------------------------------------------------------------------
# Tests: wandb NOT installed
# ---------------------------------------------------------------------------


class TestWandbUnavailable:
    """All functions should be silent no-ops when wandb cannot be imported."""

    def setup_method(self):
        self.mod = _reload_with_wandb_absent()

    def test_is_available_returns_false(self):
        assert self.mod.is_available() is False

    def test_init_run_returns_none(self):
        result = self.mod.init_run(project="test")
        assert result is None

    def test_log_metrics_noop(self):
        # Should not raise.
        self.mod.log_metrics({"acc": 0.95})

    def test_log_summary_noop(self):
        self.mod.log_summary({"final_auc": 0.88})

    def test_log_artifact_noop(self):
        self.mod.log_artifact("name", "/fake/path")

    def test_log_figure_noop(self):
        self.mod.log_figure("fig", object())

    def test_finish_noop(self):
        self.mod.finish()


# ---------------------------------------------------------------------------
# Tests: wandb installed but no active run
# ---------------------------------------------------------------------------


class TestWandbInstalledNoRun:
    """wandb is importable but no run has been initialized."""

    def setup_method(self):
        self.mod, self.fake_wandb = _reload_with_mock_wandb()
        # Ensure run is None (no active run).
        self.fake_wandb.run = None

    def test_is_available_false_without_run(self):
        assert self.mod.is_available() is False

    def test_log_metrics_noop_without_run(self):
        self.mod.log_metrics({"x": 1.0})
        self.fake_wandb.log.assert_not_called()

    def test_log_summary_noop_without_run(self):
        self.mod.log_summary({"x": 1.0})

    def test_log_artifact_noop_without_run(self):
        self.mod.log_artifact("name", "/fake/path")
        self.fake_wandb.log_artifact.assert_not_called()

    def test_log_figure_noop_without_run(self):
        self.mod.log_figure("fig", object())
        self.fake_wandb.log.assert_not_called()

    def test_finish_noop_without_run(self):
        self.mod.finish()
        self.fake_wandb.finish.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: wandb installed WITH an active run
# ---------------------------------------------------------------------------


class TestWandbWithActiveRun:
    """wandb is importable and a run is active (simulated)."""

    def setup_method(self):
        self.mod, self.fake_wandb = _reload_with_mock_wandb()
        # Simulate an active run.
        self.fake_wandb.run = mock.MagicMock()
        self.fake_wandb.run.summary = {}

    def test_is_available_true(self):
        assert self.mod.is_available() is True

    def test_init_run_calls_wandb_init(self):
        run = self.mod.init_run(project="s1s2", group="probes", name="test_run")
        self.fake_wandb.init.assert_called_once()
        assert run is not None

    def test_log_metrics_forwards(self):
        self.mod.log_metrics({"auc": 0.9, "layer": 5}, step=3)
        self.fake_wandb.log.assert_called_once_with({"auc": 0.9, "layer": 5}, step=3)

    def test_log_summary_sets_keys(self):
        self.mod.log_summary({"final_auc": 0.88, "n_layers": 32})
        assert self.fake_wandb.run.summary["final_auc"] == 0.88
        assert self.fake_wandb.run.summary["n_layers"] == 32

    def test_log_figure_calls_log(self):
        fig = mock.MagicMock()
        self.mod.log_figure("volcano", fig)
        self.fake_wandb.log.assert_called_once()
        call_args = self.fake_wandb.log.call_args
        assert "volcano" in call_args[0][0]

    def test_finish_calls_wandb_finish(self):
        self.mod.finish()
        self.fake_wandb.finish.assert_called_once()

    def test_log_artifact_file(self, tmp_path):
        """log_artifact with a file path creates an Artifact and adds the file."""
        f = tmp_path / "result.json"
        f.write_text("{}")
        self.mod.log_artifact("probes_result", str(f), artifact_type="probes_results")
        self.fake_wandb.Artifact.assert_called_once_with(
            name="probes_result", type="probes_results"
        )

    def test_log_artifact_dir(self, tmp_path):
        """log_artifact with a directory path uses add_dir."""
        d = tmp_path / "results"
        d.mkdir()
        (d / "a.json").write_text("{}")
        self.mod.log_artifact("sae_results", str(d))
        self.fake_wandb.Artifact.assert_called_once()
