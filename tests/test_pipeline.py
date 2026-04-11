"""Tests for the pipeline orchestrator: checkpointing, config hashing, reporting."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from s1s2.pipeline import (
    ALL_STAGES,
    Checkpoint,
    Pipeline,
    PipelineConfig,
    PipelineReport,
    StageResult,
    is_stage_completed,
    read_checkpoint,
    write_checkpoint,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def ckpt_dir(tmp_path: Path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture()
def default_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        stages=["validate", "extract"],
        models=["llama-3.1-8b-instruct"],
        checkpoint_dir=str(tmp_path / "ckpt"),
        seed=0,
    )


# ── Checkpoint round-trip ───────────────────────────────────────────────────


class TestCheckpointRoundTrip:
    def test_write_read(self, ckpt_dir: Path) -> None:
        """Checkpoint written by write_checkpoint is recoverable by read_checkpoint."""
        path = write_checkpoint(
            ckpt_dir,
            stage="probes",
            status="completed",
            duration=12.5,
            outputs=["results/probes"],
            config_hash="abc123",
        )
        assert path.exists()
        assert path.name == "probes.json"

        ckpt = read_checkpoint(ckpt_dir, "probes")
        assert ckpt is not None
        assert ckpt.stage == "probes"
        assert ckpt.status == "completed"
        assert ckpt.duration_seconds == 12.5
        assert ckpt.outputs == ["results/probes"]
        assert ckpt.config_hash == "abc123"
        assert ckpt.timestamp  # non-empty ISO timestamp

    def test_read_nonexistent(self, ckpt_dir: Path) -> None:
        """Reading a checkpoint for a stage that hasn't run returns None."""
        assert read_checkpoint(ckpt_dir, "nonexistent") is None

    def test_read_corrupt_json(self, ckpt_dir: Path) -> None:
        """Corrupt JSON is handled gracefully."""
        (ckpt_dir / "bad.json").write_text("not valid json{{{")
        assert read_checkpoint(ckpt_dir, "bad") is None

    def test_checkpoint_to_dict_from_dict(self) -> None:
        """Checkpoint dataclass serializes and deserializes."""
        ckpt = Checkpoint(
            stage="sae",
            status="completed",
            timestamp="2026-04-09T12:00:00+00:00",
            duration_seconds=300.0,
            outputs=["results/sae"],
            config_hash="def456",
        )
        d = ckpt.to_dict()
        restored = Checkpoint.from_dict(d)
        assert restored == ckpt

    def test_json_roundtrip(self, ckpt_dir: Path) -> None:
        """The on-disk JSON is valid and matches the in-memory checkpoint."""
        write_checkpoint(ckpt_dir, "extract", "completed", 100.0, ["data/main.h5"], "hash1")
        raw = json.loads((ckpt_dir / "extract.json").read_text())
        assert raw["stage"] == "extract"
        assert raw["config_hash"] == "hash1"


# ── Config hash invalidation ───────────────────────────────────────────────


class TestConfigHashInvalidation:
    def test_same_config_matches(self, ckpt_dir: Path) -> None:
        """Checkpoint with matching hash is recognized as completed."""
        cfg = PipelineConfig(models=["llama-3.1-8b-instruct"], seed=0)
        h = cfg.config_hash()
        write_checkpoint(ckpt_dir, "probes", "completed", 5.0, [], h)
        assert is_stage_completed(ckpt_dir, "probes", h) is True

    def test_different_seed_invalidates(self, ckpt_dir: Path) -> None:
        """Changing the seed invalidates existing checkpoints."""
        cfg1 = PipelineConfig(seed=0)
        cfg2 = PipelineConfig(seed=42)
        write_checkpoint(ckpt_dir, "probes", "completed", 5.0, [], cfg1.config_hash())
        assert is_stage_completed(ckpt_dir, "probes", cfg2.config_hash()) is False

    def test_different_models_invalidates(self, ckpt_dir: Path) -> None:
        """Changing the model list invalidates existing checkpoints."""
        cfg1 = PipelineConfig(models=["llama-3.1-8b-instruct"])
        cfg2 = PipelineConfig(models=["llama-3.1-8b-instruct", "r1-distill-llama-8b"])
        write_checkpoint(ckpt_dir, "extract", "completed", 100.0, [], cfg1.config_hash())
        assert is_stage_completed(ckpt_dir, "extract", cfg2.config_hash()) is False

    def test_failed_checkpoint_not_completed(self, ckpt_dir: Path) -> None:
        """A failed checkpoint should not count as completed."""
        cfg = PipelineConfig(seed=0)
        h = cfg.config_hash()
        write_checkpoint(ckpt_dir, "sae", "failed", 10.0, [], h)
        assert is_stage_completed(ckpt_dir, "sae", h) is False

    def test_config_hash_deterministic(self) -> None:
        """Same config produces the same hash every time."""
        c1 = PipelineConfig(models=["a", "b"], seed=7)
        c2 = PipelineConfig(models=["b", "a"], seed=7)  # order differs
        # Models are sorted in to_dict, so order shouldn't matter.
        assert c1.config_hash() == c2.config_hash()


# ── Stage skipping on existing checkpoint ───────────────────────────────────


class TestStageSkipping:
    def test_completed_stage_is_skipped(self, tmp_path: Path) -> None:
        """Pipeline skips a stage that has a valid checkpoint."""
        config = PipelineConfig(
            stages=["validate"],
            models=["llama-3.1-8b-instruct"],
            checkpoint_dir=str(tmp_path / "ckpt"),
            skip_completed=True,
        )
        h = config.config_hash()
        ckpt_dir = Path(config.checkpoint_dir)
        write_checkpoint(ckpt_dir, "validate", "completed", 0.5, [], h)

        # Mock run_stage so it would fail if called — we expect it NOT to be called.
        with patch("s1s2.pipeline.run_stage", side_effect=AssertionError("should not run")):
            pipeline = Pipeline(config, repo_root=tmp_path)
            report = pipeline.run()

        assert len(report.results) == 1
        assert report.results[0].status == "skipped"

    def test_skip_disabled_reruns(self, tmp_path: Path) -> None:
        """With skip_completed=False, stages re-run even with checkpoints."""
        config = PipelineConfig(
            stages=["validate"],
            models=["llama-3.1-8b-instruct"],
            checkpoint_dir=str(tmp_path / "ckpt"),
            skip_completed=False,
        )
        h = config.config_hash()
        ckpt_dir = Path(config.checkpoint_dir)
        write_checkpoint(ckpt_dir, "validate", "completed", 0.5, [], h)

        with patch("s1s2.pipeline.run_stage", return_value=(True, "ok", 0.1)):
            pipeline = Pipeline(config, repo_root=tmp_path)
            report = pipeline.run()

        assert report.results[0].status == "completed"


# ── Error handling ──────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_pipeline_continues_on_failure(self, tmp_path: Path) -> None:
        """By default, pipeline continues past a failed stage."""
        config = PipelineConfig(
            stages=["validate", "extract"],
            models=["llama-3.1-8b-instruct"],
            checkpoint_dir=str(tmp_path / "ckpt"),
            stop_on_error=False,
            skip_completed=False,
        )

        call_count = 0

        def mock_run_stage(stage, cfg, root):
            nonlocal call_count
            call_count += 1
            if stage == "validate":
                return (False, "validation error", 0.5)
            return (True, "ok", 1.0)

        with patch("s1s2.pipeline.run_stage", side_effect=mock_run_stage):
            pipeline = Pipeline(config, repo_root=tmp_path)
            report = pipeline.run()

        assert call_count == 2  # both stages ran
        assert report.results[0].status == "failed"
        assert report.results[1].status == "completed"
        assert report.any_failed is True

    def test_stop_on_error_halts(self, tmp_path: Path) -> None:
        """With stop_on_error, pipeline halts after the first failure."""
        config = PipelineConfig(
            stages=["validate", "extract", "probes"],
            models=["llama-3.1-8b-instruct"],
            checkpoint_dir=str(tmp_path / "ckpt"),
            stop_on_error=True,
            skip_completed=False,
        )

        call_count = 0

        def mock_run_stage(stage, cfg, root):
            nonlocal call_count
            call_count += 1
            if stage == "extract":
                return (False, "OOM", 30.0)
            return (True, "ok", 1.0)

        with patch("s1s2.pipeline.run_stage", side_effect=mock_run_stage):
            pipeline = Pipeline(config, repo_root=tmp_path)
            report = pipeline.run()

        # validate ran (ok), extract ran (fail), probes did NOT run
        assert call_count == 2
        assert len(report.results) == 2
        assert report.results[1].status == "failed"

    def test_exception_in_run_stage_caught(self, tmp_path: Path) -> None:
        """If run_stage raises, the pipeline catches it and records failure."""
        config = PipelineConfig(
            stages=["validate"],
            models=["llama-3.1-8b-instruct"],
            checkpoint_dir=str(tmp_path / "ckpt"),
            skip_completed=False,
        )

        with patch("s1s2.pipeline.run_stage", side_effect=RuntimeError("kaboom")):
            pipeline = Pipeline(config, repo_root=tmp_path)
            report = pipeline.run()

        assert report.results[0].status == "failed"
        assert "kaboom" in report.results[0].message

    def test_failed_checkpoint_written(self, tmp_path: Path) -> None:
        """A failed stage writes a 'failed' checkpoint."""
        config = PipelineConfig(
            stages=["extract"],
            models=["llama-3.1-8b-instruct"],
            checkpoint_dir=str(tmp_path / "ckpt"),
            skip_completed=False,
        )

        with patch("s1s2.pipeline.run_stage", return_value=(False, "error", 5.0)):
            pipeline = Pipeline(config, repo_root=tmp_path)
            pipeline.run()

        ckpt = read_checkpoint(Path(config.checkpoint_dir), "extract")
        assert ckpt is not None
        assert ckpt.status == "failed"

    def test_unknown_stage(self, tmp_path: Path) -> None:
        """An unknown stage name is reported as failed."""
        config = PipelineConfig(
            stages=["nonexistent_stage"],
            models=["llama-3.1-8b-instruct"],
            checkpoint_dir=str(tmp_path / "ckpt"),
            skip_completed=False,
        )
        pipeline = Pipeline(config, repo_root=tmp_path)
        report = pipeline.run()

        assert report.results[0].status == "failed"
        assert "unknown" in report.results[0].message


# ── PipelineReport ──────────────────────────────────────────────────────────


class TestPipelineReport:
    def test_empty_report(self) -> None:
        report = PipelineReport()
        assert report.n_completed == 0
        assert report.n_failed == 0
        assert report.n_skipped == 0
        assert report.any_failed is False

    def test_mixed_results(self) -> None:
        report = PipelineReport()
        report.add("validate", "completed", "ok", 0.5)
        report.add("extract", "failed", "OOM", 30.0)
        report.add("probes", "skipped", "checkpoint exists")
        assert report.n_completed == 1
        assert report.n_failed == 1
        assert report.n_skipped == 1
        assert report.any_failed is True

    def test_print_summary_no_crash(self) -> None:
        """print_summary should not raise even with rich installed."""
        report = PipelineReport()
        report.add("validate", "completed", "ok", 0.1)
        report.add("extract", "failed", "error msg", 5.0)
        # Should not raise.
        report.print_summary()

    def test_stage_result_fields(self) -> None:
        r = StageResult(name="sae", status="completed", message="ok", duration_seconds=120.0)
        assert r.name == "sae"
        assert r.duration_seconds == 120.0


# ── PipelineConfig ──────────────────────────────────────────────────────────


class TestPipelineConfig:
    def test_default_stages(self) -> None:
        cfg = PipelineConfig()
        assert cfg.stages == ALL_STAGES

    def test_to_dict(self) -> None:
        cfg = PipelineConfig(models=["a", "b"], seed=42)
        d = cfg.to_dict()
        assert d["seed"] == 42
        assert d["models"] == ["a", "b"]  # sorted

    def test_config_hash_changes_with_activations_path(self) -> None:
        c1 = PipelineConfig(activations_path="data/a.h5")
        c2 = PipelineConfig(activations_path="data/b.h5")
        assert c1.config_hash() != c2.config_hash()


# ── clean_checkpoints ──────────────────────────────────────────────────────


class TestCleanCheckpoints:
    def test_clean_removes_files(self, tmp_path: Path) -> None:
        config = PipelineConfig(checkpoint_dir=str(tmp_path / "ckpt"))
        pipeline = Pipeline(config, repo_root=tmp_path)
        ckpt_dir = Path(config.checkpoint_dir)
        write_checkpoint(ckpt_dir, "probes", "completed", 1.0, [], "h")
        write_checkpoint(ckpt_dir, "sae", "completed", 2.0, [], "h")

        n = pipeline.clean_checkpoints()
        assert n == 2
        assert list(ckpt_dir.glob("*.json")) == []

    def test_clean_empty_dir(self, tmp_path: Path) -> None:
        config = PipelineConfig(checkpoint_dir=str(tmp_path / "ckpt"))
        pipeline = Pipeline(config, repo_root=tmp_path)
        # Dir doesn't even exist yet.
        assert pipeline.clean_checkpoints() == 0
