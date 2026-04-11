"""Pipeline orchestration: stages, dependencies, checkpointing, reporting.

The Pipeline class runs the full s1s2 analysis pipeline in order, with
JSON-based checkpointing so interrupted runs can resume where they left off.
Config hashing ensures checkpoints are invalidated when parameters change.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from s1s2.utils.logging import get_logger

logger = get_logger("s1s2.pipeline")

# ── Stage ordering ──────────────────────────────────────────────────────────

ALL_STAGES: list[str] = [
    "validate",
    "extract",
    "probes",
    "sae",
    "attention",
    "geometry",
    "causal",
    "metacog",
    "figures",
]

# Stages that require GPU. The rest are CPU-only (or read from cache).
GPU_STAGES: set[str] = {"extract", "sae", "causal", "metacog"}

# Map from stage name to the script that runs it. The pipeline invokes each
# stage as a subprocess so Hydra configs and per-workstream CLIs are
# respected without coupling the orchestrator to their internals.
STAGE_SCRIPTS: dict[str, str] = {
    "validate": "scripts/generate_benchmark.py",
    "extract": "scripts/extract_all.py",
    "probes": "scripts/run_probes.py",
    "sae": "scripts/run_sae.py",
    "attention": "scripts/run_attention.py",
    "geometry": "scripts/run_geometry.py",
    "causal": "scripts/run_causal.py",
    "metacog": "scripts/run_metacog.py",
    "figures": "scripts/generate_figures.py",
}


# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Full pipeline configuration. Hashable for checkpoint invalidation."""

    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))
    models: list[str] = field(
        default_factory=lambda: [
            "llama-3.1-8b-instruct",
            "gemma-2-9b-it",
            "r1-distill-llama-8b",
            "r1-distill-qwen-7b",
        ]
    )
    activations_path: str = "data/activations/main.h5"
    results_dir: str = "results"
    figures_dir: str = "figures"
    checkpoint_dir: str = ".pipeline_checkpoints"
    seed: int = 0
    skip_completed: bool = True
    stop_on_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": self.stages,
            "models": sorted(self.models),
            "activations_path": self.activations_path,
            "results_dir": self.results_dir,
            "figures_dir": self.figures_dir,
            "seed": self.seed,
        }

    def config_hash(self) -> str:
        """Deterministic hash of the parameters that affect stage outputs.

        If any of these change between runs the checkpoints are stale and
        stages need re-running.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class StageResult:
    """Outcome of a single pipeline stage."""

    name: str
    status: str  # "completed" | "failed" | "skipped"
    message: str
    duration_seconds: float = 0.0


@dataclass
class PipelineReport:
    """Collects results from all stages and prints a summary."""

    results: list[StageResult] = field(default_factory=list)

    def add(
        self,
        name: str,
        status: str,
        message: str,
        duration: float = 0.0,
    ) -> None:
        self.results.append(StageResult(name, status, message, duration))

    @property
    def any_failed(self) -> bool:
        return any(r.status == "failed" for r in self.results)

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.results if r.status == "completed")

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    @property
    def n_skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    def print_summary(self) -> None:
        """Rich-ish summary table printed to stdout."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Pipeline Summary", show_lines=True)
            table.add_column("Stage", style="bold")
            table.add_column("Status")
            table.add_column("Duration")
            table.add_column("Message", max_width=60)

            status_style = {
                "completed": "green",
                "failed": "red bold",
                "skipped": "dim",
            }
            for r in self.results:
                style = status_style.get(r.status, "")
                dur = f"{r.duration_seconds:.1f}s" if r.duration_seconds > 0 else "-"
                table.add_row(r.name, f"[{style}]{r.status}[/]", dur, r.message)

            console.print()
            console.print(table)
            console.print(
                f"\n  completed={self.n_completed}  "
                f"failed={self.n_failed}  "
                f"skipped={self.n_skipped}"
            )
        except ImportError:
            # Fallback if rich is somehow missing at runtime.
            _print_summary_plain(self)


def _print_summary_plain(report: PipelineReport) -> None:
    """Plain-text fallback."""
    print("\n=== Pipeline Summary ===")
    width = max((len(r.name) for r in report.results), default=10) + 2
    for r in report.results:
        dur = f"{r.duration_seconds:.1f}s" if r.duration_seconds > 0 else "-"
        print(f"  {r.name:<{width}} {r.status:<10} {dur:<8} {r.message}")
    print(
        f"\n  completed={report.n_completed}  "
        f"failed={report.n_failed}  "
        f"skipped={report.n_skipped}"
    )


# ── Checkpoint I/O ──────────────────────────────────────────────────────────


@dataclass
class Checkpoint:
    """JSON-serializable checkpoint for one completed stage."""

    stage: str
    status: str
    timestamp: str
    duration_seconds: float
    outputs: list[str]
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "outputs": self.outputs,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Checkpoint:
        return cls(
            stage=d["stage"],
            status=d["status"],
            timestamp=d["timestamp"],
            duration_seconds=d["duration_seconds"],
            outputs=d.get("outputs", []),
            config_hash=d["config_hash"],
        )


def write_checkpoint(
    checkpoint_dir: Path,
    stage: str,
    status: str,
    duration: float,
    outputs: list[str],
    config_hash: str,
) -> Path:
    """Write a JSON checkpoint marker for a completed (or failed) stage."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(
        stage=stage,
        status=status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        duration_seconds=round(duration, 2),
        outputs=outputs,
        config_hash=config_hash,
    )
    path = checkpoint_dir / f"{stage}.json"
    path.write_text(json.dumps(ckpt.to_dict(), indent=2))
    return path


def read_checkpoint(checkpoint_dir: Path, stage: str) -> Checkpoint | None:
    """Read a checkpoint if it exists and is valid JSON."""
    path = checkpoint_dir / f"{stage}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return Checkpoint.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def is_stage_completed(
    checkpoint_dir: Path,
    stage: str,
    config_hash: str,
) -> bool:
    """Check whether a stage has a valid, matching checkpoint."""
    ckpt = read_checkpoint(checkpoint_dir, stage)
    if ckpt is None:
        return False
    if ckpt.status != "completed":
        return False
    if ckpt.config_hash != config_hash:
        logger.info(
            "checkpoint for '%s' has stale config_hash (%s != %s), will re-run",
            stage,
            ckpt.config_hash,
            config_hash,
        )
        return False
    return True


# ── Stage runner ────────────────────────────────────────────────────────────


def _build_stage_command(
    stage: str,
    config: PipelineConfig,
    repo_root: Path,
) -> list[str]:
    """Build the subprocess command for a stage.

    Each workstream script uses Hydra, so we pass overrides as positional
    args. The validate stage (generate_benchmark.py) uses argparse instead.
    """
    script = repo_root / STAGE_SCRIPTS[stage]
    cmd: list[str] = [sys.executable, str(script)]

    if stage == "validate":
        # argparse-based; no extra args needed for default paths
        pass
    elif stage == "extract":
        models_str = "[" + ",".join(config.models) + "]"
        cmd.extend([
            f"output_path={config.activations_path}",
            f"models_to_extract={models_str}",
            f"generation.seed={config.seed}",
        ])
    elif stage == "figures":
        cmd.extend([
            f"results_dir={config.results_dir}",
            f"output_dir={config.figures_dir}",
        ])
    else:
        # Analysis workstreams: pass activations path and models
        cmd.append(f"activations_path={config.activations_path}")
        models_str = "[" + ",".join(config.models) + "]"
        cmd.append(f"models_to_analyze={models_str}")

    return cmd


def run_stage(
    stage: str,
    config: PipelineConfig,
    repo_root: Path,
) -> tuple[bool, str, float]:
    """Execute a single pipeline stage as a subprocess.

    Returns (success, message, duration_seconds).
    """
    cmd = _build_stage_command(stage, config, repo_root)
    logger.info("running stage '%s': %s", stage, " ".join(cmd))

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=7200,  # 2h hard limit per stage
        )
        duration = time.monotonic() - t0

        if result.returncode == 0:
            return True, "ok", duration
        else:
            # Capture last 20 lines of stderr for the report.
            stderr_tail = "\n".join(result.stderr.strip().splitlines()[-20:])
            msg = f"exit code {result.returncode}"
            if stderr_tail:
                msg += f"\n{stderr_tail}"
            return False, msg, duration

    except subprocess.TimeoutExpired:
        duration = time.monotonic() - t0
        return False, "timed out (>2h)", duration
    except Exception as exc:
        duration = time.monotonic() - t0
        return False, f"{type(exc).__name__}: {exc}", duration


# ── Pipeline class ──────────────────────────────────────────────────────────


class Pipeline:
    """Orchestrates the full s1s2 analysis pipeline.

    Runs stages in sequence, writes checkpoints, and collects a report.
    Stages that already have a valid checkpoint (matching config hash) are
    skipped unless ``config.skip_completed`` is False.
    """

    def __init__(self, config: PipelineConfig, repo_root: Path | None = None):
        self.config = config
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self._config_hash = config.config_hash()

    def run(self) -> PipelineReport:
        """Run all configured stages in order."""
        report = PipelineReport()
        logger.info(
            "pipeline starting: stages=%s, config_hash=%s",
            self.config.stages,
            self._config_hash,
        )

        for stage in self.config.stages:
            if stage not in ALL_STAGES:
                report.add(stage, "failed", f"unknown stage '{stage}'")
                if self.config.stop_on_error:
                    break
                continue

            # Check checkpoint
            if self.config.skip_completed and is_stage_completed(
                self.checkpoint_dir, stage, self._config_hash
            ):
                logger.info("skipping '%s' — checkpoint exists", stage)
                report.add(stage, "skipped", "checkpoint exists")
                continue

            # Run the stage
            try:
                ok, msg, duration = run_stage(stage, self.config, self.repo_root)
            except Exception as exc:
                ok, msg, duration = False, f"{type(exc).__name__}: {exc}", 0.0

            status = "completed" if ok else "failed"
            report.add(stage, status, msg, duration)

            # Write checkpoint regardless of success (so we track failures too)
            write_checkpoint(
                self.checkpoint_dir,
                stage,
                status,
                duration,
                outputs=self._expected_outputs(stage),
                config_hash=self._config_hash,
            )

            if not ok and self.config.stop_on_error:
                logger.error("stopping pipeline — stage '%s' failed", stage)
                break

        report.print_summary()
        return report

    def _expected_outputs(self, stage: str) -> list[str]:
        """List of expected output paths for a stage (for the checkpoint)."""
        if stage == "validate":
            return ["data/benchmark/benchmark.jsonl"]
        elif stage == "extract":
            return [self.config.activations_path]
        elif stage == "figures":
            return [self.config.figures_dir]
        else:
            return [f"{self.config.results_dir}/{stage}"]

    def clean_checkpoints(self) -> int:
        """Remove all checkpoint files. Returns count of files removed."""
        if not self.checkpoint_dir.exists():
            return 0
        removed = 0
        for p in self.checkpoint_dir.glob("*.json"):
            p.unlink()
            removed += 1
        return removed


# ── Single-model runner ─────────────────────────────────────────────────────

# CPU-only analysis stages that can run on a single model's cached activations.
SINGLE_MODEL_CPU_STAGES: list[str] = [
    "probes",
    "attention",
    "geometry",
]


def run_single_model(
    model_key: str,
    config: PipelineConfig,
    repo_root: Path | None = None,
    include_extract: bool = True,
) -> PipelineReport:
    """Extract one model and run all CPU analyses on its cached activations.

    Useful for incremental progress on preemptible jobs: get one model done,
    show preliminary results, then queue the rest.
    """
    root = repo_root or Path(__file__).resolve().parent.parent
    report = PipelineReport()

    single_config = PipelineConfig(
        stages=[],  # we drive stages manually below
        models=[model_key],
        activations_path=config.activations_path,
        results_dir=config.results_dir,
        figures_dir=config.figures_dir,
        checkpoint_dir=config.checkpoint_dir,
        seed=config.seed,
        skip_completed=config.skip_completed,
        stop_on_error=config.stop_on_error,
    )
    chash = single_config.config_hash()
    ckpt_dir = Path(single_config.checkpoint_dir)

    stages_to_run: list[str] = []
    if include_extract:
        stages_to_run.append("extract")
    stages_to_run.extend(SINGLE_MODEL_CPU_STAGES)

    for stage in stages_to_run:
        ckpt_key = f"{stage}__{model_key}"

        if single_config.skip_completed and is_stage_completed(ckpt_dir, ckpt_key, chash):
            logger.info("skipping '%s' for model '%s' — checkpoint exists", stage, model_key)
            report.add(f"{stage} ({model_key})", "skipped", "checkpoint exists")
            continue

        try:
            ok, msg, duration = run_stage(stage, single_config, root)
        except Exception as exc:
            ok, msg, duration = False, f"{type(exc).__name__}: {exc}", 0.0

        status = "completed" if ok else "failed"
        report.add(f"{stage} ({model_key})", status, msg, duration)

        write_checkpoint(
            ckpt_dir,
            ckpt_key,
            status,
            duration,
            outputs=[],
            config_hash=chash,
        )

        if not ok and single_config.stop_on_error:
            break

    report.print_summary()
    return report
