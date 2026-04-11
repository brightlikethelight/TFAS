"""Optional W&B integration. All functions are no-ops if wandb is not installed.

This module provides a thin wrapper around the wandb SDK so that every
workstream can log metrics, summaries, figures, and artifacts without
caring whether W&B is available. If wandb is not installed or no run is
active, every function silently returns None / does nothing.

The import-time guard uses a module-level ``_wandb`` reference so the
``try: import wandb`` cost is paid once and never again.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

_wandb: Any = None
with contextlib.suppress(ImportError):
    import wandb as _wandb


def is_available() -> bool:
    """True only when wandb is installed AND a run is currently active."""
    return _wandb is not None and _wandb.run is not None


def init_run(
    project: str = "s1s2",
    group: str | None = None,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    mode: str = "online",
) -> Any:
    """Initialize a W&B run. Returns the run object, or None if unavailable.

    Parameters
    ----------
    project : W&B project name (default ``"s1s2"``).
    group : logical group for the run (typically the workstream name).
    name : human-readable run name shown in the dashboard.
    config : full experiment config dict — logged as the W&B run config
        for reproducibility.
    tags : list of string tags (model name, target, etc.).
    mode : ``"online"`` | ``"offline"`` | ``"disabled"``.
    """
    if _wandb is None:
        return None
    return _wandb.init(
        project=project,
        group=group,
        name=name,
        config=config,
        tags=tags,
        mode=mode,
        reinit=True,
    )


def log_metrics(metrics: dict[str, float | int], step: int | None = None) -> None:
    """Log a flat dict of numeric metrics to the active W&B run."""
    if _wandb is None or _wandb.run is None:
        return
    _wandb.log(metrics, step=step)


def log_summary(summary: dict[str, Any]) -> None:
    """Write key-value pairs into the W&B run summary (final metrics)."""
    if _wandb is None or _wandb.run is None:
        return
    for k, v in summary.items():
        _wandb.run.summary[k] = v


def log_artifact(name: str, path: str, artifact_type: str = "result") -> None:
    """Upload a file or directory as a W&B artifact.

    Uses ``add_file`` for files and ``add_dir`` for directories.
    """
    if _wandb is None or _wandb.run is None:
        return
    art = _wandb.Artifact(name=name, type=artifact_type)
    p = Path(path)
    if p.is_dir():
        art.add_dir(str(p))
    else:
        art.add_file(str(p))
    _wandb.log_artifact(art)


def log_figure(key: str, figure: Any) -> None:
    """Log a matplotlib figure to W&B as an image."""
    if _wandb is None or _wandb.run is None:
        return
    _wandb.log({key: _wandb.Image(figure)})


def finish() -> None:
    """Finish the active W&B run (flush and close)."""
    if _wandb is not None and _wandb.run is not None:
        _wandb.finish()
