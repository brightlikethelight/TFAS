"""Hydra-decorated CLI entry point for the probing workstream.

Usage::

    python scripts/run_probes.py activations_path=data/activations/main.h5 \
        models_to_probe=[llama-3.1-8b-instruct] targets=[task_type]

Reads a Hydra config (``configs/probe.yaml``), iterates over the requested
(model, target, layer, position) grid, runs one :class:`ProbeRunner` per cell,
and writes the resulting JSON files under ``results_dir``.

The main loop is simple — the heavy lifting is in :mod:`s1s2.probes.core`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import hydra
from beartype import beartype
from omegaconf import DictConfig, OmegaConf

from s1s2.probes.core import (
    ProbeRunner,
    RunnerConfig,
    apply_bh_across_layers,
    load_layer_activations,
    save_layer_result,
)
from s1s2.probes.targets import build_target
from s1s2.utils import io as ioh
from s1s2.utils.logging import get_logger
from s1s2.utils.seed import set_global_seed
from s1s2.utils.wandb_utils import (
    finish as wandb_finish,
)
from s1s2.utils.wandb_utils import (
    init_run as wandb_init,
)
from s1s2.utils.wandb_utils import (
    log_artifact as wandb_log_artifact,
)
from s1s2.utils.wandb_utils import (
    log_metrics as wandb_log_metrics,
)
from s1s2.utils.wandb_utils import (
    log_summary as wandb_log_summary,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


@beartype
def runner_config_from_hydra(cfg: DictConfig) -> RunnerConfig:
    """Translate a Hydra cfg into a :class:`RunnerConfig`."""
    probe_kwargs: dict[str, dict[str, Any]] = {}
    pk = cfg.get("probe_kwargs")
    if pk is not None:
        for name, kw in pk.items():
            probe_kwargs[str(name)] = dict(kw)
    return RunnerConfig(
        probes=tuple(cfg.probes),
        n_folds=int(cfg.cv.n_folds),
        n_seeds=int(cfg.cv.n_seeds),
        control_enabled=bool(cfg.control_task.enabled),
        control_n_shuffles=int(cfg.control_task.n_shuffles),
        n_permutations=int(cfg.permutation.n_permutations),
        n_bootstrap=int(cfg.bootstrap.n_resamples),
        run_loco=bool(cfg.get("loco", {}).get("enabled", False)),
        loco_targets=tuple(cfg.get("loco", {}).get("targets", ("task_type",))),
        probe_kwargs=probe_kwargs,
        seed=int(cfg.get("seed", 0)),
    )


# ---------------------------------------------------------------------------
# Grid iteration
# ---------------------------------------------------------------------------


@beartype
def iter_layers_for_model(cfg: DictConfig, model_key: str) -> Iterable[int]:
    """Yield layer indices to probe for a given model.

    The default is "every layer". ``cfg.layers`` may override with a list of
    explicit indices or a ``range(start, stop, step)`` dict.
    """
    layers_cfg = cfg.get("layers", "all")
    if layers_cfg == "all" or layers_cfg is None:
        n_layers = int(cfg.models[model_key].n_layers)
        return range(n_layers)
    if isinstance(layers_cfg, list | tuple):
        return [int(x) for x in layers_cfg]
    if isinstance(layers_cfg, dict) or hasattr(layers_cfg, "start"):
        start = int(layers_cfg.get("start", 0))
        stop = int(layers_cfg.get("stop", int(cfg.models[model_key].n_layers)))
        step = int(layers_cfg.get("step", 1))
        return range(start, stop, step)
    raise ValueError(f"cannot interpret layers={layers_cfg!r}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


@beartype
def run_probes(cfg: DictConfig) -> list[Path]:
    """Run the full grid. Returns the list of written JSON paths."""
    set_global_seed(int(cfg.get("seed", 0)), deterministic_torch=False)
    runner_cfg = runner_config_from_hydra(cfg)
    runner = ProbeRunner(runner_cfg)

    act_path = Path(cfg.activations_path)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Optional W&B setup -------------------------------------------------
    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    wandb_mode = str(wandb_cfg.get("mode", "disabled")) if wandb_enabled else "disabled"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    wandb_init(
        project=str(wandb_cfg.get("project", "s1s2")),
        group="probes",
        name=f"probes_{'-'.join(str(m) for m in cfg.models_to_probe)}",
        config=cfg_dict,
        tags=[str(m) for m in cfg.models_to_probe] + list(wandb_cfg.get("tags", [])),
        mode=wandb_mode,
    )

    # We'll collect results per (model, target, position) so BH-FDR can be
    # applied across layers at the end.
    grouped: dict[tuple[str, str, str], list] = {}
    written: list[Path] = []
    step_counter = 0

    with ioh.open_activations(act_path) as f:
        for model_key in cfg.models_to_probe:
            hdf5_key = cfg.models[model_key].hdf5_key
            logger.info(f"[{model_key}] hdf5_key={hdf5_key}")
            # Validate the model exists in the cache.
            if hdf5_key not in ioh.list_models(f):
                logger.warning(f"[{model_key}] not in activations file — skipping")
                continue

            for target in cfg.targets:
                logger.info(f"[{model_key}/{target}] building target")
                try:
                    td = build_target(target, f, hdf5_key)
                except Exception as e:
                    logger.error(f"[{model_key}/{target}] target build failed: {e}")
                    continue

                for position in cfg.positions:
                    for layer in iter_layers_for_model(cfg, model_key):
                        X, pos_valid = load_layer_activations(
                            act_path, hdf5_key, layer=layer, position=position
                        )
                        if not pos_valid:
                            logger.info(
                                f"[{model_key}/{target}/L{layer:02d}/{position}] "
                                "position not valid for this model — skipping"
                            )
                            continue
                        logger.info(
                            f"[{model_key}/{target}/L{layer:02d}/{position}] "
                            f"X.shape={X.shape} y.mean={td.y.mean():.3f}"
                        )
                        try:
                            res = runner.run(
                                X=X,
                                target_data=td,
                                model=model_key,
                                layer=layer,
                                position=position,
                            )
                        except Exception as e:
                            logger.error(
                                f"[{model_key}/{target}/L{layer:02d}/{position}] "
                                f"runner failed: {e}"
                            )
                            continue
                        key = (model_key, target, position)
                        grouped.setdefault(key, []).append(res)

                        # --- W&B: per-layer metrics -------------------------
                        primary = res.probes.get("logistic") or next(
                            iter(res.probes.values()), None
                        )
                        if primary is not None:
                            wandb_log_metrics(
                                {
                                    "roc_auc": primary.summary.get("roc_auc", 0.0),
                                    "selectivity": primary.summary.get("selectivity", 0.0),
                                    "layer": layer,
                                    "model": model_key,
                                    "target": target,
                                    "position": position,
                                },
                                step=step_counter,
                            )
                        step_counter += 1

    # BH-FDR across layers per (model, target, position).
    for (_model_key, _target, _position), res_list in grouped.items():
        res_list = apply_bh_across_layers(res_list, probe_name="logistic")
        for r in res_list:
            path = save_layer_result(r, results_dir)
            written.append(path)
            logger.info(f"wrote {path}")

    # --- W&B: summary + artifact upload -------------------------------------
    if written:
        wandb_log_summary({"n_result_files": len(written)})
        wandb_log_artifact(
            name=f"probes_results_{results_dir.name}",
            path=str(results_dir),
            artifact_type="probes_results",
        )
    wandb_finish()

    return written


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../../configs", config_name="probe", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for ``python -m s1s2.probes.cli``."""
    logger.info("probe config:\n" + OmegaConf.to_yaml(cfg))
    paths = run_probes(cfg)
    logger.info(f"wrote {len(paths)} result files")


if __name__ == "__main__":  # pragma: no cover
    main()
