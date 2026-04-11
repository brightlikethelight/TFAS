"""Hydra-decorated CLI entry point for the SAE workstream.

Usage::

    python scripts/run_sae.py
    python scripts/run_sae.py models=[llama-3.1-8b-instruct] layers=[16]
    python scripts/run_sae.py falsification.mode=model_forward

Reads the Hydra config ``configs/sae.yaml`` and translates it into an
:class:`s1s2.sae.core.SAERunnerConfig`. The heavy lifting is in
:class:`s1s2.sae.core.SAEAnalysisRunner`; this module is just parsing
and glue.

We deliberately keep the CLI small so the underlying class stays
Hydra-agnostic and unit-testable without spinning up Hydra.
"""

from __future__ import annotations

from typing import Any

import hydra
from beartype import beartype
from omegaconf import DictConfig, OmegaConf

from s1s2.sae.core import (
    DEFAULT_MODEL_HDF5_KEYS,
    DEFAULT_MODEL_SAE_RELEASES,
    SAEAnalysisRunner,
    SAERunnerConfig,
)
from s1s2.utils.logging import get_logger
from s1s2.utils.seed import set_global_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config translation
# ---------------------------------------------------------------------------


def _cfg_get(cfg: DictConfig, path: str, default: Any) -> Any:
    """Dotted-path lookup with a default. ``"a.b"`` -> ``cfg.a.b``.

    OmegaConf supports ``OmegaConf.select`` which almost does this, but we
    want a plain-Python API that's trivial to test and doesn't require the
    OmegaConf import chain here.
    """
    node: Any = cfg
    for key in path.split("."):
        if node is None:
            return default
        if isinstance(node, DictConfig) and key in node:
            node = node[key]
        elif hasattr(node, key):
            node = getattr(node, key)
        else:
            return default
    return node if node is not None else default


@beartype
def runner_config_from_hydra(cfg: DictConfig) -> SAERunnerConfig:
    """Translate a Hydra ``DictConfig`` into a :class:`SAERunnerConfig`.

    Accepts either the flat schema shown in the task brief
    (``reconstruction.n_samples``, ``differential.fdr_q``, etc.) or a
    fully-flattened variant, so hand-crafted overrides from the shell
    (``fdr_q=0.01``) work without needing to know the nesting.
    """

    # Models metadata (optional — pulled from configs/models.yaml if present).
    models_meta = cfg.get("models", None) if isinstance(cfg, DictConfig) else None
    model_hdf5_keys: dict[str, str] = {}
    model_sae_releases: dict[str, str] = {}
    if isinstance(models_meta, DictConfig):
        for key, entry in models_meta.items():
            if isinstance(entry, DictConfig):
                if "hdf5_key" in entry:
                    model_hdf5_keys[str(key)] = str(entry["hdf5_key"])
                if "sae_release" in entry and entry["sae_release"] is not None:
                    model_sae_releases[str(key)] = str(entry["sae_release"])
    # Fill defaults for anything still missing so the runner is robust when
    # only a subset of models is in configs/models.yaml.
    for k, v in DEFAULT_MODEL_HDF5_KEYS.items():
        model_hdf5_keys.setdefault(k, v)
    for k, v in DEFAULT_MODEL_SAE_RELEASES.items():
        model_sae_releases.setdefault(k, v)

    # The model whitelist can come from one of three places, in priority
    # order: (1) a top-level ``models_to_run`` list — used by the SAE Hydra
    # config and the test suite; (2) a top-level ``models`` list (the SAE
    # task brief style); (3) a top-level ``models`` DictConfig (the shared
    # ``configs/models.yaml`` form, where the keys are the whitelist).
    explicit_models_to_run = _cfg_get(cfg, "models_to_run", None)
    if explicit_models_to_run is not None:
        models_list = [str(m) for m in explicit_models_to_run]
    else:
        models_field = _cfg_get(cfg, "models", None)
        if isinstance(models_field, list | tuple) or (
            hasattr(models_field, "__iter__") and not isinstance(models_field, DictConfig)
        ):
            models_list = [str(m) for m in models_field]
        elif isinstance(models_field, DictConfig):
            models_list = list(models_field.keys())
        else:
            models_list = list(DEFAULT_MODEL_HDF5_KEYS.keys())

    layers_field = _cfg_get(cfg, "layers", None)
    if layers_field in (None, "none", "None", "all"):
        layers_list: list[int] | None = None
    else:
        layers_list = [int(x) for x in layers_field]

    return SAERunnerConfig(
        activations_path=str(_cfg_get(cfg, "activations_path", "data/activations/main.h5")),
        output_dir=str(_cfg_get(cfg, "output_dir", "results/sae")),
        models=models_list,
        layers=layers_list,
        position=str(_cfg_get(cfg, "position", "P0")),
        reconstruction_check_n_samples=int(_cfg_get(cfg, "reconstruction.n_samples", 256)),
        reconstruction_min_explained_variance=float(
            _cfg_get(cfg, "reconstruction.min_explained_variance", 0.5)
        ),
        fdr_q=float(_cfg_get(cfg, "differential.fdr_q", _cfg_get(cfg, "fdr_q", 0.05))),
        matched_difficulty_only=bool(_cfg_get(cfg, "differential.matched_difficulty_only", False)),
        falsification_enabled=bool(_cfg_get(cfg, "falsification.enabled", True)),
        falsification_n_random_texts=int(_cfg_get(cfg, "falsification.n_random_texts", 100)),
        falsification_n_top_tokens=int(_cfg_get(cfg, "falsification.n_top_tokens", 5)),
        falsification_threshold=float(_cfg_get(cfg, "falsification.threshold", 0.5)),
        falsification_mode=str(_cfg_get(cfg, "falsification.mode", "cheap")),
        falsification_top_k_features=int(_cfg_get(cfg, "falsification.top_k_features", 50)),
        volcano_top_k=int(_cfg_get(cfg, "volcano.top_k", 10)),
        model_hdf5_keys=model_hdf5_keys,
        model_sae_releases=model_sae_releases,
        sae_device=str(_cfg_get(cfg, "sae_device", "cpu")),
        seed=int(_cfg_get(cfg, "seed", 0)),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@beartype
def run_sae_from_hydra(cfg: DictConfig) -> dict[tuple[str, int], dict[str, Any]]:
    """Top-level driver used by the Hydra entry point and by unit tests."""
    from dataclasses import asdict

    set_global_seed(int(_cfg_get(cfg, "seed", 0)), deterministic_torch=False)
    runner_cfg = runner_config_from_hydra(cfg)
    # Avoid ``OmegaConf.structured`` on the dataclass because OmegaConf
    # does not support ``typing.Literal`` annotations. A plain dict dump
    # is equally informative and has no such type constraint.
    logger.info("SAE runner config:\n%s", OmegaConf.to_yaml(asdict(runner_cfg)))
    runner = SAEAnalysisRunner(runner_cfg)
    return runner.run()


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../../configs", config_name="sae", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for ``python -m s1s2.sae.cli`` and ``scripts/run_sae.py``."""
    logger.info("SAE raw config:\n" + OmegaConf.to_yaml(cfg))
    results = run_sae_from_hydra(cfg)
    logger.info("SAE finished: %d cells processed", len(results))


if __name__ == "__main__":  # pragma: no cover
    main()
