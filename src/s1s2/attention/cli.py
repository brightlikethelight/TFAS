"""Hydra entry point for the attention-entropy workstream.

Reads precomputed attention metrics from the activation HDF5 cache
(written by ``s1s2.extract``) and runs the per-head differential test +
multi-metric consensus head classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from s1s2.attention.core import (
    AttentionConfig,
    analyze_all_models,
    save_model_report,
)
from s1s2.utils.logging import get_logger
from s1s2.utils.seed import set_global_seed

logger = get_logger(__name__)


def runner_config_from_hydra(cfg: DictConfig) -> AttentionConfig:
    """Translate a Hydra ``DictConfig`` into the typed :class:`AttentionConfig`."""
    raw_pairs = cfg.get("matched_pairs", [["llama-3.1-8b-instruct", "r1-distill-llama-8b"]])
    matched_pairs = tuple((str(a), str(b)) for a, b in raw_pairs)
    out = AttentionConfig(
        activations_path=str(cfg.get("activations_path", "data/activations/main.h5")),
        output_dir=str(cfg.get("output_dir", "results/attention")),
        seed=int(cfg.get("seed", 0)),
        metrics=tuple(
            cfg.get(
                "metrics",
                ("entropy", "entropy_normalized", "gini", "max_attn", "focus_5"),
            )
        ),
        positions=tuple(cfg.get("positions", ("P0", "P2", "T50", "Tend"))),
        fdr_q=float(cfg.get("fdr_q", 0.05)),
        effect_size_threshold=float(cfg.get("effect_size_threshold", 0.3)),
        multi_metric_consensus=int(cfg.get("multi_metric_consensus", 3)),
        report_kv_group_aggregate=bool(cfg.get("report_kv_group_aggregate", True)),
        gemma_separate_window_layers=bool(cfg.get("gemma_separate_window_layers", True)),
        matched_pairs=matched_pairs,
    )
    out.validate()
    return out


def run_attention(cfg: DictConfig) -> dict[str, str]:
    """Run the workstream end-to-end. Returns a mapping of model -> output dir."""
    set_global_seed(int(cfg.get("seed", 0)), deterministic_torch=False)
    runner_cfg = runner_config_from_hydra(cfg)
    Path(runner_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    models_to_run = list(cfg.get("models", [])) or [a for a, _ in runner_cfg.matched_pairs]
    logger.info("attention runner config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Analyzing models: %s", models_to_run)

    reports = analyze_all_models(
        activations_path=runner_cfg.activations_path,
        model_config_keys=models_to_run,
        config=runner_cfg,
    )

    out_dirs: dict[str, str] = {}
    for model_key, report in reports.items():
        out_dir = Path(runner_cfg.output_dir) / model_key
        out_dir.mkdir(parents=True, exist_ok=True)
        save_model_report(report, out_dir)
        out_dirs[model_key] = str(out_dir)
        logger.info("saved %s -> %s", model_key, out_dir)

    # Persist the resolved config alongside results for traceability
    (Path(runner_cfg.output_dir) / "_config.json").write_text(
        json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2, default=str)
    )
    return out_dirs


@hydra.main(config_path="../../../configs", config_name="attention", version_base=None)
def main(cfg: DictConfig) -> None:
    run_attention(cfg)


if __name__ == "__main__":
    main()
