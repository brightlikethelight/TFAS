"""Hydra CLI entry point for the metacognitive monitoring workstream.

Usage::

    python scripts/run_metacog.py
    python scripts/run_metacog.py models=[r1-distill-llama-8b] layers=[20]
    python scripts/run_metacog.py activations_path=data/activations/smoke.h5

The CLI iterates over the requested ``(model, layer)`` grid, runs one
:class:`s1s2.metacog.core.DifficultyDetectorAnalysis` per cell, and
writes:

- One per-(model, layer) JSON to ``cfg.output_dir``
- One per-model gate-summary JSON
- The headline PNG figures to ``cfg.output_dir/figures``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from beartype import beartype
from omegaconf import DictConfig, ListConfig, OmegaConf

from s1s2.metacog.core import DifficultyDetectorAnalysis, MetacogConfig, PerFeatureResults
from s1s2.metacog.viz import render_all
from s1s2.sae.loaders import MockSAE, load_sae_for_model
from s1s2.utils import io as ioh
from s1s2.utils.logging import get_logger
from s1s2.utils.seed import set_global_seed

logger = get_logger("s1s2.metacog")


# Map a project model key to the HDF5 key written by the extractor.
_HDF5_KEY = {
    "llama-3.1-8b-instruct": "meta-llama_Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google_gemma-2-9b-it",
    "r1-distill-llama-8b": "deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
    "r1-distill-qwen-7b": "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
}


# ---------------------------------------------------------------------------
# Hydra cfg → MetacogConfig
# ---------------------------------------------------------------------------


@beartype
def metacog_config_from_hydra(cfg: DictConfig) -> MetacogConfig:
    """Translate the Hydra DictConfig into a MetacogConfig dataclass."""
    layers_cfg = cfg.get("layers", None)
    if layers_cfg in (None, "all", "auto"):
        layers_t = None
    elif isinstance(layers_cfg, (list, tuple, ListConfig)):
        layers_t = tuple(int(x) for x in layers_cfg)
    else:
        layers_t = (int(layers_cfg),)

    markers = tuple(cfg.get("self_correction_markers", ()))
    if not markers:
        from s1s2.metacog.trajectory import DEFAULT_MARKERS

        markers = DEFAULT_MARKERS

    return MetacogConfig(
        activations_path=str(cfg.activations_path),
        sae_results_dir=str(cfg.get("sae_results_dir", "results/sae")),
        output_dir=str(cfg.output_dir),
        seed=int(cfg.get("seed", 0)),
        layers=layers_t,
        surprise_aggregation=str(cfg.get("surprise_aggregation", "mean_full")),
        fdr_q=float(cfg.get("fdr_q", 0.05)),
        rho_threshold=float(cfg.gates.gate_1.rho_threshold),
        specificity_auc_threshold=float(cfg.gates.gate_2.min_specificity_auc),
        matched_only=bool(cfg.gates.gate_2.difficulty_matched_only),
        min_features_with_rho_gt=int(cfg.gates.gate_1.min_features_with_rho_gt),
        min_delta_p_correct=float(cfg.gates.gate_3.min_delta_p_correct),
        confidently_wrong_threshold=float(cfg.confidently_wrong.threshold_confidence),
        self_correction_markers=tuple(markers),
        self_correction_min_post_chars=int(cfg.get("self_correction_min_post_chars", 30)),
        sae_min_explained_variance=float(cfg.get("sae_min_explained_variance", 0.5)),
    )


# ---------------------------------------------------------------------------
# Layer iteration
# ---------------------------------------------------------------------------


@beartype
def iter_layers_for_model(cfg: DictConfig, model_key: str, n_layers: int) -> list[int]:
    """Yield layer indices for a model. Default = a single mid-stream layer."""
    layers_cfg = cfg.get("layers", None)
    if layers_cfg in (None, "auto"):
        # Use a single mid-stream layer; this is the metacog default
        # because the surprise correlation is dominated by mid-to-late
        # residuals (probes peak around L20 in the project's prior runs).
        return [n_layers // 2]
    if layers_cfg == "all":
        return list(range(n_layers))
    if isinstance(layers_cfg, (list, tuple, ListConfig)):
        return [int(x) for x in layers_cfg]
    return [int(layers_cfg)]


# ---------------------------------------------------------------------------
# SAE selection
# ---------------------------------------------------------------------------


@beartype
def make_sae_for_run(
    model_key: str,
    layer: int,
    *,
    hidden_dim: int,
    use_mock: bool = False,
    seed: int = 0,
):
    """Build an SAE handle for the requested (model, layer).

    Falls back to a :class:`s1s2.sae.loaders.MockSAE` when ``use_mock``
    is True (the default for tests and synthetic data) so the metacog
    pipeline can run without internet access. The mock has random
    weights and will trip the reconstruction-fidelity check on real
    data, which is the intended behavior — better to flag a fake SAE
    than to silently produce nonsense.
    """
    if use_mock:
        return MockSAE(
            hidden_dim=int(hidden_dim),
            n_features=max(64, 4 * int(hidden_dim)),
            layer=int(layer),
            seed=int(seed),
            sparsity=0.1,
        )
    try:
        return load_sae_for_model(model_key, layer=int(layer))
    except Exception as exc:
        logger.warning("load_sae_for_model failed (%s); falling back to MockSAE", exc)
        return MockSAE(
            hidden_dim=int(hidden_dim),
            n_features=max(64, 4 * int(hidden_dim)),
            layer=int(layer),
            seed=int(seed),
            sparsity=0.1,
        )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


@beartype
def run_metacog(cfg: DictConfig) -> dict[str, Any]:
    """Run the metacog grid. Returns a dict of model_key -> paths."""
    metacog_cfg = metacog_config_from_hydra(cfg)
    set_global_seed(int(metacog_cfg.seed), deterministic_torch=False)
    analysis = DifficultyDetectorAnalysis(metacog_cfg)
    use_mock = bool(cfg.get("sae", {}).get("use_mock", False))

    out_dir = Path(metacog_cfg.output_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Any] = {}

    with ioh.open_activations(metacog_cfg.activations_path) as f:
        available = ioh.list_models(f)

        for model_key in cfg.models:
            hdf5_key = _HDF5_KEY.get(str(model_key), str(model_key))
            if hdf5_key not in available:
                logger.warning(
                    "[%s] hdf5_key=%s not in activation cache (have: %s); skipping",
                    model_key,
                    hdf5_key,
                    available,
                )
                continue

            mmeta = ioh.model_metadata(f, hdf5_key)
            n_layers = int(mmeta.get("n_layers", 32))
            hidden_dim = int(mmeta.get("hidden_dim", 4096))

            per_layer_results: list[PerFeatureResults] = []
            for layer in iter_layers_for_model(cfg, str(model_key), n_layers):
                logger.info("[%s] L%02d: starting metacog pass", model_key, layer)
                sae = make_sae_for_run(
                    str(model_key),
                    layer=layer,
                    hidden_dim=hidden_dim,
                    use_mock=use_mock,
                    seed=metacog_cfg.seed,
                )
                try:
                    res = analysis.run(
                        model_key=str(model_key),
                        hdf5_key=hdf5_key,
                        layer=int(layer),
                        sae=sae,
                    )
                except Exception as exc:
                    logger.error("[%s] L%02d: pipeline failed: %s", model_key, layer, exc)
                    continue
                per_layer_results.append(res)
                analysis.write_per_layer(res)

            if not per_layer_results:
                logger.warning("[%s] no per-layer results; skipping summary", model_key)
                continue

            snapshot, gate_results = analysis.evaluate_gates(per_layer_results)
            summary_path = analysis.write_summary(snapshot, gate_results)

            # Headline figures: use the layer with the largest n_difficulty_sensitive
            best = max(
                per_layer_results,
                key=lambda r: int(
                    r.surprise_df["is_difficulty_sensitive"].sum()
                    if "is_difficulty_sensitive" in r.surprise_df.columns
                    else 0
                ),
            )
            fig_paths = render_all(
                surprise_df=best.surprise_df,
                combined_df=best.combined_df,
                gate_results=gate_results,
                out_dir=fig_dir,
                name_prefix=f"{model_key}_L{best.layer:02d}",
                rho_threshold=metacog_cfg.rho_threshold,
                auc_threshold=metacog_cfg.specificity_auc_threshold,
            )

            written[str(model_key)] = {
                "summary_path": str(summary_path),
                "per_layer_paths": [
                    str(out_dir / f"{r.model_key}_metacog_layer_{r.layer:02d}.json")
                    for r in per_layer_results
                ],
                "figure_paths": {k: str(v) for k, v in fig_paths.items()},
                "gate_decisions": [g.decision for g in gate_results],
            }

    return written


# ---------------------------------------------------------------------------
# Hydra entry
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../../configs", config_name="metacog", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover - hydra glue
    """Entry point for ``python -m s1s2.metacog.cli``."""
    logger.info("metacog config:\n%s", OmegaConf.to_yaml(cfg))
    written = run_metacog(cfg)
    logger.info("metacog: %d models processed", len(written))


if __name__ == "__main__":  # pragma: no cover
    main()
