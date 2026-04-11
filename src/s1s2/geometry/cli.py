"""Hydra entry point for the representational-geometry workstream.

Loads cached residual streams and runs cosine silhouette + permutation
tests, layer-wise CKA between matched-architecture model pairs, Two-NN
intrinsic dimensionality, and the d>>N-corrected linear separability
analysis. Writes JSON results per (model, layer).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from s1s2.geometry.cka import layer_matched_cka, within_model_cka
from s1s2.geometry.clusters import compute_silhouette_result
from s1s2.geometry.intrinsic_dim import participation_ratio, two_nn_intrinsic_dim
from s1s2.geometry.separability import linear_separability_with_d_gg_n_fix
from s1s2.utils import io as ioh
from s1s2.utils.logging import get_logger
from s1s2.utils.seed import set_global_seed

logger = get_logger(__name__)


def _layers_to_run(f, model_key: str, requested: list[int] | None) -> list[int]:
    meta = ioh.model_metadata(f, model_key)
    n_layers = int(meta.get("n_layers", 0))
    if requested:
        return [int(x) for x in requested if 0 <= int(x) < n_layers]
    return list(range(n_layers))


def _per_layer_geometry(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    n_bootstrap: int,
    n_permutations: int,
    pca_dim: int,
    n_separability_shuffles: int,
    seed: int,
) -> dict[str, Any]:
    """Run silhouette + separability + intrinsic dim on a single layer."""
    sil = compute_silhouette_result(
        X,
        labels,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        seed=seed,
    )
    sep = linear_separability_with_d_gg_n_fix(
        X,
        labels,
        pca_dim=pca_dim,
        n_shuffles=n_separability_shuffles,
        seed=seed,
    )
    return {
        "silhouette": sil.to_dict(),
        "separability": sep.to_dict(),
        "intrinsic_dim_two_nn": float(two_nn_intrinsic_dim(X.astype(np.float64))),
        "participation_ratio": float(participation_ratio(X.astype(np.float64))),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }


def _save(out_dir: Path, name: str, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(payload, indent=2, default=str))


def run_geometry(cfg: DictConfig) -> dict[str, Any]:
    """Top-level orchestrator. Returns a summary dict for the smoke test."""
    set_global_seed(int(cfg.get("seed", 0)), deterministic_torch=False)
    activations_path = str(cfg.get("activations_path", "data/activations/main.h5"))
    output_root = Path(str(cfg.get("output_dir", "results/geometry")))
    output_root.mkdir(parents=True, exist_ok=True)
    position = str(cfg.get("position", "P0"))
    n_bootstrap = int(cfg.get("n_bootstrap", 1000))
    n_permutations = int(cfg.get("n_permutations", 10_000))
    pca_dim = int(cfg.get("pca_dim", 50))
    n_separability_shuffles = int(cfg.get("n_separability_shuffles", 100))
    seed = int(cfg.get("seed", 0))

    summary: dict[str, Any] = {"models": {}, "cka_pairs": []}

    with ioh.open_activations(activations_path) as f:
        problems = ioh.load_problem_metadata(f)
        labels = problems["conflict"].astype(np.int64)
        requested_layers = list(cfg.get("layers", [])) or None
        models_to_run = list(cfg.get("models", [])) or ioh.list_models(f)

        # Per-model per-layer geometry
        per_model_per_layer: dict[str, dict[int, np.ndarray]] = {}
        for model_key in models_to_run:
            model_dir = output_root / model_key
            layers = _layers_to_run(f, model_key, requested_layers)
            per_layer_acts: dict[int, np.ndarray] = {}
            per_layer_results: dict[int, dict[str, Any]] = {}
            for layer in layers:
                X = ioh.get_residual(f, model_key, layer=layer, position=position).astype(
                    np.float32
                )
                per_layer_acts[layer] = X
                result = _per_layer_geometry(
                    X,
                    labels,
                    n_bootstrap=n_bootstrap,
                    n_permutations=n_permutations,
                    pca_dim=pca_dim,
                    n_separability_shuffles=n_separability_shuffles,
                    seed=seed,
                )
                _save(model_dir / f"layer_{layer:02d}", "geometry.json", result)
                per_layer_results[layer] = result
            per_model_per_layer[model_key] = per_layer_acts
            summary["models"][model_key] = {
                "n_layers": len(per_layer_results),
                "layers": list(per_layer_results.keys()),
            }
            logger.info("geometry: %s -> %d layers", model_key, len(per_layer_results))

        # CKA across matched-architecture pairs
        cka_pairs = list(cfg.get("cka_pairs", []))
        for pair in cka_pairs:
            pair_a, pair_b = str(pair[0]), str(pair[1])
            if pair_a not in per_model_per_layer or pair_b not in per_model_per_layer:
                logger.warning(
                    "skipping CKA pair (%s, %s): not in cache", pair_a, pair_b
                )
                continue
            shared = sorted(
                set(per_model_per_layer[pair_a].keys())
                & set(per_model_per_layer[pair_b].keys())
            )
            acts_a = [per_model_per_layer[pair_a][i] for i in shared]
            acts_b = [per_model_per_layer[pair_b][i] for i in shared]
            cka_full = layer_matched_cka(acts_a, acts_b).tolist()
            cka_s1 = layer_matched_cka(
                acts_a, acts_b, mask=labels.astype(bool)
            ).tolist()
            cka_s2 = layer_matched_cka(
                acts_a, acts_b, mask=(~labels.astype(bool))
            ).tolist()
            within_a = within_model_cka(
                acts_a, mask_a=labels.astype(bool), mask_b=(~labels.astype(bool))
            ).tolist()
            payload = {
                "model_a": pair_a,
                "model_b": pair_b,
                "layers": shared,
                "cka_full": cka_full,
                "cka_s1_only": cka_s1,
                "cka_s2_only": cka_s2,
                "within_a_s1_vs_s2": within_a,
            }
            _save(
                output_root / "cka_comparisons",
                f"{pair_a}__vs__{pair_b}.json",
                payload,
            )
            summary["cka_pairs"].append({"a": pair_a, "b": pair_b, "n_layers": len(shared)})
            logger.info("CKA %s vs %s: %d layers", pair_a, pair_b, len(shared))

    (output_root / "_config.json").write_text(
        json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2, default=str)
    )
    return summary


@hydra.main(config_path="../../../configs", config_name="geometry", version_base=None)
def main(cfg: DictConfig) -> None:
    run_geometry(cfg)


if __name__ == "__main__":
    main()
