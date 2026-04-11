"""Hydra CLI entry point for the causal-interventions workstream.

Usage::

    python scripts/run_causal.py
    python scripts/run_causal.py models=[llama-3.1-8b-instruct]
    python scripts/run_causal.py alphas=[-1,0,1]

Reads ``configs/causal.yaml`` (see that file for the full knob list),
iterates over the configured ``(model, layer, feature)`` cells, and
writes one JSON per cell plus two PDF figures (dose-response line plot,
projection-ablation bar chart).

Model loading is deliberately isolated in ``_load_model_and_tokenizer``
so tests can monkey-patch it out. The default implementation uses
``transformers.AutoModelForCausalLM`` / ``AutoTokenizer``; callers who
want custom loading (quantised, vLLM-served, etc.) can override via
``cfg.model_loader`` (not implemented here — swap in as needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
from beartype import beartype
from omegaconf import DictConfig, OmegaConf

from s1s2.benchmark.loader import BenchmarkItem, load_benchmark
from s1s2.causal.core import (
    CapabilityEvalConfig,
    CausalCellResult,
    CausalExperimentRunner,
    CausalRunnerConfig,
    RandomControlConfig,
    ScoreFn,
    load_feature_specs,
    save_cell_result,
)
from s1s2.causal.viz import (
    plot_ablation_bars,
    plot_dose_response,
    plot_feature_summary_bars,
)
from s1s2.utils.logging import get_logger
from s1s2.utils.seed import set_global_seed

logger = get_logger("s1s2.causal")


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def runner_config_from_hydra(cfg: DictConfig) -> CausalRunnerConfig:
    """Translate a Hydra cfg into a :class:`CausalRunnerConfig`.

    Note: ``@beartype`` is intentionally NOT applied to this function.
    beartype cannot resolve ``DictConfig``'s MRO without emitting a PEP
    585 deprecation warning, and there is no type-soundness gain from
    the decorator on a function whose input is an OmegaConf container.
    """
    rc_cfg = cfg.get("random_control", {})
    random_control = RandomControlConfig(
        n_directions=int(rc_cfg.get("n_directions", 5)),
        seed=int(rc_cfg.get("seed", 1)),
    )
    cap_cfg = cfg.get("capability_eval")
    capability_eval: CapabilityEvalConfig | None
    if cap_cfg is not None:
        capability_eval = CapabilityEvalConfig(
            mmlu_subset_path=(
                str(cap_cfg.get("mmlu_subset_path")) if cap_cfg.get("mmlu_subset_path") else None
            ),
            hellaswag_subset_path=(
                str(cap_cfg.get("hellaswag_subset_path"))
                if cap_cfg.get("hellaswag_subset_path")
                else None
            ),
            n_examples_per_eval=int(cap_cfg.get("n_examples_per_eval", 100)),
            max_acceptable_drop_pp=float(cap_cfg.get("max_acceptable_drop_pp", 2.0)),
        )
    else:
        capability_eval = None
    alphas = tuple(float(a) for a in cfg.get("alphas", []))
    return CausalRunnerConfig(
        alphas=alphas,
        top_features_per_layer=int(cfg.get("top_features_per_layer", 3)),
        random_control=random_control,
        capability_eval=capability_eval,
        seed=int(cfg.get("seed", 0)),
        max_new_tokens=int(cfg.get("max_new_tokens", 128)),
        n_bootstrap=int(cfg.get("n_bootstrap", 1000)),
    )


# ---------------------------------------------------------------------------
# Model loading (side-effectful, isolated so tests can bypass it)
# ---------------------------------------------------------------------------


@beartype
def _load_model_and_tokenizer(hf_model_id: str, device: str = "cpu") -> tuple[Any, Any]:
    """Load the HF causal-LM and matching tokenizer.

    Kept intentionally small so it can be monkey-patched in tests. Real
    runs should almost always pass ``device="cuda:0"`` and preload on an
    H100 before calling the runner.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading %s onto %s", hf_model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Default scoring function
# ---------------------------------------------------------------------------


@beartype
def default_score_fn(model: Any, tokenizer: Any, item: BenchmarkItem) -> bool:
    """Default zero-shot scoring via greedy generation + regex match.

    This is the "do something reasonable" fallback that works for the
    tiny test models. Real runs should override this with the
    model-family-specific chat-template prompt + thinking-trace parser.
    """
    import re

    prompt = item.prompt
    if item.system_prompt:
        prompt = f"{item.system_prompt}\n\n{prompt}"
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    # Answer-pattern match.
    try:
        return bool(re.search(item.answer_pattern, generated, flags=re.IGNORECASE))
    except re.error:
        return item.correct_answer.strip().lower() in generated.strip().lower()


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_causal(
    cfg: DictConfig,
    *,
    model_loader: Any = None,
    score_fn: ScoreFn | None = None,
) -> list[Path]:
    """Run the full causal grid defined by ``cfg``. Returns written result paths.

    ``model_loader`` is a callable ``(hf_model_id, device) -> (model,
    tokenizer)``; defaults to :func:`_load_model_and_tokenizer`. Tests
    can pass a fake to avoid touching the HuggingFace Hub.
    ``score_fn`` is optional likewise.
    """
    set_global_seed(int(cfg.get("seed", 0)), deterministic_torch=False)
    runner_cfg = runner_config_from_hydra(cfg)
    runner = CausalExperimentRunner(runner_cfg)

    benchmark_path = Path(cfg.benchmark_path)
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"benchmark file not found at {benchmark_path} — " f"aborting causal run"
        )
    benchmark: list[BenchmarkItem] = load_benchmark(benchmark_path)

    sae_results_dir = Path(cfg.sae_results_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if model_loader is None:
        model_loader = _load_model_and_tokenizer
    if score_fn is None:
        score_fn = default_score_fn

    models = list(cfg.models)
    device = str(cfg.get("device", "cpu"))

    written: list[Path] = []
    all_results: list[CausalCellResult] = []

    for model_key in models:
        hf_id = cfg.get("models_hf", {}).get(model_key, model_key)
        specs = load_feature_specs(
            sae_results_dir=sae_results_dir,
            model_key=model_key,
            top_per_layer=runner_cfg.top_features_per_layer,
        )
        if not specs:
            logger.warning(
                "no SAE features loaded for %s — skipping. Run the SAE "
                "workstream first or drop a features JSON at %s",
                model_key,
                sae_results_dir,
            )
            continue

        # Load the model once per model_key — shared across all (layer, feature) cells.
        try:
            model, tokenizer = model_loader(hf_id, device)
        except Exception as exc:
            logger.error("could not load %s (%s): %s — skipping", model_key, hf_id, exc)
            continue

        for spec in specs:
            try:
                cell = runner.run_one(
                    model=model,
                    tokenizer=tokenizer,
                    feature=spec,
                    benchmark=benchmark,
                    score_fn=score_fn,
                )
            except Exception as exc:
                logger.error(
                    "cell %s L%d F%d failed: %s",
                    spec.model_key,
                    spec.layer,
                    spec.feature_id,
                    exc,
                )
                continue
            all_results.append(cell)
            result_path = save_cell_result(cell, output_dir)
            written.append(result_path)
            # Figures.
            try:
                dose_path = plot_dose_response(
                    cell.curve,
                    output_path=figures_dir
                    / f"{cell.model}_layer{cell.layer:02d}_feature{cell.feature_id:06d}_dose.pdf",
                )
                written.append(dose_path)
            except Exception as exc:
                logger.warning("dose-response plot failed: %s", exc)
            if cell.ablation is not None:
                try:
                    abl_path = plot_ablation_bars(
                        cell,
                        output_path=figures_dir
                        / f"{cell.model}_layer{cell.layer:02d}_feature{cell.feature_id:06d}_ablation.pdf",
                    )
                    written.append(abl_path)
                except Exception as exc:
                    logger.warning("ablation bar plot failed: %s", exc)

    # Optional summary across features.
    if all_results:
        try:
            summary_path = plot_feature_summary_bars(
                all_results,
                output_path=figures_dir / "feature_summary.pdf",
            )
            written.append(summary_path)
        except Exception as exc:
            logger.warning("summary bar plot failed: %s", exc)

    return written


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../../configs", config_name="causal", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for ``python -m s1s2.causal.cli``."""
    logger.info("causal config:\n" + OmegaConf.to_yaml(cfg))
    paths = run_causal(cfg)
    logger.info("wrote %d output files", len(paths))


if __name__ == "__main__":  # pragma: no cover
    main()
