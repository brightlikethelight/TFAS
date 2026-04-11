"""Hydra-decorated CLI entry point for extraction.

Invoked via ``python scripts/extract_all.py <overrides>``. This module owns
the orchestration: loading benchmark + models, instantiating the per-model
extractor, writing the HDF5 file, running the schema validator.

It intentionally defers importing heavy dependencies (``transformers``,
``torch``) until inside :func:`run_extraction` so ``import s1s2.extract.cli``
stays cheap for tests and static analysis.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from s1s2.extract.core import (
    ExtractionConfig,
    GenerationConfig,
    ModelSpec,
    SingleModelExtractor,
    build_problem_metadata_from_items,
)
from s1s2.extract.writer import ActivationWriter, RunMetadata, validate_file
from s1s2.utils.logging import get_logger
from s1s2.utils.seed import set_global_seed

LOG = get_logger("s1s2.extract.cli")


# --------------------------------------------------------------------------- #
# Benchmark loading                                                           #
# --------------------------------------------------------------------------- #


def _load_benchmark(path: str | Path) -> list[Any]:
    """Load the benchmark from JSONL.

    We try the project's real loader first (if it exists -- built by another
    agent). If it doesn't, fall back to a minimal dict-shim loader so the
    extraction CLI still works end-to-end during development. The shim
    exposes the same field names as :class:`s1s2.extract.scoring.BenchmarkItemProto`.
    """
    try:
        from s1s2.benchmark.loader import load_benchmark  # type: ignore

        return list(load_benchmark(path))
    except Exception:
        LOG.warning("s1s2.benchmark.loader not available; using shim JSONL loader")

    from types import SimpleNamespace

    items: list[Any] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"benchmark not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: malformed JSONL: {e}") from e
            items.append(SimpleNamespace(**d))
    return items


# --------------------------------------------------------------------------- #
# Spec resolution                                                             #
# --------------------------------------------------------------------------- #


def _resolve_spec(cfg: DictConfig, key: str) -> ModelSpec:
    try:
        m = cfg.models[key]
    except Exception as e:
        raise KeyError(f"model key {key!r} not found in configs/models.yaml: {e}") from e
    return ModelSpec(
        key=key,
        hdf5_key=str(m.hdf5_key),
        hf_id=str(m.hf_id),
        family=str(m.family),
        n_layers=int(m.n_layers),
        n_heads=int(m.n_heads),
        n_kv_heads=int(m.n_kv_heads),
        hidden_dim=int(m.hidden_dim),
        head_dim=int(m.head_dim),
        is_reasoning=bool(m.is_reasoning),
    )


def _resolve_dtype(s: str):
    import torch

    s = s.lower()
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float16", "f16", "fp16", "half"):
        return torch.float16
    if s in ("float32", "f32", "fp32", "float"):
        return torch.float32
    raise ValueError(f"unknown torch_dtype: {s}")


def _resolve_device(requested: str) -> str:
    """Fall back to CPU if the requested device is unavailable."""
    import torch

    req = requested.lower()
    if req.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("cuda requested but not available; falling back to cpu")
        return "cpu"
    if req == "mps" and not getattr(torch.backends, "mps", None):
        LOG.warning("mps requested but not available; falling back to cpu")
        return "cpu"
    return req


# --------------------------------------------------------------------------- #
# Main entry                                                                  #
# --------------------------------------------------------------------------- #


def run_extraction(cfg: DictConfig) -> int:
    """Execute a full extraction pass. Returns 0 on success, nonzero on errors."""
    set_global_seed(int(cfg.generation.seed), deterministic_torch=False)

    # Resolve paths relative to the original CWD (Hydra otherwise chdirs).
    orig_cwd = Path(os.environ.get("ORIG_CWD") or os.getcwd())
    bench_path = _abs(cfg.benchmark_path, orig_cwd)
    output_path = _abs(cfg.output_path, orig_cwd)

    LOG.info("loading benchmark: %s", bench_path)
    items = _load_benchmark(bench_path)
    limit = cfg.get("limit_problems", None)
    if limit is not None and limit > 0:
        items = items[: int(limit)]
        LOG.info("limit_problems=%d, using %d items", int(limit), len(items))

    LOG.info("opening writer: %s", output_path)
    writer = ActivationWriter(output_path)
    writer.open()
    try:
        config_json = json.dumps(OmegaConf.to_container(cfg, resolve=True), default=str)
        writer.write_run_metadata(
            RunMetadata.build(
                benchmark_path=str(bench_path),
                seed=int(cfg.generation.seed),
                config_json=config_json,
            )
        )

        # Compute prompt token counts with ONE tokenizer pass (per the first
        # model) -- token counts are downstream-only, so exact matching to
        # each model is not critical. We write them from the first model's
        # tokenizer.
        prompt_token_counts = _prefill_prompt_token_counts(cfg, items)
        problem_metas = build_problem_metadata_from_items(items, prompt_token_counts)
        writer.write_problems(problem_metas)

        device = _resolve_device(str(cfg.device))
        torch_dtype = _resolve_dtype(str(cfg.torch_dtype))

        for model_key in list(cfg.models_to_extract):
            spec = _resolve_spec(cfg, str(model_key))
            gen_cfg = _build_generation_cfg(cfg, spec)
            extr_cfg = ExtractionConfig(
                dtype=str(cfg.extraction.dtype),
                attn_implementation=str(cfg.attn_implementation),
                log_every=int(cfg.logging.log_every),
            )
            extractor = SingleModelExtractor(
                spec=spec,
                generation_cfg=gen_cfg,
                extraction_cfg=extr_cfg,
                device=device,
                torch_dtype=torch_dtype,
            )
            extractor.load()
            try:
                model = extractor.model
                n_layers = _infer_n_layers(model)
                n_heads = _peek_n_q_heads(model, spec)
                hidden_dim = getattr(model.config, "hidden_size", spec.hidden_dim)
                writer.create_model_group(
                    model_key=spec.hdf5_key,
                    hf_model_id=spec.hf_id,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    n_kv_heads=spec.n_kv_heads,
                    hidden_dim=hidden_dim,
                    head_dim=spec.head_dim,
                    dtype=_disk_dtype(str(cfg.extraction.dtype)),
                    is_reasoning_model=spec.is_reasoning,
                )
                extractor.run(
                    items=items,
                    writer=writer,
                    effective_n_layers=n_layers,
                    effective_n_heads=n_heads,
                    effective_hidden_dim=hidden_dim,
                )
            finally:
                extractor.unload()
    finally:
        writer.close()

    # Post-hoc validation
    errors = validate_file(output_path)
    if errors:
        LOG.error("schema validation failed:\n  - %s", "\n  - ".join(errors))
        return 2
    LOG.info("extraction complete: %s (schema OK)", output_path)
    return 0


def _abs(pth: str | Path, base: Path) -> Path:
    p = Path(pth)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _disk_dtype(requested: str) -> str:
    """Return the dtype *actually* used on disk.

    h5py has no native bfloat16 -- we coerce to float16. The metadata must
    reflect the actual storage so downstream readers don't reinterpret bytes.
    """
    return "float16" if requested.lower() in ("bfloat16", "bf16") else requested.lower()


def _build_generation_cfg(cfg: DictConfig, spec: ModelSpec) -> GenerationConfig:
    gen = cfg.generation
    if spec.is_reasoning:
        return GenerationConfig(
            max_new_tokens=int(gen.max_new_tokens_reasoning),
            temperature=float(gen.temperature_reasoning),
            top_p=float(gen.top_p_reasoning),
            do_sample=bool(gen.do_sample_reasoning),
            seed=int(gen.seed),
        )
    return GenerationConfig(
        max_new_tokens=int(gen.max_new_tokens_standard),
        temperature=float(gen.temperature_standard),
        top_p=1.0,
        do_sample=False,
        seed=int(gen.seed),
    )


def _prefill_prompt_token_counts(cfg: DictConfig, items: list[Any]) -> list[int]:
    """Best-effort prompt token counting using the first listed model's tokenizer.

    This is metadata only -- written to /problems/prompt_token_count -- so
    it need not match each model exactly. Returns zeros if tokenizer loading
    fails.
    """
    if not items:
        return []
    try:
        from transformers import AutoTokenizer

        first_key = str(cfg.models_to_extract[0])
        hf_id = str(cfg.models[first_key].hf_id)
        tok = AutoTokenizer.from_pretrained(hf_id)
    except Exception as e:
        LOG.warning("prompt token counting disabled: %s", e)
        return [0] * len(items)
    counts: list[int] = []
    for item in items:
        text = getattr(item, "prompt", None) or getattr(item, "prompt_text", "") or ""
        sys_prompt = getattr(item, "system_prompt", None)
        try:
            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": text})
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            counts.append(int(ids.shape[1]))
        except Exception:
            counts.append(int(tok(text, return_tensors="pt").input_ids.shape[1]))
    return counts


def _infer_n_layers(model) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return 0
    for attr in ("num_hidden_layers", "n_layer", "n_layers"):
        val = getattr(cfg, attr, None)
        if val is not None:
            return int(val)
    return 0


def _peek_n_q_heads(model, spec: ModelSpec) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return spec.n_heads
    for attr in ("num_attention_heads", "n_head", "num_heads"):
        val = getattr(cfg, attr, None)
        if val is not None:
            return int(val)
    return spec.n_heads


# --------------------------------------------------------------------------- #
# Hydra wrapper                                                               #
# --------------------------------------------------------------------------- #


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="extract")
def main(cfg: DictConfig) -> None:
    # Preserve the original working directory for resolving relative paths.
    os.environ.setdefault("ORIG_CWD", hydra.utils.get_original_cwd())
    code = run_extraction(cfg)
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    # Direct invocation for debugging without Hydra sugar.
    main()  # pragma: no cover
