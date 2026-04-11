#!/usr/bin/env python3
"""Pre-download all HuggingFace models and SAEs to a local cache directory.

Separating download from extraction avoids wasted GPU-hours if a download
fails mid-run. Also useful for air-gapped clusters where you SCP a
pre-populated cache.

Usage:
    python deploy/download_models.py --cache-dir /workspace/hf_cache
    python deploy/download_models.py --cache-dir /n/holyscratch01/$USER/hf_cache
    python deploy/download_models.py --models llama-3.1-8b-instruct r1-distill-llama-8b
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
from beartype import beartype

# Ensure src/ is importable without pip install -e .
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---- Model registry --------------------------------------------------------

def _load_model_registry() -> dict[str, dict]:
    """Load model configs from configs/models.yaml."""
    cfg_path = _REPO / "configs" / "models.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["models"]


# All HF model IDs from the registry, plus SAE repos that need downloading.
SAE_REPOS: dict[str, str] = {
    "llama-scope": "fnlp/Llama-3_1-8B-Base-LXR-32x",
    "gemma-scope": "google/gemma-scope-9b-it-res",
}


# ---- Download logic ---------------------------------------------------------

@beartype
def download_model(hf_id: str, cache_dir: str) -> float:
    """Download a single HF model + tokenizer. Returns elapsed seconds."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n--- Downloading model: {hf_id} ---")
    t0 = time.time()

    # Tokenizer first (small, fast)
    AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)
    print(f"  tokenizer cached.")

    # Model weights — download only, don't load into RAM
    AutoModelForCausalLM.from_pretrained(
        hf_id,
        cache_dir=cache_dir,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        # device_map="meta" avoids actually allocating weight tensors
        device_map="meta",
    )
    elapsed = time.time() - t0
    print(f"  model cached. ({elapsed:.0f}s)")
    return elapsed


@beartype
def download_sae_repo(name: str, repo_id: str, cache_dir: str) -> float:
    """Download an SAE repo via huggingface_hub."""
    from huggingface_hub import snapshot_download

    print(f"\n--- Downloading SAE: {name} ({repo_id}) ---")
    t0 = time.time()
    try:
        snapshot_download(
            repo_id,
            cache_dir=cache_dir,
            # SAE repos can be large; only grab safetensors + config
            allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.txt", "*.md"],
        )
        elapsed = time.time() - t0
        print(f"  SAE cached. ({elapsed:.0f}s)")
        return elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  SAE download failed ({elapsed:.0f}s): {e}")
        print(f"  (Non-fatal — SAEs are optional for initial extraction.)")
        return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        help="HuggingFace cache directory (default: $HF_HOME or ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model keys to download (default: all models in configs/models.yaml)",
    )
    parser.add_argument(
        "--skip-saes",
        action="store_true",
        help="Skip SAE downloads (useful if you only need extraction models)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    print(f"Cache directory: {cache_dir}")

    registry = _load_model_registry()

    # Resolve which models to download
    if args.models:
        unknown = [m for m in args.models if m not in registry]
        if unknown:
            print(f"ERROR: unknown model keys: {unknown}")
            print(f"Valid keys: {list(registry.keys())}")
            return 1
        selected = {k: registry[k] for k in args.models}
    else:
        selected = registry

    # Download models
    total_time = 0.0
    failed: list[str] = []
    for key, cfg in selected.items():
        hf_id = cfg["hf_id"]
        try:
            elapsed = download_model(hf_id, cache_dir)
            total_time += elapsed
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(key)

    # Download SAEs
    if not args.skip_saes:
        for name, repo_id in SAE_REPOS.items():
            elapsed = download_sae_repo(name, repo_id, cache_dir)
            total_time += elapsed

    # Summary
    print("\n" + "=" * 60)
    print(f"Download complete. Total time: {total_time:.0f}s")
    print(f"Cache directory: {cache_dir}")
    if failed:
        print(f"FAILED models: {failed}")
        return 1
    print("All models downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
