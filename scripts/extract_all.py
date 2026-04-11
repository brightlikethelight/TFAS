#!/usr/bin/env python3
"""Driver script for the extraction workstream.

Thin wrapper around :mod:`s1s2.extract.cli` so we have a single canonical
invocation:

    python scripts/extract_all.py run_name=smoke limit_problems=5 \
        models_to_extract=[llama-3.1-8b-instruct]

All flags are Hydra overrides against ``configs/extract.yaml``.

A smoke run with a tiny CPU model (for debugging the pipeline wiring):

    python scripts/extract_all.py \
        run_name=tinysmoke \
        benchmark_path=data/benchmark/benchmark.jsonl \
        models_to_extract=[tiny] \
        device=cpu torch_dtype=float32 \
        generation.max_new_tokens_standard=8 \
        limit_problems=2 \
        'models.tiny={hf_id:sshleifer/tiny-gpt2, hdf5_key:tiny-gpt2, family:gpt2, n_layers:2, n_heads:2, n_kv_heads:2, hidden_dim:2, head_dim:1, is_reasoning:false}'
"""
from __future__ import annotations

import os
import sys

# Ensure the src layout works even when running without ``pip install -e .``.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from s1s2.extract.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
