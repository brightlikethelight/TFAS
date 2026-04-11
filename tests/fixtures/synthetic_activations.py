"""Standalone synthetic-HDF5 builder.

This module provides :func:`build_synthetic_hdf5`, a self-contained helper
that materialises a tiny but schema-valid activation cache for tests,
smoke runs, and downstream notebooks. It mirrors the fixture builder
in :mod:`tests.conftest` but does NOT import from conftest, so it can be
used outside the pytest collection context (smoke scripts, CLI tools).

The returned HDF5 file conforms to ``docs/data_contract.md`` and is
constructed entirely via the ``s1s2.utils.io.write_*`` helpers, so any
change to the writer API will surface here first. The cache contains:

* 20 problems (10 conflict + 10 control), 5 task categories
* 1 model subgroup (``synthetic_test-model``), 4 layers, 4 positions
* 4 query heads / 2 KV heads (group size 2)
* a planted +0.8 mean shift on residual dim 0 of layer 2 for conflict items
* a planted +0.6 entropy bump on (layer 1, head 0) for conflict items
* behaviorally realistic correctness rates (~40% conflict, ~70% control)
* lure responses on incorrect conflict items
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Make the s1s2 package importable without `pip install -e .` so smoke
# scripts work in clean dev environments.
_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# OpenMP guard for macOS — torch + numpy on the same runtime can crash
# without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from s1s2.utils import io as ioh  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants — kept in sync with tests/conftest.py                              #
# --------------------------------------------------------------------------- #

SYNTH_MODEL_KEY = "synthetic_test-model"
SYNTH_HF_ID = "synthetic/test-model"
SYNTH_N_PROBLEMS = 20
SYNTH_N_LAYERS = 4
SYNTH_HIDDEN = 32
SYNTH_N_HEADS = 4
SYNTH_N_KV_HEADS = 2
SYNTH_HEAD_DIM = 8
SYNTH_POSITIONS: tuple[str, ...] = ("P0", "P2", "T50", "Tend")
SYNTH_CATEGORIES: tuple[str, ...] = (
    "crt",
    "base_rate",
    "syllogism",
    "anchoring",
    "framing",
)


# --------------------------------------------------------------------------- #
# Builder                                                                      #
# --------------------------------------------------------------------------- #


def build_synthetic_hdf5(
    path: str | Path,
    *,
    n_problems: int = SYNTH_N_PROBLEMS,
    n_layers: int = SYNTH_N_LAYERS,
    hidden: int = SYNTH_HIDDEN,
    n_heads: int = SYNTH_N_HEADS,
    n_kv_heads: int = SYNTH_N_KV_HEADS,
    head_dim: int = SYNTH_HEAD_DIM,
    positions: tuple[str, ...] = SYNTH_POSITIONS,
    seed: int = 0,
) -> Path:
    """Materialize a fully-populated, schema-valid HDF5 cache at ``path``.

    Idempotent — overwrites any pre-existing file at ``path``. Returns
    the resolved Path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    n_positions = len(positions)
    rng = np.random.default_rng(int(seed))

    # ---- per-problem metadata --------------------------------------------
    ids: list[str] = []
    categories: list[str] = []
    conflict = np.empty(n_problems, dtype=bool)
    difficulty = np.full(n_problems, 2, dtype=np.int8)
    prompt_text: list[str] = []
    correct_answer: list[str] = []
    lure_answer: list[str] = []
    matched_pair_id: list[str] = []
    prompt_token_count = np.full(n_problems, 16, dtype=np.int32)

    for i in range(n_problems):
        is_conflict = (i % 2) == 0
        pair_idx = i // 2
        cat = SYNTH_CATEGORIES[pair_idx % len(SYNTH_CATEGORIES)]
        ids.append(f"p{i:04d}")
        categories.append(cat)
        conflict[i] = is_conflict
        prompt_text.append(
            f"[{cat}] {'lure' if is_conflict else 'neutral'} prompt #{pair_idx}"
        )
        correct_answer.append("A")
        lure_answer.append("B" if is_conflict else "")
        matched_pair_id.append(f"pair_{pair_idx:04d}")

    # ---- residuals (with planted signal at layer 2 dim 0) ----------------
    residuals: list[np.ndarray] = []
    for layer in range(n_layers):
        base = rng.normal(size=(n_problems, n_positions, hidden)).astype(np.float32)
        if layer == 2:
            shift = 0.8 * conflict.astype(np.float32)[:, None, None]
            signal = np.zeros_like(base)
            signal[:, :, 0] = shift[:, :, 0]
            base = base + signal
        residuals.append(base)

    # ---- attention metrics (planted differential on layer 1 head 0) ------
    shape = (n_problems, n_layers, n_heads, n_positions)
    base_entropy = rng.uniform(0.5, 2.5, size=shape).astype(np.float32)
    bump = 0.6 * conflict.astype(np.float32)
    base_entropy[:, 1, 0, :] += bump[:, None]
    entropy_normalized = np.clip(base_entropy / np.log2(32.0), 0.0, 1.0).astype(
        np.float32
    )
    gini = np.clip(1.0 - entropy_normalized, 0.0, 1.0).astype(np.float32)
    max_attn = np.clip(0.9 * gini + 0.05, 0.0, 1.0).astype(np.float32)
    focus_5 = np.clip(0.95 * gini + 0.05, 0.0, 1.0).astype(np.float32)
    effective_rank = np.exp2(base_entropy).astype(np.float32)
    attention_metrics = {
        "entropy": base_entropy.astype(np.float32),
        "entropy_normalized": entropy_normalized,
        "gini": gini,
        "max_attn": max_attn,
        "focus_5": focus_5,
        "effective_rank": effective_rank,
    }

    # ---- positions / token indices ---------------------------------------
    valid = np.zeros((n_problems, n_positions), dtype=bool)
    for i, label in enumerate(positions):
        if label in ("P0", "P2"):
            valid[:, i] = True
        # T-positions remain invalid; synthetic model is non-reasoning.
    token_indices = np.arange(n_positions, dtype=np.int32)
    token_indices = np.broadcast_to(token_indices, (n_problems, n_positions)).copy()

    # ---- behavioral outcomes ---------------------------------------------
    correct = np.where(
        conflict,
        rng.uniform(size=n_problems) < 0.4,
        rng.uniform(size=n_problems) < 0.7,
    )
    matches_lure = conflict & ~correct
    predicted_answer = [
        "A" if c else ("B" if l else "C") for c, l in zip(correct, matches_lure, strict=False)
    ]
    response_category = [
        "correct" if c else ("lure" if l else "other_wrong")
        for c, l in zip(correct, matches_lure, strict=False)
    ]

    # ---- token surprises (cheap synthetic) -------------------------------
    by_position_surprises = rng.uniform(
        0.5, 5.0, size=(n_problems, n_positions)
    ).astype(np.float32)

    # ---- generations (minimal valid versions) ----------------------------
    full_text = [f"[gen] {p}" for p in prompt_text]
    thinking_text = ["" for _ in range(n_problems)]
    answer_text = list(predicted_answer)
    thinking_token_count = np.zeros(n_problems, dtype=np.int32)
    answer_token_count = np.full(n_problems, 1, dtype=np.int32)

    cfg_payload = json.dumps(
        {
            "kind": "synthetic",
            "n_problems": int(n_problems),
            "n_layers": int(n_layers),
            "hidden": int(hidden),
        }
    )

    with h5py.File(path, "w") as f:
        ioh.write_run_metadata(
            f,
            benchmark_path="synthetic",
            benchmark_sha256="0" * 64,
            created_at=datetime.now(UTC).isoformat(),
            git_sha="synthetic",
            seed=int(seed),
            config_json=cfg_payload,
        )
        ioh.write_problem_metadata(
            f,
            ids=ids,
            categories=categories,
            conflict=conflict,
            difficulty=difficulty,
            prompt_text=prompt_text,
            correct_answer=correct_answer,
            lure_answer=lure_answer,
            matched_pair_id=matched_pair_id,
            prompt_token_count=prompt_token_count,
        )
        ioh.write_model_metadata(
            f,
            SYNTH_MODEL_KEY,
            hf_model_id=SYNTH_HF_ID,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=hidden,
            head_dim=head_dim,
            dtype="float32",
            extracted_at=datetime.now(UTC).isoformat(),
            is_reasoning_model=False,
        )
        for layer in range(n_layers):
            ioh.write_residual_layer(
                f, SYNTH_MODEL_KEY, layer=layer, data=residuals[layer]
            )
        ioh.write_position_index(
            f,
            SYNTH_MODEL_KEY,
            labels=list(positions),
            token_indices=token_indices,
            valid=valid,
        )
        ioh.write_attention_metrics(f, SYNTH_MODEL_KEY, attention_metrics)
        ioh.write_token_surprises(
            f,
            SYNTH_MODEL_KEY,
            by_position=by_position_surprises,
        )
        ioh.write_generations(
            f,
            SYNTH_MODEL_KEY,
            full_text=full_text,
            thinking_text=thinking_text,
            answer_text=answer_text,
            thinking_token_count=thinking_token_count,
            answer_token_count=answer_token_count,
        )
        ioh.write_behavior(
            f,
            SYNTH_MODEL_KEY,
            predicted_answer=predicted_answer,
            correct=correct,
            matches_lure=matches_lure,
            response_category=response_category,
        )

    return path


__all__ = [
    "SYNTH_CATEGORIES",
    "SYNTH_HEAD_DIM",
    "SYNTH_HF_ID",
    "SYNTH_HIDDEN",
    "SYNTH_MODEL_KEY",
    "SYNTH_N_HEADS",
    "SYNTH_N_KV_HEADS",
    "SYNTH_N_LAYERS",
    "SYNTH_N_PROBLEMS",
    "SYNTH_POSITIONS",
    "build_synthetic_hdf5",
]
