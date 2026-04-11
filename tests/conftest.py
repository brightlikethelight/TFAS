"""Shared pytest fixtures for the s1s2 test suite.

These fixtures back the unit tests for every workstream and the end-to-end
smoke test. The most important one is :func:`synthetic_hdf5_path`, which
builds a tiny but schema-valid HDF5 activation cache from
:mod:`s1s2.utils.io` writer helpers — every workstream's reader code can
load it.

Design notes
------------
* The synthetic HDF5 has a *planted signal* on layer 2: conflict items get a
  +0.8 shift on residual dim 0. Probes / geometry should pick this up;
  layer 0 is pure noise so they should fail.
* Behavior is wired so that conflict items are correct ~40% of the time and
  control items are correct ~70%, matching what we see on real cognitive
  bias benchmarks. Lure responses are emitted on incorrect conflict items.
* All position labels are valid (``T*`` positions included). The model is
  flagged as non-reasoning and we pad ``valid=False`` for ``T*`` positions
  to mimic the real extraction pipeline.
* All five attention metrics + ``effective_rank`` are written so the
  attention workstream's :func:`load_model_attention_data` reader is happy.

The fixtures are kept dependency-free (no transformers / sae-lens / hydra
imports) so the suite runs on a clean install with only the project's
``[dev]`` extras.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest

# Make sure OpenMP doesn't crash on macOS when torch + numpy share the runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Allow tests to import the package without requiring `pip install -e .`.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from s1s2.benchmark.loader import BenchmarkItem  # noqa: E402
from s1s2.utils import io as ioh  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants for the synthetic fixture                                          #
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
# RNG fixture                                                                  #
# --------------------------------------------------------------------------- #


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded numpy generator. Same seed across the suite for reproducibility."""
    return np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Synthetic benchmark items                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_benchmark_items() -> list[BenchmarkItem]:
    """Return 10 typed benchmark items: 5 conflict + 5 matched controls.

    Items 0/1, 2/3, ... are pairs sharing a ``matched_pair_id``. Categories
    cycle through the standard set so per-category masking has something to
    bite into.
    """
    items: list[BenchmarkItem] = []
    cats = ("crt", "base_rate", "syllogism", "anchoring", "framing")
    for pair_idx in range(5):
        cat = cats[pair_idx % len(cats)]
        pair_id = f"pair_{pair_idx:02d}"
        items.append(
            BenchmarkItem(
                id=f"{pair_id}__conflict",
                category=cat,  # type: ignore[arg-type]
                subcategory="synthetic",
                conflict=True,
                difficulty=2,
                prompt=f"[{cat}] Lure-eliciting prompt for pair {pair_idx}.",
                system_prompt=None,
                correct_answer="A",
                lure_answer="B",
                answer_pattern="A",
                lure_pattern="B",
                matched_pair_id=pair_id,
                source="template",
                provenance_note="synthetic test fixture",
                paraphrases=(),
            )
        )
        items.append(
            BenchmarkItem(
                id=f"{pair_id}__control",
                category=cat,  # type: ignore[arg-type]
                subcategory="synthetic",
                conflict=False,
                difficulty=2,
                prompt=f"[{cat}] Neutral prompt for pair {pair_idx}.",
                system_prompt=None,
                correct_answer="A",
                lure_answer="",
                answer_pattern="A",
                lure_pattern="",
                matched_pair_id=pair_id,
                source="template",
                provenance_note="synthetic test fixture",
                paraphrases=(),
            )
        )
    return items


# --------------------------------------------------------------------------- #
# Synthetic HDF5 fixture                                                       #
# --------------------------------------------------------------------------- #


def _build_synthetic_metadata(
    n_problems: int,
) -> tuple[
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    list[str],
    list[str],
    np.ndarray,
]:
    """Build per-problem metadata arrays for ``write_problem_metadata``.

    Half the problems are conflict, half are control. Each conflict item is
    paired with the next control item via ``matched_pair_id``. Categories
    cycle through ``SYNTH_CATEGORIES``.
    """
    ids: list[str] = []
    categories: list[str] = []
    conflict = np.empty(n_problems, dtype=bool)
    difficulty = np.empty(n_problems, dtype=np.int8)
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
        difficulty[i] = 2
        prompt_text.append(
            f"[{cat}] {'lure' if is_conflict else 'neutral'} prompt #{pair_idx}"
        )
        correct_answer.append("A")
        lure_answer.append("B" if is_conflict else "")
        matched_pair_id.append(f"pair_{pair_idx:04d}")

    return (
        ids,
        categories,
        conflict,
        difficulty,
        prompt_text,
        correct_answer,
        lure_answer,
        matched_pair_id,
        prompt_token_count,
    )


def _build_synthetic_residuals(
    n_problems: int,
    n_layers: int,
    n_positions: int,
    hidden: int,
    conflict: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Return one (n_problems, n_positions, hidden) array per layer.

    Plants a +0.8 shift on residual dim 0 of layer 2 for conflict items so
    probes / geometry have a learnable signal at exactly one layer.
    """
    out: list[np.ndarray] = []
    for layer in range(n_layers):
        base = rng.normal(size=(n_problems, n_positions, hidden)).astype(np.float32)
        if layer == 2:
            shift = 0.8 * conflict.astype(np.float32)[:, None, None]
            signal = np.zeros_like(base)
            signal[:, :, 0] = shift[:, :, 0]
            base = base + signal
        out.append(base)
    return out


def _build_synthetic_attention(
    n_problems: int,
    n_layers: int,
    n_heads: int,
    n_positions: int,
    conflict: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Synthesize all six attention metrics with a planted differential.

    On (layer, head) = (1, 0), conflict items get higher entropy than
    control items so the per-head differential test in
    :mod:`s1s2.attention.heads` finds something significant.
    """
    shape = (n_problems, n_layers, n_heads, n_positions)
    base_entropy = rng.uniform(0.5, 2.5, size=shape).astype(np.float32)
    # Plant a positive S2-direction effect on (layer 1, head 0).
    bump = 0.6 * conflict.astype(np.float32)
    base_entropy[:, 1, 0, :] += bump[:, None]
    # Approximate normalization: divide by log2(prompt_len ~ 32).
    entropy_normalized = base_entropy / np.log2(32.0)
    entropy_normalized = np.clip(entropy_normalized, 0.0, 1.0)
    # Inverse relationship for concentration metrics.
    gini = np.clip(1.0 - entropy_normalized, 0.0, 1.0).astype(np.float32)
    max_attn = np.clip(0.9 * gini + 0.05, 0.0, 1.0).astype(np.float32)
    focus_5 = np.clip(0.95 * gini + 0.05, 0.0, 1.0).astype(np.float32)
    effective_rank = np.exp2(base_entropy).astype(np.float32)
    return {
        "entropy": base_entropy.astype(np.float32),
        "entropy_normalized": entropy_normalized.astype(np.float32),
        "gini": gini,
        "max_attn": max_attn,
        "focus_5": focus_5,
        "effective_rank": effective_rank,
    }


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

    Used by both ``conftest.synthetic_hdf5_path`` and ``scripts/smoke_test.py``.
    Returns the (Path) where the file was written. Idempotent — overwrites any
    file at ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    n_positions = len(positions)
    rng = np.random.default_rng(int(seed))

    (
        ids,
        categories,
        conflict,
        difficulty,
        prompt_text,
        correct_answer,
        lure_answer,
        matched_pair_id,
        prompt_token_count,
    ) = _build_synthetic_metadata(n_problems)

    # Behavior: conflict ~40% correct, control ~70% correct, lures from wrong
    # conflict items.
    correct = np.where(
        conflict,
        rng.uniform(size=n_problems) < 0.4,
        rng.uniform(size=n_problems) < 0.7,
    )
    matches_lure = conflict & ~correct
    predicted_answer = [
        "A" if is_correct else ("B" if is_lure else "C")
        for is_correct, is_lure in zip(correct, matches_lure, strict=False)
    ]
    response_category = [
        "correct" if is_correct else ("lure" if is_lure else "other_wrong")
        for is_correct, is_lure in zip(correct, matches_lure, strict=False)
    ]

    residuals = _build_synthetic_residuals(
        n_problems=n_problems,
        n_layers=n_layers,
        n_positions=n_positions,
        hidden=hidden,
        conflict=conflict,
        rng=rng,
    )
    attention_metrics = _build_synthetic_attention(
        n_problems=n_problems,
        n_layers=n_layers,
        n_heads=n_heads,
        n_positions=n_positions,
        conflict=conflict,
        rng=rng,
    )

    # Position validity: P0 / P2 always valid; T-positions only valid if the
    # model is reasoning. We mark non-reasoning T positions as invalid so the
    # workstream readers exercise the same code path as on real models.
    valid = np.zeros((n_problems, n_positions), dtype=bool)
    for i, label in enumerate(positions):
        if label in ("P0", "P2"):
            valid[:, i] = True
        # T* positions left False — synthetic model is non-reasoning.
    token_indices = np.arange(n_positions, dtype=np.int32)
    token_indices = np.broadcast_to(token_indices, (n_problems, n_positions)).copy()

    by_position_surprises = rng.uniform(0.5, 5.0, size=(n_problems, n_positions)).astype(
        np.float32
    )
    # CSR-style ragged trace: 8 tokens per problem so the validator's
    # ``full_trace_offsets`` / ``full_trace_values`` checks pass.
    tokens_per_problem = 8
    full_trace_offsets = np.arange(
        0, (n_problems + 1) * tokens_per_problem, tokens_per_problem, dtype=np.int64
    )
    full_trace_values = rng.uniform(
        0.5, 5.0, size=int(full_trace_offsets[-1])
    ).astype(np.float32)

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
            ioh.write_residual_layer(f, SYNTH_MODEL_KEY, layer=layer, data=residuals[layer])
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
            full_trace_offsets=full_trace_offsets,
            full_trace_values=full_trace_values,
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


@pytest.fixture
def synthetic_hdf5_path(tmp_path: Path) -> Path:
    """Build a tiny HDF5 cache with 1 model, 20 problems, 4 layers.

    The cache conforms to ``docs/data_contract.md`` and is constructed
    entirely via the ``s1s2.utils.io.write_*`` helpers, so any change to
    the writer API will surface here first.
    """
    out = tmp_path / "synthetic.h5"
    return build_synthetic_hdf5(out)


@pytest.fixture
def synthetic_hdf5_with_reasoning_position(tmp_path: Path) -> Path:
    """Variant where ``T50`` is marked valid (mimics a reasoning model).

    Useful for tests that want to exercise the T-position code path without
    constructing a second HDF5 from scratch.
    """
    out = tmp_path / "synthetic_reasoning.h5"
    path = build_synthetic_hdf5(out)
    # Flip T50 to valid in-place.
    with h5py.File(path, "r+") as f:
        valid = f[f"/models/{SYNTH_MODEL_KEY}/position_index/valid"][:]
        labels = [
            s.decode() if isinstance(s, bytes) else s
            for s in f[f"/models/{SYNTH_MODEL_KEY}/position_index/labels"][:]
        ]
        if "T50" in labels:
            valid[:, labels.index("T50")] = True
        del f[f"/models/{SYNTH_MODEL_KEY}/position_index/valid"]
        f.create_dataset(
            f"/models/{SYNTH_MODEL_KEY}/position_index/valid", data=valid
        )
    return path
