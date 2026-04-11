"""Typed read accessors for the activation HDF5 cache.

The data contract is in ``docs/data_contract.md``. This module is the
single, typed read API for that schema. **Do not access HDF5 keys
directly from analysis code** — go through the helpers here so changes to
the schema only require updating one file.

The :mod:`s1s2.extract` module owns writing. Other workstreams read only.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
from beartype import beartype

SCHEMA_VERSION = 1


@contextmanager
def open_activations(path: str | Path, mode: str = "r") -> Iterator[h5py.File]:
    """Open an activations HDF5 file. Verifies schema version on read."""
    path = Path(path)
    f = h5py.File(path, mode)
    try:
        if mode.startswith("r"):
            v = int(f["/metadata"].attrs.get("schema_version", -1))
            if v != SCHEMA_VERSION:
                raise ValueError(
                    f"{path} has schema_version={v}, expected {SCHEMA_VERSION}. "
                    "Migrate or re-extract."
                )
        yield f
    finally:
        f.close()


# ----- Per-problem metadata -----


@beartype
def load_problem_metadata(f: h5py.File) -> dict[str, np.ndarray]:
    """Return all per-problem metadata as a dict of arrays.

    Decodes byte-string fields to Python strings for ergonomic use.
    """
    out: dict[str, np.ndarray] = {}
    for key in ("id", "category", "prompt_text", "correct_answer", "lure_answer", "matched_pair_id"):
        raw = f[f"/problems/{key}"][:]
        out[key] = np.array([s.decode("utf-8") if isinstance(s, bytes) else s for s in raw])
    out["conflict"] = f["/problems/conflict"][:]
    out["difficulty"] = f["/problems/difficulty"][:]
    out["prompt_token_count"] = f["/problems/prompt_token_count"][:]
    return out


@beartype
def n_problems(f: h5py.File) -> int:
    return int(f["/problems/id"].shape[0])


# ----- Per-model accessors -----


@beartype
def list_models(f: h5py.File) -> list[str]:
    return list(f["/models"].keys())


@beartype
def model_metadata(f: h5py.File, model_key: str) -> dict[str, object]:
    grp = f[f"/models/{model_key}/metadata"]
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in grp.attrs.items()}


@beartype
def get_residual(
    f: h5py.File,
    model_key: str,
    layer: int,
    position: str | None = None,
) -> np.ndarray:
    """Get residual stream activations.

    If ``position`` is None, returns shape (n_problems, n_positions, hidden_dim).
    If ``position`` is given (e.g. ``"P0"``), returns shape (n_problems, hidden_dim).
    """
    arr = f[f"/models/{model_key}/residual/layer_{layer:02d}"][:]
    if position is None:
        return arr
    labels = position_labels(f, model_key)
    if position not in labels:
        raise KeyError(f"Position {position!r} not in {labels}")
    idx = labels.index(position)
    return arr[:, idx, :]


@beartype
def position_labels(f: h5py.File, model_key: str) -> list[str]:
    raw = f[f"/models/{model_key}/position_index/labels"][:]
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw]


@beartype
def position_valid(f: h5py.File, model_key: str) -> np.ndarray:
    return f[f"/models/{model_key}/position_index/valid"][:]


@beartype
def get_attention_metric(
    f: h5py.File,
    model_key: str,
    metric: str,
) -> np.ndarray:
    """Return ``/models/{model_key}/attention/{metric}``.

    Available metrics: entropy, entropy_normalized, gini, max_attn,
    focus_5, effective_rank.
    """
    valid = {"entropy", "entropy_normalized", "gini", "max_attn", "focus_5", "effective_rank"}
    if metric not in valid:
        raise ValueError(f"metric must be one of {valid}")
    return f[f"/models/{model_key}/attention/{metric}"][:]


@beartype
def get_token_surprises(
    f: h5py.File,
    model_key: str,
    by_position_only: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return per-position surprises (n_problems, n_positions).

    If ``by_position_only=False``, also returns the full ragged trace
    as ``(offsets, values)`` for metacognitive analysis.
    """
    by_pos = f[f"/models/{model_key}/token_surprises/by_position"][:]
    if by_position_only:
        return by_pos
    offsets = f[f"/models/{model_key}/token_surprises/full_trace_offsets"][:]
    values = f[f"/models/{model_key}/token_surprises/full_trace_values"][:]
    return by_pos, (offsets, values)


@beartype
def get_behavior(f: h5py.File, model_key: str) -> dict[str, np.ndarray]:
    grp = f[f"/models/{model_key}/behavior"]
    out: dict[str, np.ndarray] = {}
    for key in ("predicted_answer", "response_category"):
        raw = grp[key][:]
        out[key] = np.array([s.decode("utf-8") if isinstance(s, bytes) else s for s in raw])
    out["correct"] = grp["correct"][:]
    out["matches_lure"] = grp["matches_lure"][:]
    return out


@beartype
def get_generations(f: h5py.File, model_key: str) -> dict[str, np.ndarray]:
    grp = f[f"/models/{model_key}/generations"]
    out: dict[str, np.ndarray] = {}
    for key in ("full_text", "thinking_text", "answer_text"):
        raw = grp[key][:]
        out[key] = np.array([s.decode("utf-8") if isinstance(s, bytes) else s for s in raw])
    out["thinking_token_count"] = grp["thinking_token_count"][:]
    out["answer_token_count"] = grp["answer_token_count"][:]
    return out


@beartype
def run_metadata(f: h5py.File) -> dict[str, object]:
    """Return run-level metadata (config, git_sha, seed, etc.)."""
    grp = f["/metadata"]
    out: dict[str, object] = {}
    for k, v in grp.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        out[k] = v
    if "config" in out and isinstance(out["config"], str):
        try:
            out["config"] = json.loads(out["config"])
        except json.JSONDecodeError:
            pass
    return out


# --------------------------------------------------------------------------- #
# Writing helpers                                                             #
# --------------------------------------------------------------------------- #
#
# The extract workstream owns writing to the activation cache. These helpers
# mirror the read API above but go the other direction: numpy arrays -> HDF5
# datasets at the canonical key. Other workstreams MUST NOT call these.


def _as_fixed_bytes(strings: list[str] | np.ndarray, max_len: int) -> np.ndarray:
    """Encode a list of Python strings to a ``(n,) S<max_len>`` numpy array.
    Strings longer than ``max_len`` are UTF-8 truncated with a best-effort
    final-byte correction (we don't want to split a multi-byte character).
    """
    out = np.empty(len(strings), dtype=f"S{max_len}")
    for i, s in enumerate(strings):
        if isinstance(s, bytes):
            b = s
        else:
            b = s.encode("utf-8", errors="replace")
        if len(b) > max_len:
            # Trim back to the nearest UTF-8 boundary
            b = b[:max_len]
            while b and (b[-1] & 0xC0) == 0x80:
                b = b[:-1]
        out[i] = b
    return out


@beartype
def write_run_metadata(
    f: h5py.File,
    *,
    benchmark_path: str,
    benchmark_sha256: str,
    created_at: str,
    git_sha: str,
    seed: int,
    config_json: str,
) -> None:
    """Create ``/metadata`` and set run-level attributes.

    Idempotent with respect to an existing ``/metadata`` group — attributes
    are overwritten, not merged.
    """
    grp = f.require_group("/metadata")
    grp.attrs["benchmark_path"] = benchmark_path
    grp.attrs["benchmark_sha256"] = benchmark_sha256
    grp.attrs["created_at"] = created_at
    grp.attrs["git_sha"] = git_sha
    grp.attrs["seed"] = int(seed)
    grp.attrs["schema_version"] = int(SCHEMA_VERSION)
    grp.attrs["config"] = config_json


@beartype
def write_problem_metadata(
    f: h5py.File,
    *,
    ids: list[str],
    categories: list[str],
    conflict: np.ndarray,
    difficulty: np.ndarray,
    prompt_text: list[str],
    correct_answer: list[str],
    lure_answer: list[str],
    matched_pair_id: list[str],
    prompt_token_count: np.ndarray,
    max_prompt_chars: int = 2048,
) -> None:
    """Write the ``/problems`` group with per-problem metadata."""
    n = len(ids)
    for arr_name, arr in (
        ("conflict", conflict),
        ("difficulty", difficulty),
        ("prompt_token_count", prompt_token_count),
    ):
        if len(arr) != n:
            raise ValueError(f"length mismatch: {arr_name}={len(arr)} vs ids={n}")
    grp = f.require_group("/problems")
    # Delete any pre-existing datasets so re-writes succeed
    for key in (
        "id",
        "category",
        "conflict",
        "difficulty",
        "prompt_text",
        "correct_answer",
        "lure_answer",
        "matched_pair_id",
        "prompt_token_count",
    ):
        if key in grp:
            del grp[key]
    grp.create_dataset("id", data=_as_fixed_bytes(ids, 64))
    grp.create_dataset("category", data=_as_fixed_bytes(categories, 32))
    grp.create_dataset("conflict", data=conflict.astype(np.bool_))
    grp.create_dataset("difficulty", data=difficulty.astype(np.int8))
    grp.create_dataset("prompt_text", data=_as_fixed_bytes(prompt_text, max_prompt_chars))
    grp.create_dataset("correct_answer", data=_as_fixed_bytes(correct_answer, 128))
    grp.create_dataset("lure_answer", data=_as_fixed_bytes(lure_answer, 128))
    grp.create_dataset("matched_pair_id", data=_as_fixed_bytes(matched_pair_id, 64))
    grp.create_dataset("prompt_token_count", data=prompt_token_count.astype(np.int32))


@beartype
def write_model_metadata(
    f: h5py.File,
    model_key: str,
    *,
    hf_model_id: str,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    hidden_dim: int,
    head_dim: int,
    dtype: str,
    extracted_at: str,
    is_reasoning_model: bool,
) -> None:
    """Create ``/models/{model_key}/metadata`` and set model-level attributes."""
    grp = f.require_group(f"/models/{model_key}/metadata")
    grp.attrs["hf_model_id"] = hf_model_id
    grp.attrs["n_layers"] = int(n_layers)
    grp.attrs["n_heads"] = int(n_heads)
    grp.attrs["n_kv_heads"] = int(n_kv_heads)
    grp.attrs["hidden_dim"] = int(hidden_dim)
    grp.attrs["head_dim"] = int(head_dim)
    grp.attrs["dtype"] = dtype
    grp.attrs["extracted_at"] = extracted_at
    grp.attrs["is_reasoning_model"] = bool(is_reasoning_model)


@beartype
def write_residual_layer(
    f: h5py.File,
    model_key: str,
    layer: int,
    data: np.ndarray,
) -> None:
    """Write ``/models/{model_key}/residual/layer_{layer:02d}``.

    ``data`` must have shape ``(n_problems, n_positions, hidden_dim)``. Stored
    in-place at the canonical key. Any existing dataset is overwritten.

    h5py does not natively support bfloat16, so if you want bf16 on disk use
    dtype ``float16`` instead (lossy) or fall back to float32 (lossless but
    2x storage). The extractor converts before calling.
    """
    if data.ndim != 3:
        raise ValueError(f"residual layer data must be 3-D; got shape {data.shape}")
    grp = f.require_group(f"/models/{model_key}/residual")
    key = f"layer_{layer:02d}"
    if key in grp:
        del grp[key]
    grp.create_dataset(key, data=data, compression="gzip", compression_opts=4)


@beartype
def write_position_index(
    f: h5py.File,
    model_key: str,
    *,
    labels: list[str],
    token_indices: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Write ``/models/{model_key}/position_index/{labels,token_indices,valid}``."""
    n_problems, n_positions = token_indices.shape
    if len(labels) != n_positions:
        raise ValueError(
            f"label count {len(labels)} != token_indices.shape[1] {n_positions}"
        )
    if valid.shape != token_indices.shape:
        raise ValueError(
            f"valid shape {valid.shape} != token_indices shape {token_indices.shape}"
        )
    grp = f.require_group(f"/models/{model_key}/position_index")
    for key in ("labels", "token_indices", "valid"):
        if key in grp:
            del grp[key]
    grp.create_dataset("labels", data=_as_fixed_bytes(labels, 16))
    grp.create_dataset("token_indices", data=token_indices.astype(np.int32))
    grp.create_dataset("valid", data=valid.astype(np.bool_))


@beartype
def write_attention_metrics(
    f: h5py.File,
    model_key: str,
    metrics: dict[str, np.ndarray],
) -> None:
    """Write the ``/models/{model_key}/attention`` subgroup.

    ``metrics`` keys must include: entropy, entropy_normalized, gini,
    max_attn, focus_5, effective_rank. Each array has shape
    ``(n_problems, n_layers, n_heads, n_positions)``.
    """
    required = {
        "entropy",
        "entropy_normalized",
        "gini",
        "max_attn",
        "focus_5",
        "effective_rank",
    }
    missing = required - set(metrics.keys())
    if missing:
        raise ValueError(f"missing attention metrics: {sorted(missing)}")
    grp = f.require_group(f"/models/{model_key}/attention")
    shape = None
    for name, arr in metrics.items():
        if name not in required:
            continue
        if arr.ndim != 4:
            raise ValueError(f"{name} must be 4-D, got shape {arr.shape}")
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError(
                f"attention metric shape mismatch: {name}={arr.shape} vs {shape}"
            )
        if name in grp:
            del grp[name]
        grp.create_dataset(
            name,
            data=arr.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )


@beartype
def write_token_surprises(
    f: h5py.File,
    model_key: str,
    *,
    by_position: np.ndarray,
    full_trace_offsets: np.ndarray | None = None,
    full_trace_values: np.ndarray | None = None,
) -> None:
    """Write ``/models/{model_key}/token_surprises``.

    ``by_position`` shape: ``(n_problems, n_positions)``. If the full ragged
    trace is provided, ``full_trace_offsets`` is CSR-style of shape
    ``(n_problems + 1,)`` and ``full_trace_values`` is the concatenation.
    """
    grp = f.require_group(f"/models/{model_key}/token_surprises")
    for key in ("by_position", "full_trace_offsets", "full_trace_values"):
        if key in grp:
            del grp[key]
    grp.create_dataset("by_position", data=by_position.astype(np.float32))
    if full_trace_offsets is not None and full_trace_values is not None:
        grp.create_dataset(
            "full_trace_offsets", data=full_trace_offsets.astype(np.int64)
        )
        grp.create_dataset(
            "full_trace_values",
            data=full_trace_values.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )


@beartype
def write_generations(
    f: h5py.File,
    model_key: str,
    *,
    full_text: list[str],
    thinking_text: list[str],
    answer_text: list[str],
    thinking_token_count: np.ndarray,
    answer_token_count: np.ndarray,
    max_full_text_chars: int = 8192,
    max_thinking_chars: int = 8192,
    max_answer_chars: int = 512,
) -> None:
    """Write the ``/models/{model_key}/generations`` subgroup."""
    grp = f.require_group(f"/models/{model_key}/generations")
    for key in (
        "full_text",
        "thinking_text",
        "answer_text",
        "thinking_token_count",
        "answer_token_count",
    ):
        if key in grp:
            del grp[key]
    grp.create_dataset("full_text", data=_as_fixed_bytes(full_text, max_full_text_chars))
    grp.create_dataset(
        "thinking_text", data=_as_fixed_bytes(thinking_text, max_thinking_chars)
    )
    grp.create_dataset("answer_text", data=_as_fixed_bytes(answer_text, max_answer_chars))
    grp.create_dataset("thinking_token_count", data=thinking_token_count.astype(np.int32))
    grp.create_dataset("answer_token_count", data=answer_token_count.astype(np.int32))


@beartype
def write_behavior(
    f: h5py.File,
    model_key: str,
    *,
    predicted_answer: list[str],
    correct: np.ndarray,
    matches_lure: np.ndarray,
    response_category: list[str],
) -> None:
    """Write the ``/models/{model_key}/behavior`` subgroup."""
    grp = f.require_group(f"/models/{model_key}/behavior")
    for key in ("predicted_answer", "correct", "matches_lure", "response_category"):
        if key in grp:
            del grp[key]
    grp.create_dataset("predicted_answer", data=_as_fixed_bytes(predicted_answer, 128))
    grp.create_dataset("correct", data=correct.astype(np.bool_))
    grp.create_dataset("matches_lure", data=matches_lure.astype(np.bool_))
    grp.create_dataset("response_category", data=_as_fixed_bytes(response_category, 16))
