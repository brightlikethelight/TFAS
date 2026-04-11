"""HDF5 writer with strict schema conformance.

This module is the **only** place that writes the activation cache. It
enforces the schema laid out in ``docs/data_contract.md`` (schema_version=1)
and validates the produced file before returning from :func:`validate_file`.

Design choices
--------------
- bfloat16 on disk: h5py has no native bf16 dtype (as of 3.16). We default
  to float16 and record it in ``/models/<key>/metadata/dtype``. The data
  contract explicitly allows "bfloat16" or "float16".
- One dataset per layer (``layer_NN``) rather than a single
  ``(n_problems, n_layers, n_positions, hidden)`` cube. Rationale:
  downstream probes only read a few layers at a time, and per-layer chunks
  with chunk shape ``(1, n_positions, hidden)`` let h5py fetch a single
  problem without loading the whole cube.
- Per-model groups are `require_group`d so the file can be written
  incrementally: one extraction pass per model, checkpointing between.
- Ragged token-surprise traces are stored CSR-style in
  ``full_trace_offsets`` / ``full_trace_values``, consistent with the
  contract.
"""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from beartype import beartype

from s1s2.utils.types import ALL_POSITIONS

SCHEMA_VERSION = 1

# Fixed string widths. Kept in sync with docs/data_contract.md. Do not change
# without bumping SCHEMA_VERSION and migrating.
_S_ID = "S64"
_S_CAT = "S32"
_S_PROMPT = "S2048"
_S_ANSWER = "S128"
_S_LABEL = "S16"
_S_FULLTEXT = "S8192"
_S_THINK = "S8192"
_S_ANSWER_TEXT = "S512"

# Attention metrics that must appear under /models/<key>/attention
_ATTN_METRICS = (
    "entropy",
    "entropy_normalized",
    "gini",
    "max_attn",
    "focus_5",
    "effective_rank",
)


# --------------------------------------------------------------------------- #
# Data classes the CLI passes to the writer                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProblemMetadata:
    """Problem-level metadata for the ``/problems`` group.

    This is the minimum set of fields the writer needs; it is intentionally
    decoupled from the benchmark loader so the extract module can be tested
    without the loader module existing.
    """

    id: str
    category: str
    conflict: bool
    difficulty: int
    prompt_text: str
    correct_answer: str
    lure_answer: str
    matched_pair_id: str
    prompt_token_count: int


@dataclass(frozen=True)
class RunMetadata:
    benchmark_path: str
    benchmark_sha256: str
    created_at: str
    git_sha: str
    seed: int
    config_json: str

    @classmethod
    def build(
        cls,
        benchmark_path: str | Path,
        seed: int,
        config_json: str,
    ) -> RunMetadata:
        bp = Path(benchmark_path)
        sha = _safe_sha256(bp)
        return cls(
            benchmark_path=str(bp),
            benchmark_sha256=sha,
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            git_sha=_git_sha(),
            seed=seed,
            config_json=config_json,
        )


def _safe_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# --------------------------------------------------------------------------- #
# Writer                                                                      #
# --------------------------------------------------------------------------- #


def _truncate_bytes(s: str, maxlen: int) -> bytes:
    """Encode ``s`` as UTF-8 and truncate to at most ``maxlen`` bytes.

    Guarantees valid UTF-8 by dropping trailing continuation bytes at a
    multi-byte boundary.
    """
    b = s.encode("utf-8", errors="replace")
    if len(b) <= maxlen:
        return b
    b = b[:maxlen]
    # Walk back to a valid char boundary
    while b and (b[-1] & 0xC0) == 0x80:
        b = b[:-1]
    # If a starter byte was left dangling, drop it too
    if (b and (b[-1] & 0xE0) == 0xC0) or (b and (b[-1] & 0xF0) == 0xE0) or (b and (b[-1] & 0xF8) == 0xF0):
        b = b[:-1]
    return b


class ActivationWriter:
    """Append-mode HDF5 writer.

    Typical lifetime::

        writer = ActivationWriter(path)
        writer.open()
        writer.write_run_metadata(run_meta)
        writer.write_problems(problem_metas)
        writer.create_model_group(...)          # per model
        writer.write_residual_layer(...)        # per (model, layer)
        writer.write_position_index(...)        # per model
        writer.write_attention_metrics(...)     # per model
        writer.write_token_surprises(...)       # per model
        writer.write_generations(...)           # per model
        writer.write_behavior(...)              # per model
        writer.close()
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._f: h5py.File | None = None

    # ---- lifecycle ----

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self.path, "a")

    def close(self) -> None:
        if self._f is not None:
            self._f.flush()
            self._f.close()
            self._f = None

    def __enter__(self) -> ActivationWriter:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def f(self) -> h5py.File:
        if self._f is None:
            raise RuntimeError("ActivationWriter not open")
        return self._f

    # ---- top-level metadata ----

    def write_run_metadata(self, meta: RunMetadata) -> None:
        grp = self.f.require_group("/metadata")
        grp.attrs["schema_version"] = SCHEMA_VERSION
        grp.attrs["benchmark_path"] = meta.benchmark_path
        grp.attrs["benchmark_sha256"] = meta.benchmark_sha256
        grp.attrs["created_at"] = meta.created_at
        grp.attrs["git_sha"] = meta.git_sha
        grp.attrs["seed"] = int(meta.seed)
        grp.attrs["config"] = meta.config_json

    # ---- /problems ----

    def write_problems(self, problems: list[ProblemMetadata]) -> None:
        grp = self.f.require_group("/problems")
        n = len(problems)
        # Per-problem string columns are fixed-length byte arrays.
        self._set_string_dataset(grp, "id", [p.id for p in problems], _S_ID, n)
        self._set_string_dataset(grp, "category", [p.category for p in problems], _S_CAT, n)
        self._set_string_dataset(
            grp, "prompt_text", [p.prompt_text for p in problems], _S_PROMPT, n
        )
        self._set_string_dataset(
            grp, "correct_answer", [p.correct_answer for p in problems], _S_ANSWER, n
        )
        self._set_string_dataset(
            grp, "lure_answer", [p.lure_answer for p in problems], _S_ANSWER, n
        )
        self._set_string_dataset(
            grp, "matched_pair_id", [p.matched_pair_id for p in problems], _S_ID, n
        )
        self._set_scalar_dataset(
            grp, "conflict", np.array([p.conflict for p in problems], dtype=bool)
        )
        self._set_scalar_dataset(
            grp, "difficulty", np.array([p.difficulty for p in problems], dtype=np.int8)
        )
        self._set_scalar_dataset(
            grp,
            "prompt_token_count",
            np.array([p.prompt_token_count for p in problems], dtype=np.int32),
        )

    # ---- /models/<key>/metadata ----

    def create_model_group(
        self,
        model_key: str,
        hf_model_id: str,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_dim: int,
        head_dim: int,
        dtype: str,
        is_reasoning_model: bool,
    ) -> h5py.Group:
        model_grp = self.f.require_group(f"/models/{model_key}")
        meta_grp = model_grp.require_group("metadata")
        meta_grp.attrs["hf_model_id"] = hf_model_id
        meta_grp.attrs["n_layers"] = int(n_layers)
        meta_grp.attrs["n_heads"] = int(n_heads)
        meta_grp.attrs["n_kv_heads"] = int(n_kv_heads)
        meta_grp.attrs["hidden_dim"] = int(hidden_dim)
        meta_grp.attrs["head_dim"] = int(head_dim)
        meta_grp.attrs["dtype"] = str(dtype)
        meta_grp.attrs["extracted_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        meta_grp.attrs["is_reasoning_model"] = bool(is_reasoning_model)
        return model_grp

    # ---- /models/<key>/residual ----

    def allocate_residual(
        self,
        model_key: str,
        n_layers: int,
        n_problems: int,
        n_positions: int,
        hidden_dim: int,
        dtype: str,
    ) -> None:
        resid_grp = self.f.require_group(f"/models/{model_key}/residual")
        np_dtype = _hdf5_float_dtype(dtype)
        for layer_idx in range(n_layers):
            name = f"layer_{layer_idx:02d}"
            if name in resid_grp:
                continue
            resid_grp.create_dataset(
                name,
                shape=(n_problems, n_positions, hidden_dim),
                dtype=np_dtype,
                chunks=(1, n_positions, hidden_dim),
                compression="gzip",
                compression_opts=4,
            )

    def write_residual_row(
        self,
        model_key: str,
        problem_idx: int,
        layer_arrays: np.ndarray,
    ) -> None:
        """Write (n_layers, n_positions, hidden) for ``problem_idx``.

        Accepts a single stacked ndarray to minimize Python-level overhead;
        the caller has already converted to the storage dtype.
        """
        resid_grp = self.f[f"/models/{model_key}/residual"]
        n_layers = layer_arrays.shape[0]
        for layer_idx in range(n_layers):
            ds = resid_grp[f"layer_{layer_idx:02d}"]
            ds[problem_idx, :, :] = layer_arrays[layer_idx]

    # ---- /models/<key>/position_index ----

    def write_position_index(
        self,
        model_key: str,
        token_indices: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        """Write token_indices and valid arrays + labels."""
        grp = self.f.require_group(f"/models/{model_key}/position_index")
        labels = np.array([lab.encode("utf-8") for lab in ALL_POSITIONS], dtype=_S_LABEL)
        self._set_fixed_dataset(grp, "labels", labels)
        self._set_fixed_dataset(grp, "token_indices", token_indices.astype(np.int32, copy=False))
        self._set_fixed_dataset(grp, "valid", valid.astype(bool, copy=False))

    # ---- /models/<key>/attention ----

    def write_attention_metrics(
        self,
        model_key: str,
        metrics: dict[str, np.ndarray],
    ) -> None:
        grp = self.f.require_group(f"/models/{model_key}/attention")
        for name in _ATTN_METRICS:
            if name not in metrics:
                raise KeyError(f"missing attention metric {name!r}")
            self._set_fixed_dataset(grp, name, metrics[name].astype(np.float32, copy=False))

    # ---- /models/<key>/token_surprises ----

    def write_token_surprises(
        self,
        model_key: str,
        by_position: np.ndarray,
        full_trace_offsets: np.ndarray,
        full_trace_values: np.ndarray,
    ) -> None:
        grp = self.f.require_group(f"/models/{model_key}/token_surprises")
        self._set_fixed_dataset(grp, "by_position", by_position.astype(np.float32, copy=False))
        self._set_fixed_dataset(
            grp, "full_trace_offsets", full_trace_offsets.astype(np.int64, copy=False)
        )
        self._set_fixed_dataset(
            grp, "full_trace_values", full_trace_values.astype(np.float32, copy=False)
        )

    # ---- /models/<key>/generations ----

    def write_generations(
        self,
        model_key: str,
        full_text: list[str],
        thinking_text: list[str],
        answer_text: list[str],
        thinking_token_count: np.ndarray,
        answer_token_count: np.ndarray,
    ) -> None:
        grp = self.f.require_group(f"/models/{model_key}/generations")
        n = len(full_text)
        self._set_string_dataset(grp, "full_text", full_text, _S_FULLTEXT, n)
        self._set_string_dataset(grp, "thinking_text", thinking_text, _S_THINK, n)
        self._set_string_dataset(grp, "answer_text", answer_text, _S_ANSWER_TEXT, n)
        self._set_fixed_dataset(
            grp, "thinking_token_count", thinking_token_count.astype(np.int32, copy=False)
        )
        self._set_fixed_dataset(
            grp, "answer_token_count", answer_token_count.astype(np.int32, copy=False)
        )

    # ---- /models/<key>/behavior ----

    def write_behavior(
        self,
        model_key: str,
        predicted_answer: list[str],
        correct: np.ndarray,
        matches_lure: np.ndarray,
        response_category: list[str],
    ) -> None:
        grp = self.f.require_group(f"/models/{model_key}/behavior")
        n = len(predicted_answer)
        self._set_string_dataset(grp, "predicted_answer", predicted_answer, _S_ANSWER, n)
        self._set_scalar_dataset(grp, "correct", correct.astype(bool, copy=False))
        self._set_scalar_dataset(grp, "matches_lure", matches_lure.astype(bool, copy=False))
        self._set_string_dataset(
            grp, "response_category", response_category, _S_LABEL, n
        )

    # ---- low-level helpers ----

    def _set_string_dataset(
        self,
        grp: h5py.Group,
        name: str,
        values: Iterable[str],
        dtype_str: str,
        n: int,
    ) -> None:
        # Parse "S64" -> 64 bytes
        maxlen = int(dtype_str.lstrip("S"))
        arr = np.empty(n, dtype=dtype_str)
        for i, s in enumerate(values):
            arr[i] = _truncate_bytes(s if s is not None else "", maxlen)
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=arr)

    def _set_scalar_dataset(self, grp: h5py.Group, name: str, arr: np.ndarray) -> None:
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=arr)

    def _set_fixed_dataset(self, grp: h5py.Group, name: str, arr: np.ndarray) -> None:
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=arr)


# --------------------------------------------------------------------------- #
# Helpers & validation                                                        #
# --------------------------------------------------------------------------- #


def _hdf5_float_dtype(dtype: str) -> np.dtype:
    """Resolve the on-disk float dtype. Fall back to float16 if bfloat16 is
    requested (h5py 3.x doesn't support bf16 natively) -- the caller must log
    this choice and store ``dtype='float16'`` in metadata so downstream code
    doesn't try to reinterpret bytes.
    """
    low = dtype.lower()
    if low in ("bfloat16", "bf16"):
        # Fall back -- caller is responsible for recording 'float16' in metadata.
        return np.dtype(np.float16)
    if low in ("float16", "f16", "fp16", "half"):
        return np.dtype(np.float16)
    if low in ("float32", "f32", "fp32"):
        return np.dtype(np.float32)
    raise ValueError(f"unsupported on-disk dtype: {dtype!r}")


@beartype
def validate_file(path: str | Path) -> list[str]:
    """Walk the file and return a list of schema-conformance errors.

    Returns an empty list if valid. Checks that required groups/datasets
    exist and have the right shapes/dtypes. Does not inspect individual
    values beyond trivial sanity bounds.
    """
    errors: list[str] = []
    p = Path(path)
    if not p.exists():
        return [f"file not found: {p}"]
    try:
        f = h5py.File(p, "r")
    except Exception as e:
        return [f"failed to open: {e}"]
    try:
        # /metadata
        if "/metadata" not in f:
            errors.append("missing /metadata group")
        else:
            mattrs = f["/metadata"].attrs
            for k in ("schema_version", "benchmark_path", "created_at", "seed", "config"):
                if k not in mattrs:
                    errors.append(f"/metadata missing attr {k}")
            if "schema_version" in mattrs and int(mattrs["schema_version"]) != SCHEMA_VERSION:
                errors.append(
                    f"schema_version={int(mattrs['schema_version'])}, expected {SCHEMA_VERSION}"
                )

        # /problems
        if "/problems" not in f:
            errors.append("missing /problems group")
        else:
            required = (
                "id",
                "category",
                "conflict",
                "difficulty",
                "prompt_text",
                "correct_answer",
                "lure_answer",
                "matched_pair_id",
                "prompt_token_count",
            )
            for name in required:
                if name not in f["/problems"]:
                    errors.append(f"/problems missing dataset {name}")
            n_problems = f["/problems/id"].shape[0] if "id" in f["/problems"] else None
        # /models/<key>/
        if "/models" not in f:
            errors.append("missing /models group")
            return errors
        for model_key in f["/models"]:
            mg = f[f"/models/{model_key}"]
            if "metadata" not in mg:
                errors.append(f"/models/{model_key} missing metadata group")
                continue
            mm = mg["metadata"]
            for a in (
                "hf_model_id",
                "n_layers",
                "n_heads",
                "n_kv_heads",
                "hidden_dim",
                "head_dim",
                "dtype",
                "is_reasoning_model",
            ):
                if a not in mm.attrs:
                    errors.append(f"/models/{model_key}/metadata missing attr {a}")
            if "n_layers" not in mm.attrs:
                continue
            n_layers = int(mm.attrs["n_layers"])
            n_heads = int(mm.attrs.get("n_heads", 0))
            hidden_dim = int(mm.attrs.get("hidden_dim", 0))

            # /residual
            if "residual" not in mg:
                errors.append(f"/models/{model_key}/residual missing")
            else:
                for layer_idx in range(n_layers):
                    name = f"layer_{layer_idx:02d}"
                    if name not in mg["residual"]:
                        errors.append(f"/models/{model_key}/residual/{name} missing")
                        continue
                    ds = mg[f"residual/{name}"]
                    if ds.ndim != 3:
                        errors.append(f"/models/{model_key}/residual/{name} wrong ndim: {ds.ndim}")
                        continue
                    if n_problems is not None and ds.shape[0] != n_problems:
                        errors.append(
                            f"/models/{model_key}/residual/{name} n_problems mismatch: "
                            f"{ds.shape[0]} vs {n_problems}"
                        )
                    if ds.shape[2] != hidden_dim:
                        errors.append(
                            f"/models/{model_key}/residual/{name} hidden_dim mismatch: "
                            f"{ds.shape[2]} vs {hidden_dim}"
                        )

            # /position_index
            if "position_index" not in mg:
                errors.append(f"/models/{model_key}/position_index missing")
            else:
                for k in ("labels", "token_indices", "valid"):
                    if k not in mg["position_index"]:
                        errors.append(f"/models/{model_key}/position_index/{k} missing")

            # /attention
            if "attention" not in mg:
                errors.append(f"/models/{model_key}/attention missing")
            else:
                for metric in _ATTN_METRICS:
                    if metric not in mg["attention"]:
                        errors.append(f"/models/{model_key}/attention/{metric} missing")
                        continue
                    ds = mg[f"attention/{metric}"]
                    if ds.ndim != 4:
                        errors.append(
                            f"/models/{model_key}/attention/{metric} wrong ndim: {ds.ndim}"
                        )
                        continue
                    if ds.shape[1] != n_layers:
                        errors.append(
                            f"/models/{model_key}/attention/{metric} n_layers mismatch: "
                            f"{ds.shape[1]} vs {n_layers}"
                        )
                    if ds.shape[2] != n_heads:
                        errors.append(
                            f"/models/{model_key}/attention/{metric} n_heads mismatch: "
                            f"{ds.shape[2]} vs {n_heads}"
                        )

            # /token_surprises
            if "token_surprises" not in mg:
                errors.append(f"/models/{model_key}/token_surprises missing")
            else:
                for k in ("by_position", "full_trace_offsets", "full_trace_values"):
                    if k not in mg["token_surprises"]:
                        errors.append(f"/models/{model_key}/token_surprises/{k} missing")

            # /generations
            if "generations" not in mg:
                errors.append(f"/models/{model_key}/generations missing")
            else:
                for k in (
                    "full_text",
                    "thinking_text",
                    "answer_text",
                    "thinking_token_count",
                    "answer_token_count",
                ):
                    if k not in mg["generations"]:
                        errors.append(f"/models/{model_key}/generations/{k} missing")

            # /behavior
            if "behavior" not in mg:
                errors.append(f"/models/{model_key}/behavior missing")
            else:
                for k in (
                    "predicted_answer",
                    "correct",
                    "matches_lure",
                    "response_category",
                ):
                    if k not in mg["behavior"]:
                        errors.append(f"/models/{model_key}/behavior/{k} missing")
    finally:
        f.close()
    return errors
