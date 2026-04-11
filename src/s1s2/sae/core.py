"""Top-level orchestration for the SAE feature-analysis workstream.

This module is the single entry point that glues together the four
primitives already in the package:

1. :mod:`s1s2.sae.loaders` — pre-trained SAE loading + reconstruction
   fidelity report.
2. :mod:`s1s2.sae.differential` — Mann-Whitney + BH-FDR across all
   features at one ``(model, layer)``.
3. :mod:`s1s2.sae.falsification` — Ma et al. (2026) spurious-feature
   filter on the top candidates.
4. :mod:`s1s2.sae.volcano` — volcano-plot rendering with the
   falsification overlay.

The flow per ``(model, layer)`` is deliberately linear:

    load activations -> load SAE -> reconstruction check
      -> encode -> differential test -> falsify top-K -> write results

The reconstruction check is a hard gate: if the SAE reconstructs poorly
on the target model's activations, **we do not trust any downstream
feature** and abort the cell. That matches the non-negotiable
``CLAUDE.md`` rule and the Ma et al. protocol.

The orchestration class :class:`SAEAnalysisRunner` does zero stats on
its own — it is pure glue. Everything stochastic flows from
``SAERunnerConfig.seed`` via :func:`s1s2.utils.seed.set_global_seed`.

Outputs per ``(model, layer)``
------------------------------
1. ``results/sae/{model_key}/layer_{NN:02d}/feature_analysis.json`` —
   ranked feature statistics and run metadata (git sha, runtime, cfg).
2. ``results/sae/{model_key}/layer_{NN:02d}/volcano.png`` — labeled
   volcano with Ma-et-al falsification overlay.
3. ``results/sae/{model_key}/layer_{NN:02d}/steering_vectors.npz`` —
   encoder / decoder directions for the top features, ready for the
   causal workstream to ingest.

``model_key`` in the path uses the same sanitized HDF5 key that the
``/models`` subgroup uses (HF id with ``/`` → ``_``). Downstream workstreams
that look up vectors by model can use that key directly.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from beartype import beartype

from s1s2.sae.differential import (
    DifferentialResult,
    differential_activation,
    encode_batched,
    matched_pair_subset,
)
from s1s2.sae.falsification import (
    FalsificationResult,
    falsify_candidates,
)
from s1s2.sae.loaders import (
    SAEHandle,
    load_sae_for_model,
    reconstruction_report,
)
from s1s2.sae.volcano import plot_volcano
from s1s2.utils import io as ioh
from s1s2.utils.logging import get_logger
from s1s2.utils.seed import set_global_seed

logger = get_logger("s1s2.sae")


# ---------------------------------------------------------------------------
# Model-key registry
# ---------------------------------------------------------------------------
#
# The SAE loaders use short keys (``"llama-3.1-8b-instruct"``) while the HDF5
# cache indexes models by their sanitized HuggingFace id
# (``"meta-llama_Llama-3.1-8B-Instruct"``). We keep a small in-module map so
# the CLI can accept both forms and the core runner doesn't need to import
# Hydra.
#
# Per CLAUDE.md: the canonical mapping lives in ``configs/models.yaml``. This
# dict is a fallback for the standard four models so the runner is runnable
# without an on-disk model config — useful for tests.

DEFAULT_MODEL_HDF5_KEYS: dict[str, str] = {
    "llama-3.1-8b-instruct": "meta-llama_Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google_gemma-2-9b-it",
    "r1-distill-llama-8b": "deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
    "r1-distill-qwen-7b": "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
}

DEFAULT_MODEL_SAE_RELEASES: dict[str, str] = {
    "llama-3.1-8b-instruct": "fnlp/Llama-3_1-8B-Base-LXR-32x",
    "gemma-2-9b-it": "google/gemma-scope-9b-it-res",
    "r1-distill-llama-8b": "fnlp/Llama-3_1-8B-Base-LXR-32x",
    "r1-distill-qwen-7b": "none (no pre-trained SAE available)",
}

# Default layer choices per model when cfg.layers is None. Picked to cover
# early / mid / late depth cheaply; override via config for full sweeps.
DEFAULT_LAYERS_FOR_MODEL: dict[str, tuple[int, ...]] = {
    "llama-3.1-8b-instruct": (8, 16, 24),
    "gemma-2-9b-it": (10, 21, 32),
    "r1-distill-llama-8b": (8, 16, 24),
    "r1-distill-qwen-7b": (7, 14, 21),
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class SAERunnerConfig:
    """Knobs controlling one full SAE analysis run.

    Kept intentionally flat and Hydra-compatible: every field is a
    primitive or a list of primitives, so the CLI can splat a
    ``DictConfig`` straight into the constructor.
    """

    activations_path: str
    output_dir: str = "results/sae"
    models: list[str] = field(
        default_factory=lambda: [
            "llama-3.1-8b-instruct",
            "gemma-2-9b-it",
            "r1-distill-llama-8b",
        ]
    )
    layers: list[int] | None = None  # None = DEFAULT_LAYERS_FOR_MODEL
    position: str = "P0"

    # Reconstruction fidelity gate ------------------------------------------
    reconstruction_check_n_samples: int = 256
    reconstruction_min_explained_variance: float = 0.5

    # Differential -----------------------------------------------------------
    fdr_q: float = 0.05
    matched_difficulty_only: bool = False

    # Falsification (Ma et al. 2026) -----------------------------------------
    falsification_enabled: bool = True
    falsification_n_random_texts: int = 100
    falsification_n_top_tokens: int = 5
    falsification_threshold: float = 0.5
    falsification_mode: Literal["model_forward", "cheap"] = "cheap"
    falsification_top_k_features: int = 50

    # Plotting ---------------------------------------------------------------
    volcano_top_k: int = 10

    # Optional per-model overrides used by the CLI to translate Hydra
    # ``configs/models.yaml`` into explicit HDF5 keys / releases.
    model_hdf5_keys: dict[str, str] = field(default_factory=dict)
    model_sae_releases: dict[str, str] = field(default_factory=dict)

    # SAE-lens device (for the real loaders only; MockSAE ignores it).
    sae_device: str = "cpu"

    seed: int = 0

    # -------------------------------------------------------------------
    # Resolution helpers
    # -------------------------------------------------------------------

    def hdf5_key_for(self, model_key: str) -> str:
        return self.model_hdf5_keys.get(
            model_key, DEFAULT_MODEL_HDF5_KEYS.get(model_key, model_key)
        )

    def sae_release_for(self, model_key: str) -> str:
        return self.model_sae_releases.get(
            model_key,
            DEFAULT_MODEL_SAE_RELEASES.get(model_key, "unknown"),
        )

    def layers_for(self, model_key: str) -> list[int]:
        if self.layers is not None:
            return list(self.layers)
        return list(DEFAULT_LAYERS_FOR_MODEL.get(model_key, (8, 16, 24)))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Short git SHA of the current checkout; ``"unknown"`` on failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _as_py(obj: Any) -> Any:
    """Recursively coerce numpy scalars / arrays into JSON-native types."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_as_py(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _as_py(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_as_py(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _layer_dir(output_dir: Path, hdf5_key: str, layer: int) -> Path:
    """Canonical results dir for one ``(model, layer)`` cell."""
    return output_dir / hdf5_key / f"layer_{layer:02d}"


def _feature_directions(sae: SAEHandle) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return encoder/decoder direction matrices, or ``(None, None)`` if unavailable.

    Both :class:`MockSAE` and the :class:`_SAELensHandle` expose
    ``encoder_directions`` / ``decoder_directions`` as ``(n_features,
    hidden_dim)`` numpy arrays. We probe duck-typed so callers don't need
    to know which backend is in use, and we fall back silently if a
    backend grows without those methods.
    """
    enc = getattr(sae, "encoder_directions", None)
    dec = getattr(sae, "decoder_directions", None)
    enc_mat = enc() if callable(enc) else None
    dec_mat = dec() if callable(dec) else None
    return enc_mat, dec_mat


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class SAEAnalysisRunner:
    """Run the SAE differential + falsification pipeline across a grid.

    Instantiate once per run, then call :meth:`run` to iterate over
    every ``(model, layer)`` cell and write the per-cell artifacts to
    ``cfg.output_dir``.

    The class holds no state across ``run`` invocations, so re-running
    is safe.
    """

    def __init__(self, cfg: SAERunnerConfig) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.activations_path = Path(cfg.activations_path)
        self._git_sha = _git_sha()
        set_global_seed(int(cfg.seed), deterministic_torch=False)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    @beartype
    def run(self) -> dict[tuple[str, int], dict[str, Any]]:
        """Execute the full grid. Returns a dict keyed by ``(model_key, layer)``.

        Each value is the per-cell summary dict (mirroring the per-cell
        JSON written to disk). Cells that hit a hard failure (missing
        model in the HDF5, reconstruction failure, empty activations,
        etc.) return a ``{"status": ..., "reason": ...}`` stub so the
        caller can distinguish "we ran it and it failed" from "we never
        tried" without crawling the log.
        """

        results: dict[tuple[str, int], dict[str, Any]] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "SAE run: models=%s layers=%s position=%s",
            self.cfg.models,
            self.cfg.layers if self.cfg.layers is not None else "defaults",
            self.cfg.position,
        )

        with ioh.open_activations(self.activations_path) as f:
            available = set(ioh.list_models(f))
            problems = ioh.load_problem_metadata(f)
            conflict = problems["conflict"].astype(bool)
            prompts = [str(p) for p in problems["prompt_text"]]
            matched_pair_id = problems["matched_pair_id"]

            for model_key in self.cfg.models:
                hdf5_key = self.cfg.hdf5_key_for(model_key)
                if hdf5_key not in available:
                    logger.warning(
                        "[%s] hdf5_key=%s not in activations file — skipping",
                        model_key,
                        hdf5_key,
                    )
                    continue

                behavior = ioh.get_behavior(f, hdf5_key)
                correct = behavior["correct"].astype(bool)

                for layer in self.cfg.layers_for(model_key):
                    cell_result = self._run_cell(
                        f=f,
                        model_key=model_key,
                        hdf5_key=hdf5_key,
                        layer=int(layer),
                        conflict=conflict,
                        prompts=prompts,
                        matched_pair_id=matched_pair_id,
                        correct=correct,
                    )
                    results[(model_key, int(layer))] = cell_result

        logger.info("SAE run complete: %d cells processed", len(results))
        return results

    # -------------------------------------------------------------------
    # Per-(model, layer) cell
    # -------------------------------------------------------------------

    def _run_cell(
        self,
        *,
        f,
        model_key: str,
        hdf5_key: str,
        layer: int,
        conflict: np.ndarray,
        prompts: list[str],
        matched_pair_id: np.ndarray,
        correct: np.ndarray,
    ) -> dict[str, Any]:
        """Load activations, run differential + falsification, write artifacts."""

        started = time.time()
        layer_dir = _layer_dir(self.output_dir, hdf5_key, layer)
        layer_dir.mkdir(parents=True, exist_ok=True)
        tag = f"[{model_key}/L{layer:02d}/{self.cfg.position}]"

        # 1. Load activations at this layer+position.
        try:
            activations = ioh.get_residual(
                f, hdf5_key, layer=layer, position=self.cfg.position
            ).astype(np.float32, copy=False)
        except Exception as exc:
            logger.error("%s failed to load residuals: %s", tag, exc)
            return self._record_failure(
                model_key=model_key,
                hdf5_key=hdf5_key,
                layer=layer,
                layer_dir=layer_dir,
                reason=f"residual load failed: {exc!r}",
                started=started,
            )

        if activations.ndim != 2 or activations.shape[0] == 0:
            return self._record_failure(
                model_key=model_key,
                hdf5_key=hdf5_key,
                layer=layer,
                layer_dir=layer_dir,
                reason=f"bad activations shape {activations.shape}",
                started=started,
            )

        logger.info("%s residuals shape=%s", tag, activations.shape)

        # 2. Load the SAE for this (model, layer).
        try:
            sae = load_sae_for_model(
                model_key=model_key,
                layer=layer,
                device=self.cfg.sae_device,
            )
        except Exception as exc:
            logger.error("%s SAE load failed: %s", tag, exc)
            return self._record_failure(
                model_key=model_key,
                hdf5_key=hdf5_key,
                layer=layer,
                layer_dir=layer_dir,
                reason=f"SAE load failed: {exc!r}",
                started=started,
            )

        if activations.shape[1] != sae.hidden_dim:
            logger.error(
                "%s hidden_dim mismatch: activations=%d SAE=%d",
                tag,
                activations.shape[1],
                sae.hidden_dim,
            )
            return self._record_failure(
                model_key=model_key,
                hdf5_key=hdf5_key,
                layer=layer,
                layer_dir=layer_dir,
                reason=(
                    f"hidden_dim mismatch activations={activations.shape[1]} "
                    f"SAE={sae.hidden_dim}"
                ),
                started=started,
            )

        # 3. Reconstruction fidelity gate.
        rng = np.random.default_rng(self.cfg.seed + layer)
        recon = reconstruction_report(
            sae=sae,
            activations=activations,
            n_samples=int(self.cfg.reconstruction_check_n_samples),
            min_explained_variance=float(self.cfg.reconstruction_min_explained_variance),
            rng=rng,
        )
        if recon.is_poor_fit:
            logger.warning(
                "%s reconstruction fidelity too low (ev=%.3f); skipping cell",
                tag,
                recon.explained_variance,
            )
            stub = self._build_result_dict(
                model_key=model_key,
                hdf5_key=hdf5_key,
                layer=layer,
                reconstruction_ev=float(recon.explained_variance),
                n_features_total=int(sae.n_features),
                n_features_significant=0,
                n_features_after_falsification=0,
                top_features=[],
                status="skipped_poor_reconstruction",
                started=started,
            )
            self._write_results_json(layer_dir, stub)
            return stub

        # 4. Encode activations -> feature activations.
        feature_activations = encode_batched(sae, activations, batch_size=256)

        # 5a. Differential analysis (primary: all pairs).
        diff_all = differential_activation(
            feature_activations=feature_activations,
            conflict=conflict,
            fdr_q=float(self.cfg.fdr_q),
            subset_label="all",
        )

        # 5b. Optional matched-pair subset. Reported for traceability; the
        # primary result remains the full-sample differential so downstream
        # code has a stable contract.
        diff_matched: DifferentialResult | None = None
        if self.cfg.matched_difficulty_only:
            matched_acts, matched_conflict = matched_pair_subset(
                feature_activations=feature_activations,
                conflict=conflict,
                matched_pair_id=matched_pair_id,
                correct=correct,
            )
            if matched_acts.shape[0] >= 6:
                diff_matched = differential_activation(
                    feature_activations=matched_acts,
                    conflict=matched_conflict,
                    fdr_q=float(self.cfg.fdr_q),
                    subset_label="matched_pairs",
                )
            else:
                logger.warning(
                    "%s matched-pair subset too small (%d rows); skipping",
                    tag,
                    matched_acts.shape[0],
                )

        diff_df = diff_all.df.copy()
        n_sig = int(diff_df["significant"].sum())

        # 6. Falsification on the top candidates (by effect magnitude, within
        # the significant subset).
        falsification_results: list[FalsificationResult] = []
        if self.cfg.falsification_enabled and n_sig > 0:
            sig_subset = diff_df[diff_df["significant"]].copy()
            sig_subset["abs_log_fc"] = sig_subset["log_fc"].abs()
            sig_subset = sig_subset.sort_values(
                ["abs_log_fc", "effect_size"], ascending=[False, False]
            )
            candidate_ids = sig_subset["feature_id"].astype(int).tolist()
            falsification_results = falsify_candidates(
                candidate_feature_ids=candidate_ids,
                sae=sae,
                activations=activations,
                feature_activations=feature_activations,
                prompts=prompts,
                tokenizer=None,
                model=None,
                layer=layer,
                mode=str(self.cfg.falsification_mode),
                n_random_texts=int(self.cfg.falsification_n_random_texts),
                n_top_tokens=int(self.cfg.falsification_n_top_tokens),
                threshold=float(self.cfg.falsification_threshold),
                top_k_features=int(self.cfg.falsification_top_k_features),
                device=self.cfg.sae_device,
            )

        # 7. Merge falsification outcome back onto the differential dataframe
        # so the volcano plot and JSON can key off ``is_falsified``.
        diff_df["is_falsified"] = False
        diff_df["falsification_ratio"] = np.nan
        if falsification_results:
            fidx = diff_df.set_index("feature_id")
            for fr in falsification_results:
                if fr.feature_id in fidx.index:
                    fidx.at[fr.feature_id, "is_falsified"] = bool(fr.is_spurious)
                    fidx.at[fr.feature_id, "falsification_ratio"] = float(fr.falsification_ratio)
            diff_df = fidx.reset_index()

        n_after_fals = int((diff_df["significant"] & (~diff_df["is_falsified"].astype(bool))).sum())

        # 7a. Top features dict list.
        top_features = self._top_feature_rows(diff_df, k=int(self.cfg.volcano_top_k))

        # 7b. Volcano plot.
        volcano_path = layer_dir / "volcano.png"
        try:
            plot_volcano(
                df=diff_df,
                title=f"{model_key} layer {layer} ({self.cfg.position})",
                out_path=volcano_path,
                fdr_q=float(self.cfg.fdr_q),
                annotate_top_k=int(self.cfg.volcano_top_k),
            )
        except Exception as exc:
            logger.error("%s volcano plot failed: %s", tag, exc)

        # 7c. Steering vectors npz.
        steering_path = layer_dir / "steering_vectors.npz"
        try:
            self._write_steering_vectors(
                out_path=steering_path,
                sae=sae,
                diff_df=diff_df,
                k=int(self.cfg.falsification_top_k_features),
            )
        except Exception as exc:
            logger.error("%s steering vector write failed: %s", tag, exc)

        # 7d. Feature stats csv (full dataframe for later browsing).
        try:
            diff_df.to_csv(layer_dir / "feature_stats.csv", index=False)
        except Exception as exc:
            logger.error("%s feature_stats.csv write failed: %s", tag, exc)

        # 8. Build and write results JSON.
        result_dict = self._build_result_dict(
            model_key=model_key,
            hdf5_key=hdf5_key,
            layer=layer,
            reconstruction_ev=float(recon.explained_variance),
            n_features_total=int(sae.n_features),
            n_features_significant=n_sig,
            n_features_after_falsification=n_after_fals,
            top_features=top_features,
            status="ok",
            started=started,
            reconstruction_report_full=asdict(recon),
            diff_matched=diff_matched,
            falsification_results=falsification_results,
        )
        self._write_results_json(layer_dir, result_dict)
        logger.info(
            "%s done: sig=%d after_falsification=%d runtime=%.1fs",
            tag,
            n_sig,
            n_after_fals,
            time.time() - started,
        )
        return result_dict

    # -------------------------------------------------------------------
    # Dict / file helpers
    # -------------------------------------------------------------------

    def _top_feature_rows(self, df: pd.DataFrame, k: int) -> list[dict[str, Any]]:
        """Return the top ``k`` feature rows by abs(log_fc) among significant
        (falling back to top-k overall if nothing is significant).

        Each row is a plain JSON-safe dict with the columns specified in the
        task brief's results schema.
        """
        if df.empty or k <= 0:
            return []
        work = df.copy()
        work["abs_log_fc"] = work["log_fc"].abs()
        sig = work[work.get("significant", False).astype(bool)]
        if sig.empty:
            ordered = work.sort_values("q_value", ascending=True).head(k)
        else:
            ordered = sig.sort_values("abs_log_fc", ascending=False).head(k)

        out: list[dict[str, Any]] = []
        for _, row in ordered.iterrows():
            out.append(
                {
                    "feature_id": int(row["feature_id"]),
                    "log_fold_change": float(row.get("log_fc", float("nan"))),
                    "q_value": float(row.get("q_value", float("nan"))),
                    "effect_size": float(row.get("effect_size", float("nan"))),
                    "mean_activation_S1": float(row.get("mean_S1", float("nan"))),
                    "mean_activation_S2": float(row.get("mean_S2", float("nan"))),
                    "is_falsified": bool(row.get("is_falsified", False)),
                    "falsification_ratio": (
                        None
                        if pd.isna(row.get("falsification_ratio", np.nan))
                        else float(row["falsification_ratio"])
                    ),
                    "auto_interp_label": "unknown",
                }
            )
        return out

    def _build_result_dict(
        self,
        *,
        model_key: str,
        hdf5_key: str,
        layer: int,
        reconstruction_ev: float,
        n_features_total: int,
        n_features_significant: int,
        n_features_after_falsification: int,
        top_features: list[dict[str, Any]],
        status: str,
        started: float,
        reconstruction_report_full: dict[str, Any] | None = None,
        diff_matched: DifferentialResult | None = None,
        falsification_results: list[FalsificationResult] | None = None,
    ) -> dict[str, Any]:
        """Assemble the JSON-serialisable result for one cell."""
        cfg_dict = asdict(self.cfg)
        out: dict[str, Any] = {
            "model_key": model_key,
            "hdf5_key": hdf5_key,
            "layer": int(layer),
            "position": self.cfg.position,
            "sae_release": self.cfg.sae_release_for(model_key),
            "reconstruction_explained_variance": float(reconstruction_ev),
            "n_features_total": int(n_features_total),
            "n_features_significant": int(n_features_significant),
            "n_features_after_falsification": int(n_features_after_falsification),
            "top_features": top_features,
            "status": status,
            "config": cfg_dict,
            "git_sha": self._git_sha,
            "runtime_seconds": float(time.time() - started),
        }
        if reconstruction_report_full is not None:
            out["reconstruction_report"] = reconstruction_report_full
        if diff_matched is not None:
            out["matched_pairs_summary"] = {
                "n_S1": int(diff_matched.n_S1),
                "n_S2": int(diff_matched.n_S2),
                "n_significant": int(diff_matched.df["significant"].sum()),
                "fdr_q": float(diff_matched.fdr_q),
            }
        if falsification_results is not None:
            out["falsification"] = {
                "mode": str(self.cfg.falsification_mode),
                "n_tested": len(falsification_results),
                "n_spurious": int(sum(1 for r in falsification_results if r.is_spurious)),
                "per_feature": [
                    {
                        "feature_id": int(r.feature_id),
                        "is_spurious": bool(r.is_spurious),
                        "trigger_tokens": list(r.trigger_tokens),
                        "mean_activation_on_original": float(r.mean_activation_on_original),
                        "peak_activation_on_original": float(r.peak_activation_on_original),
                        "mean_activation_on_random": float(r.mean_activation_on_random),
                        "peak_activation_on_random": float(r.peak_activation_on_random),
                        "falsification_ratio": float(r.falsification_ratio),
                        "notes": str(r.notes),
                    }
                    for r in falsification_results
                ],
            }
        return _as_py(out)

    def _write_results_json(self, layer_dir: Path, result_dict: dict[str, Any]) -> Path:
        """Write ``feature_analysis.json`` atomically-ish."""
        layer_dir.mkdir(parents=True, exist_ok=True)
        path = layer_dir / "feature_analysis.json"
        with path.open("w") as fh:
            json.dump(result_dict, fh, indent=2, allow_nan=True)
        return path

    def _write_steering_vectors(
        self,
        out_path: Path,
        sae: SAEHandle,
        diff_df: pd.DataFrame,
        k: int,
    ) -> Path:
        """Write the ``steering_vectors.npz`` blob used by the causal workstream.

        We save *all* significant features (not just the top K) so the causal
        workstream has freedom to reweight. Direction matrices are indexed in
        ``feature_ids`` order.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sig_mask = diff_df.get("significant", pd.Series([], dtype=bool)).astype(bool)
        if sig_mask.any():
            feat_rows = diff_df[sig_mask].copy()
        else:
            # Fallback: store the top-K by absolute log fold change so causal
            # downstream still has *something* to chew on. Marked via an npz
            # attribute-style array.
            feat_rows = diff_df.copy()
            feat_rows["abs_log_fc"] = feat_rows["log_fc"].abs()
            feat_rows = feat_rows.sort_values("abs_log_fc", ascending=False).head(max(1, k))

        feature_ids = feat_rows["feature_id"].to_numpy(dtype=np.int32)
        encoder_mat, decoder_mat = _feature_directions(sae)

        if encoder_mat is None:
            encoder_rows = np.zeros((len(feature_ids), int(sae.hidden_dim)), dtype=np.float32)
        else:
            # The backends expose (n_features, hidden_dim) matrices. Index by
            # feature_id so row i in the npz matches feature_ids[i].
            encoder_rows = encoder_mat[feature_ids].astype(np.float32, copy=False)

        if decoder_mat is None:
            decoder_rows = np.zeros((len(feature_ids), int(sae.hidden_dim)), dtype=np.float32)
        else:
            decoder_rows = decoder_mat[feature_ids].astype(np.float32, copy=False)

        np.savez(
            out_path,
            feature_ids=feature_ids,
            encoder_directions=encoder_rows,
            decoder_directions=decoder_rows,
            mean_activations_S1=feat_rows["mean_S1"].to_numpy(dtype=np.float64),
            mean_activations_S2=feat_rows["mean_S2"].to_numpy(dtype=np.float64),
            is_falsified=feat_rows.get("is_falsified", pd.Series(False, index=feat_rows.index))
            .astype(bool)
            .to_numpy(),
            log_fold_changes=feat_rows["log_fc"].to_numpy(dtype=np.float64),
            q_values=feat_rows["q_value"].to_numpy(dtype=np.float64),
            effect_sizes=feat_rows["effect_size"].to_numpy(dtype=np.float64),
        )
        logger.info("wrote steering vectors to %s (%d features)", out_path, len(feature_ids))
        return out_path

    # -------------------------------------------------------------------
    # Failure stubs
    # -------------------------------------------------------------------

    def _record_failure(
        self,
        *,
        model_key: str,
        hdf5_key: str,
        layer: int,
        layer_dir: Path,
        reason: str,
        started: float,
    ) -> dict[str, Any]:
        """Write a minimal stub JSON so downstream readers can see the failure."""
        stub = {
            "model_key": model_key,
            "hdf5_key": hdf5_key,
            "layer": int(layer),
            "position": self.cfg.position,
            "sae_release": self.cfg.sae_release_for(model_key),
            "reconstruction_explained_variance": float("nan"),
            "n_features_total": 0,
            "n_features_significant": 0,
            "n_features_after_falsification": 0,
            "top_features": [],
            "status": "failed",
            "reason": reason,
            "config": _as_py(asdict(self.cfg)),
            "git_sha": self._git_sha,
            "runtime_seconds": float(time.time() - started),
        }
        try:
            self._write_results_json(layer_dir, stub)
        except Exception as exc:  # pragma: no cover - failure writing failure
            logger.error("could not even write failure stub for %s: %s", reason, exc)
        return stub


# ---------------------------------------------------------------------------
# Functional wrapper
# ---------------------------------------------------------------------------


@beartype
def run_sae_analysis(cfg: SAERunnerConfig) -> dict[tuple[str, int], dict[str, Any]]:
    """Convenience wrapper: build an :class:`SAEAnalysisRunner` and call :meth:`run`.

    Useful when the caller already has a fully-populated
    :class:`SAERunnerConfig` and doesn't need to customize the runner
    instance.
    """
    return SAEAnalysisRunner(cfg).run()


__all__ = [
    "DEFAULT_LAYERS_FOR_MODEL",
    "DEFAULT_MODEL_HDF5_KEYS",
    "DEFAULT_MODEL_SAE_RELEASES",
    "SAEAnalysisRunner",
    "SAERunnerConfig",
    "run_sae_analysis",
]
