"""Difficulty-detector analysis orchestrator.

This module owns the metacognitive monitoring pipeline. It coordinates
the four sub-analyses defined in this workstream:

1. Surprise-feature correlation (:mod:`s1s2.metacog.surprise`)
2. S1/S2 specificity test (:mod:`s1s2.metacog.difficulty`)
3. Confidently-wrong test (:mod:`s1s2.metacog.difficulty`)
4. Self-correction trajectory analysis (:mod:`s1s2.metacog.trajectory`)
   — for reasoning models only

It then packages the per-feature results into a
:class:`s1s2.metacog.gates.DifficultyDetectorResults` snapshot and
hands it to :func:`s1s2.metacog.gates.evaluate_gates` for the
pre-registered 4-gate go/no-go evaluation.

The orchestrator does NOT do interventions — that is the causal
workstream's job. It DOES expose a hook
(:meth:`DifficultyDetectorAnalysis.attach_causal`) that the causal
workstream can call to retroactively populate Gate 3.

Design notes
------------
- Stateless functions where possible. The class only holds the
  Hydra-derived config and a logger; analyses are functions that
  return immutable result objects.
- All randomness flows through a seeded ``np.random.Generator``.
- Activations are encoded through the SAE in batches; the SAE handle
  is whatever ``s1s2.sae.loaders.load_sae_for_model`` returns
  (real SAE, MockSAE fallback for tests, etc.).
- The pipeline writes one JSON file per (model, layer) under
  ``output_dir`` and one corpus-level JSON summarizing the gate
  decisions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from beartype import beartype

from s1s2.metacog.difficulty import (
    confidently_wrong_batch,
    difficulty_sensitive_features,
    s1s2_specificity_batch,
)
from s1s2.metacog.gates import (
    DifficultyDetectorResults,
    GateConfig,
    GateResult,
    evaluate_gates,
    gates_to_dict,
)
from s1s2.metacog.surprise import (
    aggregate_surprise,
    surprise_feature_correlation,
)
from s1s2.metacog.trajectory import (
    DEFAULT_MARKERS,
    difficulty_trajectory_means,
    parse_trace_corpus,
)
from s1s2.sae.differential import encode_batched
from s1s2.sae.loaders import SAEHandle, reconstruction_report
from s1s2.utils import io as ioh
from s1s2.utils.logging import get_logger
from s1s2.utils.seed import set_global_seed

logger = get_logger("s1s2.metacog")


# ---------------------------------------------------------------------------
# Configuration / dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MetacogConfig:
    """Knobs for one :class:`DifficultyDetectorAnalysis` run.

    Mirrors ``configs/metacog.yaml``. Kept as a plain dataclass so the
    orchestrator stays Hydra-independent and unit-testable.
    """

    activations_path: str = "data/activations/main.h5"
    sae_results_dir: str = "results/sae"
    output_dir: str = "results/metacog"
    seed: int = 0
    layers: tuple[int, ...] | None = None  # None => probe layer per model
    surprise_aggregation: str = "mean_full"
    fdr_q: float = 0.05
    rho_threshold: float = 0.3
    specificity_auc_threshold: float = 0.65
    matched_only: bool = True
    min_features_with_rho_gt: int = 20
    min_delta_p_correct: float = 0.15
    confidently_wrong_threshold: float = -0.5
    self_correction_markers: tuple[str, ...] = DEFAULT_MARKERS
    self_correction_min_post_chars: int = 30
    sae_min_explained_variance: float = 0.5


@dataclass
class PerFeatureResults:
    """Per-(model, layer) feature-level summary.

    Carries the per-feature dataframes from each sub-analysis plus the
    layer-level aggregate metrics that the gates consume. The
    :meth:`to_json` method returns a serializable dict — the
    :class:`DifficultyDetectorAnalysis` writes one JSON per layer.
    """

    model_key: str
    layer: int
    n_problems: int
    n_features: int
    surprise_df: pd.DataFrame
    specificity_df: pd.DataFrame
    confidently_wrong_df: pd.DataFrame
    combined_df: pd.DataFrame
    trajectory_metrics: dict[str, Any]
    sae_explained_variance: float
    sae_is_poor_fit: bool
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable view, dropping huge raw fields."""
        # Cap the per-feature dataframes at the top-50 candidates so the
        # JSON stays small. The full pickle of the dataframes is dumped
        # alongside in a separate ``.parquet`` file when requested.
        top = self.combined_df.head(50).copy()
        # Convert numpy ints/bools to Python types for JSON.
        records = top.to_dict(orient="records")
        clean = []
        for rec in records:
            clean_rec = {}
            for k, v in rec.items():
                if isinstance(v, (np.integer,)):
                    clean_rec[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean_rec[k] = float(v)
                elif isinstance(v, (np.bool_,)):
                    clean_rec[k] = bool(v)
                else:
                    clean_rec[k] = v
            clean.append(clean_rec)
        return {
            "model_key": self.model_key,
            "layer": int(self.layer),
            "n_problems": int(self.n_problems),
            "n_features": int(self.n_features),
            "n_difficulty_sensitive": (
                int(self.surprise_df["is_difficulty_sensitive"].sum())
                if "is_difficulty_sensitive" in self.surprise_df.columns
                else 0
            ),
            "n_passing_specificity": (
                int(self.specificity_df["passes_specificity"].sum())
                if "passes_specificity" in self.specificity_df.columns
                else 0
            ),
            "n_metacognitive": (
                int(self.confidently_wrong_df["is_metacognitive"].sum())
                if (
                    not self.confidently_wrong_df.empty
                    and "is_metacognitive" in self.confidently_wrong_df.columns
                )
                else 0
            ),
            "trajectory_metrics": self.trajectory_metrics,
            "sae_explained_variance": float(self.sae_explained_variance),
            "sae_is_poor_fit": bool(self.sae_is_poor_fit),
            "top_features": clean,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class DifficultyDetectorAnalysis:
    """Run the metacognitive monitoring pipeline for one or more (model, layer) cells.

    Typical usage::

        cfg = MetacogConfig(activations_path="data/activations/smoke.h5")
        analysis = DifficultyDetectorAnalysis(cfg)
        per_layer = analysis.run(model_key="r1-distill-llama-8b",
                                 hdf5_key="deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
                                 layer=20,
                                 sae=my_sae)
        gate_results = analysis.evaluate_gates(per_layer)

    The class deliberately separates the per-layer pipeline (cheap,
    deterministic) from the gate evaluation (a pure function over the
    aggregated state) so unit tests can exercise either half in
    isolation.
    """

    def __init__(self, cfg: MetacogConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.seed))
        set_global_seed(int(cfg.seed), deterministic_torch=False)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- per-layer ----------------------------------------------------

    @beartype
    def run(
        self,
        *,
        model_key: str,
        hdf5_key: str,
        layer: int,
        sae: SAEHandle,
        residual: np.ndarray | None = None,
        feature_activations: np.ndarray | None = None,
    ) -> PerFeatureResults:
        """Execute the metacog pipeline for one (model, layer) cell.

        Loads activations, encodes through the SAE, computes
        surprise correlations, and runs the specificity and
        confidently-wrong tests on the candidate set.

        Parameters
        ----------
        model_key
            Project model key (e.g. ``"r1-distill-llama-8b"``).
        hdf5_key
            HDF5 model key inside the activation cache.
        layer
            Transformer block index.
        sae
            A SAE handle (real or mock) to encode the residuals through.
        residual
            Optional pre-loaded residual matrix of shape
            ``(n_problems, hidden_dim)``. If ``None`` we read it from
            the activation cache at the ``P0`` slice.
        feature_activations
            Optional pre-encoded feature matrix of shape
            ``(n_problems, n_features)``. If supplied, ``sae`` is only
            used for the reconstruction-fidelity check (and the SAE
            handle's metadata is taken at face value).
        """

        notes = ""
        with ioh.open_activations(self.cfg.activations_path) as f:
            problems = ioh.load_problem_metadata(f)
            behavior = ioh.get_behavior(f, hdf5_key)
            generations = ioh.get_generations(f, hdf5_key)
            position_labels = ioh.position_labels(f, hdf5_key)

            # Load residual if not supplied
            if residual is None:
                residual = ioh.get_residual(f, hdf5_key, layer=layer, position="P0")
            assert residual is not None
            n_problems = int(residual.shape[0])

            # Surprise: by_position is always present; full ragged trace
            # is optional and read directly off the HDF5 path to avoid the
            # known beartype return-type mismatch on
            # ``get_token_surprises(..., by_position_only=False)``.
            by_position = np.asarray(ioh.get_token_surprises(f, hdf5_key, by_position_only=True))
            full_offsets: np.ndarray | None = None
            full_values: np.ndarray | None = None
            ts_group = f[f"/models/{hdf5_key}/token_surprises"]
            if "full_trace_offsets" in ts_group and "full_trace_values" in ts_group:
                full_offsets = ts_group["full_trace_offsets"][:]
                full_values = ts_group["full_trace_values"][:]

        # SAE fidelity (silent on success, warn on failure).
        rep = reconstruction_report(
            sae,
            np.asarray(residual, dtype=np.float32),
            min_explained_variance=self.cfg.sae_min_explained_variance,
            rng=self.rng,
        )
        sae_ev = float(rep.explained_variance)
        sae_poor = bool(rep.is_poor_fit)
        if sae_poor:
            notes += "SAE poor fit; downstream feature analyses may be unreliable. "

        # Encode through the SAE
        if feature_activations is None:
            feature_activations = encode_batched(sae, np.asarray(residual, dtype=np.float32))
        feature_activations = np.asarray(feature_activations, dtype=np.float32)
        n_features = int(feature_activations.shape[1])

        # ----- 1. Surprise-feature correlation -----
        surprise_vec = aggregate_surprise(
            by_position=by_position.astype(np.float32),
            position_labels=position_labels,
            full_trace_offsets=full_offsets,
            full_trace_values=full_values,
            method=self.cfg.surprise_aggregation,
        )
        sc_result = surprise_feature_correlation(
            feature_activations=feature_activations,
            surprise=surprise_vec,
            rho_threshold=self.cfg.rho_threshold,
            fdr_q=self.cfg.fdr_q,
            layer=int(layer),
            model_key=str(model_key),
            aggregation=self.cfg.surprise_aggregation,
        )
        surprise_df = sc_result.df

        # ----- 2. S1/S2 specificity on candidates -----
        candidate_ids = sc_result.difficulty_sensitive_ids.tolist()
        if not candidate_ids:
            # Even with no candidates, we still want a row-shaped df
            specificity_df = pd.DataFrame(
                columns=[
                    "feature_id",
                    "auc",
                    "u_stat",
                    "p_value",
                    "q_value",
                    "rank_biserial",
                    "n_S1",
                    "n_S2",
                    "passes_specificity",
                    "notes",
                ]
            )
        else:
            specificity_df = s1s2_specificity_batch(
                feature_activations=feature_activations,
                conflict=problems["conflict"].astype(bool),
                matched_pair_id=problems["matched_pair_id"],
                feature_ids=candidate_ids,
                auc_threshold=self.cfg.specificity_auc_threshold,
                fdr_q=self.cfg.fdr_q,
                matched_only=self.cfg.matched_only,
            )

        # ----- 3. Confidently-wrong test on candidates -----
        # Synthesize an output-confidence proxy: the negative mean
        # surprise on the answer span. Higher value <=> more confident.
        # We don't have answer log-probs cached separately, so we use
        # this proxy. The metacognition test cares only about the *sign*
        # of the difference, so monotone proxies are fine.
        if "by_position:P2" in [f"by_position:{lbl}" for lbl in position_labels]:
            p2_idx = position_labels.index("P2")
            answer_surprise = by_position[:, p2_idx].astype(np.float32)
        else:
            answer_surprise = surprise_vec
        # Confidence = -surprise. NaNs and infs go to the floor (low confidence).
        output_confidence = -answer_surprise
        output_confidence = np.nan_to_num(output_confidence, nan=-1e6, posinf=-1e6, neginf=-1e6)
        if candidate_ids:
            cw_df = confidently_wrong_batch(
                feature_activations=feature_activations,
                is_correct=behavior["correct"].astype(bool),
                output_confidence=output_confidence,
                feature_ids=candidate_ids,
                threshold_confidence=self.cfg.confidently_wrong_threshold,
            )
        else:
            cw_df = pd.DataFrame()

        combined = (
            difficulty_sensitive_features(
                surprise_df=surprise_df.loc[surprise_df["feature_id"].isin(candidate_ids)],
                specificity_df=specificity_df,
                confidently_wrong_df=cw_df,
                rho_threshold=self.cfg.rho_threshold,
                auc_threshold=self.cfg.specificity_auc_threshold,
            )
            if candidate_ids
            else pd.DataFrame()
        )

        # ----- 4. Self-correction trajectory (reasoning models only) -----
        thinking_texts = list(generations.get("thinking_text", []))
        # decode bytes to str if needed
        thinking_texts = [
            t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else str(t)
            for t in thinking_texts
        ]
        any_thinking = any(len(t) > 0 for t in thinking_texts)
        if any_thinking and candidate_ids:
            parses = parse_trace_corpus(
                thinking_texts,
                markers=self.cfg.self_correction_markers,
                min_post_chars=self.cfg.self_correction_min_post_chars,
            )
            # Trajectory means: pick the top candidate feature for the analysis.
            top_id = (
                int(combined.iloc[0]["feature_id"]) if not combined.empty else int(candidate_ids[0])
            )
            traj_metrics = difficulty_trajectory_means(
                feature_activations=feature_activations[:, top_id : top_id + 1],
                parse_results=parses,
            )
            traj_metrics["analyzed_feature_id"] = int(top_id)
            traj_metrics["n_traces"] = len(parses)
        else:
            traj_metrics = {
                "n_self_correcting": 0,
                "mean_pre": 0.0,
                "mean_post": 0.0,
                "delta_post_minus_pre": 0.0,
                "n_baseline": 0,
                "baseline_mean": 0.0,
                "analyzed_feature_id": -1,
                "n_traces": len(thinking_texts),
                "skipped_reason": ("no thinking traces" if not any_thinking else "no candidates"),
            }

        return PerFeatureResults(
            model_key=str(model_key),
            layer=int(layer),
            n_problems=int(n_problems),
            n_features=int(n_features),
            surprise_df=surprise_df,
            specificity_df=specificity_df,
            confidently_wrong_df=cw_df,
            combined_df=combined,
            trajectory_metrics=traj_metrics,
            sae_explained_variance=sae_ev,
            sae_is_poor_fit=sae_poor,
            notes=notes,
        )

    # ---- cross-layer / cross-model gate evaluation -------------------

    @beartype
    def evaluate_gates(
        self,
        per_layer: PerFeatureResults | list[PerFeatureResults],
        causal_delta_p_correct: float | None = None,
    ) -> tuple[DifficultyDetectorResults, list[GateResult]]:
        """Run the 4-gate framework on per-layer results.

        ``per_layer`` may be a single :class:`PerFeatureResults` (one
        model+layer) or a list (e.g. all layers of one model). When a
        list, the gates are evaluated on the *aggregated* statistics:
        n_difficulty_sensitive sums over layers, max specificity AUC is
        the global max, etc. This implements the "best layer wins" rule
        which matches the project's pre-registered analysis plan.
        """
        if isinstance(per_layer, PerFeatureResults):
            layers = [per_layer]
        else:
            layers = list(per_layer)

        if not layers:
            return (
                DifficultyDetectorResults(
                    infrastructure_ok=False,
                    notes="no per-layer results passed to evaluate_gates",
                ),
                evaluate_gates(
                    DifficultyDetectorResults(infrastructure_ok=False),
                    self._gate_config(),
                ),
            )

        n_diff = sum(
            int(p.surprise_df["is_difficulty_sensitive"].sum())
            for p in layers
            if "is_difficulty_sensitive" in p.surprise_df.columns
        )
        rho_dist = []
        for p in layers:
            if "rho" in p.surprise_df.columns:
                rho_dist.extend(p.surprise_df["rho"].tolist())

        max_auc = 0.5
        n_pass_spec = 0
        for p in layers:
            if "auc" in p.specificity_df.columns and len(p.specificity_df) > 0:
                max_auc = max(max_auc, float(p.specificity_df["auc"].max()))
                n_pass_spec += int(p.specificity_df["passes_specificity"].sum())

        meta_scores: list[float] = []
        n_meta = 0
        for p in layers:
            if (
                not p.confidently_wrong_df.empty
                and "metacognition_score" in p.confidently_wrong_df.columns
            ):
                meta_scores.extend(p.confidently_wrong_df["metacognition_score"].tolist())
                n_meta += int(p.confidently_wrong_df["is_metacognitive"].sum())

        # Gate 0 = "did the pipeline run?" — a poor SAE fit is NOT an
        # infrastructure failure (the code succeeded), it's a downstream
        # caveat that shows up in the per-layer notes and on the gate
        # snapshot. Infra fails only when we have zero layers.
        infra_ok = bool(layers)
        any_poor_fit = any(p.sae_is_poor_fit for p in layers)
        combined_notes = "; ".join(p.notes for p in layers if p.notes)
        if any_poor_fit and "poor fit" not in combined_notes.lower():
            combined_notes = (combined_notes + " SAE poor fit on at least one layer.").strip()

        snapshot = DifficultyDetectorResults(
            model_key=layers[0].model_key,
            layer=layers[0].layer if len(layers) == 1 else -1,
            n_difficulty_sensitive_features=int(n_diff),
            rho_distribution=[float(r) for r in rho_dist],
            max_specificity_auc=float(max_auc),
            n_features_passing_specificity=int(n_pass_spec),
            metacognition_scores=[float(s) for s in meta_scores],
            n_metacognitive_features=int(n_meta),
            causal_delta_p_correct=causal_delta_p_correct,
            infrastructure_ok=bool(infra_ok),
            notes=combined_notes,
        )
        gate_results = evaluate_gates(snapshot, self._gate_config())
        return snapshot, gate_results

    def _gate_config(self) -> GateConfig:
        return GateConfig(
            min_features_with_rho_gt=int(self.cfg.min_features_with_rho_gt),
            rho_threshold=float(self.cfg.rho_threshold),
            fdr_q=float(self.cfg.fdr_q),
            min_specificity_auc=float(self.cfg.specificity_auc_threshold),
            difficulty_matched_only=bool(self.cfg.matched_only),
            min_delta_p_correct=float(self.cfg.min_delta_p_correct),
        )

    @beartype
    def attach_causal(
        self,
        snapshot: DifficultyDetectorResults,
        delta_p_correct: float,
    ) -> tuple[DifficultyDetectorResults, list[GateResult]]:
        """Re-evaluate gates after the causal workstream lands its number."""
        snapshot.causal_delta_p_correct = float(delta_p_correct)
        return snapshot, evaluate_gates(snapshot, self._gate_config())

    # ---- writing -----------------------------------------------------

    @beartype
    def write_per_layer(self, per_layer: PerFeatureResults) -> Path:
        """Dump a per-(model, layer) JSON to ``cfg.output_dir``."""
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{per_layer.model_key}_metacog_layer_{per_layer.layer:02d}.json"
        with path.open("w") as fh:
            json.dump(per_layer.to_json(), fh, indent=2)
        logger.info("wrote %s", path)
        return path

    @beartype
    def write_summary(
        self,
        snapshot: DifficultyDetectorResults,
        gate_results: list[GateResult],
        suffix: str = "summary",
    ) -> Path:
        """Dump a per-model gate summary JSON."""
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{snapshot.model_key}_{suffix}.json"
        payload = {
            "snapshot": asdict(snapshot),
            "gates": gates_to_dict(gate_results),
            "config": {
                "fdr_q": self.cfg.fdr_q,
                "rho_threshold": self.cfg.rho_threshold,
                "specificity_auc_threshold": self.cfg.specificity_auc_threshold,
                "min_features_with_rho_gt": self.cfg.min_features_with_rho_gt,
                "min_delta_p_correct": self.cfg.min_delta_p_correct,
                "surprise_aggregation": self.cfg.surprise_aggregation,
                "matched_only": self.cfg.matched_only,
                "seed": self.cfg.seed,
            },
        }
        with path.open("w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("wrote %s", path)
        return path


__all__ = [
    "DifficultyDetectorAnalysis",
    "MetacogConfig",
    "PerFeatureResults",
]
