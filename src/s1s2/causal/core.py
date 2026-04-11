"""Causal experiment runner — orchestrates steering/ablation sweeps.

Given a model, a tokenizer, a benchmark split, and a list of (layer,
feature_id, direction) triples from the SAE workstream, this module runs
the dose-response experiment (steering over a grid of ``alpha`` values
plus a random-direction control) and optionally the projection-ablation
experiment. Results are written per (model, layer, feature) to the
configured output directory.

The main entry point is :class:`CausalExperimentRunner`. It takes a
:class:`CausalRunnerConfig` at construction time and exposes a ``run_one``
method for a single (layer, feature) cell and a ``run`` method that
iterates over every configured cell.

Pipeline per cell
-----------------

1. Load the feature direction (from the SAE results JSON + the SAE
   handle) and compute its unit norm.
2. For each item in the benchmark split:
   a. Measure baseline P(correct) (no hook).
   b. For each alpha in the grid: measure P(correct) under
      :class:`SteeringHook`.
   c. For each alpha and each random seed in the random-control grid:
      measure P(correct) under :class:`SteeringHook` with a randomly
      sampled unit direction.
3. Aggregate via :func:`dose_response.build_curve`.
4. Run :class:`AblationHook` once for the "projection-ablation" point
   and record the drop in accuracy on conflict items.
5. Optionally run capability preservation against MMLU / HellaSwag
   fixtures.
6. Serialize a JSON record to ``output_dir``.

A "scoring function" is abstracted out — the runner doesn't care how you
decide whether the model answered correctly, it just calls
``score_fn(model, tokenizer, item) -> bool`` inside the hook context.
This keeps the runner decoupled from the exact prompting / generation
protocol, which varies across reasoning and non-reasoning models.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from beartype import beartype

from s1s2.benchmark.loader import BenchmarkItem
from s1s2.causal.ablation import AblationHook
from s1s2.causal.capability import (
    CapabilityComparison,
    CapabilityItem,
    compare_capability,
    load_capability_jsonl,
    score_capability,
)
from s1s2.causal.dose_response import (
    DoseResponseCurve,
    build_curve,
    is_canonical_s2_signature,
)
from s1s2.causal.steering import SteeringHook, random_unit_direction
from s1s2.utils.logging import get_logger
from s1s2.utils.seed import set_global_seed

logger = get_logger("s1s2.causal")


# ---------------------------------------------------------------------------
# Scoring function signature
# ---------------------------------------------------------------------------

#: A scoring function takes ``(model, tokenizer, item)`` and returns True if
#: the model answered correctly. It is called inside the hook context; the
#: runner does NOT wrap it in any additional state management, so the scoring
#: function must leave the model in the same state it found it.
ScoreFn = Callable[[Any, Any, BenchmarkItem], bool]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RandomControlConfig:
    n_directions: int = 5
    seed: int = 1


@dataclass(frozen=True, slots=True)
class CapabilityEvalConfig:
    mmlu_subset_path: str | None = None
    hellaswag_subset_path: str | None = None
    n_examples_per_eval: int = 100
    max_acceptable_drop_pp: float = 2.0


@dataclass(frozen=True, slots=True)
class CausalRunnerConfig:
    """Knobs that govern one causal experiment run.

    Kept as a plain dataclass so the runner is trivially unit-testable
    without Hydra. The CLI in :mod:`s1s2.causal.cli` builds one of these
    from the Hydra config.
    """

    alphas: tuple[float, ...] = (-5.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 5.0)
    top_features_per_layer: int = 3
    random_control: RandomControlConfig = field(default_factory=RandomControlConfig)
    capability_eval: CapabilityEvalConfig | None = None
    seed: int = 0
    max_new_tokens: int = 128
    n_bootstrap: int = 1000
    #: If True, save intermediate per-alpha vectors in addition to the aggregated
    #: curve. Useful for debugging, expensive on disk for full runs.
    save_raw_vectors: bool = False


# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One (model, layer, feature) with its unit-norm direction.

    Produced by :func:`load_feature_specs` and consumed by
    :class:`CausalExperimentRunner`. The direction is pre-computed so
    we don't have to keep the SAE handle around during the experiment.
    """

    model_key: str
    layer: int
    feature_id: int
    direction: np.ndarray  # shape (hidden_dim,), unit norm
    effect_size: float = 0.0
    q_value: float = 1.0
    source: str = "sae_differential"


@beartype
def load_feature_specs(
    sae_results_dir: str | Path,
    model_key: str,
    *,
    top_per_layer: int = 3,
    hidden_dim: int | None = None,
) -> list[FeatureSpec]:
    """Load candidate feature specs from the SAE workstream's results directory.

    Expected layout::

        {sae_results_dir}/{model_key}_layer{NN}_features.json

    with the schema::

        {
            "model": "...",
            "layer": 16,
            "top_features": [
                {"feature_id": 1234, "effect_size": 0.73, "q_value": 0.0001,
                 "direction": [...hidden_dim floats...]},
                ...
            ]
        }

    If no such files exist (e.g. the SAE workstream hasn't produced them
    yet in a smoke run), returns an empty list. Callers can then feed
    synthetic specs in directly via :meth:`CausalExperimentRunner.run_one`.

    ``hidden_dim``, if provided, is validated against every loaded
    direction so a corrupt file fails loudly rather than silently.
    """
    root = Path(sae_results_dir)
    if not root.exists():
        logger.info("SAE results dir %s does not exist — no features loaded.", root)
        return []

    pattern = re.compile(rf"^{re.escape(model_key)}_layer(\d+)_features\.json$")
    specs: list[FeatureSpec] = []
    for entry in sorted(root.iterdir()):
        m = pattern.match(entry.name)
        if m is None:
            continue
        layer = int(m.group(1))
        try:
            payload = json.loads(entry.read_text())
        except Exception as exc:
            logger.warning("failed to read %s: %s", entry, exc)
            continue
        top = payload.get("top_features", [])
        for _rank, feat in enumerate(top[:top_per_layer]):
            direction = np.asarray(feat.get("direction", []), dtype=np.float32)
            if direction.size == 0:
                logger.warning(
                    "feature %s at %s has empty direction — skipping",
                    feat.get("feature_id"),
                    entry,
                )
                continue
            if hidden_dim is not None and direction.shape[0] != hidden_dim:
                raise ValueError(
                    f"{entry}: direction dim {direction.shape[0]} != "
                    f"expected hidden_dim {hidden_dim}"
                )
            norm = float(np.linalg.norm(direction))
            if norm == 0.0:
                logger.warning("zero-norm direction at %s feature %s", entry, feat)
                continue
            specs.append(
                FeatureSpec(
                    model_key=model_key,
                    layer=layer,
                    feature_id=int(feat["feature_id"]),
                    direction=direction / norm,
                    effect_size=float(feat.get("effect_size", 0.0)),
                    q_value=float(feat.get("q_value", 1.0)),
                    source=str(payload.get("source", "sae_differential")),
                )
            )
    logger.info("loaded %d feature specs for %s from %s", len(specs), model_key, root)
    return specs


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    """Result of the projection-ablation experiment on one feature."""

    baseline_p_correct_conflict: float
    baseline_p_correct_no_conflict: float
    ablated_p_correct_conflict: float
    ablated_p_correct_no_conflict: float

    @property
    def conflict_drop(self) -> float:
        return float(self.baseline_p_correct_conflict - self.ablated_p_correct_conflict)

    @property
    def no_conflict_drop(self) -> float:
        return float(self.baseline_p_correct_no_conflict - self.ablated_p_correct_no_conflict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_p_correct_conflict": float(self.baseline_p_correct_conflict),
            "baseline_p_correct_no_conflict": float(self.baseline_p_correct_no_conflict),
            "ablated_p_correct_conflict": float(self.ablated_p_correct_conflict),
            "ablated_p_correct_no_conflict": float(self.ablated_p_correct_no_conflict),
            "conflict_drop": self.conflict_drop,
            "no_conflict_drop": self.no_conflict_drop,
        }


@dataclass
class CausalCellResult:
    """Complete results for one (model, layer, feature) cell."""

    model: str
    layer: int
    feature_id: int
    curve: DoseResponseCurve
    ablation: AblationResult | None
    canonical_s2: bool
    capability: list[CapabilityComparison]
    config: dict[str, Any]
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "layer": int(self.layer),
            "feature_id": int(self.feature_id),
            "curve": self.curve.to_dict(),
            "ablation": self.ablation.to_dict() if self.ablation is not None else None,
            "canonical_s2": bool(self.canonical_s2),
            "capability": [c.to_dict() for c in self.capability],
            "config": self.config,
            "elapsed_s": float(self.elapsed_s),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class CausalExperimentRunner:
    """Run dose-response + ablation experiments on a set of feature specs.

    Stateless w.r.t. to an in-memory model: every ``run_one`` call is
    given the model / tokenizer / scoring function explicitly. That way
    the same runner instance can be used to sweep across models without
    tying its lifetime to a particular HF handle.
    """

    def __init__(self, config: CausalRunnerConfig) -> None:
        self.config = config

    # -- Direct benchmark iteration ----------------------------------------

    @beartype
    def _score_under_hook(
        self,
        model: Any,
        tokenizer: Any,
        items: list[BenchmarkItem],
        score_fn: ScoreFn,
        hook: Any | None,
    ) -> np.ndarray:
        """Return a 0/1 correctness vector after scoring ``items`` under ``hook``.

        ``hook`` is any context manager. If ``None``, items are scored
        with no hook attached (the baseline path).
        """
        if hook is None:
            return np.asarray(
                [bool(score_fn(model, tokenizer, it)) for it in items],
                dtype=np.float32,
            )
        with hook:
            return np.asarray(
                [bool(score_fn(model, tokenizer, it)) for it in items],
                dtype=np.float32,
            )

    @beartype
    def _split_items(
        self, items: list[BenchmarkItem]
    ) -> tuple[list[BenchmarkItem], list[BenchmarkItem]]:
        """Partition a benchmark into conflict and no-conflict items."""
        conflict = [it for it in items if it.conflict]
        no_conflict = [it for it in items if not it.conflict]
        return conflict, no_conflict

    # -- One (layer, feature) cell -----------------------------------------

    @beartype
    def run_one(
        self,
        *,
        model: Any,
        tokenizer: Any,
        feature: FeatureSpec,
        benchmark: list[BenchmarkItem],
        score_fn: ScoreFn,
        capability_suites: list[tuple[str, list[CapabilityItem]]] | None = None,
    ) -> CausalCellResult:
        """Run the full causal pipeline on one (model, layer, feature) cell.

        Parameters
        ----------
        model
            The HuggingFace model, already moved to the right device.
        tokenizer
            Matching tokenizer.
        feature
            The feature to intervene on.
        benchmark
            Items to evaluate. Mix of conflict and no-conflict items; the
            runner will split them internally.
        score_fn
            Callable ``(model, tokenizer, item) -> bool`` returning True
            if the model answered correctly. Must not leak model state
            outside the hook context.
        capability_suites
            Optional list of ``(benchmark_name, items)`` pairs for the
            capability-preservation check. If ``None`` and the runner's
            config has capability paths set, the runner loads them.
        """
        started = time.time()
        set_global_seed(self.config.seed, deterministic_torch=False)

        conflict_items, no_conflict_items = self._split_items(benchmark)
        logger.info(
            "[%s/L%d/F%d] conflict=%d no_conflict=%d alphas=%d",
            feature.model_key,
            feature.layer,
            feature.feature_id,
            len(conflict_items),
            len(no_conflict_items),
            len(self.config.alphas),
        )

        # Dose-response sweep.
        conflict_by_alpha: dict[float, np.ndarray] = {}
        no_conflict_by_alpha: dict[float, np.ndarray] = {}
        random_by_alpha_seed: dict[float, dict[int, np.ndarray]] = {}

        direction = torch.from_numpy(feature.direction.astype(np.float32))

        for alpha in self.config.alphas:
            if alpha == 0.0:
                # alpha=0 is the baseline. Short-circuit: no hook at all
                # so we don't pay the register/remove overhead.
                conflict_by_alpha[alpha] = self._score_under_hook(
                    model, tokenizer, conflict_items, score_fn, hook=None
                )
                no_conflict_by_alpha[alpha] = self._score_under_hook(
                    model, tokenizer, no_conflict_items, score_fn, hook=None
                )
            else:
                sh = SteeringHook(model, layer=feature.layer, direction=direction, alpha=alpha)
                conflict_by_alpha[alpha] = self._score_under_hook(
                    model, tokenizer, conflict_items, score_fn, hook=sh
                )
                sh2 = SteeringHook(model, layer=feature.layer, direction=direction, alpha=alpha)
                no_conflict_by_alpha[alpha] = self._score_under_hook(
                    model, tokenizer, no_conflict_items, score_fn, hook=sh2
                )

            # Random-direction control: N independent seeds.
            random_by_alpha_seed[alpha] = {}
            hidden_dim = direction.shape[0]
            for i in range(self.config.random_control.n_directions):
                rnd_seed = (
                    self.config.random_control.seed
                    + i * 101
                    + int((alpha * 1000) if alpha != 0 else 0)
                )
                rnd_dir = random_unit_direction(int(hidden_dim), seed=rnd_seed)
                rh = SteeringHook(model, layer=feature.layer, direction=rnd_dir, alpha=alpha)
                random_by_alpha_seed[alpha][i] = self._score_under_hook(
                    model, tokenizer, conflict_items, score_fn, hook=rh
                )

        curve = build_curve(
            model=feature.model_key,
            layer=feature.layer,
            feature_id=feature.feature_id,
            alphas=list(self.config.alphas),
            conflict_correct_by_alpha=conflict_by_alpha,
            no_conflict_correct_by_alpha=no_conflict_by_alpha,
            random_correct_by_alpha_seed=random_by_alpha_seed,
            n_bootstrap=self.config.n_bootstrap,
            seed=self.config.seed,
        )

        # Projection ablation.
        ablation_result: AblationResult | None = None
        try:
            base_c = conflict_by_alpha.get(0.0)
            base_nc = no_conflict_by_alpha.get(0.0)
            if base_c is None:
                base_c = self._score_under_hook(
                    model, tokenizer, conflict_items, score_fn, hook=None
                )
            if base_nc is None:
                base_nc = self._score_under_hook(
                    model, tokenizer, no_conflict_items, score_fn, hook=None
                )
            ah_c = AblationHook(model, layer=feature.layer, direction=direction)
            abl_c = self._score_under_hook(model, tokenizer, conflict_items, score_fn, hook=ah_c)
            ah_nc = AblationHook(model, layer=feature.layer, direction=direction)
            abl_nc = self._score_under_hook(
                model, tokenizer, no_conflict_items, score_fn, hook=ah_nc
            )
            ablation_result = AblationResult(
                baseline_p_correct_conflict=float(np.mean(base_c)) if base_c.size else float("nan"),
                baseline_p_correct_no_conflict=(
                    float(np.mean(base_nc)) if base_nc.size else float("nan")
                ),
                ablated_p_correct_conflict=float(np.mean(abl_c)) if abl_c.size else float("nan"),
                ablated_p_correct_no_conflict=(
                    float(np.mean(abl_nc)) if abl_nc.size else float("nan")
                ),
            )
        except Exception as exc:
            logger.warning("ablation step failed: %s", exc)

        canonical = is_canonical_s2_signature(curve)

        # Capability preservation.
        capability_outs: list[CapabilityComparison] = []
        if capability_suites is None and self.config.capability_eval is not None:
            capability_suites = self._load_capability_suites()
        if capability_suites:
            capability_outs = self._run_capability_eval(
                model=model,
                tokenizer=tokenizer,
                feature=feature,
                suites=capability_suites,
            )

        elapsed = time.time() - started
        return CausalCellResult(
            model=feature.model_key,
            layer=feature.layer,
            feature_id=feature.feature_id,
            curve=curve,
            ablation=ablation_result,
            canonical_s2=bool(canonical),
            capability=capability_outs,
            config=self._config_snapshot(),
            elapsed_s=float(elapsed),
        )

    # -- Capability suites -------------------------------------------------

    @beartype
    def _load_capability_suites(
        self,
    ) -> list[tuple[str, list[CapabilityItem]]]:
        """Load MMLU / HellaSwag subsets from the paths in config."""
        ce = self.config.capability_eval
        if ce is None:
            return []
        suites: list[tuple[str, list[CapabilityItem]]] = []
        if ce.mmlu_subset_path and Path(ce.mmlu_subset_path).exists():
            items = load_capability_jsonl(ce.mmlu_subset_path)[: ce.n_examples_per_eval]
            suites.append(("mmlu", items))
        if ce.hellaswag_subset_path and Path(ce.hellaswag_subset_path).exists():
            items = load_capability_jsonl(ce.hellaswag_subset_path)[: ce.n_examples_per_eval]
            suites.append(("hellaswag", items))
        return suites

    @beartype
    def _run_capability_eval(
        self,
        *,
        model: Any,
        tokenizer: Any,
        feature: FeatureSpec,
        suites: list[tuple[str, list[CapabilityItem]]],
    ) -> list[CapabilityComparison]:
        """Score the MMLU/HellaSwag subsets under baseline and under a +1 steer.

        The intervention we test is the "target" alpha = +1 (the default
        midpoint of the dose-response grid). Readers should interpret the
        resulting drop as "what does the canonical S2-push cost on general
        capabilities?", not as a full grid-to-grid sweep.
        """
        ce = self.config.capability_eval
        if ce is None:
            return []
        direction = torch.from_numpy(feature.direction.astype(np.float32))
        out: list[CapabilityComparison] = []
        for name, items in suites:
            baseline = score_capability(model, tokenizer, items, benchmark_name=name)
            with SteeringHook(model, layer=feature.layer, direction=direction, alpha=1.0):
                intervention = score_capability(model, tokenizer, items, benchmark_name=name)
            cmp = compare_capability(
                baseline,
                intervention,
                max_acceptable_drop_pp=ce.max_acceptable_drop_pp,
            )
            if cmp.exceeded_max_drop:
                logger.warning(
                    "capability drop on %s: %.2f pp (baseline %.3f -> %.3f); "
                    "steering compromises capability",
                    cmp.benchmark,
                    -cmp.delta_pp,
                    cmp.baseline_accuracy,
                    cmp.intervention_accuracy,
                )
            out.append(cmp)
        return out

    # -- Utilities ---------------------------------------------------------

    def _config_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the config for provenance."""
        ce = self.config.capability_eval
        return {
            "alphas": list(self.config.alphas),
            "top_features_per_layer": int(self.config.top_features_per_layer),
            "random_control": {
                "n_directions": int(self.config.random_control.n_directions),
                "seed": int(self.config.random_control.seed),
            },
            "capability_eval": (
                {
                    "mmlu_subset_path": (
                        str(ce.mmlu_subset_path) if ce and ce.mmlu_subset_path else None
                    ),
                    "hellaswag_subset_path": (
                        str(ce.hellaswag_subset_path) if ce and ce.hellaswag_subset_path else None
                    ),
                    "n_examples_per_eval": int(ce.n_examples_per_eval) if ce else 0,
                    "max_acceptable_drop_pp": float(ce.max_acceptable_drop_pp) if ce else 0.0,
                }
                if ce is not None
                else None
            ),
            "seed": int(self.config.seed),
            "max_new_tokens": int(self.config.max_new_tokens),
            "n_bootstrap": int(self.config.n_bootstrap),
        }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


@beartype
def save_cell_result(result: CausalCellResult, output_dir: str | Path) -> Path:
    """Write one :class:`CausalCellResult` as JSON."""
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    fname = f"{result.model}_layer{result.layer:02d}_feature{result.feature_id:06d}.json"
    path = d / fname
    with path.open("w") as fh:
        json.dump(result.to_dict(), fh, indent=2, allow_nan=True)
    return path


@beartype
def config_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of a config dict for provenance."""
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


__all__ = [
    "AblationResult",
    "CapabilityEvalConfig",
    "CausalCellResult",
    "CausalExperimentRunner",
    "CausalRunnerConfig",
    "FeatureSpec",
    "RandomControlConfig",
    "ScoreFn",
    "config_hash",
    "load_feature_specs",
    "save_cell_result",
]
