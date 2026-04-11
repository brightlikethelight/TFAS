"""Causal intervention pipeline.

SAE feature steering (amplify/ablate/clamp), activation patching, dose-
response curves, and capability-preservation checks (MMLU/HellaSwag).
Includes random-direction controls.

Public entry points
-------------------
* :class:`CausalExperimentRunner` — orchestration for one (model, feature) cell.
* :class:`CausalRunnerConfig` — knobs (alphas, random_control, capability_eval).
* :func:`run_causal` — high-level driver used by ``scripts/run_causal.py``.

Hook primitives
---------------
* :class:`SteeringHook` — ``+alpha * d`` residual-stream injection.
* :class:`AblationHook` — projection ablation of a direction.
* :class:`FeatureClampHook` — SAE encode -> clamp -> decode.
* :func:`random_unit_direction` — reproducible random-control baseline.
* :func:`ablate_direction` — pure-function projection ablation.

Aggregation / viz
-----------------
* :func:`build_curve` — assemble a :class:`DoseResponseCurve`.
* :func:`fit_curve` — per-group slope + monotonicity.
* :func:`is_canonical_s2_signature` — boolean shape check.
* :func:`plot_dose_response`, :func:`plot_ablation_bars` — figures.

Capability preservation
-----------------------
* :func:`load_capability_jsonl`, :func:`score_capability`,
  :func:`compare_capability` — MMLU / HellaSwag log-likelihood scoring.
"""

from __future__ import annotations

from s1s2.causal.ablation import (
    AblationHook,
    FeatureClampHook,
    ablate_direction,
)
from s1s2.causal.capability import (
    CapabilityComparison,
    CapabilityItem,
    CapabilityResult,
    compare_capability,
    load_capability_jsonl,
    save_capability_jsonl,
    score_capability,
)
from s1s2.causal.cli import (
    default_score_fn,
    run_causal,
    runner_config_from_hydra,
)
from s1s2.causal.core import (
    AblationResult,
    CapabilityEvalConfig,
    CausalCellResult,
    CausalExperimentRunner,
    CausalRunnerConfig,
    FeatureSpec,
    RandomControlConfig,
    ScoreFn,
    config_hash,
    load_feature_specs,
    save_cell_result,
)
from s1s2.causal.dose_response import (
    DoseResponseCurve,
    DoseResponsePoint,
    aggregate_p_correct,
    build_curve,
    fit_curve,
    is_canonical_s2_signature,
)
from s1s2.causal.steering import (
    StackedSteeringHook,
    SteeringHook,
    normalize_direction,
    random_unit_direction,
)
from s1s2.causal.viz import (
    plot_ablation_bars,
    plot_dose_response,
    plot_feature_summary_bars,
)

__all__ = [
    "AblationHook",
    "AblationResult",
    "CapabilityComparison",
    "CapabilityEvalConfig",
    "CapabilityItem",
    "CapabilityResult",
    "CausalCellResult",
    "CausalExperimentRunner",
    "CausalRunnerConfig",
    "DoseResponseCurve",
    "DoseResponsePoint",
    "FeatureClampHook",
    "FeatureSpec",
    "RandomControlConfig",
    "ScoreFn",
    "StackedSteeringHook",
    "SteeringHook",
    "ablate_direction",
    "aggregate_p_correct",
    "build_curve",
    "compare_capability",
    "config_hash",
    "default_score_fn",
    "fit_curve",
    "is_canonical_s2_signature",
    "load_capability_jsonl",
    "load_feature_specs",
    "normalize_direction",
    "plot_ablation_bars",
    "plot_dose_response",
    "plot_feature_summary_bars",
    "random_unit_direction",
    "run_causal",
    "runner_config_from_hydra",
    "save_capability_jsonl",
    "save_cell_result",
    "score_capability",
]
