"""Metacognitive monitoring (stretch goal).

Surprise-feature correlation, difficulty detector identification, and
self-correction trajectory probing in reasoning models. The 4-gate
go/no-go framework lives in :func:`s1s2.metacog.gates.evaluate_gates`.

Public API
----------
* :class:`DifficultyDetectorAnalysis` — orchestrator
* :class:`MetacogConfig` — knobs
* :class:`PerFeatureResults` — per-(model, layer) result bundle
* :func:`surprise_feature_correlation` — surprise rho test
* :func:`s1s2_specificity_test` — matched-pair AUC test
* :func:`confidently_wrong_test` — metacognition vs confidence falsifier
* :func:`evaluate_gates` — pre-registered 4-gate framework
* :func:`parse_self_correction` — thinking-trace marker parser
* :func:`run_metacog` — top-level Hydra-driven driver
"""

from __future__ import annotations

from s1s2.metacog.core import (
    DifficultyDetectorAnalysis,
    MetacogConfig,
    PerFeatureResults,
)
from s1s2.metacog.difficulty import (
    SpecificityResult,
    auc_from_mannwhitney,
    confidently_wrong_batch,
    confidently_wrong_test,
    difficulty_sensitive_features,
    s1s2_specificity_batch,
    s1s2_specificity_test,
)
from s1s2.metacog.gates import (
    DifficultyDetectorResults,
    GateConfig,
    GateDecision,
    GateResult,
    evaluate_gates,
    gate_0_infrastructure,
    gate_1_signal_existence,
    gate_2_specificity,
    gate_3_causal,
    gates_to_dict,
)
from s1s2.metacog.surprise import (
    SurpriseAggregation,
    SurpriseCorrelationResult,
    aggregate_surprise,
    merge_correlation_results,
    surprise_feature_correlation,
)
from s1s2.metacog.trajectory import (
    DEFAULT_MARKERS,
    SelfCorrectionEvent,
    TraceParseResult,
    difficulty_trajectory_means,
    parse_self_correction,
    parse_trace_corpus,
)

__all__ = [
    "DEFAULT_MARKERS",
    "DifficultyDetectorAnalysis",
    "DifficultyDetectorResults",
    "GateConfig",
    "GateDecision",
    "GateResult",
    "MetacogConfig",
    "PerFeatureResults",
    "SelfCorrectionEvent",
    "SpecificityResult",
    "SurpriseAggregation",
    "SurpriseCorrelationResult",
    "TraceParseResult",
    "aggregate_surprise",
    "auc_from_mannwhitney",
    "confidently_wrong_batch",
    "confidently_wrong_test",
    "difficulty_sensitive_features",
    "difficulty_trajectory_means",
    "evaluate_gates",
    "gate_0_infrastructure",
    "gate_1_signal_existence",
    "gate_2_specificity",
    "gate_3_causal",
    "gates_to_dict",
    "merge_correlation_results",
    "parse_self_correction",
    "parse_trace_corpus",
    "s1s2_specificity_batch",
    "s1s2_specificity_test",
    "surprise_feature_correlation",
]


def run_metacog(*args, **kwargs):  # pragma: no cover - thin wrapper
    """Lazily import and call :func:`s1s2.metacog.cli.run_metacog`.

    Importing ``s1s2.metacog.cli`` at module load time would pull in
    Hydra/OmegaConf, which we want to keep optional for unit tests.
    """
    from s1s2.metacog.cli import run_metacog as _run

    return _run(*args, **kwargs)
