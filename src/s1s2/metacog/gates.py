"""Pre-registered 4-gate go/no-go framework for the metacog workstream.

The point of pre-registering these gates is to commit to *what would
falsify the hypothesis* before looking at any results. The four gates
are progressively more demanding:

- **Gate 0 — Infrastructure**: code runs end-to-end on the smoke
  pipeline. Always passes if the orchestrator finishes without
  raising. This is a sanity check, not a scientific gate.

- **Gate 1 — Signal existence**: at least ``min_features_with_rho_gt``
  SAE features (default 20) have a Spearman correlation with token
  surprise > ``rho_threshold`` (default 0.3) AND survive BH-FDR. If
  Gate 1 fails, no internal "difficulty signal" exists at the SAE
  feature level — the entire downstream pipeline is moot.

- **Gate 2 — S1/S2 specificity**: at least one of those features
  achieves matched-pair AUC > ``min_specificity_auc`` (default 0.65)
  on conflict vs. matched-difficulty controls. If Gate 2 fails, the
  features track *generic* difficulty (e.g. token rarity) rather than
  S1/S2-relevant cognitive load.

- **Gate 3 — Causal evidence**: ablating or boosting the candidate
  features changes accuracy on conflict items by at least
  ``min_delta_p_correct`` (default 0.15). The causal workstream owns
  this; the metacog gate just records the criterion and exposes a
  "marginal" status until the causal results land. This is the gate
  that distinguishes a *correlate* of difficulty from a *driver* of
  S1/S2 processing.

Each gate is :class:`GateResult` with fields:

- ``gate_id``, ``name``: identification
- ``criteria``: dict of pre-registered thresholds
- ``metrics``: dict of measured values that the criteria are checked against
- ``decision``: ``"go"`` | ``"marginal"`` | ``"no_go"``
- ``rationale``: one-line human-readable summary

The orchestrator runs the gates in order; later gates are only
*evaluated* (not skipped) so the report shows the user exactly which
criterion failed and by how much.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from beartype import beartype

from s1s2.utils.logging import get_logger

logger = get_logger("s1s2.metacog")


GateDecision = Literal["go", "marginal", "no_go"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Outcome of a single gate evaluation.

    Use ``to_dict`` to get a JSON-serializable view.
    """

    gate_id: int
    name: str
    criteria: dict[str, Any]
    metrics: dict[str, Any]
    decision: GateDecision
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateConfig:
    """Pre-registered gate thresholds.

    These map 1:1 onto the ``gates`` block in ``configs/metacog.yaml``.
    Defaults reflect the project task brief.
    """

    # Gate 1: signal existence
    min_features_with_rho_gt: int = 20
    rho_threshold: float = 0.3
    fdr_q: float = 0.05

    # Gate 2: specificity
    min_specificity_auc: float = 0.65
    difficulty_matched_only: bool = True

    # Gate 3: causal (recorded only)
    min_delta_p_correct: float = 0.15


# ---------------------------------------------------------------------------
# The orchestration interface needs a forward reference
# ---------------------------------------------------------------------------


@dataclass
class DifficultyDetectorResults:
    """Snapshot of the metacog pass that the gates consume.

    The fields are intentionally a small, flat structure of plain
    Python types so the gates module has no dependency on the
    orchestrator's internals (and the gates can be unit-tested with
    synthetic dicts).

    The orchestrator (:class:`s1s2.metacog.core.DifficultyDetectorAnalysis`)
    fills this in.
    """

    model_key: str = ""
    layer: int = -1
    n_difficulty_sensitive_features: int = 0
    rho_distribution: list[float] = field(default_factory=list)
    max_specificity_auc: float = 0.5
    n_features_passing_specificity: int = 0
    metacognition_scores: list[float] = field(default_factory=list)
    n_metacognitive_features: int = 0
    causal_delta_p_correct: float | None = None  # filled by causal workstream
    infrastructure_ok: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# Per-gate evaluators
# ---------------------------------------------------------------------------


@beartype
def gate_0_infrastructure(
    results: DifficultyDetectorResults,
    cfg: GateConfig,
) -> GateResult:
    """Gate 0 — infrastructure works.

    Trivially passes if the orchestrator successfully populated the
    results dataclass. Fails only when ``results.infrastructure_ok``
    is set to ``False`` by the orchestrator (e.g. SAE failed to load,
    HDF5 file unreadable, no problems in the cache).
    """
    decision: GateDecision = "go" if results.infrastructure_ok else "no_go"
    rationale = (
        "Pipeline executed end-to-end."
        if results.infrastructure_ok
        else f"Infrastructure failure: {results.notes or 'see logs'}"
    )
    return GateResult(
        gate_id=0,
        name="infrastructure",
        criteria={"requires": "pipeline runs without error"},
        metrics={"infrastructure_ok": bool(results.infrastructure_ok)},
        decision=decision,
        rationale=rationale,
    )


@beartype
def gate_1_signal_existence(
    results: DifficultyDetectorResults,
    cfg: GateConfig,
) -> GateResult:
    """Gate 1 — at least N features survive surprise correlation + FDR."""
    n = int(results.n_difficulty_sensitive_features)
    threshold = int(cfg.min_features_with_rho_gt)
    decision: GateDecision
    if n >= threshold:
        decision = "go"
        rationale = (
            f"{n} features have rho > {cfg.rho_threshold} and pass FDR "
            f"(threshold: {threshold})."
        )
    elif n >= max(1, threshold // 2):
        decision = "marginal"
        rationale = (
            f"Only {n} features pass the rho/FDR criterion "
            f"(threshold {threshold}); a difficulty signal may exist but is weak."
        )
    else:
        decision = "no_go"
        rationale = (
            f"{n} features pass the rho/FDR criterion (threshold {threshold}); "
            "no internal difficulty signal detected at the SAE feature level."
        )
    return GateResult(
        gate_id=1,
        name="signal_existence",
        criteria={
            "min_features_with_rho_gt": threshold,
            "rho_threshold": float(cfg.rho_threshold),
            "fdr_q": float(cfg.fdr_q),
        },
        metrics={"n_difficulty_sensitive_features": n},
        decision=decision,
        rationale=rationale,
    )


@beartype
def gate_2_specificity(
    results: DifficultyDetectorResults,
    cfg: GateConfig,
) -> GateResult:
    """Gate 2 — at least one feature passes the matched-pair AUC threshold."""
    auc = float(results.max_specificity_auc)
    n_pass = int(results.n_features_passing_specificity)
    threshold = float(cfg.min_specificity_auc)
    decision: GateDecision
    if auc > threshold and n_pass >= 1:
        decision = "go"
        rationale = (
            f"{n_pass} features pass the matched-pair AUC > {threshold} criterion "
            f"(best AUC = {auc:.3f})."
        )
    elif auc > (threshold - 0.05):
        decision = "marginal"
        rationale = (
            f"Best matched-pair AUC = {auc:.3f}; within 0.05 of the {threshold} "
            "threshold but no feature strictly clears it."
        )
    else:
        decision = "no_go"
        rationale = (
            f"Best matched-pair AUC = {auc:.3f}; below {threshold}. "
            "Difficulty features track surface confounds, not S1/S2."
        )
    return GateResult(
        gate_id=2,
        name="s1_s2_specificity",
        criteria={
            "min_specificity_auc": threshold,
            "difficulty_matched_only": bool(cfg.difficulty_matched_only),
        },
        metrics={
            "max_specificity_auc": auc,
            "n_features_passing_specificity": n_pass,
        },
        decision=decision,
        rationale=rationale,
    )


@beartype
def gate_3_causal(
    results: DifficultyDetectorResults,
    cfg: GateConfig,
) -> GateResult:
    """Gate 3 — causal evidence from the causal workstream.

    The metacog workstream cannot evaluate Gate 3 on its own because
    it does not run interventions. We record the criterion and emit
    ``"marginal"`` until ``results.causal_delta_p_correct`` is filled
    in by the causal pipeline (which writes the value back into the
    results JSON the metacog CLI emits).
    """
    threshold = float(cfg.min_delta_p_correct)
    delta = results.causal_delta_p_correct
    decision: GateDecision
    if delta is None:
        decision = "marginal"
        rationale = (
            f"Causal workstream has not been run yet; threshold for go = "
            f"|delta_p_correct| >= {threshold}."
        )
        metrics: dict[str, Any] = {"causal_delta_p_correct": None}
    else:
        delta_f = float(delta)
        metrics = {"causal_delta_p_correct": delta_f}
        if abs(delta_f) >= threshold:
            decision = "go"
            rationale = (
                f"|delta_p_correct| = {abs(delta_f):.3f} >= {threshold}: "
                "intervening on candidate features causally changes accuracy."
            )
        elif abs(delta_f) >= 0.5 * threshold:
            decision = "marginal"
            rationale = (
                f"|delta_p_correct| = {abs(delta_f):.3f} is below {threshold} "
                "but above half-threshold; weak causal evidence."
            )
        else:
            decision = "no_go"
            rationale = (
                f"|delta_p_correct| = {abs(delta_f):.3f} is well below {threshold}; "
                "intervening on candidate features does not change accuracy."
            )
    return GateResult(
        gate_id=3,
        name="causal_evidence",
        criteria={"min_delta_p_correct": threshold},
        metrics=metrics,
        decision=decision,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


@beartype
def evaluate_gates(
    results: DifficultyDetectorResults,
    cfg: GateConfig | None = None,
) -> list[GateResult]:
    """Run all four gates in order and return the result list.

    The gates are *all* evaluated, even after a no_go, so the report
    shows what failed AND how each later check would have looked. This
    matches the project pre-registration: we don't short-circuit on
    Gate 1 because the Gate 2/3 numbers are useful for write-up even
    when Gate 1 fails (they tell us how much downstream change we'd
    need to flip the conclusion).
    """
    if cfg is None:
        cfg = GateConfig()

    gate_results = [
        gate_0_infrastructure(results, cfg),
        gate_1_signal_existence(results, cfg),
        gate_2_specificity(results, cfg),
        gate_3_causal(results, cfg),
    ]

    logger.info(
        "[%s/L%02d] gate decisions: %s",
        results.model_key or "model",
        results.layer,
        " | ".join(f"G{g.gate_id}={g.decision}" for g in gate_results),
    )
    return gate_results


@beartype
def gates_to_dict(gate_results: list[GateResult]) -> list[dict[str, Any]]:
    """Convenience: ``[g.to_dict() for g in gate_results]`` for JSON dumps."""
    return [g.to_dict() for g in gate_results]


__all__ = [
    "DifficultyDetectorResults",
    "GateConfig",
    "GateDecision",
    "GateResult",
    "evaluate_gates",
    "gate_0_infrastructure",
    "gate_1_signal_existence",
    "gate_2_specificity",
    "gate_3_causal",
    "gates_to_dict",
]
