"""Difficulty-sensitive feature analyses.

This module owns the two tests that distinguish a *metacognitive*
feature from a *confidence proxy*:

1. **S1/S2 specificity test** — does the candidate feature activate
   more strongly on conflict (S1-lure) items than on matched-difficulty
   non-conflict controls? Mann-Whitney U on the matched subset, with
   AUC > 0.65 as the gate threshold.

2. **Confidently-wrong test** — the critical falsifier. A feature that
   merely tracks output confidence will be silent when the model
   confidently produces a wrong answer (the model "thinks" it's right).
   A feature that tracks *internal processing difficulty* will fire
   even on confidently-wrong cases. We measure the activation
   difference between the two cells (confident-wrong vs confident-right)
   and call it the "metacognition score": positive => metacognition,
   ~0 => confidence proxy.

Both tests are stateless functions returning plain dicts. The
:class:`DifficultyDetectorAnalysis` orchestrator in
``metacog.core`` calls them per-feature for the candidate set
identified by the surprise-correlation pass.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Bool, Float
from scipy import stats

from s1s2.utils.logging import get_logger
from s1s2.utils.stats import bh_fdr

logger = get_logger("s1s2.metacog")


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------


@beartype
def auc_from_mannwhitney(
    activations_pos: Float[np.ndarray, "n_pos"],
    activations_neg: Float[np.ndarray, "n_neg"],
) -> tuple[float, float, float]:
    """Compute (AUC, U, p_value) for a Mann-Whitney U two-sample test.

    AUC is computed via ``U / (n_pos * n_neg)``: this is the
    probability that a random ``pos`` sample exceeds a random ``neg``
    sample, which is the canonical definition. Returns 0.5 if either
    sample is empty.
    """
    n_pos, n_neg = len(activations_pos), len(activations_neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0, 1.0
    try:
        u, p = stats.mannwhitneyu(activations_pos, activations_neg, alternative="two-sided")
    except ValueError:
        return 0.5, 0.0, 1.0
    auc = float(u) / (n_pos * n_neg)
    return float(auc), float(u), float(p)


# ---------------------------------------------------------------------------
# S1/S2 specificity (matched subset)
# ---------------------------------------------------------------------------


@dataclass
class SpecificityResult:
    """Per-feature S1/S2 specificity test on the matched-pair subset."""

    feature_id: int
    auc: float
    u_stat: float
    p_value: float
    q_value: float
    rank_biserial: float
    n_S1: int
    n_S2: int
    passes_specificity: bool
    notes: str = ""


@beartype
def s1s2_specificity_test(
    feature_activations: Float[np.ndarray, "n_problems"],
    conflict: Bool[np.ndarray, "n_problems"],
    matched_pair_id: np.ndarray,
    *,
    feature_id: int = -1,
    auc_threshold: float = 0.65,
    difficulty: np.ndarray | None = None,
    matched_only: bool = True,
) -> SpecificityResult:
    """Run the S1/S2 specificity test on a single feature.

    The "matched only" mode (default) restricts to the conflict/control
    pairs linked by ``matched_pair_id`` — every kept S1 item shares
    surface form and intended difficulty with its S2 control, so any
    activation difference cannot be explained by surface confounds.

    A feature passes if AUC(conflict vs matched control) > auc_threshold.
    The default threshold of 0.65 is the Gate-2 criterion in the
    pre-registered framework.
    """

    if feature_activations.ndim != 1:
        raise ValueError(
            f"expected 1D feature activations for one feature, got shape "
            f"{feature_activations.shape}"
        )
    if feature_activations.shape[0] != conflict.shape[0]:
        raise ValueError(
            f"problem axis mismatch: features={feature_activations.shape[0]} "
            f"vs conflict={conflict.shape[0]}"
        )

    notes = ""

    if matched_only:
        # Build conflict↔control pairs from matched_pair_id.
        if matched_pair_id.dtype == object:
            ids = np.asarray(
                [
                    s.decode("utf-8") if isinstance(s, bytes | bytearray) else str(s)
                    for s in matched_pair_id
                ]
            )
        else:
            ids = matched_pair_id.astype(str)

        by_pair: dict[str, dict[str, int]] = {}
        for row_idx, pid in enumerate(ids):
            if pid in ("", "b''"):
                continue
            kind = "S1" if conflict[row_idx] else "S2"
            by_pair.setdefault(pid, {})[kind] = row_idx

        kept: list[int] = []
        kept_conf: list[bool] = []
        for _pid, members in by_pair.items():
            if "S1" in members and "S2" in members:
                kept.extend([members["S1"], members["S2"]])
                kept_conf.extend([True, False])
        if not kept:
            notes = "no matched pairs available; specificity not testable"
            return SpecificityResult(
                feature_id=int(feature_id),
                auc=0.5,
                u_stat=0.0,
                p_value=1.0,
                q_value=1.0,
                rank_biserial=0.0,
                n_S1=0,
                n_S2=0,
                passes_specificity=False,
                notes=notes,
            )
        rows = np.asarray(kept, dtype=np.int64)
        feat_kept = feature_activations[rows]
        is_conf = np.asarray(kept_conf, dtype=bool)
    else:
        feat_kept = feature_activations
        is_conf = conflict.astype(bool)

    s1 = feat_kept[is_conf]
    s2 = feat_kept[~is_conf]
    auc, u, p = auc_from_mannwhitney(s1, s2)
    # Rank-biserial: positive means S1 stochastically larger.
    n_s1, n_s2 = len(s1), len(s2)
    rb = float(1.0 - 2.0 * u / (n_s1 * n_s2)) if (n_s1 * n_s2) > 0 else 0.0
    # mannwhitneyu returns U for x; AUC = U / (n_x*n_y) corresponds to
    # P(x > y). If S1 is larger, U is large, AUC > 0.5. The "passes"
    # criterion is "AUC strictly greater than the threshold." We don't
    # enforce a sign on the direction of the difference here — that's
    # what AUC itself encodes.

    passes = auc > auc_threshold

    return SpecificityResult(
        feature_id=int(feature_id),
        auc=float(auc),
        u_stat=float(u),
        p_value=float(p),
        q_value=float(p),  # filled in by the multi-feature wrapper
        rank_biserial=float(rb),
        n_S1=int(n_s1),
        n_S2=int(n_s2),
        passes_specificity=bool(passes),
        notes=notes,
    )


@beartype
def s1s2_specificity_batch(
    feature_activations: Float[np.ndarray, "n_problems n_features"],
    conflict: Bool[np.ndarray, "n_problems"],
    matched_pair_id: np.ndarray,
    feature_ids: Iterable[int],
    *,
    auc_threshold: float = 0.65,
    fdr_q: float = 0.05,
    matched_only: bool = True,
) -> pd.DataFrame:
    """Run the specificity test across many features and BH-correct.

    Returns a dataframe with one row per feature and the columns
    described in :class:`SpecificityResult`. The ``q_value`` column
    is the BH-adjusted p-value across the requested features only.
    """

    fids = list(feature_ids)
    if not fids:
        return pd.DataFrame(
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

    rows: list[SpecificityResult] = []
    for fid in fids:
        if fid < 0 or fid >= feature_activations.shape[1]:
            raise IndexError(f"feature_id {fid} out of range")
        col = feature_activations[:, int(fid)]
        rows.append(
            s1s2_specificity_test(
                feature_activations=col,
                conflict=conflict,
                matched_pair_id=matched_pair_id,
                feature_id=int(fid),
                auc_threshold=auc_threshold,
                matched_only=matched_only,
            )
        )

    p_vals = np.asarray([r.p_value for r in rows], dtype=np.float64)
    _, q_vals = bh_fdr(p_vals, q=fdr_q)
    for r, q in zip(rows, q_vals, strict=False):
        r.q_value = float(q)

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df


# ---------------------------------------------------------------------------
# Confidently-wrong test (the critical metacognition vs confidence falsifier)
# ---------------------------------------------------------------------------


@beartype
def confidently_wrong_test(
    feature_activations: Float[np.ndarray, "n_problems"],
    is_correct: Bool[np.ndarray, "n_problems"],
    output_confidence: Float[np.ndarray, "n_problems"],
    *,
    threshold_confidence: float = -0.5,
    feature_id: int = -1,
) -> dict[str, float | int]:
    """Distinguish metacognition from confidence calibration.

    A feature that tracks *output* confidence will be uniformly
    low-magnitude on confident-output cases regardless of correctness;
    a feature that tracks *internal processing difficulty* will spike
    on confidently-wrong cases (the model is sure of its answer but
    something inside the network knows that answer is the easy/lure
    response).

    We define "confident" as the answer log-probability exceeding
    ``threshold_confidence``. The metacognition score is

        score = mean(activation | confident & wrong)
              - mean(activation | confident & right)

    Positive score => metacognitive feature; ~0 score => confidence
    proxy. The standardized version (divided by the pooled std) is
    returned in ``"metacognition_score_std"`` for cross-feature
    comparison; the unstandardized score is returned for the original
    units.
    """

    if not (feature_activations.shape == is_correct.shape == output_confidence.shape):
        raise ValueError(
            f"shape mismatch: features={feature_activations.shape}, "
            f"correct={is_correct.shape}, conf={output_confidence.shape}"
        )

    confident = output_confidence > threshold_confidence
    confident_wrong = confident & (~is_correct)
    confident_correct = confident & is_correct

    n_cw = int(confident_wrong.sum())
    n_cc = int(confident_correct.sum())

    if n_cw == 0 or n_cc == 0:
        # Cannot distinguish — return a neutral result.
        return {
            "feature_id": int(feature_id),
            "n_confident_wrong": n_cw,
            "n_confident_correct": n_cc,
            "mean_activation_confident_wrong": (
                0.0 if n_cw == 0 else float(feature_activations[confident_wrong].mean())
            ),
            "mean_activation_confident_correct": (
                0.0 if n_cc == 0 else float(feature_activations[confident_correct].mean())
            ),
            "metacognition_score": 0.0,
            "metacognition_score_std": 0.0,
            "p_value": 1.0,
            "is_metacognitive": False,
        }

    cw_acts = feature_activations[confident_wrong].astype(np.float64)
    cc_acts = feature_activations[confident_correct].astype(np.float64)
    mean_cw = float(cw_acts.mean())
    mean_cc = float(cc_acts.mean())
    score = mean_cw - mean_cc

    pooled_std = float(np.sqrt(0.5 * (cw_acts.var(ddof=0) + cc_acts.var(ddof=0))))
    score_std = score / pooled_std if pooled_std > 1e-12 else 0.0

    # Significance test: one-sided MWU asking whether confident_wrong
    # activations stochastically exceed confident_correct activations.
    try:
        _, p = stats.mannwhitneyu(cw_acts, cc_acts, alternative="greater")
    except ValueError:
        p = 1.0

    is_metacog = (score > 0.0) and (p < 0.05)

    return {
        "feature_id": int(feature_id),
        "n_confident_wrong": n_cw,
        "n_confident_correct": n_cc,
        "mean_activation_confident_wrong": mean_cw,
        "mean_activation_confident_correct": mean_cc,
        "metacognition_score": float(score),
        "metacognition_score_std": float(score_std),
        "p_value": float(p),
        "is_metacognitive": bool(is_metacog),
    }


@beartype
def confidently_wrong_batch(
    feature_activations: Float[np.ndarray, "n_problems n_features"],
    is_correct: Bool[np.ndarray, "n_problems"],
    output_confidence: Float[np.ndarray, "n_problems"],
    feature_ids: Iterable[int],
    *,
    threshold_confidence: float = -0.5,
) -> pd.DataFrame:
    """Run the confidently-wrong test across many features."""
    rows = []
    for fid in feature_ids:
        rows.append(
            confidently_wrong_test(
                feature_activations=feature_activations[:, int(fid)],
                is_correct=is_correct,
                output_confidence=output_confidence,
                threshold_confidence=threshold_confidence,
                feature_id=int(fid),
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Composite "difficulty-sensitive feature" tagging
# ---------------------------------------------------------------------------


@beartype
def difficulty_sensitive_features(
    surprise_df: pd.DataFrame,
    specificity_df: pd.DataFrame,
    confidently_wrong_df: pd.DataFrame,
    *,
    rho_threshold: float = 0.3,
    auc_threshold: float = 0.65,
    require_metacog: bool = False,
) -> pd.DataFrame:
    """Combine the three per-feature tests into one tagged dataframe.

    Joins on ``feature_id``. The output has columns from each input
    plus three booleans:

    - ``is_difficulty_sensitive``: surprise-rho passes
    - ``passes_specificity``: matched-pair AUC passes
    - ``is_metacognitive``: confidently-wrong score is positive and significant

    And one integer ``score`` column counting how many of the three
    boxes the feature ticks. Features with ``score >= 2`` are the
    headline candidates.
    """

    df = surprise_df.merge(
        specificity_df, on="feature_id", how="inner", suffixes=("_corr", "_spec")
    )
    if not confidently_wrong_df.empty:
        df = df.merge(
            confidently_wrong_df,
            on="feature_id",
            how="left",
            suffixes=("", "_cw"),
        )
    if "is_metacognitive" not in df.columns:
        df["is_metacognitive"] = False

    df["score"] = (
        df["is_difficulty_sensitive"].astype(int)
        + df["passes_specificity"].astype(int)
        + df["is_metacognitive"].astype(int)
    )
    if require_metacog:
        df = df[df["is_metacognitive"]]
    return df.sort_values(by=["score", "rho"], ascending=[False, False]).reset_index(drop=True)


__all__ = [
    "SpecificityResult",
    "auc_from_mannwhitney",
    "confidently_wrong_batch",
    "confidently_wrong_test",
    "difficulty_sensitive_features",
    "s1s2_specificity_batch",
    "s1s2_specificity_test",
]
