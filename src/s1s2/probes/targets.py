"""Label generators for the four probing targets.

A *target* is a binary classification task whose label is derived from per-problem
metadata (``/problems/...``) or per-model behaviour (``/models/{key}/behavior/...``)
stored in the activation HDF5 file. This module centralises the mapping so the
probing core never touches HDF5 keys directly.

The four targets are described in ``CLAUDE.md`` / the project task brief:

1. ``task_type``          — conflict (1) vs no-conflict control (0). Headline result.
2. ``correctness``        — model answered correctly (1) vs not (0).
3. ``bias_susceptible``   — model gave the S1 lure answer (1) vs not (0). Only
                             defined on conflict items; control items are dropped.
4. ``processing_mode``    — matched-pair contrast: same problem shown in
                             conflict (1) vs control (0) format. Equivalent to
                             ``task_type`` restricted to items that have a partner
                             so within-pair contrasts are clean.

Each generator returns a ``TargetData`` bundle with the binary label vector, a
mask over problems that are *applicable* to the target (e.g. for
``bias_susceptible`` we mask out control items), a stratification key per problem
(used by the CV splitter), and an optional paired-group id (used by LOCO / matched
pair analyses).

We do NOT return any activations here — that's the caller's job. This keeps the
module trivially unit-testable with synthetic arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import h5py
import numpy as np
from beartype import beartype

from s1s2.utils import io as ioh
from s1s2.utils.types import ProbeTarget

__all__ = [
    "ALL_TARGETS",
    "TargetData",
    "build_target",
]

ALL_TARGETS: Final[tuple[ProbeTarget, ...]] = (
    "task_type",
    "correctness",
    "bias_susceptible",
    "processing_mode",
)


@dataclass(frozen=True, slots=True)
class TargetData:
    """Bundle returned by a target generator.

    Attributes
    ----------
    target
        Name of the target (one of :data:`ALL_TARGETS`).
    y
        Binary labels shaped ``(n_applicable,)``. Always int8 in ``{0, 1}``.
    mask
        Boolean mask shaped ``(n_problems,)`` selecting which rows of the
        activation array are *applicable* to this target. The caller applies
        this mask to the activations before slicing ``y``.
    stratify_key
        Per-applicable-item stratification key shaped ``(n_applicable,)``. We
        combine this with ``y`` (via cantor pairing) for stratified CV splits
        so folds preserve both target-label and task-category balance.
    group_id
        Per-applicable-item group identifier shaped ``(n_applicable,)``. Used
        by leave-one-category-out (LOCO) transfer: all items sharing a group id
        must end up in the same fold. For ``task_type`` this is the task
        category; for ``processing_mode`` this is the matched_pair_id (so the
        two members of a pair never split across train/test).
    category
        Per-applicable-item category string shaped ``(n_applicable,)``. Kept
        separately from ``group_id`` because some targets use a different
        grouping than the category (see ``processing_mode``).
    meta
        Free-form description of how the target was constructed (for logging).
    """

    target: ProbeTarget
    y: np.ndarray
    mask: np.ndarray
    stratify_key: np.ndarray
    group_id: np.ndarray
    category: np.ndarray
    meta: dict[str, object]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@beartype
def build_target(
    target: str,
    f: h5py.File,
    model_key: str,
) -> TargetData:
    """Dispatch to the right label generator.

    Parameters
    ----------
    target
        One of :data:`ALL_TARGETS`.
    f
        An open HDF5 activations file (reader context from
        :func:`s1s2.utils.io.open_activations`).
    model_key
        HDF5 model key (see the data contract: HF id with ``/`` -> ``_``).
    """
    if target not in ALL_TARGETS:
        raise ValueError(f"Unknown target {target!r}; expected one of {ALL_TARGETS}")
    problems = ioh.load_problem_metadata(f)

    if target == "task_type":
        return _target_task_type(problems)
    if target == "correctness":
        behavior = ioh.get_behavior(f, model_key)
        return _target_correctness(problems, behavior)
    if target == "bias_susceptible":
        behavior = ioh.get_behavior(f, model_key)
        return _target_bias_susceptible(problems, behavior)
    if target == "processing_mode":
        return _target_processing_mode(problems)
    # Unreachable — guarded above — but keeps mypy happy.
    raise RuntimeError(f"unreachable: {target}")


# ---------------------------------------------------------------------------
# Individual target builders
# ---------------------------------------------------------------------------


def _target_task_type(problems: dict[str, np.ndarray]) -> TargetData:
    """conflict (1) vs no-conflict (0).

    Every item is applicable. Stratify by category so folds keep category
    balance. Group by category so a LOCO split is a pure held-out category.
    """
    y = problems["conflict"].astype(np.int8)
    mask = np.ones_like(y, dtype=bool)
    category = problems["category"].astype(str)
    return TargetData(
        target="task_type",
        y=y,
        mask=mask,
        stratify_key=category,
        group_id=category,
        category=category,
        meta={"source": "problems/conflict"},
    )


def _target_correctness(
    problems: dict[str, np.ndarray],
    behavior: dict[str, np.ndarray],
) -> TargetData:
    """Model answered correctly (1) vs not (0).

    Applicable to all items. A ``refusal`` response category is treated as
    incorrect (label 0) — it contributes noise to the probe but discarding
    refusals would bias the test set toward "easy" problems.
    """
    y = behavior["correct"].astype(np.int8)
    mask = np.ones_like(y, dtype=bool)
    category = problems["category"].astype(str)
    return TargetData(
        target="correctness",
        y=y,
        mask=mask,
        stratify_key=category,
        group_id=category,
        category=category,
        meta={"source": "behavior/correct"},
    )


def _target_bias_susceptible(
    problems: dict[str, np.ndarray],
    behavior: dict[str, np.ndarray],
) -> TargetData:
    """Lure match (1) vs not (0), restricted to conflict items.

    ``matches_lure`` is only meaningful when the prompt contains a lure. On
    control items the lure_answer field is empty and ``matches_lure`` is
    always False, so including them would artificially boost the "not-lure"
    class and give the probe a trivial "detect conflict vs control" shortcut.
    We therefore mask control items out.
    """
    is_conflict = problems["conflict"].astype(bool)
    y_full = behavior["matches_lure"].astype(np.int8)
    mask = is_conflict
    y = y_full[mask]
    category = problems["category"].astype(str)[mask]
    return TargetData(
        target="bias_susceptible",
        y=y,
        mask=mask,
        stratify_key=category,
        group_id=category,
        category=category,
        meta={"source": "behavior/matches_lure", "restricted_to": "conflict_items_only"},
    )


def _target_processing_mode(problems: dict[str, np.ndarray]) -> TargetData:
    """Matched-pair contrast: conflict (1) vs control (0) for paired items only.

    Equivalent to ``task_type`` but restricted to problems that have a
    matched partner. We use the ``matched_pair_id`` as the group_id so pairs
    are never split across train/test — otherwise the probe can cheat by
    memorising the prompt.
    """
    pair_id = problems["matched_pair_id"].astype(str)
    counts: dict[str, int] = {}
    for pid in pair_id:
        counts[pid] = counts.get(pid, 0) + 1
    has_partner = np.array([counts[pid] >= 2 and pid != "" for pid in pair_id])
    mask = has_partner
    y = problems["conflict"][mask].astype(np.int8)
    category = problems["category"].astype(str)[mask]
    return TargetData(
        target="processing_mode",
        y=y,
        mask=mask,
        stratify_key=category,
        group_id=pair_id[mask],
        category=category,
        meta={"source": "problems/conflict", "restricted_to": "matched_pairs_only"},
    )
