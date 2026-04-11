"""Linear probing pipeline.

Mass-mean, logistic regression, MLP, and CCS probes with Hewitt-Liang control
tasks and permutation tests for significance. The probing workstream asks: is
information about processing mode, correctness, or bias susceptibility linearly
decodable from residual stream activations across layers?

Public entry points
-------------------
* :class:`ProbeRunner`  — orchestrates training, CV, controls, bootstrap, perms.
* :class:`RunnerConfig` — knobs (n_folds, n_seeds, n_permutations, ...).
* :func:`build_target`  — turn the HDF5 cache into a ``(mask, y, strata)`` bundle.
* :func:`run_probes`    — high-level driver used by ``scripts/run_probes.py``.

Probe classes
-------------
* :class:`MassMeanProbe` — zero-parameter sanity check.
* :class:`LogisticRegressionProbe` — PRIMARY result. sklearn LogisticRegressionCV.
* :class:`MLPProbe` — nonlinear decoder for quantifying residual nonlinearity.
* :class:`CCSProbe` — Burns et al. 2022 unsupervised probe (correctness only).
"""

from __future__ import annotations

from s1s2.probes.cli import run_probes, runner_config_from_hydra
from s1s2.probes.controls import run_control_task, shuffled_labels
from s1s2.probes.core import (
    FoldResult,
    LayerResult,
    ProbeResult,
    ProbeRunner,
    RunnerConfig,
    apply_bh_across_layers,
    git_sha,
    layer_result_to_dict,
    load_layer_activations,
    loco_split_iter,
    make_stratify_key,
    save_layer_result,
)
from s1s2.probes.probes import (
    CCSProbe,
    LogisticRegressionProbe,
    MassMeanProbe,
    MLPProbe,
    Probe,
    get_probe_class,
)
from s1s2.probes.targets import ALL_TARGETS, TargetData, build_target

__all__ = [
    "ALL_TARGETS",
    "CCSProbe",
    "FoldResult",
    "LayerResult",
    "LogisticRegressionProbe",
    "MLPProbe",
    "MassMeanProbe",
    "Probe",
    "ProbeResult",
    "ProbeRunner",
    "RunnerConfig",
    "TargetData",
    "apply_bh_across_layers",
    "build_target",
    "get_probe_class",
    "git_sha",
    "layer_result_to_dict",
    "load_layer_activations",
    "loco_split_iter",
    "make_stratify_key",
    "run_control_task",
    "run_probes",
    "runner_config_from_hydra",
    "save_layer_result",
    "shuffled_labels",
]
