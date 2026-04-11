"""Reproducibility helpers.

Use :func:`set_global_seed` at the top of every script. It seeds Python's
random module, NumPy, PyTorch (CPU + CUDA), and sets ``CUBLAS_WORKSPACE_CONFIG``
for deterministic cuBLAS GEMMs.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed all RNGs used in the project.

    Why deterministic_torch is opt-in: ``torch.use_deterministic_algorithms(True)``
    is slower and raises if any op lacks a deterministic implementation. We
    enable it for tests and final runs but allow opt-out for speed.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch not installed in this environment — fine for analysis-only sessions
        pass
