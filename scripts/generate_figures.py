#!/usr/bin/env python3
"""Top-level entry point for the unified figure generator.

Reads ``configs/figures.yaml`` (Hydra), then calls
:func:`s1s2.viz.figures.generate_all_figures` to rebuild every paper
figure from the per-workstream result JSONs under ``results_dir``.

Example::

    python scripts/generate_figures.py
    python scripts/generate_figures.py figure_format=png
    python scripts/generate_figures.py include=[figure_2_probe_layer_curves]
"""

from __future__ import annotations

import os
import sys

# Ensure the src layout works even when running without ``pip install -e .``.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _run() -> None:
    try:
        import hydra
        from omegaconf import DictConfig, OmegaConf

        from s1s2.utils.logging import get_logger
        from s1s2.viz.figures import generate_all_figures
    except ImportError as exc:
        sys.stderr.write(
            "[generate_figures] failed to import dependencies: "
            f"{exc}\n"
            "Install the package in editable mode first:\n"
            "    pip install -e .[dev]\n"
        )
        raise SystemExit(1) from exc

    logger = get_logger("s1s2.scripts.generate_figures")

    @hydra.main(
        config_path="../configs",
        config_name="figures",
        version_base=None,
    )
    def main(cfg: DictConfig) -> None:
        logger.info("figures config:\n" + OmegaConf.to_yaml(cfg))
        report = generate_all_figures(
            results_dir=cfg.results_dir,
            output_dir=cfg.output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        )
        if report.n_error > 0:
            # Non-zero exit when any figure raised an error — CI should fail.
            raise SystemExit(1)

    main()


if __name__ == "__main__":
    _run()
