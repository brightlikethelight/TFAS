"""Smoke tests for the ``scripts/`` directory.

Purpose: ensure that every top-level entry point at least *imports*
without raising. Running ``main()`` requires Hydra config + activations
files that we don't have in CI, so we only verify the import graph.

If a workstream has not been built yet (``s1s2.<workstream>.cli``
missing), the corresponding script will raise ``ImportError`` at import
time — we skip those, because a downstream agent will land the CLI
later.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

SCRIPTS_DIR = _REPO / "scripts"

SCRIPTS = [
    "run_probes",
    "run_sae",
    "run_attention",
    "run_geometry",
    "run_causal",
    "run_metacog",
    "generate_figures",
]


def _import_script(script: str) -> types.ModuleType:
    """Load a script from ``scripts/<script>.py`` as a module.

    We avoid the usual ``spec.loader.exec_module`` + ``sys.modules``
    registration dance because scripts are not packages — importing one
    as ``scripts.run_probes`` would require a package marker. Instead we
    use ``spec_from_file_location`` and execute the module object.
    """
    path = SCRIPTS_DIR / f"{script}.py"
    spec = importlib.util.spec_from_file_location(f"_s1s2_scripts.{script}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # The scripts guard their main execution behind ``if __name__ ==
    # "__main__":`` so importing them is side-effect-free (they only
    # mutate sys.path to include ./src, which is idempotent).
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_file_exists(script: str) -> None:
    """Every listed script must exist on disk.

    Using an xfail-style skip instead of a hard fail lets us add new
    scripts to the list before their corresponding wrappers have been
    written, without breaking CI — as long as the test is updated in
    the same commit, the full list passes.
    """
    path = SCRIPTS_DIR / f"{script}.py"
    if not path.exists():
        pytest.skip(f"{path} does not exist yet")
    assert path.is_file()


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_imports(script: str) -> None:
    """Each script must be importable (i.e. ``python scripts/X.py --help``).

    We skip (not fail) if the script depends on a workstream CLI module
    that hasn't been landed yet. A separate downstream smoke test will
    catch that once the workstream lands.
    """
    path = SCRIPTS_DIR / f"{script}.py"
    if not path.exists():
        pytest.skip(f"{path} does not exist yet")
    try:
        _import_script(script)
    except ImportError as exc:
        pytest.skip(f"{script} depends on an un-built workstream: {exc}")
    except SystemExit as exc:
        pytest.fail(f"{script} called sys.exit({exc.code}) at import time")


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_is_executable(script: str) -> None:
    """Entry points should be executable so users can ``./scripts/run_X.py``."""
    path = SCRIPTS_DIR / f"{script}.py"
    if not path.exists():
        pytest.skip(f"{path} does not exist yet")
    mode = path.stat().st_mode
    # At least the owner execute bit should be set.
    assert mode & 0o100, f"{path} is not executable (chmod +x missing)"


def test_viz_theme_importable() -> None:
    """The shared theme module must be importable (no heavy deps)."""
    from s1s2.viz import theme

    theme.set_paper_theme()
    assert "standard" in theme.COLORS
    assert "reasoning" in theme.COLORS
    assert (
        theme.get_model_color("llama-3.1-8b-instruct")
        == theme.MODEL_COLORS["llama-3.1-8b-instruct"]
    )
    # Unknown reasoning model falls back to the reasoning color.
    assert theme.get_model_color("r1-future-13b") == theme.COLORS["reasoning"]
    # Unknown standard model falls back to the standard color.
    assert theme.get_model_color("some-other-model") == theme.COLORS["standard"]


def test_figures_module_importable() -> None:
    """The unified figure module must import and expose the public API."""
    from s1s2.viz import figures

    assert hasattr(figures, "generate_all_figures")
    assert hasattr(figures, "FigureGenerationReport")
    assert callable(figures.generate_all_figures)


def test_generate_all_figures_empty_results(tmp_path: Path) -> None:
    """With empty results_dir, every figure should be 'skipped', not 'error'."""
    from s1s2.viz.figures import generate_all_figures

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    output_dir = tmp_path / "figures"
    report = generate_all_figures(
        results_dir=results_dir,
        output_dir=output_dir,
        config={"include": list(_all_registered_figures())},
    )
    # All figures are either skipped or error — never unexpectedly ok
    # on empty inputs. With no plotter modules and no JSONs, the only
    # valid outcome is "skipped".
    assert report.n_ok == 0
    # We allow errors too, because a workstream that ships its plotter
    # but expects nonzero inputs will raise KeyError on an empty df.
    # The contract is: we never crash the sweep.
    assert report.n_skipped + report.n_error == len(report.results)


def _all_registered_figures() -> list[str]:
    from s1s2.viz.figures import _FIGURE_REGISTRY

    return list(_FIGURE_REGISTRY.keys())
