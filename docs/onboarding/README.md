# Onboarding Guide -- The Deliberation Gradient (s1s2)

**Goal**: Go from "never seen this repo" to "running your first analysis" in one session (~4 hours).

**Prerequisites**: Python 3.11+, git, a working terminal. No GPU required -- everything here runs on CPU.

---

## Step 1: Environment setup (30 min)

```bash
git clone <repo-url>
cd s1s2
pip install -e ".[sae,dev]"
```

Verify the install:

```bash
make test          # should see 365+ tests pass
make smoke         # end-to-end smoke test on synthetic data; should see all workstreams pass
```

`make test` runs the full pytest suite. `make smoke` runs
`scripts/smoke_test.py`, which builds a synthetic HDF5 activation
cache on the fly and pipes it through every analysis workstream on
CPU in under 60 seconds. If either command fails, fix the issue
before proceeding -- every downstream step depends on a working
install.

Optional extras you may want:

```bash
# Jupyter for the exploration notebooks
pip install -e ".[dev]"   # already includes jupyterlab + ipykernel

# Pre-commit hooks (auto-runs ruff + black on commit)
pre-commit install
```

---

## Step 2: Understand the project (1 hr)

Read these files in order. All paths are relative to the repo root.

| File | Time | What you learn |
|------|------|----------------|
| `README.md` | 5 min | The 30-second pitch: 4 models, 7 bias categories, 5 mechanistic methods, graded-not-binary framing |
| `CLAUDE.md` | 15 min | The project bible. Coding standards, statistical rules, model table, confound controls, gotchas. Read every word. |
| `docs/architecture.md` | 10 min | Pipeline schematic, module layout, dependency graph, where to extend |
| `docs/data_contract.md` | 10 min | The HDF5 schema that every workstream reads from. Understand the position labels (P0, P2, T0...Tend) and the per-model subgroup layout. |
| `notebooks/00_benchmark_exploration.ipynb` | 20 min | Open and run it. See the benchmark items, per-category counts, matched-pair structure. |

Key concepts to internalize:

- **S1/S2 is operational, not a cognitive claim.** We never say "LLMs have System 1 and System 2." We say "S1-like / S2-like processing signatures" or "deliberation-intensity gradient."
- **Matched pairs are the core experimental design.** Every conflict item (S1 lure present) has a no-conflict control with the same surface structure. This controls for task difficulty vs. processing mode.
- **The HDF5 activation cache is write-once, read-many.** Only `s1s2.extract` writes to it. All five analysis workstreams are read-only consumers.
- **Statistical standards are non-negotiable.** BH-FDR for multiple comparisons, Hewitt-Liang controls for probes, Ma et al. falsification for SAE features, multi-seed for everything stochastic.

---

## Step 3: Explore the benchmark (30 min)

The benchmark is a JSONL file at `data/benchmark/benchmark.jsonl` with
284+ cognitive bias problems across 7 categories (CRT, base-rate
neglect, syllogism, anchoring, framing, conjunction fallacy,
multi-step arithmetic).

```bash
# Print per-category counts, conflict ratios, difficulty distribution
python -m s1s2.benchmark.cli stats
```

```bash
# Validate the benchmark structure
python -m s1s2.benchmark.cli validate
```

Then open `notebooks/00_benchmark_exploration.ipynb` and step through it.
Look at a few example items -- notice how each conflict item has a
matched control with the same `matched_pair_id`.

---

## Step 4: Run your first analysis (1 hr)

The smoke test already ran the full pipeline on synthetic data. Now
let's do it manually so you understand each step.

### 4a. Build a synthetic HDF5 cache

The smoke test script builds a synthetic cache automatically. You can
also build one from a Python session:

```python
import tempfile
from pathlib import Path

# The conftest helper builds a schema-valid HDF5 with planted signals
import sys; sys.path.insert(0, "tests")
from conftest import build_synthetic_hdf5, SYNTH_MODEL_KEY

cache_path = Path(tempfile.mkdtemp()) / "my_first_cache.h5"
build_synthetic_hdf5(cache_path)
print(f"Cache written to {cache_path}")
```

### 4b. Inspect the HDF5 file

```python
from s1s2.utils.io import (
    open_activations,
    load_problem_metadata,
    list_models,
    get_residual,
    get_behavior,
    position_labels,
)

with open_activations(str(cache_path)) as f:
    # Problem metadata
    meta = load_problem_metadata(f)
    print(f"Problems: {len(meta['id'])}")
    print(f"Categories: {set(meta['category'])}")
    print(f"Conflict items: {meta['conflict'].sum()}")

    # Models in the cache
    models = list_models(f)
    print(f"Models: {models}")

    # Position labels for the first model
    positions = position_labels(f, models[0])
    print(f"Positions: {positions}")

    # Residual stream activations at layer 0, position P0
    X = get_residual(f, models[0], layer=0, position="P0")
    print(f"Activations shape: {X.shape}")  # (n_problems, hidden_dim)

    # Behavioral outcomes
    beh = get_behavior(f, models[0])
    print(f"Accuracy: {beh['correct'].mean():.2f}")
    print(f"Lure rate: {beh['matches_lure'].mean():.2f}")
```

### 4c. Understand the planted signal

The synthetic cache has a deliberate signal planted at **layer 2,
residual dimension 0**: conflict items get a +0.8 shift. This means:

- **Layer 2** should be easy for probes and geometry to separate.
- **Layer 0** is pure noise -- probes should fail there.

Try it yourself:

```python
import numpy as np

with open_activations(str(cache_path)) as f:
    conflict = load_problem_metadata(f)["conflict"]
    model = list_models(f)[0]

    # Layer 0: no signal
    X0 = get_residual(f, model, layer=0, position="P0")
    print(f"Layer 0 dim-0 mean (conflict): {X0[conflict, 0].mean():.3f}")
    print(f"Layer 0 dim-0 mean (control):  {X0[~conflict, 0].mean():.3f}")

    # Layer 2: planted signal
    X2 = get_residual(f, model, layer=2, position="P0")
    print(f"Layer 2 dim-0 mean (conflict): {X2[conflict, 0].mean():.3f}")
    print(f"Layer 2 dim-0 mean (control):  {X2[~conflict, 0].mean():.3f}")
```

You should see a clear separation at layer 2 (~0.8 difference) and
noise at layer 0 (~0.0 difference).

---

## Step 5: Pick your role (15 min)

Read `docs/presentation/team_roles.md` for the full role descriptions.
The roles are:

| Role | People | Difficulty | Key codebase area |
|------|--------|------------|-------------------|
| Project Lead | 1 | 3/3 | All (coordination) |
| Benchmark Lead | 1 | 1/3 | `src/s1s2/benchmark/` |
| Infrastructure Lead | 1 | 3/3 | `src/s1s2/extract/` |
| Probes + Geometry | 2 | 2/3 | `src/s1s2/probes/`, `src/s1s2/geometry/` |
| SAE + Causal | 2 | 3/3 | `src/s1s2/sae/`, `src/s1s2/causal/` |
| Attention + Metacog | 1-2 | 2-3/3 | `src/s1s2/attention/`, `src/s1s2/metacog/` |
| Writing Lead | 1 | 2/3 | `docs/`, `figures/` |

Then skim `docs/onboarding/coding_exercises.md`. The exercises are
graded by difficulty and aligned with different roles -- pick the one
that matches your interest.

---

## Step 6: Do a coding exercise (1-2 hrs)

Pick one exercise from `docs/onboarding/coding_exercises.md` and
complete it. All exercises are solvable with the synthetic data on
CPU -- no GPU, no model downloads, no internet required.

Recommended first exercise by role:

| Your role | Start with |
|-----------|------------|
| Benchmark Lead | Exercise 5 (build a new benchmark item) |
| Infrastructure Lead | Exercise 2 (inspect activations) |
| Probes + Geometry | Exercise 3 (train your first probe) |
| SAE + Causal | Exercise 4 (run SAE analysis) |
| Attention + Metacog | Exercise 2 (inspect activations) |
| Writing Lead | Exercise 1 (read the benchmark) |

---

## What's next

After completing onboarding:

1. **Read `docs/CONTRIBUTING.md`** for the full contribution workflow
   (formatting, linting, testing, commit hygiene).
2. **Read your workstream's code** under `src/s1s2/<workstream>/`.
   Each module follows the same layout: `core.py` (logic), `stats.py`
   (statistics), `viz.py` (plots), `cli.py` (entry point),
   `__init__.py` (public API).
3. **Read the background papers** in
   `docs/onboarding/background_reading.md`. Start with the
   "must-read" tier before your first contribution.
4. **Check `docs/onboarding/faq.md`** if anything is unclear.
5. **For GPU work**: see `deploy/README.md` for RunPod and FASRC setup.

---

## Quick reference

| Task | Command |
|------|---------|
| Run tests | `make test` |
| Run smoke test | `make smoke` |
| Lint | `make lint` |
| Format code | `make format` |
| Benchmark stats | `python -m s1s2.benchmark.cli stats` |
| Validate benchmark | `python -m s1s2.benchmark.cli validate` |
| Generate benchmark | `python -m s1s2.benchmark.cli generate` |
| Run probes | `make probes` |
| Run SAE analysis | `make sae` |
| Run attention analysis | `make attention` |
| Run geometry analysis | `make geometry` |
| Run everything | `make all-analyses` |
