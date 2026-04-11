# FAQ -- The Deliberation Gradient

Common questions from new team members.

---

## Setup and environment

### "How do I install the project?"

```bash
pip install -e ".[sae,dev]"
```

The `sae` extra pulls in `sae-lens`, `transformer-lens`, and
`nnsight` for SAE loading. The `dev` extra pulls in pytest, ruff,
black, mypy, jupyterlab, and pre-commit. Both are needed for full
functionality.

### "make test is failing. What do I do?"

First, check that you installed with all extras:
```bash
pip install -e ".[sae,dev]"
```

If specific tests fail, read the error. Common causes:
- **Missing `sae-lens`**: Install with `pip install -e ".[sae,dev]"`,
  not just `pip install -e ".[dev]"`.
- **macOS OpenMP conflict**: If you see `OMP: Error #15`, set
  `export KMP_DUPLICATE_LIB_OK=TRUE` in your shell.
- **Python version**: We require 3.11+. Check with `python --version`.

### "How do I run something on GPU?"

See `deploy/README.md` for full instructions. The short version:

- **RunPod**: Best for development. SCP the repo, SSH in, run
  `pip install -e ".[sae,dev]"`, then run scripts normally.
- **FASRC / Kempner**: Best for production runs (free compute). Use
  the Slurm scripts in `deploy/` (`slurm_extract.sh`,
  `slurm_probes.sh`, etc.).

Nothing in the onboarding exercises requires a GPU.

### "What Python version do I need?"

Python 3.11 or later. The codebase uses features from 3.11 (e.g.,
`datetime.UTC`, modern type hint syntax). Check with:
```bash
python --version
```

---

## Project concepts

### "What is the S1/S2 distinction?"

It comes from Kahneman's dual-process theory: System 1 is fast,
automatic, heuristic-based processing; System 2 is slow, deliberate,
effortful processing.

**Critical framing**: We do NOT claim LLMs literally have two
processing systems. We use S1/S2 as an operational vocabulary to ask:
"Do models process conflict tasks (where intuition and deliberation
disagree) differently from no-conflict tasks (where they agree)?
And does reasoning training amplify this difference?"

The project-preferred term is "deliberation-intensity gradient."

### "Why do we need matched conflict/control pairs?"

To distinguish "the model processes S1 vs S2 differently" from "the
model processes easy vs hard differently."

A conflict item has a System 1 lure -- an answer that "feels" right
but is wrong. The no-conflict control has the same surface structure
and difficulty but no S1 lure. If we see a difference in internal
representations between conflict and control, it cannot be attributed
to difficulty alone.

Example:
- **Conflict**: "A guitar and a case cost $110 total. The guitar costs
  $100 more than the case. How much does the case cost?"
  (Lure: $10. Correct: $5.)
- **Control**: "A guitar costs $100 and a case costs $10. How much
  do they cost together?" (Answer: $110. No lure.)

### "What is the headline comparison?"

Llama-3.1-8B-Instruct vs. DeepSeek-R1-Distill-Llama-8B. Same
architecture (32 layers, 32 heads, 4096 hidden dim), different
training (standard instruction tuning vs. reasoning distillation).
Any differences in internal processing are attributable to the
reasoning training, not architecture.

### "What are the 5 analysis workstreams?"

1. **Linear probes** (`src/s1s2/probes/`): Train classifiers on
   hidden states to test what information is linearly decodable at
   each layer.
2. **SAE feature analysis** (`src/s1s2/sae/`): Find which sparse
   autoencoder features activate differentially on conflict vs
   no-conflict tasks.
3. **Attention entropy** (`src/s1s2/attention/`): Measure whether
   attention patterns differ (more focused vs more diffuse) in
   S1-eliciting vs S2-requiring conditions.
4. **Representational geometry** (`src/s1s2/geometry/`): Test whether
   conflict and no-conflict activations occupy distinguishable
   regions of activation space (silhouette scores, CKA, UMAP).
5. **Causal interventions** (`src/s1s2/causal/`): Ablate or amplify
   specific SAE features and measure whether behavior changes
   (steering experiments).

Each workstream reads from the shared HDF5 activation cache and
writes results to `results/<workstream>/`. They never import from
each other.

---

## Methodology

### "What is the difference between a probe and an SAE feature?"

A **probe** asks: "Is this information linearly decodable from the
representation?" You train a classifier (logistic regression, MLP)
on the hidden states and see if it can predict conflict vs.
no-conflict. High probe accuracy means the information is *present*
in the representation.

An **SAE feature** asks: "Which specific direction in activation space
fires differentially?" A sparse autoencoder decomposes the hidden
state into interpretable features, and you test which features have
statistically different activations on conflict vs. no-conflict items.
A significant feature gives you a *direction* you can interpret and
causally intervene on.

Probes tell you "the information is there." SAE features tell you
"here is the specific feature encoding it."

### "What is the Hewitt-Liang control task and why is it mandatory?"

Hewitt & Liang (2019) showed that probes can achieve high accuracy
on *random labels* if the representation is high-dimensional relative
to the number of examples (the d >> N problem). A probe with 90%
accuracy is meaningless if a probe trained on random labels also
gets 85%.

The control: train the same probe architecture on *permuted labels*
(destroying any real signal). **Selectivity** = real AUC - control
AUC. If selectivity < 5 percentage points, the probe is exploiting
its own expressiveness, not information in the representation.

This is mandatory for all probes in this project. See
`src/s1s2/probes/controls.py`.

### "What is the Ma et al. falsification and why is it non-negotiable?"

Ma et al. (2026) demonstrated that **45-90% of SAE features** that
activate on "reasoning" text are actually triggered by specific tokens
(like "Let", "First", "wait"), not by the reasoning process itself.

The test: take the top tokens that activate a candidate feature.
Inject those tokens into 100 random sentences that have nothing to do
with cognitive bias tasks. If the feature still fires, it is a token
artifact, not a processing-mode feature.

This test is implemented in `src/s1s2/sae/falsification.py` and is
run on every FDR-significant feature before we count it. Skipping it
is not acceptable.

### "What is the difference between TransformerLens and HuggingFace?"

- **HuggingFace Transformers**: We use it for model loading and text
  generation, plus custom forward hooks for activation extraction
  (see `src/s1s2/extract/`).
- **TransformerLens**: An alternative mechanistic interpretability
  library. We use it indirectly through `sae-lens` for loading
  pre-trained SAEs. We do NOT use TransformerLens for activation
  extraction (we use HuggingFace hooks instead for better control
  over generation and thinking-trace parsing).

### "What are position labels (P0, P2, T0, T25, etc.)?"

These label specific token positions where we extract activations:

| Label | Meaning | Available for |
|-------|---------|---------------|
| P0 | Last token of the prompt (pre-generation) | All models |
| P2 | Final answer token | All models |
| T0 | First token after `<think>` | Reasoning models only |
| T25 | 25% through thinking trace | Reasoning models only |
| T50 | 50% through thinking trace | Reasoning models only |
| T75 | 75% through thinking trace | Reasoning models only |
| Tend | Last token before `</think>` | Reasoning models only |
| Tswitch | First token after `</think>` | Reasoning models only |

For non-reasoning models (Llama-Instruct, Gemma-IT), the T-positions
exist in the HDF5 but are marked `valid=False`. This avoids ragged
data structures.

---

## Codebase navigation

### "Where does the data live?"

- **Benchmark**: `data/benchmark/benchmark.jsonl` (284+ items, 7
  categories, version controlled).
- **Activations**: `data/activations/{run_name}.h5` (gitignored,
  generated by the extraction pipeline). The `smoke` run is built
  by `scripts/smoke_test.py` with synthetic data.
- **Raw model outputs**: `data/raw_outputs/` (gitignored).
- **Results**: `results/{workstream}/` (per-workstream JSON files
  with statistics, configs, and W&B run IDs).
- **Figures**: `figures/` (gitignored, regeneratable from results).

### "Where is the code for X?"

| What | Where |
|------|-------|
| Benchmark loading and validation | `src/s1s2/benchmark/` |
| Activation extraction | `src/s1s2/extract/` |
| Linear probes | `src/s1s2/probes/` |
| SAE feature analysis | `src/s1s2/sae/` |
| Attention entropy | `src/s1s2/attention/` |
| Representational geometry | `src/s1s2/geometry/` |
| Causal interventions | `src/s1s2/causal/` |
| Metacognitive monitoring | `src/s1s2/metacog/` |
| Plotting utilities | `src/s1s2/viz/` |
| Shared IO, seeding, stats | `src/s1s2/utils/` |
| Hydra configs | `configs/` |
| GPU deployment | `deploy/` |
| Tests | `tests/` |

Every workstream module follows the same internal layout:
`core.py` (logic), `stats.py` (statistics), `viz.py` (plots),
`cli.py` (Hydra entry point), `__init__.py` (public API).

### "How do I read activations from the HDF5 cache?"

Always use the typed accessors in `src/s1s2/utils/io.py`. Never
construct HDF5 key strings manually in analysis code.

```python
from s1s2.utils.io import (
    open_activations,
    load_problem_metadata,
    list_models,
    get_residual,
    get_behavior,
    get_attention_metric,
    position_labels,
)

with open_activations("data/activations/main.h5") as f:
    meta = load_problem_metadata(f)
    models = list_models(f)
    X = get_residual(f, models[0], layer=16, position="P0")
```

### "How do workstreams communicate?"

They don't import each other. Communication flows through two shared
artifacts:

1. **HDF5 activation cache** (`data/activations/*.h5`): written by
   `s1s2.extract`, read by everyone else.
2. **Result JSON files** (`results/<workstream>/`): written by each
   workstream, read by `s1s2.viz` and `s1s2.causal`.

This decoupling means you can work on your workstream without
understanding the internals of other workstreams.

---

## Statistical questions

### "Why BH-FDR instead of Bonferroni?"

Bonferroni is overly conservative for exploratory analyses with
thousands of tests (e.g., testing all SAE features). Benjamini-
Hochberg controls the *false discovery rate* (expected proportion of
false positives among rejected hypotheses) at q=0.05 rather than the
*family-wise error rate*. This is the standard in genomics and other
high-dimensional fields.

We use Bonferroni only for hypothesis-confirmatory tests (small
number of pre-registered hypotheses).

### "What is the North et al. +1 correction?"

For permutation tests, the naive p-value is
`n_extreme / n_permutations`. This gives p=0 when no permutation
exceeds the observed statistic, which is misleading. The correction:
`p = (n_extreme + 1) / (n_permutations + 1)`. This ensures p > 0
and gives a valid conservative bound.

### "Why do we need 3+ seeds?"

Any single stochastic result (random train/test split, random
initialization, random permutation) could be a lucky or unlucky draw.
Running 3+ seeds and reporting mean plus/minus std gives a measure of
stability. If a finding only appears in 1 of 3 seeds, it is not
robust.

---

## Git and workflow

### "What is the pre-commit hook?"

Running `make install` sets up pre-commit hooks that automatically
run `ruff` (linting) and `black` (formatting) on staged files before
each commit. If a check fails, the commit is rejected -- fix the
issue and try again. Do not bypass with `--no-verify`.

### "What is the smoke test?"

`make smoke` (or `python scripts/smoke_test.py`) is the canonical
"is everything wired correctly?" check. It builds a 20-problem
synthetic HDF5 cache, runs each workstream against it, and reports
pass/fail. Run it before committing any change that touches
cross-workstream code.

### "How do I format my code?"

```bash
make format    # runs black + ruff --fix
make lint      # runs ruff check (no auto-fix)
```

Standards: black with line-length 100, ruff with the rule set in
`pyproject.toml`. Type hints on all public functions, `@beartype`
decorator on functions exported in `__init__.py`, `jaxtyping` for
tensor shapes.
