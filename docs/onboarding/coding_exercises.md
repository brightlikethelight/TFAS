# Coding Exercises -- The Deliberation Gradient

Five exercises that teach the codebase by doing. All exercises use the
synthetic HDF5 cache and run on CPU -- no GPU, no model downloads, no
internet required.

**Setup** (run this once before starting any exercise):

```python
import sys
sys.path.insert(0, "tests")
sys.path.insert(0, "src")

import tempfile
from pathlib import Path

from conftest import build_synthetic_hdf5, SYNTH_MODEL_KEY
from s1s2.utils.io import (
    open_activations,
    load_problem_metadata,
    list_models,
    get_residual,
    get_behavior,
    get_attention_metric,
    position_labels,
    position_valid,
)

# Build a synthetic cache in a temp directory
CACHE_PATH = Path(tempfile.mkdtemp()) / "exercise.h5"
build_synthetic_hdf5(CACHE_PATH)
print(f"Synthetic cache: {CACHE_PATH}")
```

---

## Exercise 1: Read the benchmark (easy, 30 min)

**Goal**: Understand the benchmark structure and the matched-pair design.

**Task**: Load the benchmark JSONL, filter by category, find all
matched pairs, and print a formatted table showing conflict prompts
side by side with their no-conflict controls.

```python
from s1s2.benchmark.loader import load_benchmark

# 1. Load the benchmark
items = load_benchmark("data/benchmark/benchmark.jsonl")
print(f"Total items: {len(items)}")

# 2. Count items per category
from collections import Counter
cats = Counter(item.category for item in items)
for cat, count in sorted(cats.items()):
    print(f"  {cat}: {count}")

# 3. Build a dict of matched pairs: pair_id -> {True: conflict_item, False: control_item}
pairs: dict[str, dict[bool, object]] = {}
for item in items:
    pid = item.matched_pair_id
    if pid not in pairs:
        pairs[pid] = {}
    pairs[pid][item.conflict] = item

# 4. Print a formatted table for CRT items
print(f"\n{'='*80}")
print("CRT Matched Pairs")
print(f"{'='*80}")
for pid, pair in sorted(pairs.items()):
    if True not in pair or False not in pair:
        continue
    if pair[True].category != "crt":
        continue
    print(f"\nPair: {pid}")
    print(f"  CONFLICT:   {pair[True].prompt[:80]}...")
    print(f"  CONTROL:    {pair[False].prompt[:80]}...")
    print(f"  Correct:    {pair[True].correct_answer}")
    print(f"  Lure:       {pair[True].lure_answer}")
    print(f"  Difficulty: {pair[True].difficulty}")
```

**Check your understanding**: Why does every conflict item need a
matched control? (Answer: to distinguish "the model processes S1 vs
S2 differently" from "the model processes easy vs hard differently.")

---

## Exercise 2: Inspect activations (easy, 30 min)

**Goal**: Navigate the HDF5 activation cache and verify the planted
signal in the synthetic data.

**Task**: Load the synthetic HDF5, extract layer 0 and layer 2
activations at position P0, compute the mean activation per class
(conflict vs no-conflict) on dimension 0, and verify the planted
signal.

```python
import numpy as np

with open_activations(str(CACHE_PATH)) as f:
    meta = load_problem_metadata(f)
    conflict = meta["conflict"]
    model = list_models(f)[0]

    # 1. Extract activations at layer 0 and layer 2, position P0
    X0 = get_residual(f, model, layer=0, position="P0")
    X2 = get_residual(f, model, layer=2, position="P0")

    print(f"Shape: {X0.shape}")  # (20, 32) -- 20 problems, 32 hidden dims

    # 2. Compute per-class means on dimension 0
    for layer_name, X in [("Layer 0 (noise)", X0), ("Layer 2 (signal)", X2)]:
        mean_conflict = X[conflict, 0].mean()
        mean_control = X[~conflict, 0].mean()
        diff = mean_conflict - mean_control
        print(f"{layer_name}:  conflict={mean_conflict:.3f}  "
              f"control={mean_control:.3f}  diff={diff:.3f}")

    # 3. Try all 4 layers -- which one has the signal?
    print("\nPer-layer difference on dim 0:")
    for layer in range(4):
        X = get_residual(f, model, layer=layer, position="P0")
        diff = X[conflict, 0].mean() - X[~conflict, 0].mean()
        print(f"  Layer {layer}: {diff:+.3f}" +
              (" <-- planted signal" if abs(diff) > 0.5 else ""))
```

**Extension**: Plot the distribution of dimension 0 activations at
layer 2 for conflict vs control items using matplotlib. Do they
separate cleanly?

```python
import matplotlib.pyplot as plt

with open_activations(str(CACHE_PATH)) as f:
    meta = load_problem_metadata(f)
    conflict = meta["conflict"]
    model = list_models(f)[0]
    X2 = get_residual(f, model, layer=2, position="P0")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(X2[conflict, 0], bins=15, alpha=0.6, label="Conflict (S1 lure)")
ax.hist(X2[~conflict, 0], bins=15, alpha=0.6, label="No-conflict (control)")
ax.set_xlabel("Residual dim 0 activation")
ax.set_ylabel("Count")
ax.set_title("Layer 2, position P0 -- planted signal")
ax.legend()
plt.tight_layout()
plt.savefig("exercise2_signal.png", dpi=150)
print("Saved exercise2_signal.png")
```

---

## Exercise 3: Train your first probe (medium, 45 min)

**Goal**: Use the probing methodology to classify conflict vs
no-conflict items from residual stream activations, and understand
per-layer accuracy curves.

**Task**: For each of the 4 layers in the synthetic HDF5, train a
logistic regression probe to classify conflict vs no-conflict at
position P0. Plot the per-layer accuracy curve. Verify that layer 2
has the best signal.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

with open_activations(str(CACHE_PATH)) as f:
    meta = load_problem_metadata(f)
    conflict = meta["conflict"].astype(int)
    model = list_models(f)[0]

    results = []
    for layer in range(4):
        X = get_residual(f, model, layer=layer, position="P0")

        # 5-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        for train_idx, test_idx in cv.split(X, conflict):
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X[train_idx], conflict[train_idx])
            proba = clf.predict_proba(X[test_idx])[:, 1]
            auc = roc_auc_score(conflict[test_idx], proba)
            fold_aucs.append(auc)

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        results.append((layer, mean_auc, std_auc))
        print(f"Layer {layer}: AUC = {mean_auc:.3f} +/- {std_auc:.3f}")

# Plot the per-layer accuracy curve
import matplotlib.pyplot as plt

layers, aucs, stds = zip(*results)
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(layers, aucs, yerr=stds, marker="o", capsize=5)
ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
ax.set_xlabel("Layer")
ax.set_ylabel("ROC-AUC")
ax.set_title("Probe accuracy by layer (conflict vs no-conflict)")
ax.set_xticks(layers)
ax.legend()
plt.tight_layout()
plt.savefig("exercise3_probe_curve.png", dpi=150)
print("Saved exercise3_probe_curve.png")
```

**Now add the Hewitt-Liang control**: Train the same probe on
*random labels* (permuted `conflict` array) and compute selectivity
= real AUC - control AUC. Per `CLAUDE.md`, selectivity < 5
percentage points means the signal is probe expressiveness, not
representation content.

```python
# Hewitt-Liang control: random-label baseline
with open_activations(str(CACHE_PATH)) as f:
    meta = load_problem_metadata(f)
    conflict = meta["conflict"].astype(int)
    model = list_models(f)[0]

    rng = np.random.default_rng(42)
    random_labels = rng.permutation(conflict)

    for layer in [0, 2]:  # Compare noise layer vs signal layer
        X = get_residual(f, model, layer=layer, position="P0")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        real_aucs, ctrl_aucs = [], []
        for train_idx, test_idx in cv.split(X, conflict):
            # Real labels
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X[train_idx], conflict[train_idx])
            real_aucs.append(roc_auc_score(
                conflict[test_idx], clf.predict_proba(X[test_idx])[:, 1]
            ))
            # Random labels (control)
            clf_ctrl = LogisticRegression(max_iter=1000, random_state=42)
            clf_ctrl.fit(X[train_idx], random_labels[train_idx])
            ctrl_aucs.append(roc_auc_score(
                random_labels[test_idx], clf_ctrl.predict_proba(X[test_idx])[:, 1]
            ))

        selectivity = np.mean(real_aucs) - np.mean(ctrl_aucs)
        print(f"Layer {layer}: real={np.mean(real_aucs):.3f}  "
              f"control={np.mean(ctrl_aucs):.3f}  "
              f"selectivity={selectivity:+.3f}")
```

**Check your understanding**: Why might selectivity be low at layer 0
even though the probe might achieve AUC > 0.5? (Hint: with
hidden_dim=32 and only 20 samples, a flexible model can memorize
random labels too -- this is the d >> N pitfall from `CLAUDE.md`.)

---

## Exercise 4: Run SAE analysis (medium, 1 hr)

**Goal**: Understand the SAE analysis pipeline end-to-end: encode
activations through an SAE, run differential analysis, and inspect
the results.

**Task**: Create a MockSAE, encode the synthetic activations through
it, run differential activation analysis, and produce a volcano plot.

```python
import torch
import numpy as np
from s1s2.sae.loaders import MockSAE
from s1s2.sae.differential import differential_activation, encode_batched

with open_activations(str(CACHE_PATH)) as f:
    meta = load_problem_metadata(f)
    conflict = meta["conflict"]
    model = list_models(f)[0]

    # 1. Load activations at layer 2 (where the signal is)
    X = get_residual(f, model, layer=2, position="P0")
    print(f"Activations: {X.shape}")  # (20, 32)

# 2. Create a MockSAE (random weights, no download needed)
hidden_dim = X.shape[1]
n_features = 128  # small for testing
sae = MockSAE(hidden_dim=hidden_dim, n_features=n_features, seed=42)

# 3. Encode activations through the SAE
feature_acts = encode_batched(sae, X)  # (20, 128)
print(f"Feature activations: {feature_acts.shape}")
print(f"Sparsity: {(feature_acts == 0).mean():.2f}")

# 4. Run differential analysis (Mann-Whitney U + BH-FDR across features)
result = differential_activation(
    feature_activations=feature_acts,
    conflict=conflict,
)
print(f"\nDifferential analysis results:")
print(f"  Features tested: {len(result.df)}")
print(f"  Significant (FDR < 0.05): {result.df['significant'].sum()}")

# 5. Inspect the top features (sorted by p-value)
top = result.df.nsmallest(5, "p_value")
for _, row in top.iterrows():
    print(f"  Feature {int(row['feature_id'])}: p={row['p_value']:.4f}  "
          f"effect={row['effect_size']:+.3f}  "
          f"significant={row['significant']}")
```

**Extension**: Make a volcano plot (log2 fold change vs -log10
p-value). With random SAE weights the features will be largely
non-significant, but the pipeline mechanics should work correctly.

```python
import matplotlib.pyplot as plt

log2_fc = result.df["log_fc"].values
neg_log10_p = -np.log10(np.clip(result.df["p_value"].values, 1e-10, None))
significant = result.df["significant"].values

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(log2_fc[~significant], neg_log10_p[~significant],
           alpha=0.5, s=20, color="gray", label="Not significant")
ax.scatter(log2_fc[significant], neg_log10_p[significant],
           alpha=0.8, s=30, color="red", label="FDR < 0.05")
ax.axhline(-np.log10(0.05), color="black", linestyle="--", alpha=0.3)
ax.set_xlabel("log2(fold change)")
ax.set_ylabel("-log10(p-value)")
ax.set_title("SAE Feature Volcano Plot (synthetic data)")
ax.legend()
plt.tight_layout()
plt.savefig("exercise4_volcano.png", dpi=150)
print("Saved exercise4_volcano.png")
```

**Check your understanding**: Why is the Ma et al. falsification test
necessary even when features pass the statistical test? (Answer:
because a feature might fire on specific tokens like "Let" or "First"
that correlate with but do not encode reasoning. The statistical test
catches spurious *differences* but not spurious *causes* of those
differences.)

---

## Exercise 5: Build a new benchmark item (hard, 1 hr)

**Goal**: Understand the benchmark generation system and contribute a
new item.

**Task**: Using the generators in `s1s2.benchmark.generators`, create
a new CRT bat-and-ball variant with a novel cover story. Add it to
`build.py`, regenerate the benchmark, and verify it passes validation.

### Step 1: Understand the generator API

```python
from s1s2.benchmark.generators import (
    anchoring_isomorph,
    base_rate_isomorph,
    conjunction_isomorph,
    framing_isomorph,
    syllogism_isomorph,
)
from s1s2.benchmark.templates import bat_ball_isomorph

# Look at how an existing CRT generator works
help(bat_ball_isomorph)
```

### Step 2: Create a new item

Write a new bat-and-ball variant. The classic problem: "A bat and a
ball cost $1.10 total. The bat costs $1.00 more than the ball. How
much does the ball cost?" (Lure: $0.10. Correct: $0.05.)

Your task: create a **novel cover story** (not bat-and-ball, not
any classic CRT surface form) that has the same algebraic structure:

- Two items that together cost X.
- One costs Y more than the other.
- The lure answer is X - Y (the "fast" intuitive response).
- The correct answer requires algebra.

Example novel cover story: "A notebook and a pen together cost $2.20.
The notebook costs $2.00 more than the pen. How much does the pen
cost?" (Lure: $0.20. Correct: $0.10.)

### Step 3: Register it in build.py

Look at `src/s1s2/benchmark/build.py` to see how existing items are
registered. Each item is a spec dict passed to a generator function.
Add your new item following the same pattern.

### Step 4: Regenerate and validate

```bash
# Regenerate the benchmark
python -m s1s2.benchmark.cli generate

# Validate it
python -m s1s2.benchmark.cli validate

# Check your item shows up
python -m s1s2.benchmark.cli stats
```

**Check your understanding**: Why do we use novel cover stories
instead of the classic bat-and-ball? (Answer: contamination. Models
have seen the classic CRT in training data and may have memorized the
answer. Novel structural isomorphs test whether the model *reasons*
about the structure vs. *recalls* a memorized answer.)

---

## Difficulty summary

| Exercise | Difficulty | Time | Best for role |
|----------|-----------|------|---------------|
| 1. Read the benchmark | Easy | 30 min | Writing Lead, anyone new |
| 2. Inspect activations | Easy | 30 min | Infrastructure, Attention |
| 3. Train your first probe | Medium | 45 min | Probes + Geometry |
| 4. Run SAE analysis | Medium | 1 hr | SAE + Causal |
| 5. Build a new benchmark item | Hard | 1 hr | Benchmark Lead |
