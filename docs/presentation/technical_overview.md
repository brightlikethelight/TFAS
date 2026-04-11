---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 22px;
  }
  h1 {
    color: #1a1a2e;
  }
  h2 {
    color: #16213e;
  }
  table {
    font-size: 18px;
  }
  blockquote {
    border-left: 4px solid #e94560;
    padding-left: 1em;
    color: #555;
    font-size: 20px;
  }
  code {
    font-size: 16px;
  }
---

# The Deliberation Gradient

### Technical Overview -- Team Onboarding

**HUSAI Research Project**
Spring/Summer 2026

---

# Theoretical Background: Dual-Process Theory

## Kahneman's System 1 / System 2 (2011)

- **System 1**: Fast, automatic, heuristic -- "the bat costs $0.10"
- **System 2**: Slow, effortful, deliberative -- "wait, let me set up the equation"

## The modern view: it's not binary

- Melnikoff & Bargh (2018), "The Mythical Number Two" -- the strict binary is a simplification
- Evans & Stanovich themselves now endorse a continuum
- Processing modes share computational substrate; the distinction is **graded**

**Our framing**: deliberation-intensity gradient, not a binary S1/S2 switch.

---

# Why "Graded" Matters for Our Design

## What kills a dual-process paper in review

1. **"You're reifying a folk-psychological dichotomy"** -- the Melnikoff critique
2. **"S1/S2 is just difficulty in disguise"** -- the most predictable fatal objection
3. **"Behavioral differences don't imply mechanistic differences"** -- why we need interpretability

## How we handle each

1. Graded framing from day 1. We never claim binary modes exist.
2. **2x2 difficulty x congruence design** -- matched controls at every difficulty level.
3. Five complementary mechanistic methods, not just behavioral benchmarks.

> A graded finding is *more* publishable -- more nuanced, defensible, and preempts the strongest critiques.

---

# Related Work: Key Precedents

## What exists

- **Coda-Forno et al. (2025)**: dual-architecture models show *overlapping* S1/S2 subspaces -- supports graded view
- **Ziabari et al. (2025)**: "Reasoning on a Spectrum" -- monotonic interpolation between S1/S2-aligned models
- **He et al. (2026)**: latent reasoning mode identified -- supports our hypothesis that a mode exists

## What we build on directly

- **Zhang et al. (2025)**: probed R1 hidden states for self-verification -- we extend to S1/S2 characterization
- **Fartale et al. (2025)**: attention-based recall vs. reasoning distinction -- we add SAE + causal methods
- **Ji-An et al. (2025)**: metacognitive monitoring in activations -- our stretch goal extends this

---

# Related Work: Threats and Competitors

## CogBias (Huang et al., Apr 2026) -- closest competitor

- Cognitive bias benchmark + linear probes + activation steering
- Achieved 26-32% bias reduction via steering
- **Our differentiators**: (1) mode characterization not bias mitigation, (2) multi-method not probe-only, (3) standard vs reasoning model comparison, (4) built-in falsification

## Ma et al. (Jan 2026) -- critical threat

- 45-90% of claimed SAE "reasoning features" are spurious token-level artifacts
- **We build their falsification framework in from day 1**

## "Can Aha Moments Be Fake?" (Oct 2025)

- Only ~2.3% of CoT steps causally influence final answers
- Implication: look in **activations**, not tokens, for genuine reasoning shifts

---

# One-Sentence Novelty

> "First unified mechanistic characterization of dual-process cognition in LLMs, combining probing, SAEs, attention entropy, and representational geometry on contrastively-designed cognitive bias benchmarks, with systematic comparison across standard and reasoning model architectures and built-in feature falsification."

---

# Benchmark Design: 7 Categories

| Category | Conflict | Control | Source | Key Design Choice |
|----------|---------|---------|--------|-------------------|
| CRT Variants | 60 | 60 | Hagendorff OSF + templates | Novel items only (classics memorized) |
| Base Rate Neglect | 40 | 40 | Custom + Bayesian verification | Exact posterior computed for each |
| Belief-Bias Syllogisms | 40 | 40 | Evans taxonomy | 4 conditions: believable x valid |
| Anchoring | 30 | 30 | arXiv 2511.05766 + extensions | Validated anchoring magnitudes |
| Framing Effects | 25 | 25 | Tversky & Kahneman variants | Gain/loss matched pairs |
| Conjunction Fallacy | 20 | 20 | Linda problem isomorphs | Novel protagonists/scenarios |
| Multi-Step Arithmetic | 40 | 40 | Graded 1-5 steps | With/without carrying traps |

**Total**: ~255 conflict + ~255 control = ~510 items (compact set; full set ~580)

Every conflict item has a structurally identical no-conflict control where S1 and S2 cue the same answer.

---

# Benchmark Design: Scoring and Controls

## Three-tier scoring

1. **Binary**: correct / incorrect
2. **Categorical** (primary for probing): correct / S1-lure / other-wrong / refusal
3. **Continuous**: distance from S2-correct vs S1-predicted answer

## Contamination handling

- Classic CRT items run as contamination baselines only (never primary)
- Memorization test: prompt model to complete original CRT verbatim
- Minimum 5 paraphrases per problem; report variance across paraphrases

## Dual format

- **Constrained free-response**: "Answer with just a number:" (for activation analysis)
- **Multiple choice**: (for behavioral validation and comparison with literature)

---

# Activation Extraction: Token Positions

| ID | Position | Which Models | Purpose |
|----|----------|-------------|---------|
| **P0** | Last token of prompt | All | **Primary** -- pre-generation state, no autoregressive contamination |
| P2 | Answer token | All | Final commitment |
| T0 | First token after `<think>` | Reasoning only | Problem encoding in reasoning mode |
| T25/T50/T75 | 25/50/75% through thinking | Reasoning only | Deliberation trajectory |
| Tend | Last token before `</think>` | Reasoning only | Answer crystallization |

Plus: full per-token sweep on 50 stratified problems for empirical position optimization.

---

# Activation Extraction: HDF5 Cache

## Storage format

All workstreams read from a single HDF5 file per model (see `docs/data_contract.md`).

```
data/activations/{model_key}.h5
  /{problem_id}/
    /P0/          # shape: (n_layers, hidden_dim) -- BF16
    /P2/
    /T0/          # reasoning models only
    ...
    /metadata     # JSON: problem_id, category, condition, model_answer, ...
```

## Storage estimates

- Main experiment (4 models x ~500 problems x 32 layers x 4096 hidden x BF16): **~3 GB**
- Position sweep (50 problems x full sequence): **~26 GB**
- Attention metrics (computed on-the-fly, not stored as full matrices)

Hardware: 1x A100-80GB, 15-30 hours for all models.

---

# Activation Extraction: Reasoning Model Traces

## R1-Distill `<think>...</think>` blocks

```
<think>
Let me work through this step by step.      <- T0
The problem says a bat and ball cost...      <- T25
Wait, I initially thought $0.10 but...      <- T50  (self-correction!)
So the ball must cost $0.05.                 <- T75
</think>                                     <- Tend
The answer is $0.05.                         <- P2
```

## Self-correction detection

- Parse thinking traces for correction markers ("wait", "actually", "let me reconsider")
- Expected yield: 20-40% of problems show self-correction (~80-160 instances)
- Extract activations at T_lure, T_correct, T_answer for trajectory probing
- **Warning**: only ~2.3% causally influence the answer. Must validate in activation space.

---

# Workstream 1: Linear Probes

## Probe hierarchy (per target, per layer, per position)

1. **Mass-mean probe** -- sanity check, no training
2. **L2-regularized logistic regression** -- primary result (`LogisticRegressionCV`, nested 5-fold)
3. **2-layer MLP** -- quantifies nonlinearity
4. **CCS** (Contrast-Consistent Search) -- unsupervised corroboration

## Probing targets

| Target | Question |
|--------|----------|
| Task type | Can we decode conflict vs. no-conflict? |
| Correctness | Will the model answer correctly? |
| Bias susceptibility | Is the model producing the S1 lure? |
| Processing mode | Same problem, conflict vs. no-conflict format |

---

# Linear Probes: Controls (Non-Negotiable)

## Hewitt & Liang control tasks

Train probes on **random labels**. Measure selectivity:

```
selectivity = real_accuracy - random_label_accuracy
```

If selectivity < 5 percentage points, the probe is exploiting expressiveness, not representation. **This kills the finding.**

## Additional controls

- **Permutation tests**: 1000 shuffles for significance (with 4096 dims and ~500 points, chance accuracy from noise is a real risk)
- **Leave-one-category-out**: train on 6 task categories, test on 7th -- tests whether the probe learned S1/S2 or domain
- **Difficulty matching**: match S1 and S2 problems by model accuracy (within 0.1); probe on matched subset

## Key figure

Layer-wise probe accuracy curves -- 4 models, bootstrap 95% CIs, permutation significance threshold.

---

# Linear Probes: Self-Correction Trajectory

## Novel analysis for reasoning models

When R1-Distill initially states the S1 lure then corrects itself:

1. Extract activations at **T_lure** (lure answer stated)
2. Extract activations at **T_correct** ("wait, actually...")
3. Extract activations at **T_answer** (correct answer stated)

Probe for the S1-to-S2 transition at these exact positions.

## Expected outcome

- Probe confidence should shift from "S1-like" to "S2-like" across the trajectory
- ~80-160 instances expected (20-40% of problems)
- If the transition is visible in activation space but not just in tokens, this is strong evidence for a genuine mechanistic shift

**Key figure**: Probe probability across thinking trace -- the "deliberation trajectory."

---

# Workstream 2: SAE Feature Analysis

## Available pre-trained SAEs

| SAE | Model | Layers | Features | Source |
|-----|-------|--------|----------|--------|
| Llama Scope | Llama-3.1-8B-Base | All 32 | 32K / 128K | HuggingFace `fnlp/` |
| Gemma Scope | Gemma-2-9B-IT | Multiple | 16K - 1M | Google DeepMind |
| Goodfire R1 | DeepSeek-R1 (671B) | Layer 37 | -- | Known features: Backtracking #15204, Self-reference #24186 |

**Caveat**: Llama Scope trained on Base, not Instruct. Verify reconstruction loss before trusting.
**Caveat**: For R1-Distill-Llama-8B, pre-trained SAEs may not fit. Test first; train custom via SAELens if needed.

---

# SAE Features: Analysis Protocol

## Differential activation analysis

1. For each feature: compute mean activation on conflict vs. no-conflict tasks
2. Mann-Whitney U per feature, BH-FDR corrected
3. **Volcano plot** (key figure): x = log fold change, y = -log10(p-value)
   - Upper-right: "deliberation features" (more active on S2-requiring tasks)
   - Upper-left: "heuristic features" (more active on S1-easy tasks)

## Ma et al. falsification (MANDATORY)

For every candidate S1/S2 feature:
1. Identify top-3 activating tokens
2. Inject those tokens into 100 random non-cognitive-bias texts
3. If feature still activates -> **spurious** (token-level artifact)
4. Report falsification rate for all candidates
5. Only features surviving falsification are "real"

Ma et al. found 45-90% spurious. We must expect and report this.

---

# SAE Features: Causal Validation

## Ablation experiments

- **Ablate S2-associated features** -> does model become more bias-prone on S1-lure problems?
- **Amplify S2-associated features** -> does model resist biases better?
- **Random feature ablation** as control (must show no systematic effect)

## Side effect monitoring

- Check MMLU / HellaSwag to ensure general capability is preserved
- A "deliberation feature" that also destroys language modeling is not specific

## Dose-response curve

- Vary steering coefficient: 0.5, 1.0, 2.0, 3.0, 5.0
- Plot bias susceptibility vs. steering strength
- Monotonic relationship = strong causal evidence
- Non-monotonic = feature is doing something more complex

---

# Workstream 3: Attention Entropy

## Metrics (per head, per position)

| Metric | Formula | Why |
|--------|---------|-----|
| Shannon entropy | H = -sum(a_i log2 a_i) | Standard |
| Normalized entropy | H / log2(t) | Controls for sequence length |
| Gini coefficient | Scale-invariant | Co-primary metric |

## Head classification (following Fartale et al.)

- **S2-specialized**: significantly higher entropy on S2 tasks in >= 3 of 5 metrics, BH-FDR corrected, effect size |r_rb| >= 0.3
- **S1-specialized**: opposite direction
- Report **proportion** (not count) for cross-architecture comparison

---

# Attention Entropy: Complications

## GQA non-independence

- Llama: 32 query heads share 8 KV heads (groups of 4)
- Heads in the same KV group are NOT statistically independent
- Report at both query-head and KV-group granularity

## Gemma-2 sliding window

- Odd layers use 4096-token local window; even layers use global attention
- Analyze separately. Never pool.

## Full matrix materialization

- Never store full attention matrices (~128 GB per layer at 32K tokens)
- Compute entropy/Gini **incrementally** inside forward hooks
- Cannot use FlashAttention (need materialized attention per token)

## Statistical plan

Mann-Whitney U per head, BH-FDR at q=0.05 across all heads x metrics (~5120 tests for Llama). Rank-biserial correlation for effect size.

---

# Workstream 4: Representational Geometry

## Methods

- **PCA** (primary, linear) + **UMAP** (secondary, nonlinear) + **random projection** (control)
- Random projection baseline is non-negotiable: if clusters visible after random Gaussian projection to 2D (100 repeats), structure is genuine

## Key metric: layer-wise cosine silhouette score

X-axis = layer, Y-axis = silhouette, 4 model lines with bootstrap CIs.

## Critical pitfall: d >> N

With d=4096 and N~500, **any** two random classes are linearly separable (Cover's theorem).
**Must PCA to 50-100 dims before SVM.** Compare margin against shuffled-label baseline.

## CKA analysis

Layer-matched CKA between Llama-3.1-8B and R1-Distill-Llama-8B.
**Prediction**: CKA on S2 tasks drops faster in late layers than S1 tasks (reasoning training specifically changed S2 processing).

---

# Workstream 5: Causal Interventions

## Activation steering

1. Identify "deliberation direction" from probe weights or top SAE features
2. Add scaled direction to residual stream at inference time
3. Measure: does P(correct on S1-lure problems) increase?

## Controls

- **Random direction steering**: must show no systematic effect
- **Dose-response**: vary coefficient 0.5 to 5.0
- **Side effects**: MMLU/HellaSwag capability check

## Integration with SAE ablation

- SAE feature ablation (workstream 2) provides one causal test
- Activation steering (this workstream) provides an independent one
- Convergent causal evidence from both methods = strongest result

---

# Go/No-Go Decision Points

| Week | Gate | Pass Criterion | If Fail |
|------|------|---------------|---------|
| **2** | Behavioral validation | Models show >30% lure responses on conflict items | Revise tasks or pivot research question |
| **3** | Infrastructure | Activations cached for 2+ models | Descope to 2 models |
| **5** | Probes work | Probe ROC-AUC > 0.6 for S1/S2 at some layer | Try nonlinear probes, different positions |
| **7** | SAE features | Significant differential features after Ma et al. falsification | Drop SAE workstream, focus probes + attention |
| **9** | Full assessment | Which workstreams have results? | Commit to final paper scope |

**Philosophy**: fail fast, scope honestly. Every outcome is publishable (see next slide).

---

# Results Scenarios

| Scenario | What we find | Narrative | Venue |
|----------|-------------|-----------|-------|
| **Strong positive** | Clear S1/S2 separation, SAE features found, causal interventions work | "First mechanistic evidence for dual-process computation in LLMs" | ICML/NeurIPS main |
| **Mixed** (most likely) | Probes work, some SAE features, noisy attention/geometry | "The gradient of deliberation -- S1/S2 is continuous, not binary" | ICML MechInterp Workshop, ICLR 2027 |
| **Negative** | No S1/S2 distinction in internal representations | "Against dual-process theory in LLMs -- behavioral mimicry without mechanistic substrate" | Alignment Forum, NeurIPS SoLaR |
| **Surprising** | Distinction exists but not where expected | "Deliberation is not where we expected" | Often the most interesting |

**All four scenarios are publishable.** This is by design.

---

# Publication Strategy

## Timeline

| Date | Milestone | Format |
|------|-----------|--------|
| **June 2026** | Alignment Forum post | Probe + SAE preliminary results (establishes priority) |
| **July 2026** | ICML MechInterp Workshop | 4-6 pages, probes + SAE on all models |
| **October 2026** | ICLR 2027 submission | 8 pages + supplement, all workstreams + causal validation |

## Paper structure (recommended)

**Main paper (8 pages)**: probes + SAE features + attention entropy (3 strongest methods)
**Supplement**: geometry, metacognitive monitoring, per-task breakdowns, robustness

## Key figures (6)

1. Benchmark schematic with S1/S2 examples
2. Layer-wise probe accuracy curves (4 models, CIs, permutation threshold)
3. SAE volcano plot (differential feature activation)
4. Causal intervention bar chart (ablate, amplify, random control)
5. Attention entropy ridge plots (S1 vs S2 by layer group)
6. Self-correction trajectory (probe probability across thinking trace)

---

# Next Steps and Current Blockers

## Immediate priorities (weeks 1-2)

1. **FASRC access**: need faculty sponsor for GPU allocation (or RunPod fallback at ~$80-360)
2. **Benchmark finalization**: Hagendorff OSF dataset download, novel item generation, validation
3. **First extraction run**: smoke test on 10 problems x 1 model to validate HDF5 pipeline

## Current blockers

- [ ] FASRC allocation (or RunPod budget approval)
- [ ] Hagendorff OSF dataset access confirmation
- [ ] Team role assignments finalized
- [ ] `docs/data_contract.md` reviewed and signed off by all workstream leads

## Compute budget

| Scenario | GPU-hours | Cost (RunPod) |
|----------|-----------|---------------|
| Conservative (pre-trained SAEs) | ~60 hours | $80-120 |
| With custom SAE training | ~150 hours | $200-360 |
| With FASRC | Same hours | Free |

**First milestone**: behavioral validation by end of week 2.
