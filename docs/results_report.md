# Results Report: Mechanistic Signatures of S1/S2 Processing in LLMs

**Project**: s1s2 (HUSAI Semester Project)
**Date**: 2026-04-09
**Status**: Behavioral validation complete; probing and geometry analyses on initial model subset complete; SAE, attention, and causal workstreams pending.

---

## Overview

This report synthesizes all experimental results obtained so far in the s1s2 project, which investigates whether large language models encode mechanistically distinguishable processing modes analogous to the dual-process (System 1 / System 2) framework from cognitive psychology. We test this using a 330-item cognitive bias benchmark across five model configurations, combined with linear probing, representational geometry, and cross-model comparison of residual stream activations.

We stress at the outset: the dual-process framing is operational. We do not claim LLMs "have" System 1 and System 2. We ask whether there exist graded, linearly decodable signatures in the residual stream that correlate with heuristic-prone versus deliberation-requiring task conditions, and whether reasoning training modulates those signatures.

---

## 1. Behavioral Results: Category-Specific Vulnerability

We evaluated five model configurations on the full 330-item benchmark (7 categories, matched conflict/control pairs). The "lure rate" is the percentage of conflict items where the model produces the heuristic (S1) answer rather than the normatively correct answer.

| Model | Overall | base_rate | conjunction | syllogism | CRT | arithmetic | framing | anchoring |
|-------|---------|-----------|-------------|-----------|-----|------------|---------|-----------|
| Llama-3.1-8B-Instruct | 27.3% | 84% | 55% | 52% | 0% | 0% | 0% | 0% |
| R1-Distill-Llama-8B | 2.4% | 4% | 0% | 0% | 3% | 0% | 0% | 0% |
| R1-Distill-Qwen-7B | ~0% | ~0% | ~0% | ~0% | 0% | 0% | 0% | 0% |
| Qwen 3-8B (no-think) | 21% | 56% | 95% | 0% | 0% | 0% | 0% | 0% |
| Qwen 3-8B (think) | 7% | 4% | 55% | 0% | 0% | 0% | 0% | 0% |

Three findings stand out.

**Only three of seven categories elicit heuristic responses.** Base rate neglect, conjunction fallacy, and syllogistic reasoning (belief-bias) are the vulnerable categories. The remaining four (CRT variants, multi-step arithmetic, framing, anchoring) show floor-level lure rates across all models. This is not a failure of the benchmark. It indicates that these particular LLMs are not susceptible to heuristic lures in those four domains -- likely because the lure structures in CRT, arithmetic, framing, and anchoring are either well-represented in training data or do not trigger the same shortcut mechanisms in transformer architectures. The immune categories serve as built-in negative controls for the mechanistic analyses.

**Reasoning training dramatically reduces susceptibility.** The Llama-3.1-8B to R1-Distill-Llama-8B comparison is the cleanest test: identical architecture, identical parameter count, differing only in reasoning distillation training. The base rate lure rate drops from 84% to 4%. Conjunction drops from 55% to 0%. Syllogism drops from 52% to 0%. The overall lure rate falls from 27.3% to 2.4%. This is a large, consistent effect across all three vulnerable categories and directly supports the hypothesis that reasoning distillation suppresses heuristic-prone processing.

**The within-model thinking toggle partially replicates this.** Qwen 3-8B with identical weights but explicit chain-of-thought (`/think` mode) reduces the overall lure rate from 21% to 7%. The base rate category shows the clearest effect (56% to 4%). However, the conjunction category remains elevated (95% to 55%), suggesting that explicit deliberation helps but does not fully eliminate heuristic susceptibility -- particularly for conjunction fallacy items, which may require a qualitatively different kind of probabilistic reasoning rather than simply "more thinking." This partial effect is informative: it suggests deliberation is not a single switch but a graded intervention whose efficacy varies by task structure.

---

## 2. Probing Results: Linear Decodability of S1/S2 Processing Mode

We trained L2-regularized logistic regression probes on residual stream activations at the last prompt token to classify conflict versus control items. Probes were trained under 5-fold stratified cross-validation with Hewitt-Liang random-label controls for selectivity.

| Model | Vulnerable peak AUC | Layer | Immune peak AUC | Layer | All-category peak AUC | Layer |
|-------|---------------------|-------|-----------------|-------|-----------------------|-------|
| Llama-3.1-8B-Instruct | 0.999 | 14 | 1.000 | 1 | 0.999 | 14 |
| R1-Distill-Llama-8B | 0.929 | 14 | 1.000 | 0 | 0.985 | 16 |
| Qwen 3-8B (no-think) | 0.971 | 34 | -- | -- | 0.997 | 24 |

The headline result: on the three vulnerable categories, Llama achieves AUC 0.999 at layer 14, while R1-Distill achieves AUC 0.929 at the same layer. Both models peak at the same depth in the network, but the reasoning-trained model shows reduced separability. This is consistent with the hypothesis that reasoning distillation integrates S2-like processing into the forward pass, blurring the boundary between processing modes in the residual stream representation. The "where" does not change; the "how sharply" does.

Qwen 3-8B (no-think) peaks later (layer 34 of its deeper architecture) with AUC 0.971, falling between the standard and reasoning-trained Llama models in separability.

### The confound: immune category probes

This is the result that demands the most honest discussion. Probes trained on immune categories (CRT, arithmetic, framing, anchoring) achieve AUC 1.0 at early layers (layer 0-1). These categories have 0% lure rates -- the models never produce heuristic answers on them -- yet a linear probe can perfectly distinguish conflict from control items.

This means the probes are detecting something about the task structure itself (e.g., surface-level differences between conflict and control item phrasings), not exclusively the model's processing mode. The conflict/control distinction is, to some degree, encoded in the input representation before any "processing" has occurred.

This does not invalidate the vulnerable-category probing results, but it constrains their interpretation. The meaningful signal is the **inter-model delta on vulnerable categories**: the drop from AUC 0.999 (Llama) to 0.929 (R1-Distill) at the same layer. If probes were merely detecting input structure, this delta would be zero -- both models receive identical inputs. The fact that reasoning training reduces probe separability specifically on categories where it also reduces lure rates is the genuinely informative finding. Cross-category prediction experiments (train on vulnerable, test on immune, and vice versa) are needed to determine how much of the probe signal is task-structural versus processing-mode-specific.

---

## 3. Geometry Results: Modest but Real Separation

Representational geometry analysis assessed whether conflict and control activations form distinguishable clusters in the residual stream.

**Silhouette scores.** Cosine silhouette scores (computed after PCA to 50 dimensions to address the d >> N pitfall) are positive but low. Llama peaks at 0.079 at layer 13; R1-Distill peaks at 0.059 at layer 0. These are statistically significant (above permutation null) but indicate substantially overlapping distributions rather than clean clusters. For reference, silhouette scores above 0.5 indicate strong cluster structure; scores below 0.1 indicate that the two classes largely overlap in representational space.

**CKA (Centered Kernel Alignment).** Cross-model CKA between Llama and R1-Distill ranges from 0.379 to 0.985 across layers. High CKA in early layers (near 1.0) indicates that the two models represent inputs similarly at the embedding level, which is expected given their shared architecture. The decline in later layers (down to 0.379) quantifies how much reasoning distillation diverges the representation in the layers where task-relevant computation occurs. This is consistent with the probing results: the models start from the same representation and diverge as processing deepens.

The geometry results support a picture of graded, overlapping processing-mode representations rather than discrete clusters. This is actually consistent with the theoretical framing: if S1/S2 is a continuum (as Evans and Stanovich themselves argue), we should not expect clean geometric separation. The modest silhouette scores are not a weakness -- they are what the theory predicts.

---

## 4. What These Results Mean

Taking the behavioral, probing, and geometry results together, four conclusions are warranted at this stage:

**Reasoning training suppresses heuristic processing, and this is visible in the residual stream.** The behavioral effect (84% to 4% on base rate) is mirrored by a mechanistic effect (AUC 0.999 to 0.929 on vulnerable-category probes). The direction is consistent: reasoning training makes the model less susceptible to heuristic lures AND makes the conflict/control distinction less linearly separable. This is what we would expect if reasoning distillation integrates deliberative processing throughout the forward pass rather than segregating it.

**The effect is category-specific, not universal.** Only 3 of 7 bias categories show any heuristic susceptibility. This constrains the generality of any dual-process claim: whatever S1-like processing these models engage in, it is specific to certain probabilistic reasoning tasks (base rates, conjunctions, belief-logic conflicts), not a general heuristic mode.

**Explicit deliberation helps but is not equivalent to reasoning training.** The Qwen thinking toggle reduces but does not eliminate lure rates, and the reduction is uneven across categories. This suggests that the reasoning distillation in R1-Distill produces a qualitatively different internal change than simply prepending a chain-of-thought -- it reshapes the representation, not just the output strategy.

**Probe results require careful interpretation due to the input-structure confound.** The perfect AUC on immune categories at early layers means we cannot claim probes purely detect "processing mode." The inter-model delta on vulnerable categories is the more defensible signal, and cross-prediction experiments are essential before making stronger claims.

---

## 5. What Comes Next

The results so far establish the behavioral phenomenon and provide initial mechanistic evidence. Several analyses are needed to strengthen or constrain these claims:

**Cross-category prediction.** Train probes on vulnerable categories, test on immune categories (and vice versa). If vulnerable-trained probes generalize to immune categories, the signal is task-structural. If they do not, there is a processing-mode-specific component. This is the single most important experiment for resolving the confound.

**SAE feature analysis.** The Goodfire L19 SAE (trained on the exact Llama-3.1-8B-Instruct model, layer 19) enables feature-level analysis. Identifying individual SAE features that differentially activate on conflict versus control items -- and survive Ma et al. falsification -- would provide interpretable mechanistic evidence beyond probe decodability.

**Attention entropy analysis.** Per-head attention entropy comparisons between conflict and control conditions, with proper GQA-aware statistical testing, will determine whether the S1/S2 distinction is reflected in attention patterns or is purely a residual-stream phenomenon.

**Causal interventions.** Steering along S2-associated feature directions (identified via SAE analysis) to shift model behavior from heuristic-prone to deliberative. This is the strongest possible test: if we can causally shift behavior by intervening on identified features, the mechanistic story moves from correlation to causation.

**Additional model pairs.** The Ministral-3-8B pair (same weights, thinking toggle) would provide a third within-architecture comparison. Replication across architectures is essential for generality claims.

**Hewitt-Liang selectivity reporting.** Full selectivity scores (real AUC minus random-label control AUC) must be computed and reported for all probe results. Raw AUC alone is insufficient given the high dimensionality of the representation space.

---

## Summary of Hypothesis Status

| Hypothesis | Status | Key Evidence |
|------------|--------|-------------|
| H1: Linear decodability | **Supported** (with caveats) | AUC 0.999 on vulnerable categories; but immune-category confound needs resolution |
| H2: Reasoning training amplification | **Direction reversed from pre-registration** | R1-Distill shows *lower* AUC (0.929) than Llama (0.999); reasoning training *reduces* separability rather than amplifying it. The behavioral result (lower lure rates) is clear; the mechanistic interpretation requires reframing. |
| H3: SAE feature specificity | **Pending** | Goodfire L19 SAE ready; analysis not yet run |
| H4: Causal efficacy | **Pending** | Depends on H3 |
| H5: Attention entropy | **Pending** | Extraction complete; analysis not yet run |
| H6: Geometric separability | **Weakly supported** | Silhouette > 0 but low (0.079); overlapping distributions |

The H2 result deserves emphasis: the pre-registration predicted reasoning models would show *stronger* S1/S2 separation. The data show the opposite -- reasoning training *blurs* the distinction. This is not a failure; it is an informative surprise. It suggests reasoning distillation does not create a stronger "deliberation mode" that is more distinct from heuristic processing. Instead, it integrates deliberation-like computation throughout the forward pass, making the two modes less distinguishable. The behavioral improvement (fewer lure responses) coexists with reduced representational separability. This reframing -- that effective reasoning training makes S1/S2 less separable, not more -- is arguably a more interesting finding than the pre-registered prediction.
