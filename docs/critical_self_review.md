# Critical Self-Review: s1s2 Project

**Date**: 2026-04-12
**Purpose**: Adversarial review of our own claims before submission. Every finding is interrogated for overclaiming, statistical weakness, and alternative explanations. This document is for internal use -- be honest about what we know and what we are pretending to know.

---

## 1. Are we overclaiming? "Mechanistic signatures of dual-process cognition"

**Yes, we are overclaiming.** Here is what we actually showed:

- Linear probes can distinguish conflict items from control items in the residual stream.
- This distinguishability is slightly lower in a reasoning-distilled model than in a standard model.
- Inference-time CoT does not change this distinguishability at P0.

Here is what we are *calling* that: "mechanistic signatures of dual-process cognition."

The gap between the evidence and the framing is large. A linear probe distinguishing conflict from control items is detecting *task-type features*, not *processing-mode features*, unless we can rule out every way the two item types differ besides processing mode. The cross-prediction test (0.378 AUC) is our best evidence against the surface-confound alternative, but it has its own problems (see point 7 below).

The word "mechanistic" implies causal understanding. We have zero causal evidence. We have not shown that ablating or steering along these directions changes behavior. Without causal interventions, "mechanistic" is aspirational, not descriptive. The correct term is "representational signatures," and even that is generous -- "linear-probe-decodable correlates" is what we literally have.

**Recommendation**: The paper title should use "representational" not "mechanistic." The abstract should never use "dual-process cognition" without "signatures consistent with" or "correlates of." The current elevator pitch is close to honest but still too confident.

---

## 2. The probe AUC gap (0.974 vs 0.930): meaningful or noise?

The bootstrap CIs are [0.952, 0.992] vs [0.894, 0.960]. These do not overlap, so the difference is statistically significant by the non-overlapping-CI heuristic. But several concerns:

**Concern A: Non-overlapping CIs is a conservative test, but the real question is the CI on the *difference*.** Non-overlapping individual CIs guarantees the difference is significant, but the CIs for these two models were computed on *different data partitions* (the bootstrap resamples items within each model independently). The correct test is a paired bootstrap on the difference, resampling the same items through both models. The preregistration specifies this (Section 4.2: "paired bootstrap CI on AUC difference"), but the reported CIs appear to be unpaired. If the paired bootstrap was not actually computed, the claimed significance could be wrong.

**Concern B: Training data differences, not "reorganized representations."** R1-Distill differs from Llama in everything that happened during distillation fine-tuning, not just "reasoning training." The lower AUC could reflect any change in the fine-tuning pipeline -- different data mix, different learning rate schedule, different regularization. Attributing the AUC drop specifically to "reasoning distillation reorganizes representations" is a causal claim we cannot support without an ablation study on the training process itself. We do not control the training; we downloaded the checkpoints.

**Concern C: The AUC numbers are inconsistent across documents.** The scientific narrative reports peak AUC 0.974 at L16 (Llama) and 0.930 at L31 (R1-Distill). The SESSION_STATE reports 0.999 at L14 (Llama) and 0.929 at L14 (R1-Distill). These are different numbers, different layers, and different stories (same layer vs. shifted layer). This inconsistency must be resolved before submission -- either the bootstrap CIs apply to the L14 numbers or the L16/L31 numbers, not both. The "peak layer shifts from L16 to L31" narrative and the "both peak at the same layer L14" narrative are contradictory claims present in different documents.

**Recommendation**: Resolve the number inconsistency immediately. Compute the paired bootstrap CI on the difference. Report both the individual CIs and the paired difference CI. Do not claim "peak layer shift" unless the CI analysis was actually done at the shifted layers.

---

## 3. The OLMo gap (0.998 vs 0.993): noise until proven otherwise

This is a 0.005 AUC difference. Both numbers are essentially at ceiling. There are no bootstrap CIs for OLMo (the documents explicitly note "Bootstrap CIs for Qwen -- Need per-fold data from the pod" and OLMo CIs are not mentioned as computed).

At AUC 0.998, the probe is essentially perfect. The 0.005 drop to 0.993 could be:
- Random variation in the 5-fold CV splits
- One or two items flipping between folds
- A rounding artifact in how AUC is aggregated

Claiming this "replicates" the Llama/R1-Distill pattern is a serious stretch. The Llama/R1-Distill gap is 0.044 AUC (or 0.070 per SESSION_STATE -- another inconsistency). The OLMo gap is 0.005. These are not the same phenomenon at the same magnitude. The direction is consistent (reasoning model lower), but without CIs, the OLMo gap is consistent with zero.

Furthermore, OLMo Think and OLMo Instruct are not the same kind of comparison as Llama vs R1-Distill. OLMo Think is (presumably) a reasoning-enhanced variant of OLMo, but the training differences are different from DeepSeek's distillation process. Treating these as equivalent comparisons requires an argument that is never made.

**Recommendation**: Compute bootstrap CIs for OLMo before claiming replication. If the CI on the difference includes zero, report it as "directionally consistent but not significant." Do not present OLMo as confirming the mechanistic finding. At best it confirms the *behavioral* pattern (14.9% vs 0.9% lure rate, which is genuinely large).

---

## 4. SAE features: 41 out of 65,536 -- signal or noise?

**The honest answer: we cannot tell.** Here is why:

**Problem A: Base rate of false discovery.** With 65,536 tests at BH-FDR q=0.05, we expect ~3,277 false positives under the null. Finding 41 significant features means the actual discovery rate is far below the expected false positive count. This is actually *good* -- it means the BH correction is working and these 41 passed a stringent threshold. But 41 features out of 65,536 (0.063%) is a tiny fraction. The question is whether these 41 features represent a real, meaningful signal about S1/S2 processing, or whether they detect dataset-specific surface patterns.

**Problem B: Trigger tokens are scenario-scaffolding words.** The SAE analysis document itself flags this clearly: "professional" (83% of features), "respondent" (73%), "only" (49%). These are not cognitive-process tokens; they are vocabulary items that differ between conflict and control item wordings. All 41 features could be detecting "this is a base rate problem phrased in survey language" rather than "the model is in System 1 processing mode."

**Problem C: The top two features by effect size are probably spurious.** Features 5402 and 19622 have falsification ratios of 0.476 and 0.295 respectively. The SAE analysis itself notes these are "borderline" and recommends re-running falsification in full mode. These are the #1 and #3 features. If the top features are borderline, the entire feature set is suspect.

**Problem D: We cannot distinguish interpretations #1 (task-structure detectors) from #2 (bias-specific processing).** The SAE analysis correctly identifies three possible interpretations and admits the evidence supports #2 at best. But #1 (these features just detect the surface form of base-rate problems) is equally consistent with all available data. Without cross-category transfer tests for the SAE features -- training an SAE-based classifier on base_rate+conjunction and testing on syllogism -- we cannot rule out #1.

**Problem E: No semantic labels, no causal evidence.** We report feature IDs. We do not know what these features "mean." We have not patched or ablated them. 41 opaque numbers with a confounded trigger-token pattern is not a publishable SAE result.

**Recommendation**: The SAE findings should be relegated to supplementary material or future work. They are not ready for a primary claim. Reporting them as "SAE features survive falsification" in the key findings list is misleading when the top features have borderline falsification ratios and the trigger tokens are scenario-specific.

---

## 5. Natural frequency finding: N=10 and a known scoring bug

This is the weakest empirical claim in the entire project. The problems stack:

**Problem A: N=10.** Ten items. The Llama result (10/10 lure) is striking as a point estimate, but the Clopper-Pearson lower bound is 69.2%. The comparison to 84% (21/25) on probability format gives an increase of +16pp. With these sample sizes, a Fisher exact test p-value for the difference between 10/10 and 21/25 is not significant (p = 0.319). The "natural frequency makes it worse" claim for Llama does not survive a formal significance test.

**Problem B: The R1-Distill result is invalid.** The natural_freq_failure_analysis.md demonstrates, with 100% prediction accuracy, that all R1-Distill natural frequency verdicts are artifacts of a Unicode BPE scoring bug and response truncation. Every single verdict is determined by whether the answer string contains a space, not by model behavior. The "40% lure rate" for R1-Distill is meaningless. The "96% to 40%" reversal -- the most dramatic claim in the entire Gigerenzer analysis -- is based on invalid data.

**Problem C: The scientific narrative still presents the R1-Distill result as real.** Finding 6 ("Natural frequency framing produces a paradoxical reversal in reasoning models") reports R1-Distill going from 4% to 40% lure. The gigerenzer_analysis.md flags the bug in a warning banner but still devotes multiple sections to interpreting the 40% number. The reviewer_response_prep.md (Objection 6) defends the N=10 result without mentioning the scoring bug. These documents are internally inconsistent and partially based on data known to be invalid.

**Problem D: Even the valid Llama result has confounds.** Natural frequency items are longer, have different syntactic structure, introduce explicit counts (10, 1000), and use different vocabulary than the probability-format items. The gigerenzer_analysis itself identifies "token-level distraction" as a confound (Section 3.2). Without controlling for prompt length and syntactic complexity, we cannot attribute the lure rate increase to the frequency format specifically.

**Recommendation**: The R1-Distill natural frequency claim must be retracted until re-collected with the fixed pipeline. The Llama-only result (N=10, non-significant by Fisher exact test, confounded by prompt length) should be presented as a preliminary observation at most, not a "finding." It should not appear in the abstract, key findings list, or as a numbered finding in the scientific narrative. The entire Gigerenzer section should be conditional on re-collection with a larger item set.

---

## 6. The Qwen dissociation: trivially expected, not deep

We probe at P0 -- the last prompt token, before the model has generated any output. Qwen THINK and NO_THINK share identical weights. At P0, the model has processed the input through its transformer layers using those identical weights. Of course the representations are identical at P0. This is not a finding about CoT failing to reshape representations; it is a consequence of measuring before CoT has started.

The interesting comparison -- probing at positions within the thinking trace (T0, T25, T50, T75, Tend) -- is explicitly listed as "planned for the full paper" but has not been done. Without it, the dissociation between training and inference effects is asserted, not demonstrated.

**Counterargument we should anticipate**: "But the behavioral difference between THINK and NO_THINK shows that CoT matters." True, but the CoT happens *after* P0. The fact that P0 is the same just means the model reads the problem the same way regardless of what it will do next. This is trivially true for any autoregressive model with the same weights -- the forward pass up to P0 is deterministic given the input.

**What would be non-trivial**: If probing at Tend (after the full reasoning trace) showed the same AUC as P0, *that* would mean CoT does not reshape representations at all. That is a strong and interesting claim. But we have not tested it.

**Recommendation**: Reframe the Qwen finding honestly. "At the point of initial problem encoding (P0), representations are identical regardless of whether the model will subsequently generate a reasoning trace." Do not claim this demonstrates that "inference-time reasoning does NOT reshape the residual stream." We have not tested the residual stream during or after reasoning. The finding is about initial encoding, which is expected to be identical given identical weights.

---

## 7. Cross-prediction at 0.378: interesting but fragile

A transfer AUC of 0.378 is below chance (0.5). The interpretation is that the probe direction learned from vulnerable categories is anti-correlated with whatever direction exists in immune categories. This is used to argue the probe is "processing-mode-specific."

**Problem A: Anti-correlation vs. specificity.** Below-chance transfer means the vulnerable-category direction actively anti-predicts in immune categories. This could mean: (a) vulnerable and immune categories have opposite processing-mode directions (interesting, supports specificity), or (b) the probe learned some confound that is inverted between category types (e.g., conflict items in vulnerable categories have feature X, while conflict items in immune categories have anti-feature X, where X is a surface feature). We cannot distinguish these without more analysis.

**Problem B: Small sample sizes per category.** The vulnerable set has approximately 57 conflict items (20 base_rate + 12 conjunction + 25 syllogism) and 57 matched controls = 114 items. The immune set used for transfer testing has roughly 85 conflict + 85 control = 170 items. Training a probe on 114 items and testing on 170 items is feasible but noisy. The 0.378 AUC is a single point estimate. The reviewer_response_prep says "the confidence interval excludes 0.5 from above" but does not report the actual CI bounds. Without seeing these bounds, we cannot assess how robustly below-chance this is.

**Problem C: The result does not replicate cleanly in R1-Distill.** R1-Distill shows 0.878 at early layers (high transfer, meaning the probe IS detecting surface features there) and 0.385 at late layers (low transfer, similar to Llama). The narrative says this is "itself a finding about reorganization," but a simpler reading is that the cross-prediction test gives unstable results that depend on which layer you look at. Cherry-picking the late-layer result and calling it "processing-specific" while ignoring the early-layer result that says "surface-feature" is selective reporting.

**Recommendation**: Report the full layer-wise cross-prediction curve, not selected layers. Report CIs on the transfer AUC. Acknowledge that early-layer high transfer in R1-Distill is evidence *against* the specificity claim at those layers. Do not use the word "resolves" -- the cross-prediction test *partially addresses* the confound, it does not resolve it.

---

## 8. Statistical power: the item count problem

The vulnerable set has approximately:
- Base rate neglect: 20 conflict + 20 control = 40 items
- Conjunction fallacy: 12 conflict + 12 control = 24 items
- Syllogism: 25 conflict + 25 control = 50 items
- **Total vulnerable: ~57 conflict + 57 control = 114 items**

All probe AUCs, cross-predictions, transfer matrices, and geometry results are computed on this 114-item set (or subsets of it for per-category analyses). For the transfer matrix, we are training a probe on 20+12=32 conflict items from base_rate+conjunction and testing on the 25 from syllogism (or similar splits). These are very small numbers for estimating AUC with any precision.

The bootstrap CIs for Llama (width 0.040) and R1-Distill (width 0.066) are computed on this ~114-item set. These widths are plausible for this sample size but leave meaningful uncertainty. The CI width difference (0.040 vs 0.066) itself suggests R1-Distill has more variability, possibly because the signal is weaker and more sensitive to which items are in each fold.

The per-category analyses (base_rate AUC, conjunction AUC, syllogism AUC) are based on 24-50 items each. Any per-category AUC estimate has wide uncertainty bands. The transfer matrix entry of 0.993 (base_rate-trained probe tested on conjunction) is based on testing on 24 items. At N=24, even AUC=0.993 has a non-trivial bootstrap CI.

**Recommendation**: Report per-category sample sizes explicitly in all tables. Compute and report bootstrap CIs for the transfer matrix entries. Do not treat 0.993 as meaningfully different from, say, 0.95 without a CI. The "remarkable degree" of transfer between base_rate and conjunction could partly reflect shared surface features in these two categories (both involve probability scenarios with personality descriptions).

---

## 9. The "S2-by-default" interpretation: is there a simpler explanation?

The S2-by-default interpretation says: reasoning training makes the model apply deliberation to everything, so it no longer needs to distinguish conflict from control items, hence lower probe AUC.

A simpler alternative: **reasoning distillation training just teaches the model to produce correct answers on probability problems.** The training data for R1-Distill includes mathematical reasoning traces. The model learned to solve base-rate problems, conjunction problems, and syllogisms from the training data. The lower AUC reflects the fact that the model no longer processes these items as "special" because it has seen similar problems during training and has memorized (or generalized from) the solution patterns.

This alternative makes the same predictions as S2-by-default (lower AUC, better accuracy) but requires no dual-process theory at all. The model is not "applying deliberation to everything" -- it has simply learned the answer patterns.

Evidence that might distinguish these:
- If R1-Distill shows S2-by-default processing on *novel* problem types it was not trained on, that supports the general interpretation.
- If R1-Distill only improves on the specific problem types present in its training data, that supports the memorization interpretation.

The natural frequency result (if the R1-Distill data were valid) would be relevant here: if R1-Distill fails on the same logical problems in a different format, it suggests format-specific learning rather than general deliberation. But the R1-Distill NF data is invalid.

**Recommendation**: Present the S2-by-default interpretation as one possible explanation, not as the established interpretation. Explicitly state the memorization/generalization alternative. Note that the two cannot be distinguished without testing on out-of-distribution problem types.

---

## 10. Minimum publishable unit: what's the actual finding?

Strip away all the framing. The actual findings are:

1. **Instruction-tuned LLMs are bad at some bias categories and good at others.** This is known (Hagendorff et al. 2023, Rao et al. 2024, CogBias 2026).

2. **Linear probes can distinguish conflict from control items in the residual stream.** This is expected given the items differ in textual content (Cover's theorem, d >> N). The cross-prediction test provides partial evidence against the pure-surface-feature interpretation.

3. **The probe AUC is lower in a reasoning-distilled model.** This is novel and interesting, with non-overlapping CIs on the Llama/R1-Distill pair. The interpretation (S2-by-default) is one of several possible explanations.

4. **At P0, Qwen THINK and NO_THINK have identical probe AUCs.** This is expected given identical weights and identical inputs.

**The minimum publishable unit is finding #3** -- the observation that reasoning distillation reduces the linear decodability of conflict vs. control, paired with the behavioral improvement. Combined with the cross-prediction test (#2), this makes a case that reasoning training changes something about how the model represents these tasks internally. Everything else is either known, expected, or underpowered.

For a 4-page workshop paper, this is sufficient if presented honestly. The novelty is the reasoning-model comparison with statistical rigor (bootstrap CIs, cross-prediction control). The Qwen dissociation adds flavor but is weaker than it appears (see point 6). The SAE results are not ready. The natural frequency results are partially invalid. The OLMo "replication" is not yet statistically supported.

---

## The 3 Weakest Claims

### Weak #1: Natural frequency reversal (Finding 6)

- R1-Distill result is entirely invalid (scoring bug, 100% predictable from bug mechanism)
- Llama result is N=10, non-significant by Fisher exact test, confounded by prompt length
- The scientific narrative still presents the R1-Distill number as real
- **Risk**: If a reviewer discovers the scoring bug, the entire paper's credibility is damaged

### Weak #2: SAE features survive falsification (Finding 8)

- 41 features out of 65,536; trigger tokens are scenario-scaffolding words
- Top two features by effect size have borderline falsification ratios (0.476, 0.295)
- No semantic labels, no causal evidence, no cross-category transfer
- Cannot distinguish "base-rate problem detector" from "S1/S2 processing mode detector"
- **Risk**: Any reviewer who reads the trigger token list will immediately see the confound

### Weak #3: OLMo replication of the mechanistic pattern

- The AUC gap is 0.005 (0.998 vs 0.993), no bootstrap CIs computed
- Both values are at ceiling; the gap could be noise
- The behavioral replication (14.9% vs 0.9%) is strong, but the probe replication is not
- **Risk**: Claiming "three independent architectures confirm" when the third has no statistical support for the mechanistic claim

---

## The 3 Strongest Claims

### Strong #1: Reasoning distillation reduces probe separability (Llama vs R1-Distill)

- Non-overlapping bootstrap CIs: [0.952, 0.992] vs [0.894, 0.960]
- Paired with massive behavioral improvement (27.3% to 2.4% lure rate)
- Same architecture, same parameter count -- the only difference is training
- P0 lure susceptibility sign flip (+0.422 to -0.326) corroborates with a continuous measure
- The pre-registered prediction was wrong (predicted amplification, found compression), which adds credibility
- **Caveat**: Resolve the AUC number inconsistency across documents (0.974/0.930 vs 0.999/0.929)

### Strong #2: Cross-prediction specificity for Llama

- Transfer AUC 0.378 at the peak probe layer -- below chance
- Directly addresses the most serious confound (surface features)
- Immune categories serve as a built-in negative control
- Combined with Finding 1 (category-specific vulnerability), this builds a coherent story: the probe detects something about how Llama processes items it is *vulnerable to*, not just task structure
- **Caveat**: Needs CI bounds reported; R1-Distill result is mixed (high transfer at early layers)

### Strong #3: Category-specific vulnerability profile

- Sharp boundaries: 84% lure on base rate, 0% on CRT, 0% on sunk cost, 0% on arithmetic
- Replicated across Llama, Qwen, and OLMo families
- Not a trivial finding -- it constrains what "cognitive bias in LLMs" means (it is domain-specific, not a general processing deficit)
- The behavioral data is large enough (20-30 items per category) to be reliable for the categories tested
- **Caveat**: This is a behavioral finding, not a mechanistic one. It is well-supported but also the least novel contribution (consistent with prior work).

---

## Internal Consistency Issues That Must Be Resolved

1. **AUC numbers**: 0.974 vs 0.999 (Llama), 0.930 vs 0.929 (R1-Distill) across different documents. These may reflect different analysis runs, different item subsets, or different layer selections. Pin down which numbers are canonical and use them consistently.

2. **Peak layer**: L16 vs L14 (Llama), L31 vs L14 (R1-Distill). The "peak layer shifts" narrative and the "same peak layer" narrative cannot both be true. Determine which is correct and retract the other.

3. **Natural frequency R1-Distill**: The scoring bug is documented but the invalidated numbers still appear in the scientific narrative, reviewer_response_prep, and findings lists. All documents must be updated to flag these numbers as invalid.

4. **OLMo CIs**: Listed as "pending" but the OLMo result is already being cited as confirming the mechanistic pattern. Either compute the CIs or downgrade the OLMo claim to behavioral replication only.

---

## What a Skeptical Reviewer Would Say

> "You trained logistic regression probes to distinguish two sets of text items that differ in surface content (conflict items contain personality descriptions and lure-inviting phrasing; control items do not). You found the probes work. This is unsurprising in a 4096-dimensional space with ~114 items. Your cross-prediction test partially addresses this confound for one model at one layer. Your reasoning-model comparison is interesting but the interpretation is speculative. Your SAE results are confounded by trigger tokens. Your natural frequency result is N=10 with a known scoring bug. Your OLMo replication has no confidence intervals. Strip away the dual-process framing and you have a modest finding about representation differences between two fine-tuning checkpoints, dressed up in cognitive science terminology."

That is the review you need to be prepared to answer. The core finding (lower probe AUC after reasoning training, with CIs) can survive this critique, but only if presented honestly and without overclaiming. Everything else needs either more data, more analysis, or honest downgrading.
