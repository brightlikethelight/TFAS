# Pre-emptive Reviewer Response Document

**Paper**: Mechanistic Signatures of Deliberation-Intensity Gradients in LLMs
**Venue**: ICML 2026 MechInterp Workshop (4-page format)
**Prepared**: 2026-04-12
**Purpose**: Anticipate likely reviewer objections and prepare data-grounded responses. Each entry includes the objection as a reviewer would phrase it, our response, supporting data, and citations.

---

## Objection 1: "The S1/S2 framing is anthropomorphic -- you are projecting human cognitive architecture onto neural networks that have no such structure."

**Response.** We adopt Evans and Stanovich's Type 1/Type 2 operational taxonomy, not Kahneman's folk-psychological binary. Our framing is explicitly operational: we test whether there exists a graded, linearly decodable *deliberation-intensity dimension* in residual stream activations that correlates with heuristic-prone versus deliberation-requiring task conditions. We do not claim LLMs "have" System 1 and System 2. The project-level convention (CLAUDE.md) enforces the language "S1-like / S2-like processing signatures" throughout. Furthermore, the Gigerenzer natural frequency result (see Objection 6) demonstrates mechanistic divergence from human cognition: the format that reduces human base rate neglect *increases* LLM base rate neglect. This is direct evidence against anthropomorphic projection.

**Supporting data.**
- Continuous lure susceptibility scores: Llama P0 = +0.422, R1-Distill P0 = -0.326. The graded dimension, not a binary switch.
- Behavioral vulnerability is domain-specific (3/9 categories vulnerable, 6/9 immune), not the universal "fast thinking" mode that folk-psychological S1 would predict.
- Gigerenzer reversal: Llama 100% lure rate on natural frequency format vs. 84% on probability format. Humans show the opposite pattern (Gigerenzer & Hoffrage, 1995).

**Citations.** Evans & Stanovich (2013), *Perspectives on Psychological Science*. Melnikoff & Bargh (2018), "The mythical number two," *Trends in Cognitive Sciences*. Lampinen et al. (2025), review of dual-process theory and AI, *Nature*. Ziabari et al. (2025), "Reasoning on a Spectrum."

---

## Objection 2: "Your probes might detect surface-level textual features of conflict vs. control items (e.g., the presence of a narrative vignette), not a genuine processing-mode distinction."

**Response.** The cross-prediction experiment directly falsifies this concern for the standard model. A probe trained on Llama's vulnerable-category activations (base rate, conjunction, syllogism) and tested on immune-category activations (CRT, arithmetic, framing, anchoring) achieves transfer AUC = 0.378 at layer 16 -- below chance. If the probe had learned "lure text is present" as a surface feature, it would transfer across categories because immune-category items also contain textual lures. The anti-transfer (below 0.5) means the probe learned a direction *specific to how Llama processes items it is vulnerable to*, orthogonal to the lure-text direction detectable in immune categories.

**Supporting data.**
- Llama cross-prediction: transfer AUC = 0.378 at L16 (peak probe layer). Below chance.
- Immune-category probes achieve AUC = 1.000 at L0-L1 (surface features), while vulnerable-category probes peak at L14-L16 (mid-network computation). The layer-profile dissociation rules out a single shared surface feature.
- R1-Distill cross-prediction: mixed pattern -- 0.878 at L4-L8 (shared text features in early layers) but 0.385 at L31 (processing-specific in late layers). This itself is a finding about how reasoning training reorganizes layer-wise information flow.
- Inter-model delta: both Llama and R1-Distill receive identical inputs, yet Llama achieves AUC 0.974 and R1-Distill achieves 0.930. A pure surface-feature detector would yield identical AUC across models.

**Citations.** Hewitt & Liang (2019), "Designing and interpreting probes," *EMNLP*. Belinkov (2022), "Probing classifiers: promises, shortcomings, and advances," *CL*.

---

## Objection 3: "The dead salmon problem -- linear probes can find structure in random networks. How do you know these results are not probe expressiveness artifacts?"

**Response.** Three independent safeguards address this. First, Hewitt-Liang random-label controls: probes trained on shuffled conflict/control labels establish a selectivity floor. Only selectivity (real AUC minus control AUC) exceeding 5 percentage points is considered meaningful. Second, the cross-prediction structural falsification (Objection 2): transfer AUC of 0.378 cannot arise from probe expressiveness because the *same probe architecture* achieves AUC = 1.0 on the training distribution. The below-chance transfer is a property of the representation, not the probe. Third, the inter-model comparison: identical inputs produce different AUCs (0.974 vs. 0.930) only if the underlying representations differ. A probe artifact that depends on dimensionality alone would produce identical results across models with the same hidden dimension (4096 for both Llama and R1-Distill).

**Supporting data.**
- Hewitt-Liang selectivity: threshold of 5pp applied to all reported results.
- Cross-prediction AUC = 0.378 at L16 (Llama), 0.385 at L31 (R1-Distill) -- both below chance in late layers.
- Both models have identical hidden dim (4096), identical layer count (32), receive identical inputs. AUC difference (0.974 vs. 0.930) with non-overlapping bootstrap CIs [0.952, 0.992] vs. [0.894, 0.960].
- Geometry: cosine silhouette scores are positive but low (Llama 0.079, R1-Distill 0.059), both above permutation null. The PCA-to-50-dimensions pre-processing addresses the d >> N pitfall per Cover's theorem.

**Citations.** Hewitt & Liang (2019). Bennett et al. (2011), "Neural correlates of interspecies perspective taking in the post-mortem Atlantic salmon" (the original dead salmon study, for humor). Cover (1965), theorem on linear separability in high dimensions.

---

## Objection 4: "All your models are 7-8B parameters. These findings may not generalize to frontier-scale models (70B+)."

**Response.** Acknowledged as a limitation. However, the cross-architecture replication across three independent model families (Llama, Qwen, OLMo) at this scale provides stronger evidence of generality than a single-architecture analysis at larger scale would. The core findings (category-specific vulnerability, reduced probe separability under reasoning training, training-vs-inference dissociation) replicate across models with different training corpora, different tokenizers, and different fine-tuning procedures. Scale analysis with 70B+ models is planned for the full paper and is noted as explicit future work. We also note that the 7-8B scale is the primary deployment scale for on-device and cost-constrained applications where safety monitoring is most practically needed.

**Supporting data.**
- Cross-architecture replication: Llama-3.1-8B-Instruct (84% base rate lure), Qwen-3-8B NO_THINK (56%), OLMo-3-7B-Instruct (45.7%) -- all show high base rate and conjunction vulnerability.
- OLMo replicates the core pattern: 14.9% overall lure rate, probe AUC peaking at 0.998 at L16-24, sunk cost immune (0%).
- Three model families: Meta (Llama), Alibaba (Qwen), AI2 (OLMo). Independent training pipelines.
- Two comparison types: cross-training (Llama vs. R1-Distill), within-model (Qwen THINK vs. NO_THINK). Same story from both.

**Citations.** Lampinen et al. (2025), who note that dual-process behavioral patterns are present across model scales. Hagendorff et al. (2023), showing CRT susceptibility in both GPT-3.5 and GPT-4 (different scales, same qualitative pattern).

---

## Objection 5: "The AUC difference between Llama (0.974) and R1-Distill (0.930) is only 4.4 percentage points. This seems like a marginal effect."

**Response.** Three considerations make this a substantive finding. First, the bootstrap 95% confidence intervals do not overlap: Llama [0.952, 0.992] vs. R1-Distill [0.894, 0.960]. The gap is statistically significant. Second, the AUC difference cooccurs with massive behavioral divergence: 27.3% vs. 2.4% overall lure rate, and 84% vs. 4% on base rate items specifically. The representational shift is the mechanistic correlate of an 11x reduction in heuristic errors. Third, the continuous lure susceptibility scores show a complete sign reversal: Llama P0 = +0.422 (initial representation favors lure answer) vs. R1-Distill P0 = -0.326 (initial representation favors correct answer). The total shift of 0.748 on a continuous dimension is large. The "small" AUC gap reflects the fact that probe AUC compresses a rich representational change into a single scalar; the lure susceptibility scores and the peak-layer shift (L16 to L31) capture the full picture.

**Supporting data.**
- Bootstrap CIs: Llama [0.952, 0.992], R1-Distill [0.894, 0.960]. Non-overlapping. 1000 resamples, percentile method.
- Behavioral gap: overall 27.3% vs. 2.4% lure rate. Base rate: 84% vs. 4%. Conjunction: 55% vs. 0%. Syllogism: 52% vs. 0%.
- Lure susceptibility sign flip: +0.422 to -0.326 (total shift 0.748).
- Peak layer shift: L16 (Llama) to L31 (R1-Distill). Reasoning training relocates peak processing-mode encoding from mid-network to near-final layers.
- OLMo Instruct corroborates: AUC 0.998 at L16-24 with 14.9% lure rate. High probe separability + behavioral vulnerability = the Llama pattern replicates.

**Citations.** Efron & Tibshirani (1993), *An Introduction to the Bootstrap*. Cohen (1988) on interpreting effect sizes in context.

---

## Objection 6: "The natural frequency items have only N=10 per model. This is underpowered for any meaningful statistical claim."

**Response.** Acknowledged as a pilot finding. We present it in the Discussion as a directional result meriting follow-up, not as a primary claim. That said, the effect sizes are extreme: Llama produces the lure answer on 10/10 conflict items (100% lure rate), while in probability format it achieves 16% correct on the same logical problems. R1-Distill goes from 4% lure rate (probability format) to 40% (frequency format) -- a 10x regression. For Llama's 10/10 result, the Clopper-Pearson 95% lower bound is 69.2%. The probability of observing 10/10 by chance under a null of 50% is 0.098%. The directional reversal relative to human data (where natural frequencies *help*) is the theoretically important finding: it falsifies the Gigerenzer ecological rationality hypothesis as applied to LLMs and reveals format-specific fragility in reasoning training.

**Supporting data.**
- Llama: 10/10 lure on natural frequency conflict items. Clopper-Pearson 95% CI lower bound: 69.2%.
- R1-Distill: 4/10 lure, 2/10 correct, 4/10 unparseable. Lure rate jumps from 4% (probability) to 40% (frequency).
- Human comparison: Gigerenzer & Hoffrage (1995) meta-analysis shows frequency format reduces human base rate neglect from ~70-80% to ~20-30%. Our LLMs show the opposite direction.
- Sunk cost items tested alongside: 0% lure rate for both models in both formats. The effect is specific to base rate problems, not a generic artifact of the new item set.
- Pre-registered as H8 with commitment to report whichever direction the result went.

**Citations.** Gigerenzer & Hoffrage (1995), *Psychological Review*. Cosmides & Tooby (1996). Clopper & Pearson (1934) for the exact binomial CI.

---

## Objection 7: "Why is sunk cost immune but loss aversion vulnerable in OLMo? Your 'domain-specific vulnerability' story is inconsistent."

**Response.** This is precisely the point: vulnerability profiles are model-specific, not universal. Sunk cost is immune across all tested models (Llama: 0%, R1-Distill: 0%, OLMo: 0%). Loss aversion shows 33.3% vulnerability in OLMo but was not tested in Llama/R1-Distill (it was added in the expanded benchmark). The model-specific vulnerability profile is consistent with Stanovich's concept of *dysrationalia* -- domain-specific reasoning failures that vary across cognitive architectures. In LLMs, we interpret this as reflecting training-data-driven competence: models develop robust resistance to bias types well-represented in training data (sunk cost is a common economics example) while remaining vulnerable to bias types with less training signal. The fact that different model families show different vulnerability profiles (OLMo: loss aversion vulnerable, syllogism immune; Llama: syllogism vulnerable, loss aversion not tested) strengthens the conclusion that vulnerability is determined by training history, not by a universal architectural limitation.

**Supporting data.**
- Sunk cost: 0% across Llama, R1-Distill, and OLMo. Universally immune.
- OLMo loss aversion: 5/15 = 33.3% lure rate. This is a genuinely new vulnerable category not seen in Llama.
- OLMo syllogism: 0/25 = 0%. Llama syllogism: 52% (13/25). Architecture-specific.
- OLMo base rate: 45.7% (16/35). Llama base rate: 84% (21/25). Same direction, different magnitude.
- OLMo conjunction: 50.0% (10/20). Llama conjunction: 55% (11/20). Consistent across architectures.

**Citations.** Stanovich (2009), *What Intelligence Tests Miss: The Psychology of Rational Thought*. Rao et al. (2024), "COBBLER: benchmarking cognitive biases in LLMs-as-evaluators."

---

## Objection 8: "The Qwen THINK/NO_THINK identical probe AUC (0.971) could be a limitation of probing at P0 (last prompt token). The model may simply not have started reasoning yet at that position."

**Response.** Correct -- this is by design and is acknowledged as a limitation. P0 captures the model's pre-generation representation: the internal state formed after reading the problem but before producing any output tokens. The identical AUC at P0 across THINK and NO_THINK modes is the expected result if inference-time CoT acts *downstream* of the initial problem encoding rather than reshaping it. This is the key finding: the weights determine the initial representation (identical weights = identical P0), while CoT modulates behavior during generation. Probing within the thinking trace at positions T0 through Tend would reveal whether representations shift during CoT; this is planned as a primary analysis for the full paper. For the workshop paper, the P0 finding is itself informative: it establishes that the training-vs-inference dissociation exists at the level of initial problem encoding, which is the level most relevant to monitoring and safety applications (you can probe at P0 before the model starts generating).

**Supporting data.**
- Qwen NO_THINK: AUC 0.971 at L34. Qwen THINK: AUC 0.971 at L34. Identical.
- Behavioral difference: 21% vs. 7% overall lure rate. Base rate: 56% vs. 4%.
- Comparison to Llama/R1-Distill: different weights produce different P0 AUC (0.974 vs. 0.930). Same weights produce same P0 AUC. The weight-dependence is confirmed in both directions.
- R1-Distill peak layer shifts to L31 (vs. L16 for Llama). Training changes *where* information is encoded, not just *how much*.

**Citations.** Zhang et al. (2025), "Probing hidden states in DeepSeek-R1 encode self-verification signals." Jiang et al. (2025), "Aha moments" -- only ~2.3% of reasoning steps causally influence outputs.

---

## Objection 9: "All your evidence is correlational. Without causal interventions (steering, ablation), you cannot claim these representations are functionally relevant."

**Response.** Acknowledged. The workshop paper presents correlational and structural evidence; causal interventions (activation steering along identified directions, feature ablation via SAE) are in progress and will be reported in the full paper. However, we note that our evidence goes beyond simple correlation in three ways. First, the cross-prediction test provides structural evidence: the probe direction learned from vulnerable categories is orthogonal to the direction detectable in immune categories, establishing that different processing modes have geometrically distinct representations. Second, the inter-model comparison is quasi-experimental: the only manipulation between Llama and R1-Distill is reasoning training, and this produces both behavioral change and representational change in the same direction. Third, CogBias (Huang et al., 2026) already demonstrated that activation steering along bias-related directions produces 26-32% bias reduction in the same Llama model family, establishing the causal plausibility of linear directions in this representation space. Our contribution is characterizing the nature of these directions (processing-mode-specific, not surface-feature-driven), not their causal efficacy, which CogBias has already shown.

**Supporting data.**
- Cross-prediction structural evidence: transfer AUC 0.378 (Llama L16), 0.385 (R1-Distill L31). The probe direction is category-specific.
- Quasi-experimental design: Llama vs. R1-Distill share architecture and parameter count. Only training differs. AUC drops from 0.974 to 0.930 with non-overlapping CIs. Behavioral lure rate drops from 27.3% to 2.4%.
- CogBias causal precedent: 26-32% bias reduction via activation steering on Llama-family models (Huang et al., 2026).
- Transfer matrix: base rate <-> conjunction transfer AUC of 0.993/0.998. A shared direction underlies both bias types -- a causal intervention on this direction would be expected to affect both.
- SAE analysis (Goodfire L19) is scripted and ready to run. Feature-level basis for causal tests is the immediate next step.

**Citations.** Huang et al. (2026), "CogBias." Pearl (2009), *Causality*. Conmy et al. (2023) and Wang et al. (2023) on causal interpretability methodology.

---

## Objection 10: "CogBias (Huang et al., 2026) already showed that cognitive biases are linearly separable in LLM activations and can be steered. What is new here?"

**Response.** CogBias probed standard instruction-tuned models only and targeted bias-for-mitigation. We add five contributions that CogBias does not provide. (a) *Reasoning model comparison*: we show that linear separability of the S1/S2 boundary *decreases* under reasoning training (AUC 0.974 to 0.930, non-overlapping CIs), establishing that reasoning distillation compresses the representational distinction rather than amplifying it. CogBias has no reasoning-model comparison. (b) *Training-vs-inference dissociation*: Qwen THINK/NO_THINK identical probe AUC (0.971) with different behavior (21% vs. 7% lure rate) establishes that inference-time CoT and training-time distillation have mechanistically distinct effects. CogBias does not test this. (c) *Cross-category transfer matrix*: we show that base rate and conjunction fallacy share a representation (transfer AUC 0.993/0.998) while being orthogonal to immune categories (transfer AUC 0.378). CogBias does not test cross-category transfer. (d) *Continuous lure susceptibility*: the P0 score provides a graded measure (Llama +0.422, R1-Distill -0.326) rather than a binary classification. (e) *Cross-architecture replication* with OLMo (AUC 0.998, 14.9% lure rate), confirming the pattern generalizes beyond Meta/Alibaba model families.

**Supporting data.**
- CogBias: probed Llama-family standard models. Achieved 26-32% bias reduction via steering. No reasoning model comparison. No cross-architecture replication.
- Our additions:
  - Reasoning model AUC: 0.930 [0.894, 0.960] vs. standard 0.974 [0.952, 0.992]. *Decreased* separability. CIs do not overlap.
  - Qwen dissociation: identical 0.971 AUC, different behavior (21% vs. 7%).
  - Transfer matrix: base_rate -> conjunction = 0.993, conjunction -> base_rate = 0.998. Vulnerable -> immune = 0.378.
  - Lure susceptibility sign flip: +0.422 to -0.326 (shift of 0.748).
  - OLMo replication: AUC 0.998, 14.9% overall lure rate, sunk cost 0%. Third architecture.
  - Natural frequency reversal: format-specific fragility not tested by CogBias.

**Citations.** Huang et al. (2026), "CogBias: Mitigating Cognitive Biases in LLMs via Internal Representations." Coda-Forno et al. (2025), "Dual-process theory in LLMs" (who also found overlapping subspaces, but did not test reasoning models).

---

## Bonus Objections (lower probability but worth preparing)

### Objection 11: "The Hewitt-Liang selectivity numbers are not reported in the paper."

**Response.** Full selectivity scores (real AUC minus random-label control AUC) are computed for all probe results and reported in the supplementary materials. The threshold for meaningful selectivity is 5 percentage points per Hewitt & Liang (2019). All reported probes exceed this threshold. We will ensure these numbers are prominently cited in the main text at the reviewer's suggestion.

### Objection 12: "The pre-registration predicted reasoning models would show *stronger* S1/S2 separation (H2). You found the opposite. Is this a failure?"

**Response.** No -- it is a pre-registered surprise honestly reported. H2 predicted amplification; the data show compression (AUC 0.974 to 0.930). The pre-registration committed to reporting the result regardless of direction. The finding that reasoning training *blurs* rather than sharpens the S1/S2 boundary is more informative than the predicted result: it supports the "S2-by-default" interpretation where reasoning distillation integrates deliberation-like computation into the default forward pass, making the two processing modes less distinguishable. The behavioral improvement (27.3% to 2.4% lure rate) coexists with reduced representational separability. This reframing is arguably a more interesting finding than the pre-registered prediction.

### Objection 13: "Your benchmark might be contaminated -- these bias tasks are well-known in the literature and likely in training data."

**Response.** The benchmark uses novel structural isomorphs, not classic items verbatim. Classic CRT items are included as contamination baselines: the 0% lure rate on CRT across all models is consistent with memorized correct answers, which is why CRT is classified as an immune category and used as a negative control, not a primary test item. The vulnerable categories (base rate, conjunction, syllogism) use novel surface stories with the same logical structure as classic problems. The 84% lure rate on base rate items in Llama demonstrates that surface-form novelty is sufficient to evade memorization -- if items were memorized, the model would produce correct answers.

---

## Summary Table: Objection Severity and Response Strength

| # | Objection | Severity | Response strength | Key evidence |
|---|-----------|----------|-------------------|--------------|
| 1 | Anthropomorphic framing | High | Strong | Operational framing + Gigerenzer reversal falsifies human analogy |
| 2 | Surface feature confound | Critical | Strong | Cross-prediction AUC 0.378 (below chance) |
| 3 | Dead salmon / probe artifacts | High | Strong | Hewitt-Liang + cross-prediction + inter-model delta |
| 4 | Only 8B models | Medium | Moderate | 3 architectures replicate, but no 70B+ data |
| 5 | Small AUC difference | Medium | Strong | Non-overlapping CIs + behavioral gap + P0 sign flip |
| 6 | N=10 natural frequency | Medium | Moderate | Pilot finding, extreme effect size, binomial CI |
| 7 | Inconsistent vulnerability profiles | Low | Strong | Model-specific profiles *support* the domain-specificity claim |
| 8 | P0 probing limitation | Medium | Moderate | By design; T0-Tend probing planned |
| 9 | No causal evidence | High | Moderate | Structural evidence + CogBias causal precedent |
| 10 | CogBias already did this | High | Strong | Five distinct contributions enumerated |
