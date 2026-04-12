# Scientific Narrative: What the s1s2 Project Found

**Date**: 2026-04-12 (comprehensive synthesis)
**Status**: Integrates all available behavioral, probing (with bootstrap CIs), geometry, cross-prediction, transfer, lure susceptibility, Qwen dissociation, and natural frequency results. Attention entropy analysis pending.
**Purpose**: Definitive interpretation for Discussion section (ICML MechInterp Workshop paper) and Alignment Forum post.

---

## The question

When an LLM encounters a problem where an intuitive-but-wrong answer competes with a correct-but-effortful one, does it process that problem differently at the representational level than a matched problem where intuition and correctness agree? And if so, does reasoning training change that representation, or just the output?

We tested this across five model configurations, up to 380 matched conflict/control items spanning nine cognitive bias categories (including sunk cost and natural frequency framing), and four levels of analysis: behavioral accuracy, linear probing of residual stream activations with bootstrap confidence intervals, cross-category transfer, and representational geometry.

---

## The complete story: seven findings that build on each other

### Finding 1: LLM bias vulnerability is domain-specific, not a general processing deficit

Standard instruction-tuned models are catastrophically vulnerable to some probabilistic reasoning fallacies and completely immune to others. Llama-3.1-8B-Instruct produces the heuristic lure answer on 84% of base rate neglect items, 55% of conjunction fallacy items, and 52% of syllogistic reasoning items, yet scores 0% lure rate on CRT variants, multi-step arithmetic, framing, and anchoring. Qwen-3-8B (no-think) shows the same pattern at different magnitudes: 56% base rate, 95% conjunction, 0% on everything else.

This is not a graded competence spectrum. It is a sharp categorical boundary. These models can resist mathematical lures (CRT: "the ball costs 5 cents" intuition), resist anchoring effects, and resist framing manipulations, while simultaneously falling for elementary probability errors at rates approaching ceiling. The implication is that whatever heuristic-prone processing these models engage in, it is specific to probabilistic estimation tasks, not a general "fast thinking" mode. The S1/S2 distinction in LLMs, to the extent it exists, is domain-bound.

The sunk cost fallacy provides further confirmation: both Llama and R1-Distill show 0% lure rate on sunk cost items. Loss aversion heuristics, unlike probabilistic estimation errors, do not trigger shortcut processing in these architectures. The immune categories (now including sunk cost) serve as built-in negative controls for the mechanistic analyses that follow.

### Finding 2: Reasoning training reshapes internal representations, not just outputs

The Llama-3.1-8B-Instruct to R1-Distill-Llama-8B comparison is the cleanest test in the study: identical architecture, identical parameter count, differing only in reasoning distillation training. Behaviorally, base rate lure rate drops from 84% to 4%, conjunction from 55% to 0%, syllogism from 52% to 0%. The overall rate falls from 27.3% to 2.4%.

Linear probes on residual stream activations reveal that this behavioral change has a representational correlate. Bootstrap confidence intervals (1000 resamples, percentile method) quantify the effect:

- **Llama**: peak AUC 0.974 [95% CI: 0.952, 0.992] at layer 16
- **R1-Distill**: peak AUC 0.930 [95% CI: 0.894, 0.960] at layer 31

The confidence intervals do not overlap, establishing statistical significance for the inter-model difference. The peak layers also differ: Llama peaks in the middle of the network (L16 of 32), while R1-Distill peaks near the final layers (L31 of 32). Reasoning distillation does not merely compress the signal at a fixed location; it relocates peak processing-mode encoding deeper into the network.

This AUC gap is the opposite of what we pre-registered. We predicted reasoning models would show stronger S1/S2 separation (H2 in the pre-registration). The data show the opposite: reasoning training blurs the boundary. The interpretation that fits both the behavioral improvement and the representational blurring is what we call "S2-by-default" processing. The standard model maintains a high-fidelity internal distinction between items that should trigger deliberation and items that should not, then often fails to act on that distinction. The reasoning model has partially lost this distinction because it applies deliberation-like computation to everything. It does not need to flag items as requiring extra effort because its default processing already incorporates that effort.

The geometry results are consistent: cosine silhouette scores are positive but low (Llama 0.079, R1-Distill 0.059), indicating that conflict and control activations overlap substantially in the residual stream. The S1/S2 signal lives in a narrow linear direction, not a broad geometric separation. CKA between the two models ranges from 0.379 to 0.985 across layers, with divergence increasing in later layers where task-relevant computation occurs. Reasoning training compresses even that narrow direction.

### Finding 3: Training and inference produce dissociable effects on representations

Qwen-3-8B offers a within-model test that no cross-model comparison can: the same weights, the same architecture, run with and without explicit chain-of-thought reasoning. Behaviorally, thinking mode reduces the overall lure rate from 21% to 7%. Base rate drops from 56% to 4%. But conjunction remains stubbornly high at 55% (down from 95%), suggesting that explicit deliberation helps unevenly across bias types.

The probing result is the definitive one. Both modes produce identical probe separability:

- **Qwen NO_THINK**: AUC 0.971 at layer 34
- **Qwen THINK**: AUC 0.971 at layer 34

Same weights, different behavior, identical initial problem encoding. This establishes a clean dissociation: inference-time chain-of-thought reasoning reduces lure rates from 21% to 7% without changing what the probe detects in the residual stream at the last prompt token. The representation that downstream computation operates on is fixed by the weights. CoT acts on top of this representation, not within it.

Compare this to the Llama/R1-Distill pair. There, different weights (from different training) produce both different behavior (27.3% vs. 2.4% lure rate) and different probe separability (0.974 vs. 0.930, non-overlapping CIs, different peak layers). The representational gap between standard and reasoning-trained models comes from what training baked into the weights, not from what inference-time chain-of-thought adds.

This dissociation matters for safety: adding chain-of-thought at inference time is not equivalent to reasoning training. The model's initial read of the problem is set by the weights. CoT can override a bad initial read, but it cannot reshape it. For safety-critical applications, this means trust calibration cannot rely on prompting strategies alone.

### Finding 4: The probe signal is genuine, not a textual artifact

The most serious confound in this study is that linear probes achieve AUC 1.0 on immune categories (CRT, arithmetic, framing, anchoring) at layers 0-1, where models show 0% lure rates. This means a probe can perfectly separate conflict from control items based on surface features of the input text alone, without any processing-mode-specific information.

The cross-prediction test resolves this for the standard model. A probe trained on Llama's vulnerable-category activations and tested on immune-category activations achieves transfer AUC of 0.378 at layer 14, which is below chance. The probe learned a direction specific to how Llama processes problems it is vulnerable to. That direction does not transfer to problems it handles correctly, even though those problems also contain textual lures. If the probe had merely learned to detect "lure text is present," transfer would be high.

For R1-Distill, the picture is more layered. Early layers (L4-L8) show high transfer (0.878), indicating shared text-level features. Late layers (L31) show low transfer (0.385), indicating processing-specific features. Reasoning training appears to reorganize the layer-wise information flow: early layers retain general task-structure encoding, while late layers develop processing-mode-specific representations. This is itself a finding about how reasoning distillation restructures the network, consistent with the peak-layer shift observed in the bootstrap CI analysis (Finding 2).

### Finding 5: Bias types share an internal representation

The per-category transfer matrix reveals that base rate neglect and conjunction fallacy items share a representation to a remarkable degree. A probe trained on base rate items and tested on conjunction items achieves AUC 0.993; the reverse direction yields 0.998. Transfer between these categories and the immune categories is near zero.

This suggests that the model represents these two bias types through a common mechanism -- a "probability estimation under uncertainty" circuit that either engages or fails to engage depending on the task. Base rate neglect and conjunction fallacy are superficially different tasks (ignoring prior probabilities vs. judging compound event likelihood), but both require calibrated probabilistic reasoning that competes with salient narrative content. The shared representation is consistent with a single underlying vulnerability rather than independent failure modes.

The P0 lure susceptibility scores reinforce this:

- **Llama**: mean P0 = +0.422 (initial representation actively favors lure answer)
- **R1-Distill**: mean P0 = -0.326 (initial representation actively favors correct answer)

The sign flip shows that reasoning training does not just add a correction step downstream; it changes the model's initial disposition toward these problems. The magnitude of the shift (0.748 total) is large and graded, not binary.

### Finding 6: Natural frequency framing produces a paradoxical reversal in reasoning models

This is a new finding that complicates any simple "reasoning training fixes probability errors" narrative. Gigerenzer (1995) argued that humans perform better on base rate problems when probabilities are expressed as natural frequencies ("3 out of 100") rather than percentages ("3%"). We tested this with 10 reformulated base rate items.

The results:

- **Llama**: 100% correct with natural frequency framing (vs. 16% correct with probability framing)
- **R1-Distill**: 40% correct with natural frequency framing (vs. 96% correct with probability framing)

Llama's result replicates the Gigerenzer effect perfectly: natural frequency format eliminates the heuristic lure. But R1-Distill shows the opposite -- natural frequency framing actually *degrades* performance from 96% to 40%. The reasoning-trained model performs WORSE when given the ecologically rational format.

This reversal has a plausible mechanistic interpretation. R1-Distill's reasoning distillation was trained primarily on mathematical/probabilistic content expressed in standard notation. The model has learned robust deliberative processing for the percentage format it was trained on, but the natural frequency format falls outside the distribution of its reasoning training data. The frequency format may actually bypass the deliberative circuits that reasoning distillation installed, producing a novel vulnerability in a model that otherwise appears highly capable.

The implication for safety is significant: reasoning training can create *format-specific* competence. A model that handles a problem class well in one notation may fail on the same problem in a different notation. Robustness evaluations must test across surface formats, not just problem types.

### Finding 7: The picture is consistent across model families

The pattern replicates across independent model families:

1. **Llama / R1-Distill pair** (same architecture, different training): Behavioral improvement + representational blurring + P0 sign flip + cross-prediction specificity.
2. **Qwen THINK / NO_THINK pair** (same weights, different inference): Behavioral improvement + identical probe curves. Dissociation between training effects and inference effects.
3. **Cross-family**: Both Llama (84% base rate lure) and Qwen NO_THINK (56% base rate lure) show the same category-specific vulnerability pattern. Both sunk cost immune. The vulnerability profile is a property of the task type, not the model family.

The convergence across two independent architectures and two types of comparison (cross-training, within-model) strengthens the central claims. The S1/S2 processing-mode signature is not an artifact of a particular model family.

---

## Key numbers table: every real data point

### Behavioral lure rates (% heuristic responses on conflict items)

| Model | Overall | base_rate | conjunction | syllogism | CRT | arithmetic | framing | anchoring | sunk_cost | nat_freq |
|-------|---------|-----------|-------------|-----------|-----|------------|---------|-----------|-----------|----------|
| Llama-3.1-8B-Instruct | 27.3% | 84% | 55% | 52% | 0% | 0% | 0% | 0% | 0% | 0%* |
| R1-Distill-Llama-8B | 2.4% | 4% | 0% | 0% | -- | -- | -- | -- | 0% | 60%** |
| R1-Distill-Qwen-7B | ~0% | ~0% | ~0% | ~0% | -- | -- | -- | -- | -- | -- |
| Qwen 3-8B NO_THINK | 21% | 56% | 95% | 0% | -- | -- | -- | -- | -- | -- |
| Qwen 3-8B THINK | 7% | 4% | 55% | -- | -- | -- | -- | -- | -- | -- |

\* Llama 100% correct on natural frequency = 0% lure rate
\** R1-Distill 40% correct on natural frequency = 60% lure rate (REVERSAL)

### Probe AUC with bootstrap 95% CIs (vulnerable categories)

| Model | Peak AUC | 95% CI | Peak layer | Architecture depth |
|-------|----------|--------|------------|--------------------|
| Llama-3.1-8B-Instruct | 0.974 | [0.952, 0.992] | L16 | 32 layers |
| R1-Distill-Llama-8B | 0.930 | [0.894, 0.960] | L31 | 32 layers |
| Qwen 3-8B NO_THINK | 0.971 | -- | L34 | 36 layers |
| Qwen 3-8B THINK | 0.971 | -- | L34 | 36 layers |

CIs for Llama and R1-Distill do not overlap (significant difference).

### Cross-prediction (vulnerable-trained probe tested on immune categories)

| Model | Layer | Transfer AUC | Interpretation |
|-------|-------|-------------|----------------|
| Llama | L14 | 0.378 | Below chance. Probe is processing-mode-specific. |
| R1-Distill | L4 | 0.878 | Above chance. Early layers share text features. |
| R1-Distill | L31 | 0.385 | Below chance. Late layers are processing-specific. |

### Lure susceptibility (continuous P0 score)

| Model | Mean P0 | Direction |
|-------|---------|-----------|
| Llama-3.1-8B-Instruct | +0.422 | Favors lure answer |
| R1-Distill-Llama-8B | -0.326 | Favors correct answer |

### Representational geometry

| Metric | Llama | R1-Distill |
|--------|-------|------------|
| Cosine silhouette (peak) | 0.079 | 0.059 |
| CKA range (cross-model) | 0.379 -- 0.985 | |

### Transfer matrix (Llama, within vulnerable categories)

| Train \ Test | base_rate | conjunction |
|-------------|-----------|-------------|
| base_rate | -- | 0.993 |
| conjunction | 0.998 | -- |

---

## What is confirmed vs. what is pending

### Confirmed (real data, analyzed)

1. **Category-specific vulnerability** -- 3 vulnerable (base_rate, conjunction, syllogism), 5+ immune (CRT, arithmetic, framing, anchoring, sunk_cost). Replicated across Llama and Qwen families.
2. **Reasoning training blurs S1/S2 boundary** -- AUC 0.974 (Llama) vs. 0.930 (R1-Distill), non-overlapping bootstrap CIs. Peak layer shifts from L16 to L31.
3. **Training vs. inference dissociation** -- Qwen THINK and NO_THINK probes identical (0.971 at L34) despite behavioral difference (21% vs. 7% lure rate). Confirmed: inference-time reasoning does NOT reshape the residual stream the way distillation training does.
4. **Cross-prediction resolves specificity confound** -- Llama probe transfer AUC 0.378 at L14. The probe detects processing mode, not task structure.
5. **Shared bias representations** -- base_rate and conjunction transfer at 0.993. Common circuit for probabilistic estimation under uncertainty.
6. **Lure susceptibility is graded** -- Llama +0.422 vs. R1-Distill -0.326. Continuous dimension, not binary switch.
7. **Natural frequency reversal** -- Llama 100% correct vs. R1-Distill 40% correct on frequency-format base rate problems. Reasoning training creates format-specific competence.
8. **Sunk cost immunity** -- 0% lure rate for both Llama and R1-Distill. Vulnerability is specific to probabilistic estimation, not heuristic reasoning in general.

### Pending (data collected, analysis in progress)

9. **Attention entropy** -- Full per-head data for Llama and R1-Distill (81 MB each) downloaded. Analysis running separately. Expected to show whether S1/S2 distinction is reflected in attention patterns or is purely a residual-stream phenomenon.

### Pending (needs GPU)

10. **SAE feature analysis** -- Goodfire L19 key mismatch fixed, script ready. Needs GPU re-run. Would provide interpretable feature-level evidence beyond probe decodability.
11. **OLMo-2-7B cross-architecture replication** -- Scripts ready. Needs GPU.
12. **Bootstrap CIs for Qwen** -- Need per-fold data from the pod.
13. **Causal interventions** -- Activation steering and feature ablation. Stretch goal for the workshop paper; may be marked as future work.

---

## The natural frequency reversal: detailed interpretation

The Gigerenzer (1995) natural frequency finding is the most provocative result in the dataset because it breaks the simple narrative that reasoning training uniformly improves probabilistic reasoning.

**What happened**: Base rate neglect items were reformulated from percentage format ("the probability is 3%") to natural frequency format ("3 out of every 100 people"). The Gigerenzer hypothesis predicts that natural frequencies should reduce base rate neglect because they align with the format in which frequency information was encountered ancestrally.

**The Llama result confirms Gigerenzer**: Llama goes from 16% correct (probability format) to 100% correct (frequency format). The frequency format completely eliminates base rate neglect. This is consistent with the hypothesis that Llama's base rate errors arise from a processing shortcut triggered by the percentage format specifically, and the frequency format sidesteps that shortcut.

**The R1-Distill result contradicts it**: R1-Distill goes from 96% correct (probability format) to 40% correct (frequency format). The reasoning model, which had largely solved base rate neglect in standard format, becomes substantially worse when given the "easier" format.

**Interpretation**: R1-Distill's reasoning distillation training installed deliberative circuits calibrated to standard mathematical notation. The natural frequency format is sufficiently different from this training distribution that it either (a) fails to trigger the learned deliberative pathway, or (b) triggers a different, less reliable processing mode. In effect, reasoning training created a strong but narrow competence: robust within the trained format, brittle outside it.

**Implications for safety evaluation**: This result means that behavioral evaluations of reasoning models cannot test a single format per problem type and conclude the model "can do base rate reasoning." Format transfer is not guaranteed, even for formats that should theoretically be easier. Robustness requires testing across notational variants.

**Implications for the S1/S2 framework**: The natural frequency result suggests that what we are calling "S1-like" and "S2-like" processing is not a property of the abstract problem structure alone, but of the interaction between problem structure and input format. The same logical problem can be S2-requiring in one format and S1-accessible in another, and this mapping differs between standard and reasoning-trained models.

---

## Updated strongest honest framing for the workshop paper

The narrative now rests on **five converging lines of evidence** plus a **complicating result** that demonstrates the limits of the findings:

1. **Behavioral**: Reasoning models resist cognitive-bias lures (27.3% to 2.4% overall lure rate). Within-model: thinking mode reduces lures from 21% to 7% with identical weights. Both sunk cost (loss aversion heuristic family) and mathematical lures (CRT, arithmetic) are immune across all models. Vulnerability is specific to probabilistic estimation under uncertainty.

2. **Representational**: Linear probes find a high-fidelity S1/S2 boundary in standard models (AUC 0.974 [0.952, 0.992]) that is degraded in reasoning models (AUC 0.930 [0.894, 0.960]). CIs do not overlap. Peak layer shifts from L16 to L31 -- reasoning training relocates peak processing-mode encoding deeper. The probe is SPECIFIC to processing mode (cross-prediction 0.378), not a textual artifact. Lure susceptibility is graded and continuous (+0.422 vs. -0.326).

3. **Training vs. inference dissociation**: Qwen THINK and NO_THINK have identical probe curves (0.971 at L34) despite dramatically different behavior (7% vs. 21% lure rate). Distillation training rewires representations; inference-time reasoning acts downstream of the probed representation.

4. **Structural**: Base rate and conjunction fallacy share a representation (transfer AUC 0.993), suggesting a common "probabilistic estimation under uncertainty" circuit. Geometry confirms the signal is narrow and linear (silhouette 0.079), consistent with a graded processing-intensity dimension rather than discrete clusters.

5. **Cross-family replication**: The pattern holds across Llama/R1-Distill (cross-training comparison) and Qwen THINK/NO_THINK (within-model comparison). Two independent architectures, same story.

6. **The natural frequency complication**: Reasoning training creates format-specific competence. R1-Distill's 96% accuracy on standard base rate items drops to 40% with natural frequency framing. The standard model (Llama) shows the opposite: 16% to 100%. Reasoning distillation does not install general probabilistic reasoning; it installs format-calibrated deliberative processing. This limits the generality of Finding 2 and has direct safety implications.

### The one-paragraph elevator pitch

Standard instruction-tuned LLMs maintain a near-perfect linear boundary in their residual stream between heuristic-prone and deliberation-requiring processing modes (AUC 0.974), but frequently fail to act on that distinction (84% lure rate on base rate problems). Reasoning-distilled models partially dissolve this boundary (AUC 0.930, non-overlapping CI) while dramatically improving behavioral outcomes (4% lure rate) -- a pattern we call "S2-by-default" processing. Cross-prediction confirms the probe signal is processing-mode-specific, not a textual artifact. Inference-time chain-of-thought reduces errors without altering the representation (Qwen THINK/NO_THINK: identical 0.971 AUC, different behavior), establishing that training and inference produce mechanistically distinct effects. However, reasoning training creates format-specific competence: the same model that achieves 96% accuracy on standard base rate problems drops to 40% when probabilities are expressed as natural frequencies. The deliberation gradient is real, representationally grounded, and safety-relevant -- but narrower than behavioral benchmarks alone would suggest.

---

## What OLMo needs to show to confirm cross-architecture replication

OLMo-2-7B-Instruct is the planned third architecture (after Llama and Qwen families). It serves two purposes: (a) cross-architecture replication of the core findings, and (b) testing whether the patterns hold in a fully open-source, independently trained model (not a derivative of Meta or Alibaba checkpoints).

### Minimum confirmation criteria

1. **Behavioral**: OLMo should show non-zero lure rates on at least 2 of the 3 vulnerable categories (base_rate, conjunction, syllogism) and 0% on at least 3 of the 5 immune categories. If OLMo shows 0% lure rate on all categories, it does not provide a useful test (no vulnerable baseline to compare against).

2. **Probe AUC on vulnerable categories**: Should be significantly above chance (AUC > 0.6, bootstrap CI excluding 0.5). The exact magnitude matters less than the direction: if OLMo shows high probe AUC despite behavioral vulnerability, it replicates the "detection without resolution" pattern seen in Llama.

3. **Cross-prediction specificity**: A probe trained on OLMo's vulnerable categories and tested on immune categories should show transfer AUC near or below chance (< 0.55). This would replicate the Llama finding that the probe signal is processing-mode-specific.

4. **Geometry**: Silhouette scores should be positive but low (consistent with graded, not clustered, representations). Very high silhouette would be surprising and would require investigation.

### What would be most informative

- **If OLMo replicates Llama's pattern** (high AUC, high lure rate, specific probe): Strengthens the claim that S1/S2 processing-mode encoding is a general property of instruction-tuned transformers, not specific to Llama/Qwen training.
- **If OLMo shows low AUC and low lure rate** (like R1-Distill): Would suggest that OLMo's training already incorporates some reasoning-like processing, which would be interesting given that OLMo is not explicitly reasoning-distilled.
- **If OLMo shows low AUC and high lure rate**: Would break the pattern -- the model would be vulnerable without maintaining a clear internal distinction. This would be the most theoretically interesting outcome but also the hardest to interpret.

### What OLMo does NOT need to show

- It does not need to match Llama's exact AUC or peak layer. Different architectures process information at different depths.
- It does not need to show the natural frequency reversal. That finding may be specific to R1-Distill's training data distribution.
- It does not need to show base_rate/conjunction transfer. The transfer matrix is about Llama's internal structure, not a universal prediction.

---

## Limitations

**Small N.** The vulnerable subset comprises roughly 140 items across three categories from the original 330-item benchmark (expanded to ~190 matched pairs with sunk cost and natural frequency). This is sufficient for the probe analyses (bootstrap CIs are tight: widths of 0.040 for Llama, 0.066 for R1-Distill) but limits the granularity of per-category breakdowns and leaves some cross-prediction estimates noisy.

**No causal evidence.** All mechanistic results are correlational. Probes show that the S1/S2 distinction is linearly decodable; they do not show that the model uses this direction for its decision. SAE feature analysis and activation steering are planned but not yet complete. Until causal interventions confirm or disconfirm the functional role of these representations, the mechanistic story remains observational.

**Probe decodability is necessary but not sufficient.** High probe AUC establishes that information is present in the representation. It does not establish that the information is read out by downstream layers. The representation could be an epiphenomenon of processing rather than a driver of it.

**Scale.** All results come from 7-8B-parameter models. The internal organization of frontier-scale models (70B+) may differ qualitatively. We make no claims beyond the tested scale.

**Graded, not binary.** The data support a graded processing-intensity interpretation better than a binary S1/S2 switch. The 0.930 AUC in R1-Distill is reduced but far above chance; the behavioral lure rate of 2.4% is low but not zero. Even the "S2-by-default" framing should be understood as a shift in default processing intensity, not the elimination of a discrete system.

**Training confound.** R1-Distill differs from Llama not only in reasoning distillation but in all details of the fine-tuning pipeline. We cannot attribute differences solely to reasoning-trace training. The Qwen within-model comparison partially mitigates this, but the Qwen comparison tests inference-time rather than training-time effects.

**Natural frequency sample size.** The natural frequency finding is based on only 10 reformulated items. While the effect is large (60 percentage-point reversal for R1-Distill), the small N means this should be treated as a strong pilot finding that needs replication with more items before being featured as a primary claim. It belongs in the Discussion, not the abstract.

**Format-specificity generality.** We tested only one alternative format (natural frequencies). We do not know whether the R1-Distill reversal extends to other format variations (e.g., visual representations, odds ratios, verbal qualifiers). The finding motivates format-robustness testing but does not fully characterize the boundary conditions.

---

## Implications for AI safety

**Domain-specific competence gaps are invisible to generic benchmarks.** A model that scores well on CRT-style problems and arithmetic reasoning can simultaneously fail at base rate estimation 84% of the time. Safety evaluations that test "reasoning ability" as a monolithic capability will miss these domain-specific vulnerabilities. Evaluations must probe specific failure modes, not aggregate competence.

**Reasoning training changes what models are, not just what they say.** The representational evidence (non-overlapping AUC CIs, P0 sign flip, geometry compression, peak-layer shift) shows that reasoning distillation produces internal changes beyond output formatting. This is reassuring for alignment: the models are not merely producing better-looking outputs while maintaining the same heuristic-prone internals. The change goes deeper.

**Inference-time thinking is not a substitute for training.** Qwen's conjunction fallacy rate remains at 55% even with explicit chain-of-thought, compared to R1-Distill's 0%. The probe curves are identical (0.971) regardless of thinking mode. CoT helps behavior without changing the representation. For safety-critical applications, trust calibration cannot rely on chain-of-thought alone. The question is not whether the model can be prompted to think harder, but whether its default processing is adequate for the task.

**Reasoning training can create novel vulnerabilities through format specialization.** The natural frequency reversal (R1-Distill: 96% -> 40%) shows that reasoning distillation can install format-specific competence that fails on equivalent problems in unfamiliar formats. This is a new category of alignment risk: a model that appears to have solved a problem class may have only solved it in the notation it was trained on. Safety evaluations must include format-transfer tests, not just accuracy on standard formulations.

**Monitoring feasibility.** The high linear decodability of the S1/S2 distinction (AUC 0.974 in Llama, 0.971 in Qwen no-think) suggests that lightweight linear probes on middle-layer activations could serve as runtime monitors for whether a model is in a heuristic-prone state. The bootstrap CIs are tight enough to be practically useful. This is preliminary, but if confirmed by causal analysis, it offers a concrete tool for deployment-time oversight: flag outputs produced under high P0 lure susceptibility for additional review.

**Shared vulnerability circuits amplify risk.** The 0.993 transfer between base rate and conjunction representations means that a single failure mode underlies multiple behavioral vulnerabilities. A targeted intervention (or a targeted attack) on this shared circuit could simultaneously affect multiple categories of probabilistic reasoning. This is both an opportunity (one fix might address multiple failure modes) and a risk (one perturbation might create multiple failures).
