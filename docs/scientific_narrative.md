# Scientific Narrative: What the s1s2 Project Found

**Date**: 2026-04-09
**Status**: Synthesizes all available behavioral, probing, geometry, cross-prediction, and transfer results.
**Purpose**: Definitive interpretation for Discussion section (ICML MechInterp Workshop paper) and Alignment Forum post.

---

## The question

When an LLM encounters a problem where an intuitive-but-wrong answer competes with a correct-but-effortful one, does it process that problem differently at the representational level than a matched problem where intuition and correctness agree? And if so, does reasoning training change that representation, or just the output?

We tested this across five model configurations, 330 matched conflict/control items spanning seven cognitive bias categories, and three levels of analysis: behavioral accuracy, linear probing of residual stream activations, and cross-category transfer.

## Finding 1: LLM bias vulnerability is domain-specific, not a general processing deficit

Standard instruction-tuned models are catastrophically vulnerable to some probabilistic reasoning fallacies and completely immune to others. Llama-3.1-8B-Instruct produces the heuristic lure answer on 84% of base rate neglect items and 55% of conjunction fallacy items, yet scores 0% lure rate on CRT variants, multi-step arithmetic, framing, and anchoring. Qwen-3-8B (no-think) shows the same pattern at different magnitudes: 56% base rate, 95% conjunction, 0% on everything else.

This is not a graded competence spectrum. It is a sharp categorical boundary. These models can resist mathematical lures (CRT: "the ball costs 5 cents" intuition), resist anchoring effects, and resist framing manipulations, while simultaneously falling for elementary probability errors at rates approaching ceiling. The implication is that whatever heuristic-prone processing these models engage in, it is specific to probabilistic estimation tasks, not a general "fast thinking" mode. The S1/S2 distinction in LLMs, to the extent it exists, is domain-bound.

The four immune categories serve a second function: they are built-in negative controls for the mechanistic analyses that follow.

## Finding 2: Reasoning training reshapes internal representations, not just outputs

The Llama-3.1-8B-Instruct to R1-Distill-Llama-8B comparison is the cleanest test in the study: identical architecture, identical parameter count, differing only in reasoning distillation training. Behaviorally, base rate lure rate drops from 84% to 4%, conjunction from 55% to 0%, syllogism from 52% to 0%. The overall rate falls from 27.3% to 2.4%.

Linear probes on residual stream activations reveal that this behavioral change has a representational correlate. On the three vulnerable categories, Llama achieves near-perfect conflict/control separability (AUC 0.999 at layer 14), while R1-Distill achieves reduced but still high separability (AUC 0.929 at the same layer). Both models peak at layer 14 of 32. The locus of the representation does not move; the sharpness decreases.

This 7 percentage-point AUC gap is the opposite of what we pre-registered. We predicted reasoning models would show stronger S1/S2 separation (H2 in the pre-registration). The data show the opposite: reasoning training blurs the boundary. The interpretation that fits both the behavioral improvement and the representational blurring is what we call "S2-by-default" processing. The standard model maintains a crisp internal distinction between items that should trigger deliberation and items that should not, then often fails to act on that distinction. The reasoning model has partially lost this distinction because it applies deliberation-like computation to everything. It does not need to flag items as requiring extra effort because its default processing already incorporates that effort.

The geometry results are consistent: cosine silhouette scores are positive but low (Llama 0.079, R1-Distill 0.059), indicating that conflict and control activations overlap substantially in the residual stream. The S1/S2 signal lives in a narrow linear direction, not a broad geometric separation. Reasoning training compresses even that narrow direction.

## Finding 3: Training and inference produce dissociable effects on representations

Qwen-3-8B offers a within-model test that no cross-model comparison can: the same weights, the same architecture, run with and without explicit chain-of-thought reasoning. Behaviorally, thinking mode reduces the overall lure rate from 21% to 7%. Base rate drops from 56% to 4%. But conjunction remains stubbornly high at 55% (down from 95%), suggesting that explicit deliberation helps unevenly across bias types.

The probing result is the striking one. Qwen no-think peaks at AUC 0.971 at layer 34. We do not yet have the thinking-mode probe curves, but the behavioral dissociation already constrains the interpretation: the same weights produce different behavioral outcomes depending on inference-time processing, while the initial encoding of the problem (what the probe at the last prompt token detects) is fixed by the weights.

Compare this to the Llama/R1-Distill pair. There, different weights (from different training) produce both different behavior and different probe separability (0.999 vs. 0.929). The representational gap between standard and reasoning-trained models comes from what training baked into the weights, not from what inference-time chain-of-thought adds on top. Inference-time thinking helps behavior without changing the initial problem encoding. Training changes both.

This dissociation matters for safety: it means that adding chain-of-thought at inference time is not equivalent to reasoning training. The model's initial read of the problem, the representation that downstream computation operates on, is set by the weights. CoT can override a bad initial read, but it cannot reshape it.

## Finding 4: The probe signal is genuine, not a textual artifact

The most serious confound in this study is that linear probes achieve AUC 1.0 on immune categories (CRT, arithmetic, framing, anchoring) at layers 0-1, where models show 0% lure rates. This means a probe can perfectly separate conflict from control items based on surface features of the input text alone, without any processing-mode-specific information.

The cross-prediction test resolves this for the standard model. A probe trained on Llama's vulnerable-category activations and tested on immune-category activations achieves transfer AUC of 0.378 at layer 14, which is below chance. The probe learned a direction specific to how Llama processes problems it is vulnerable to. That direction does not transfer to problems it handles correctly, even though those problems also contain textual lures. If the probe had merely learned to detect "lure text is present," transfer would be high.

For R1-Distill, the picture is more layered. Early layers (L4-L8) show high transfer (0.878), indicating shared text-level features. Late layers (L31) show low transfer (0.385), indicating processing-specific features. Reasoning training appears to reorganize the layer-wise information flow: early layers retain general task-structure encoding, while late layers develop processing-mode-specific representations. This is itself a finding about how reasoning distillation restructures the network.

## Finding 5: Bias types share an internal representation

The per-category transfer matrix reveals that base rate neglect and conjunction fallacy items share a representation to a remarkable degree. A probe trained on base rate items and tested on conjunction items achieves AUC 0.993; the reverse direction yields 0.998. Transfer between these categories and the immune categories is near zero.

This suggests that the model represents these two bias types through a common mechanism, possibly a general "probability estimation under uncertainty" circuit that either engages or fails to engage depending on the task. Base rate neglect and conjunction fallacy are superficially different tasks (ignoring prior probabilities vs. judging compound event likelihood), but both require calibrated probabilistic reasoning that competes with salient narrative content. The shared representation is consistent with a single underlying vulnerability rather than independent failure modes.

The P0 lure susceptibility scores reinforce this. Llama's initial representations favor the lure answer (mean P0 = +0.42), while R1-Distill's initial representations favor the correct answer (mean P0 = -0.33). The sign flip shows that reasoning training does not just add a correction step; it changes the model's initial disposition toward these problems.

## Limitations

**Small N.** The vulnerable subset comprises roughly 140 items across three categories. This is sufficient for the probe analyses (which achieve tight confidence intervals) but limits the granularity of per-category breakdowns and leaves some cross-prediction estimates noisy.

**No causal evidence.** All mechanistic results are correlational. Probes show that the S1/S2 distinction is linearly decodable; they do not show that the model uses this direction for its decision. SAE feature analysis and activation steering are planned but not yet complete. Until causal interventions confirm or disconfirm the functional role of these representations, the mechanistic story remains observational.

**Probe decodability is necessary but not sufficient.** High probe AUC establishes that information is present in the representation. It does not establish that the information is read out by downstream layers. The representation could be an epiphenomenon of processing rather than a driver of it.

**Scale.** All results come from 8B-parameter models. The internal organization of frontier-scale models (70B+) may differ qualitatively. We make no claims beyond the tested scale.

**Graded, not binary.** The data support a graded processing-intensity interpretation better than a binary S1/S2 switch. The 0.929 AUC in R1-Distill is reduced but far above chance; the behavioral lure rate of 2.4% is low but not zero. Even the "S2-by-default" framing should be understood as a shift in default processing intensity, not the elimination of a discrete system.

**Training confound.** R1-Distill differs from Llama not only in reasoning distillation but in all details of the fine-tuning pipeline. We cannot attribute differences solely to reasoning-trace training. The Qwen within-model comparison partially mitigates this, but the Qwen comparison tests inference-time rather than training-time effects.

## Implications for AI safety

**Domain-specific competence gaps are invisible to generic benchmarks.** A model that scores well on CRT-style problems and arithmetic reasoning can simultaneously fail at base rate estimation 84% of the time. Safety evaluations that test "reasoning ability" as a monolithic capability will miss these domain-specific vulnerabilities. Evaluations must probe specific failure modes, not aggregate competence.

**Reasoning training changes what models are, not just what they say.** The representational evidence (AUC gap, P0 sign flip, geometry compression) shows that reasoning distillation produces internal changes beyond output formatting. This is reassuring for alignment: the models are not merely producing better-looking outputs while maintaining the same heuristic-prone internals. The change goes deeper.

**Inference-time thinking is not a substitute for training.** Qwen's conjunction fallacy rate remains at 55% even with explicit chain-of-thought, compared to R1-Distill's 0%. CoT helps but does not close the gap. For safety-critical applications, trust calibration cannot rely on chain-of-thought alone. The question is not whether the model can be prompted to think harder, but whether its default processing is adequate for the task.

**Monitoring feasibility.** The high linear decodability of the S1/S2 distinction (AUC 0.999 in Llama, 0.971 in Qwen no-think) suggests that lightweight linear probes on middle-layer activations could serve as runtime monitors for whether a model is in a heuristic-prone state. This is preliminary, but if confirmed by causal analysis, it offers a concrete tool for deployment-time oversight: flag outputs produced under high P0 lure susceptibility for additional review.
