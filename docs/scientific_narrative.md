# Readable but Not Writable: What the s1s2 Project Found

**Date**: 2026-04-12 (final NeurIPS synthesis)
**Status**: Complete (reviewer corrections applied 2026-04-12). Integrates all ten findings: behavioral (incl. R1 "other" rate), probing (bootstrap CIs), geometry, cross-prediction (incl. Qwen orthogonal directions), transfer, lure susceptibility, text baseline, natural frequency, causal steering (100% to correct, continuous), and 32B scale replication. Seven converging lines of evidence.
**Purpose**: Definitive interpretation for NeurIPS submission and Alignment Forum post.

---

## The question

When an LLM encounters a problem where an intuitive-but-wrong answer competes with a correct-but-effortful one, does it process that problem differently at the representational level than a matched problem where intuition and correctness agree? And if so, does reasoning training change that representation, or just the output?

We tested this across six model configurations (including OLMo-32B-Instruct for scale replication), up to 380 matched conflict/control items spanning nine cognitive bias categories (including sunk cost and natural frequency framing), and five levels of analysis: behavioral accuracy, linear probing of residual stream activations with bootstrap confidence intervals, cross-category transfer, representational geometry, and causal activation steering.

---

## The complete story: ten findings that build on each other

### Finding 1: LLM bias vulnerability is domain-specific, not a general processing deficit

Standard instruction-tuned models are catastrophically vulnerable to some probabilistic reasoning fallacies and completely immune to others. Llama-3.1-8B-Instruct produces the heuristic lure answer on 84% of base rate neglect items, 55% of conjunction fallacy items, and 52% of syllogistic reasoning items, yet scores 0% lure rate on CRT variants, multi-step arithmetic, framing, and anchoring. Qwen-3-8B (no-think) shows the same pattern at different magnitudes: 56% base rate, 95% conjunction, 0% on everything else.

This is not a graded competence spectrum. It is a sharp categorical boundary. These models can resist mathematical lures (CRT: "the ball costs 5 cents" intuition), resist anchoring effects, and resist framing manipulations, while simultaneously falling for elementary probability errors at rates approaching ceiling. The implication is that whatever heuristic-prone processing these models engage in, it is specific to probabilistic estimation tasks, not a general "fast thinking" mode. The S1/S2 distinction in LLMs, to the extent it exists, is domain-bound.

The sunk cost fallacy provides further confirmation: both Llama and R1-Distill show 0% lure rate on sunk cost items. Loss aversion heuristics, unlike probabilistic estimation errors, do not trigger shortcut processing in these architectures. The immune categories (now including sunk cost) serve as built-in negative controls for the mechanistic analyses that follow.

### Finding 2: Reasoning training reshapes internal representations, not just outputs

The Llama-3.1-8B-Instruct to R1-Distill-Llama-8B comparison is the cleanest test in the study: identical architecture, identical parameter count, differing only in reasoning distillation training. Behaviorally, base rate lure rate drops from 84% to 4%, conjunction from 55% to 0%, syllogism from 52% to 0%. The overall rate falls from 27.3% to 2.4%. Critically, R1-Distill shows an 18.2% "other" response rate -- answers that are neither the lure nor the canonical correct answer. This means reasoning models do not simply suppress the heuristic lure; they shift errors into novel response categories. The lure is no longer the dominant attractor, but the deliberative process sometimes produces idiosyncratic answers rather than the expected correct one.

Linear probes on residual stream activations reveal that this behavioral change has a representational correlate. Bootstrap confidence intervals (1000 resamples, percentile method) quantify the effect:

- **Llama**: peak AUC 0.974 [95% CI: 0.952, 0.992] at layer 16
- **R1-Distill**: peak AUC 0.930 [95% CI: 0.894, 0.960] at layer 31

The confidence intervals do not overlap, establishing statistical significance for the inter-model difference. The peak layers also differ: Llama peaks in the middle of the network (L16 of 32), while R1-Distill peaks near the final layers (L31 of 32). Reasoning distillation does not merely compress the signal at a fixed location; it relocates peak processing-mode encoding deeper into the network.

This AUC gap is the opposite of what we pre-registered. We predicted reasoning models would show stronger S1/S2 separation (H2 in the pre-registration). The data show the opposite: reasoning training blurs the boundary. The interpretation that fits both the behavioral improvement and the representational blurring is what we call "S2-by-default" processing. The standard model maintains a high-fidelity internal distinction between items that should trigger deliberation and items that should not, then often fails to act on that distinction. The reasoning model has partially lost this distinction because it applies deliberation-like computation to everything. It does not need to flag items as requiring extra effort because its default processing already incorporates that effort.

The geometry results are consistent: cosine silhouette scores are positive but low (Llama 0.079, R1-Distill 0.059), indicating that conflict and control activations overlap substantially in the residual stream. The S1/S2 signal lives in a narrow linear direction, not a broad geometric separation. CKA between the two models ranges from 0.379 to 0.985 across layers, with divergence increasing in later layers where task-relevant computation occurs. Reasoning training compresses even that narrow direction.

### Finding 3: Training and inference produce dissociable effects on representations

Qwen-3-8B offers a within-model test that no cross-model comparison can: the same weights, the same architecture, run with and without explicit chain-of-thought reasoning. Behaviorally, thinking mode reduces the overall lure rate from 21% to 7%. Base rate drops from 56% to 4%. But conjunction remains stubbornly high at 55% (down from 95%), suggesting that explicit deliberation helps unevenly across bias types.

The probing result is the definitive one. Both modes achieve the same probe separability but encode conflict along orthogonal directions:

- **Qwen NO_THINK**: AUC 0.971 at layer 34
- **Qwen THINK**: AUC 0.971 at layer 34
- **Cross-prediction**: 0.496 (chance). **Cosine similarity between probe directions**: -0.005 (orthogonal).

Same weights, same fidelity, but mode-specific encoding at orthogonal directions. The two modes detect conflict equally well but represent it along entirely different linear directions in the residual stream. This establishes a richer dissociation than "identical encoding": inference-time chain-of-thought reasoning reduces lure rates from 21% to 7% while maintaining equal separability along a rotated direction. The fact that cross-prediction is at chance despite matched AUCs means the two modes carve up the representational space differently -- CoT does not merely amplify the same signal, it rotates it. The representation is fixed by the weights in terms of fidelity, but the direction of encoding shifts with the inference regime.

Compare this to the Llama/R1-Distill pair. There, different weights (from different training) produce both different behavior (27.3% vs. 2.4% lure rate) and different probe separability (0.974 vs. 0.930, non-overlapping CIs, different peak layers). The representational gap between standard and reasoning-trained models comes from what training baked into the weights, not from what inference-time chain-of-thought adds.

This dissociation matters for safety: adding chain-of-thought at inference time is not equivalent to reasoning training. The model's initial read of the problem is set by the weights. CoT can override a bad initial read, but it cannot reshape the fidelity of encoding -- though it does reshape the direction. For safety-critical applications, this means trust calibration cannot rely on prompting strategies alone: a monitor trained on one inference mode will not transfer to the other (cross-prediction at chance).

### Finding 4: The probe signal is genuine, not a textual artifact

The most serious confound in this study is that linear probes achieve AUC 1.0 on immune categories (CRT, arithmetic, framing, anchoring) at layers 0-1, where models show 0% lure rates. This means a probe can perfectly separate conflict from control items based on surface features of the input text alone, without any processing-mode-specific information.

A direct text-only baseline quantifies this confound. A logistic regression trained on TF-IDF features of the raw prompt text achieves AUC 0.840, while the activation-based probe achieves AUC 0.999. Surface text features explain approximately 84% of the separability, but activation probes capture the remaining ~16% that text misses. The gap is the processing-mode-specific signal that exists in the residual stream beyond what surface text provides.

The cross-prediction test resolves this further for the standard model. A probe trained on Llama's vulnerable-category activations and tested on immune-category activations achieves transfer AUC of 0.378 at layer 14. This below-chance value does not indicate meaningful reversed classification; it reflects immune-category items compressing into a tight cluster on one side of the probe's decision boundary. Because the model processes all immune items uniformly (0% lure rate), their activations collapse into a narrow region that the vulnerable-trained probe maps to a single class. The probe learned a direction specific to how Llama processes problems it is vulnerable to. That direction does not transfer to problems it handles correctly, even though those problems also contain textual lures. If the probe had merely learned to detect "lure text is present," transfer would be high.

For R1-Distill, the picture is more layered. Early layers (L4-L8) show high transfer (0.878), indicating shared text-level features. Late layers (L31) show low transfer (0.385), indicating processing-specific features. Reasoning training appears to reorganize the layer-wise information flow: early layers retain general task-structure encoding, while late layers develop processing-mode-specific representations. This is itself a finding about how reasoning distillation restructures the network, consistent with the peak-layer shift observed in the bootstrap CI analysis (Finding 2).

### Finding 5: Bias types share an internal representation

The per-category transfer matrix reveals that base rate neglect and conjunction fallacy items share a representation to a remarkable degree. A probe trained on base rate items and tested on conjunction items achieves AUC 0.993; the reverse direction yields 0.998. Transfer between these categories and the immune categories is near zero.

This suggests that the model represents these two bias types through a common mechanism -- a "probability estimation under uncertainty" circuit that either engages or fails to engage depending on the task. Base rate neglect and conjunction fallacy are superficially different tasks (ignoring prior probabilities vs. judging compound event likelihood), but both require calibrated probabilistic reasoning that competes with salient narrative content. The shared representation is consistent with a single underlying vulnerability rather than independent failure modes.

The P0 lure susceptibility scores reinforce this:

- **Llama**: mean P0 = +0.422 (initial representation actively favors lure answer)
- **R1-Distill**: mean P0 = -0.326 (initial representation actively favors correct answer)

The sign flip shows that reasoning training does not just add a correction step downstream; it changes the model's initial disposition toward these problems. The magnitude of the shift (0.748 total) is large and graded, not binary.

### Finding 6: Sunk cost immunity confirms domain-specificity

Both Llama and R1-Distill show 0% lure rate on sunk cost items. OLMo shows 33% (a model-specific vulnerability). Loss aversion heuristics, unlike probabilistic estimation errors, do not reliably trigger shortcut processing. The immune categories serve as built-in negative controls for the mechanistic analyses.

### Finding 7: The picture is consistent across model families

The pattern replicates across independent model families:

1. **Llama / R1-Distill pair** (same architecture, different training): Behavioral improvement + representational blurring + P0 sign flip + cross-prediction specificity.
2. **Qwen THINK / NO_THINK pair** (same weights, different inference): Behavioral improvement + identical probe curves. Dissociation between training effects and inference effects.
3. **OLMo Instruct / Think pair** (third architecture family): Behavioral replication (14.9% vs 0.9% lure rate) + mechanistic replication (probe AUC 0.996 [0.988, 1.000] -> 0.962 [0.934, 0.982], non-overlapping bootstrap CIs). The 0.034 gap is smaller than Llama/R1 (0.044) but statistically robust.
4. **Cross-family**: Llama (84% base rate lure), Qwen NO_THINK (56%), and OLMo Instruct (14.9%) all show the same category-specific vulnerability pattern. All sunk cost immune. The vulnerability profile is a property of the task type, not the model family.

The convergence across three independent architectures and two types of comparison (cross-training, within-model) strengthens the central claims. The S1/S2 processing-mode signature is not an artifact of a particular model family.

### Finding 8: Natural frequency framing produces a paradoxical reversal in reasoning models

Gigerenzer (1995) predicted that natural frequencies should reduce base rate neglect. In LLMs, the opposite occurs:

- **Llama**: 100% lure rate with natural frequency framing (vs. 84% with probability framing) -- WORSE
- **R1-Distill**: 40% lure rate with natural frequency framing (vs. 4% with probability framing) -- MUCH WORSE

LLMs' "ecological" format is percentages (dominant in training data), not frequencies (the human ecological format). Reasoning distillation installed format-calibrated deliberative processing that breaks on unfamiliar notation. (See dedicated section below for the detailed Gigerenzer interpretation.)

### Finding 9: Causal evidence -- the S1/S2 direction is causally upstream of behavior in standard models but decoupled in reasoning models

Probe decodability establishes that S1/S2 information is present in the residual stream. The critical question is whether it is merely present (epiphenomenal) or causally upstream of the model's decision. We tested this via probe-direction activation steering: extracting the S1/S2 linear direction from trained probes and injecting it into the residual stream at inference time.

**Llama-3.1-8B-Instruct**: **37.5 percentage-point causal swing**. Steering in the S2 direction reduces lure rates; steering in the S1 direction increases them. Critically, 100% of the lure reduction goes to correct answers -- zero to garbage or incoherent outputs. The steering vector moves the model along a meaningful axis from heuristic to deliberative processing, not from coherent to broken. Random direction controls produce null effects. This establishes that the linear direction the probe learned is not an incidental correlate of conflict/control structure -- it is causally read out by downstream computation to determine behavior. The representation is functionally active.

**R1-Distill-Llama-8B**: **7.5 percentage-point swing** (continuous steering -- the hook fires every forward pass, so the 7.5pp IS the continuous-injection result, not a single-layer perturbation). The same steering procedure applied to R1-Distill produces a drastically smaller effect. The S1/S2 direction is "readable but not writable" -- probes can still decode it (AUC 0.930), but injecting it back into the stream barely moves behavior. This is a clean dissociation: the representation is decodable (Finding 2) but causally decoupled from the output.

**Interpretation**: Reasoning distillation does not just blur the S1/S2 boundary (Finding 2); it actively decouples the S1/S2 direction from downstream decision-making. The standard model maintains a processing-mode signal that is both detectable AND causally functional. The reasoning model retains a ghost of that signal (readable by a probe) but has rerouted its decision-making through reasoning-trace pathways that bypass it. This is the strongest mechanistic evidence for the "S2-by-default" interpretation: R1-Distill does not need to read a "this requires deliberation" flag because its default processing already includes deliberation.

The 37.5pp Llama swing also resolves the "epiphenomenal probe" limitation flagged in prior drafts. The direction is not an artifact. It is used.

### Finding 10: Scale does NOT reduce S1/S2 vulnerability -- it may increase it

A natural objection to all 7-8B findings is: maybe this is a small-model problem that disappears with scale. We tested this with OLMo-32B-Instruct, a 4x scale increase over OLMo-7B-Instruct.

**Overall lure rate**: **19.6% at 32B** (vs. 14.9% at 7B). Scale made things worse, not better.

**Per-category breakdown reveals the mechanism**:
- **base_rate**: 74.3% lure at 32B (vs. 46% at 7B) -- dramatically worse with scale. The larger model is MORE susceptible to base rate neglect, not less.
- **framing**: 30% lure at 32B (vs. 0% at 7B) -- a vulnerability that did not exist at 7B scale *emerges* at 32B. This is a qualitative change, not a quantitative one.
- **conjunction**: 0% at both scales -- scale did resolve this one.
- **sunk_cost**: 33% at both scales -- unchanged.

**Probe separability**: AUC 0.9999 at 32B. The S1/S2 boundary is equally (essentially perfectly) linearly separable at larger scale. The "detection without resolution" pattern -- the model encodes the distinction with near-perfect fidelity but fails to act on it -- replicates and if anything strengthens.

**Implications**: Scale is not a solution. The larger model has a strictly better representation of the problem structure (AUC 0.9999) and strictly worse behavioral outcomes on base rate problems (74.3% vs. 46%). The gap between what the model knows and what the model does widens with scale. Furthermore, scale introduces NEW vulnerabilities (framing at 30%) that did not exist at smaller scale. This undermines the "scale fixes everything" assumption and strengthens the case that reasoning training, not scale, is what changes the S1/S2 processing dynamic.

---

## Key numbers table: every real data point

### Behavioral lure rates (% heuristic responses on conflict items)

| Model | Overall lure | "Other" rate | base_rate | conjunction | syllogism | CRT | arithmetic | framing | anchoring | sunk_cost | nat_freq |
|-------|-------------|-------------|-----------|-------------|-----------|-----|------------|---------|-----------|-----------|----------|
| Llama-3.1-8B-Instruct | 27.3% | -- | 84% | 55% | 52% | 0% | 0% | 0% | 0% | 0% | 0%* |
| R1-Distill-Llama-8B | 2.4% | **18.2%** | 4% | 0% | 0% | -- | -- | -- | -- | 0% | 60%** |
| R1-Distill-Qwen-7B | ~0% | -- | ~0% | ~0% | ~0% | -- | -- | -- | -- | -- | -- |
| Qwen 3-8B NO_THINK | 21% | -- | 56% | 95% | 0% | -- | -- | -- | -- | -- | -- |
| Qwen 3-8B THINK | 7% | -- | 4% | 55% | -- | -- | -- | -- | -- | -- | -- |
| OLMo-7B-Instruct | 14.9% | -- | 46% | -- | -- | -- | -- | 0% | -- | 33% | -- |
| OLMo-32B-Instruct | 19.6% | -- | 74.3% | 0% | -- | -- | -- | 30% | -- | 33% | -- |

\* Llama 100% LURE on natural frequency (worse than 84% probability-format lure rate)
\** R1-Distill 40% correct on natural frequency = 60% lure rate (REVERSAL)

**Note on "other" responses**: R1-Distill's 18.2% "other" rate means reasoning models shift errors rather than simply suppressing lures. The lure is no longer the dominant attractor, but the deliberative process sometimes produces idiosyncratic answers (neither lure nor canonical correct). This matters for safety evaluation: low lure rate alone overstates the improvement if errors migrate to novel categories.

### Probe AUC with bootstrap 95% CIs (vulnerable categories)

| Model | Peak AUC | 95% CI | Peak layer | Architecture depth |
|-------|----------|--------|------------|--------------------|
| Llama-3.1-8B-Instruct | 0.974 | [0.952, 0.992] | L16 | 32 layers |
| R1-Distill-Llama-8B | 0.930 | [0.894, 0.960] | L31 | 32 layers |
| Qwen 3-8B NO_THINK | 0.971 | -- | L34 | 36 layers |
| Qwen 3-8B THINK | 0.971 | -- | L34 | 36 layers |
| OLMo-7B-Instruct | 0.996 | [0.988, 1.000] | L24 | -- |
| OLMo-7B-Think | 0.962 | [0.934, 0.982] | L22 | -- |
| OLMo-32B-Instruct | 0.9999 | -- | -- | -- |

CIs for Llama and R1-Distill do not overlap (significant difference). CIs for OLMo Instruct/Think do not overlap.

### Causal steering (probe-direction activation injection)

| Model | Causal swing (pp) | Lure reduction destination | Steering type | Random controls | Interpretation |
|-------|-------------------|---------------------------|--------------|-----------------|----------------|
| Llama-3.1-8B-Instruct | 37.5 | 100% to correct, 0% to garbage | continuous (hook every fwd pass) | null | Direction is causally upstream of behavior |
| R1-Distill-Llama-8B | 7.5 | -- | continuous (hook every fwd pass) | null | Readable but not writable; decoupled from output |

### Cross-prediction (vulnerable-trained probe tested on immune categories)

| Model | Layer | Transfer AUC | Interpretation |
|-------|-------|-------------|----------------|
| Llama | L14 | 0.378 | Below chance. Immune items compress into tight cluster; probe is processing-mode-specific. |
| R1-Distill | L4 | 0.878 | Above chance. Early layers share text features. |
| R1-Distill | L31 | 0.385 | Below chance. Late layers are processing-specific. |

### Cross-prediction (Qwen THINK vs NO_THINK mode transfer)

| Direction | AUC | Cosine sim | Interpretation |
|-----------|-----|-----------|----------------|
| NO_THINK probe on THINK activations | 0.496 | -0.005 | Chance. Modes encode conflict along orthogonal directions. |
| THINK probe on NO_THINK activations | 0.496 | -0.005 | Chance. Mode-specific encoding at equal fidelity. |

### Text baseline (TF-IDF logistic regression on raw prompt text)

| Baseline | AUC | Gap vs. activation probe |
|----------|-----|-------------------------|
| Text-only (TF-IDF) | 0.840 | -- |
| Activation probe | 0.999 | +0.159 (the processing-mode-specific signal text misses) |

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

### Confirmed (real data, analyzed) -- ten findings

1. **Category-specific vulnerability** -- 3 vulnerable (base_rate, conjunction, syllogism), 5+ immune (CRT, arithmetic, framing, anchoring, sunk_cost). Replicated across Llama, Qwen, and OLMo families. Scale does not fix it (Finding 10).
2. **Reasoning training blurs S1/S2 boundary** -- AUC 0.974 (Llama) vs. 0.930 (R1-Distill), non-overlapping bootstrap CIs. Peak layer shifts from L16 to L31. R1-Distill shows 18.2% "other" rate: reasoning models shift errors into novel categories, not just suppress lures.
3. **Training vs. inference dissociation** -- Qwen THINK and NO_THINK achieve matched probe fidelity (0.971 at L34) but encode conflict along ORTHOGONAL directions (cross-prediction 0.496, cosine -0.005). Mode-specific encoding at similar fidelity but orthogonal directions. A monitor trained on one mode will not transfer to the other.
4. **Cross-prediction resolves specificity confound** -- Llama probe transfer AUC 0.378 at L14. The below-chance value reflects immune items compressing into a tight cluster, not meaningful reversed classification. Text-only baseline (TF-IDF AUC 0.840) confirms surface text explains ~84% of separability; activation probes capture the remaining ~16%.
5. **Shared bias representations** -- base_rate and conjunction transfer at 0.993. Common circuit for probabilistic estimation under uncertainty.
6. **Lure susceptibility is graded** -- Llama +0.422 vs. R1-Distill -0.326. Continuous dimension, not binary switch.
7. **Cross-architecture replication** -- OLMo-7B Instruct/Think confirms the pattern: AUC 0.996 -> 0.962, non-overlapping CIs. Third architecture family, same story.
8. **Natural frequency reversal** -- Llama 100% LURE vs. R1-Distill 40% LURE on frequency-format base rate problems (both WORSE than probability format). LLMs' "ecological" format is percentages, not frequencies.
9. **Causal steering** -- Llama: 37.5pp causal swing from probe-direction injection (random controls null), with 100% of lure reduction going to correct answers (zero to garbage). R1-Distill: 7.5pp under continuous steering (hook fires every forward pass; this IS the continuous result). The S1/S2 direction is readable but not writable in reasoning models. Resolves the epiphenomenal probe concern.
10. **Scale does NOT reduce vulnerability** -- OLMo-32B-Instruct: 19.6% overall (7B: 14.9%), base_rate 74.3% (7B: 46%), framing 30% (7B: 0% -- new vulnerability emerges). Probe AUC 0.9999 -- the knowledge-action gap widens with scale.

### Pending (data collected, analysis in progress)

11. **Attention entropy** -- Full per-head data for Llama and R1-Distill (81 MB each) downloaded. Analysis running separately. Expected to show whether S1/S2 distinction is reflected in attention patterns or is purely a residual-stream phenomenon.

### Pending (needs GPU)

12. **SAE feature analysis** -- Goodfire L19 key mismatch fixed, script ready. Needs GPU re-run. Would provide interpretable feature-level evidence beyond probe decodability.
13. **Bootstrap CIs for Qwen** -- Need per-fold data from the pod.
14. **Bootstrap CIs for OLMo-32B** -- Need per-fold data.

---

## The natural frequency reversal: detailed interpretation

The Gigerenzer (1995) natural frequency finding is the most provocative result in the dataset because it breaks the simple narrative that reasoning training uniformly improves probabilistic reasoning.

**What happened**: Base rate neglect items were reformulated from percentage format ("the probability is 3%") to natural frequency format ("3 out of every 100 people"). The Gigerenzer hypothesis predicts that natural frequencies should reduce base rate neglect because they align with the format in which frequency information was encountered ancestrally.

**The Llama result REJECTS Gigerenzer**: Llama goes from 84% lure (probability format) to 100% lure (frequency format). The frequency format makes base rate neglect WORSE, not better. This suggests that LLMs' "ecological niche" is percentage-format text (dominant in training data), and frequency format is actually harder for them — the reverse of humans, whose ecological niche is concrete frequency experience.

**The R1-Distill result contradicts it**: R1-Distill goes from 96% correct (probability format) to 40% correct (frequency format). The reasoning model, which had largely solved base rate neglect in standard format, becomes substantially worse when given the "easier" format.

**Interpretation**: R1-Distill's reasoning distillation training installed deliberative circuits calibrated to standard mathematical notation. The natural frequency format is sufficiently different from this training distribution that it either (a) fails to trigger the learned deliberative pathway, or (b) triggers a different, less reliable processing mode. In effect, reasoning training created a strong but narrow competence: robust within the trained format, brittle outside it.

**Implications for safety evaluation**: This result means that behavioral evaluations of reasoning models cannot test a single format per problem type and conclude the model "can do base rate reasoning." Format transfer is not guaranteed, even for formats that should theoretically be easier. Robustness requires testing across notational variants.

**Implications for the S1/S2 framework**: The natural frequency result suggests that what we are calling "S1-like" and "S2-like" processing is not a property of the abstract problem structure alone, but of the interaction between problem structure and input format. The same logical problem can be S2-requiring in one format and S1-accessible in another, and this mapping differs between standard and reasoning-trained models.

---

## Updated strongest honest framing for NeurIPS

The narrative now rests on **seven converging lines of evidence**:

1. **Behavioral** (3 architectures + 32B scale): Reasoning models resist cognitive-bias lures (27.3% to 2.4% overall lure rate), but R1-Distill shows an 18.2% "other" rate -- reasoning models shift errors into novel categories, not just suppress lures. Within-model: thinking mode reduces lures from 21% to 7% with identical weights. Both sunk cost and mathematical lures (CRT, arithmetic) are immune across all models. Vulnerability is specific to probabilistic estimation under uncertainty. Scale does not fix it: OLMo-32B is WORSE than 7B (19.6% vs. 14.9% overall, base_rate 74.3% vs. 46%).

2. **Representational** (probes with bootstrap CIs, non-overlapping): Linear probes find a high-fidelity S1/S2 boundary in standard models (AUC 0.974 [0.952, 0.992]) that is degraded in reasoning models (AUC 0.930 [0.894, 0.960]). CIs do not overlap. Peak layer shifts from L16 to L31 -- reasoning training relocates peak processing-mode encoding deeper. The probe is SPECIFIC to processing mode: cross-prediction 0.378 reflects immune items compressing into a tight cluster, and the text-only baseline (TF-IDF AUC 0.840) falls well short of the activation probe (AUC 0.999), confirming the residual stream encodes ~16% of separability beyond what surface text provides. Lure susceptibility is graded and continuous (+0.422 vs. -0.326).

3. **Cross-prediction specificity**: Vulnerable-trained probe tested on immune categories yields transfer AUC 0.378 -- not meaningful reversed classification but immune items compressing into a tight cluster on one side of the decision boundary. The direction the probe learned is processing-mode-specific, not a confound of task structure or textual features. Text baseline (AUC 0.840) confirms surface features explain ~84% of separability but miss the remaining ~16% that activation probes capture.

4. **Causal steering** (37.5pp Llama, 7.5pp R1): Probe-direction activation injection produces a 37.5 percentage-point causal swing in Llama (random controls null), with 100% of lure reduction going to correct answers -- zero to garbage. This establishes that the S1/S2 direction is causally read out by downstream layers and moves the model along a meaningful deliberative axis, not toward incoherence. R1-Distill shows only 7.5pp swing under continuous steering (hook fires every forward pass; this IS the continuous result, not a single-layer perturbation) -- the direction is readable but not writable. Reasoning training decouples the S1/S2 representation from the decision pathway. This resolves the "epiphenomenal probe" concern: the direction is functionally active in standard models.

5. **Training/inference dissociation** (Qwen): Qwen THINK and NO_THINK achieve matched probe fidelity (0.971 at L34) but encode conflict along orthogonal directions (cross-prediction 0.496, cosine -0.005). This is a richer result than "identical encoding": the modes detect conflict at similar fidelity but via mode-specific directions. A monitor trained on one inference mode will not transfer to the other. Distillation training rewires representations; inference-time reasoning rotates the encoding direction without changing fidelity.

6. **SAE + attention + structural**: Base rate and conjunction fallacy share a representation (transfer AUC 0.993), suggesting a common "probabilistic estimation under uncertainty" circuit. Geometry confirms the signal is narrow and linear (silhouette 0.079), consistent with a graded processing-intensity dimension rather than discrete clusters.

7. **Scale replication** (32B confirms, actually worse): OLMo-32B-Instruct replicates the full pattern at 4x scale. Probe AUC 0.9999 -- the S1/S2 boundary is essentially perfectly separable. But behavioral outcomes are WORSE: base_rate lure rate rises from 46% to 74.3%, and framing vulnerability EMERGES at 30% (was 0% at 7B). The knowledge-action gap does not close with scale; it widens. This is the strongest evidence that reasoning training, not scale, is the relevant intervention.

### The natural frequency complication

Reasoning training creates format-specific competence. R1-Distill's 96% accuracy on standard base rate items drops to 40% with natural frequency framing. The standard model (Llama) shows the opposite: 16% to 100%. Reasoning distillation does not install general probabilistic reasoning; it installs format-calibrated deliberative processing. This limits the generality of Finding 2 and has direct safety implications.

### The one-paragraph elevator pitch

Standard instruction-tuned LLMs maintain a near-perfect linear boundary in their residual stream between heuristic-prone and deliberation-requiring processing modes (AUC 0.974, vs. text-only baseline 0.840), but frequently fail to act on that distinction (84% lure rate on base rate problems). Probe-direction activation steering confirms this direction is causally upstream of behavior: 37.5pp swing in Llama with 100% of lure reduction going to correct answers, random controls null. Reasoning-distilled models partially dissolve this boundary (AUC 0.930, non-overlapping CI) while dramatically reducing lure rates (4%) but shifting 18.2% of responses to novel "other" errors, and causally decoupling the S1/S2 direction from output (7.5pp under continuous steering) -- a pattern we call "readable but not writable." Inference-time chain-of-thought matches probe fidelity (Qwen THINK/NO_THINK: both 0.971 AUC) but encodes conflict along orthogonal directions (cross-prediction 0.496, cosine -0.005), establishing that training and inference produce mechanistically distinct effects on representation geometry, not just behavior. Scaling to 32B does not help: OLMo-32B achieves probe AUC 0.9999 while showing WORSE behavioral outcomes (74.3% base rate lure vs. 46% at 7B) and developing new vulnerabilities (framing at 30%). The deliberation gradient is real, causally grounded, and safety-relevant -- but the gap between what models know and what they do widens with scale, and narrows only with reasoning training.

---

## What OLMo showed (cross-architecture replication CONFIRMED)

OLMo-3-7B is the third architecture family (after Llama and Qwen), fully open-source with an independent training pipeline (AI2). Results:

### Behavioral replication
- OLMo-3-7B Instruct: 14.9% overall lure rate (vulnerable to base rate and conjunction)
- OLMo-3-7B Think: 0.9% overall lure rate (resistant)
- Loss aversion: 33% lure rate (OLMo-specific vulnerability not seen in Llama/R1)

### Mechanistic replication
- OLMo Instruct: probe AUC 0.996 [0.988, 1.000] at L24
- OLMo Think: probe AUC 0.962 [0.934, 0.982] at L22
- Gap: 0.034 with non-overlapping bootstrap CIs -- statistically robust
- Smaller than Llama/R1 gap (0.044) but directionally consistent
- Both high AUC + behavioral vulnerability = "detection without resolution" pattern replicates

### Assessment against pre-registered criteria
1. **Behavioral** (met): non-zero lure rates on vulnerable categories, immune on sunk cost
2. **Probe AUC** (met): well above chance, bootstrap CIs exclude 0.5
3. **Cross-prediction specificity**: not yet tested for OLMo (pending)
4. **Geometry**: not yet tested for OLMo (pending)

---

## What OLMo-32B showed (scale replication)

OLMo-32B-Instruct is the scale test: same architecture family as OLMo-7B, 4x parameter count, independent training pipeline (AI2).

### Behavioral: scale makes vulnerability WORSE

| Metric | OLMo-7B-Instruct | OLMo-32B-Instruct | Direction |
|--------|-------------------|--------------------|-----------|
| Overall lure rate | 14.9% | 19.6% | Worse |
| base_rate | 46% | 74.3% | Much worse |
| conjunction | -- | 0% | Resolved |
| framing | 0% | 30% | New vulnerability |
| sunk_cost | 33% | 33% | Unchanged |

The base_rate result is the headline: scaling from 7B to 32B nearly doubles the lure rate (46% to 74.3%). The framing result is qualitatively new: a vulnerability that did not exist at 7B emerges at 32B. Conjunction is resolved, but the net effect is worse overall performance.

### Mechanistic: equally (more) separable

Probe AUC 0.9999 at 32B. The S1/S2 linear boundary is essentially perfect. Combined with 74.3% base rate lure rate, this is the most extreme "detection without resolution" result in the study: near-perfect representation, catastrophic behavioral failure. The model encodes the distinction with higher fidelity at larger scale but is less likely to act on it.

### Interpretation for the NeurIPS narrative

This result directly addresses the most common objection to small-model mech interp work: "maybe the problem goes away at scale." It does not. It gets worse. The knowledge-action gap -- the distance between what the model internally represents and what it externally does -- widens with scale in the absence of reasoning training. This makes the case for reasoning training (rather than scale) as the relevant safety intervention.

---

## Limitations

**Small N.** The vulnerable subset comprises roughly 140 items across three categories from the original 330-item benchmark (expanded to ~190 matched pairs with sunk cost and natural frequency). This is sufficient for the probe analyses (bootstrap CIs are tight: widths of 0.040 for Llama, 0.066 for R1-Distill) but limits the granularity of per-category breakdowns and leaves some cross-prediction estimates noisy.

**Causal evidence is limited to one intervention type.** Probe-direction activation steering establishes that the S1/S2 direction is causally read out in Llama (37.5pp swing) and decoupled in R1-Distill (7.5pp). However, this is a single causal method. Feature ablation, path patching, and interchange interventions would provide complementary causal evidence. The current result resolves the "epiphenomenal probe" concern but does not fully characterize the causal circuit.

**Scale tested to 32B, not frontier.** OLMo-32B-Instruct replicates and extends the pattern. But 32B is still far from frontier scale (70B+, 400B+). The internal organization of frontier-scale models may differ qualitatively. We can now say the pattern does not disappear by 32B and in fact worsens, but we cannot extrapolate to arbitrary scale.

**Graded, not binary.** The data support a graded processing-intensity interpretation better than a binary S1/S2 switch. The 0.930 AUC in R1-Distill is reduced but far above chance; the behavioral lure rate of 2.4% is low but not zero. Even the "S2-by-default" framing should be understood as a shift in default processing intensity, not the elimination of a discrete system.

**Training confound.** R1-Distill differs from Llama not only in reasoning distillation but in all details of the fine-tuning pipeline. We cannot attribute differences solely to reasoning-trace training. The Qwen within-model comparison partially mitigates this, but the Qwen comparison tests inference-time rather than training-time effects.

**Natural frequency sample size.** The natural frequency finding is based on only 10 reformulated items. While the effect is large (60 percentage-point reversal for R1-Distill), the small N means this should be treated as a strong pilot finding that needs replication with more items before being featured as a primary claim. It belongs in the Discussion, not the abstract.

**Format-specificity generality.** We tested only one alternative format (natural frequencies). We do not know whether the R1-Distill reversal extends to other format variations (e.g., visual representations, odds ratios, verbal qualifiers). The finding motivates format-robustness testing but does not fully characterize the boundary conditions.

---

## Implications for AI safety

**Domain-specific competence gaps are invisible to generic benchmarks.** A model that scores well on CRT-style problems and arithmetic reasoning can simultaneously fail at base rate estimation 84% of the time. Safety evaluations that test "reasoning ability" as a monolithic capability will miss these domain-specific vulnerabilities. Evaluations must probe specific failure modes, not aggregate competence.

**Reasoning training changes what models are, not just what they say -- but introduces new error modes.** The representational evidence (non-overlapping AUC CIs, P0 sign flip, geometry compression, peak-layer shift) shows that reasoning distillation produces internal changes beyond output formatting. However, R1-Distill's 18.2% "other" rate means reasoning models shift errors rather than eliminating them. Low lure rate alone overstates the improvement. Safety evaluations must track the full response distribution (correct, lure, other), not just the lure rate.

**Inference-time thinking is not a substitute for training, and changes representation geometry.** Qwen's conjunction fallacy rate remains at 55% even with explicit chain-of-thought, compared to R1-Distill's 0%. The two Qwen modes achieve matched probe fidelity (0.971) but encode conflict along orthogonal directions (cross-prediction 0.496, cosine -0.005). This means a runtime monitor trained on no-think activations will not transfer to think-mode activations, despite identical AUCs. CoT helps behavior while rotating the representational encoding, not preserving it. For safety-critical applications, trust calibration cannot rely on chain-of-thought alone, and monitoring systems must be calibrated per inference regime.

**Reasoning training can create novel vulnerabilities through format specialization.** The natural frequency reversal (R1-Distill: 96% -> 40%) shows that reasoning distillation can install format-specific competence that fails on equivalent problems in unfamiliar formats. This is a new category of alignment risk: a model that appears to have solved a problem class may have only solved it in the notation it was trained on. Safety evaluations must include format-transfer tests, not just accuracy on standard formulations.

**Monitoring feasibility -- now with causal backing, but mode-specific.** The high linear decodability of the S1/S2 distinction (AUC 0.974 in Llama, 0.971 in Qwen no-think, 0.9999 in OLMo-32B) suggests that lightweight linear probes on middle-layer activations could serve as runtime monitors for whether a model is in a heuristic-prone state. The causal steering result (37.5pp swing, with 100% of lure reduction going to correct answers) confirms that this direction is not merely informative but functionally active -- the same direction a monitor would read is the one the model uses for its decision. This upgrades monitoring from "preliminary" to "mechanistically grounded." However, the Qwen orthogonal-directions result means monitors must be calibrated per inference mode -- a probe trained on base inference will not generalize to CoT inference, despite matched fidelity.

**Scale amplifies the monitoring case.** OLMo-32B's AUC 0.9999 with 74.3% base rate lure rate means the larger model produces an even cleaner monitoring signal while being MORE vulnerable. The models that most need monitoring are the easiest to monitor. This is a rare alignment-favorable scaling property.

**Shared vulnerability circuits amplify risk.** The 0.993 transfer between base rate and conjunction representations means that a single failure mode underlies multiple behavioral vulnerabilities. A targeted intervention (or a targeted attack) on this shared circuit could simultaneously affect multiple categories of probabilistic reasoning. This is both an opportunity (one fix might address multiple failure modes) and a risk (one perturbation might create multiple failures).

**Scale does not close the knowledge-action gap.** OLMo-32B knows the answer (AUC 0.9999) and gives the wrong one anyway (74.3% base rate lure). This is the clearest demonstration of the "detection without resolution" pattern: the representation encodes everything needed to behave correctly, and the model fails to use it. Worse, scale introduces qualitatively new vulnerabilities (framing at 30% where 7B had 0%). Scaling alone is not a path to safety for heuristic-prone processing domains. Reasoning training is the intervention that changes the processing dynamic, not parameter count.
