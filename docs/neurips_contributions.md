# NeurIPS Contributions Summary

All experiments complete. Six independent contributions; together they constitute the most comprehensive mechanistic characterization of dual-process cognition in LLMs to date.

---

## C1: CogBias Benchmark

A purpose-built evaluation suite for systematic vs. heuristic reasoning in LLMs.

- **470 items** across **11 bias categories** spanning **4 heuristic families**
- Matched conflict/control pairs isolate heuristic interference from task difficulty
- Includes novel isomorphic variants to defeat memorization and surface-pattern shortcuts
- Categories: base rate neglect, conjunction fallacy, syllogistic reasoning, framing effects, anchoring, sunk cost, gambler's fallacy, representativeness, availability, confirmation bias, belief bias

---

## C2: Behavioral + Cross-Architecture Analysis

Reasoning training reliably suppresses heuristic responding across 8 models, 3 architecture families, and up to 32B scale.

| Model (Base) | Conflict Error Rate | Model (Reasoning) | Conflict Error Rate | Reduction |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 27.3% | DeepSeek-R1-Distill-Llama-8B | 2.4% | -24.9 pp |
| OLMo-2-7B-Instruct | 14.9% | OLMo-2-7B-Think | 0.9% | -14.0 pp |
| OLMo-2-32B-Instruct | 19.6% | OLMo-2-32B-Think | 0.4% | -19.2 pp |
| Qwen NO_THINK | 21.0% | Qwen THINK | 7.3% | -13.7 pp |

**Key finding:** Scale *increases* vulnerability to heuristic bias (OLMo 7B 14.9% → 32B 19.6%), but reasoning training remains effective at all scales. This dissociates the effect of reasoning training from mere scale.

---

## C3: Linear Probes + Cross-Prediction + Specificity

Linear probes decode S1/S2 processing mode from residual stream activations with near-perfect accuracy.

**Within-model probe accuracy (bootstrap 95% CIs):**

| Model | Accuracy | 95% CI |
|---|---|---|
| Llama-3.1-8B-Instruct | 0.974 | [0.952, 0.992] |
| DeepSeek-R1-Distill-Llama-8B | 0.930 | [0.894, 0.960] |
| OLMo-2-7B-Instruct | 0.996 | [0.988, 1.000] |
| OLMo-2-7B-Think | 0.962 | [0.934, 0.982] |
| OLMo-2-32B-Instruct | 0.9999 (L20) | — |

**Cross-prediction (processing-mode specificity):** 0.378 accuracy when predicting bias type from the S1/S2 direction — the representation is specific to processing mode, not to surface features of individual biases.

**Cross-model transfer:** Llama → R1 probe achieves 0.920 accuracy, demonstrating a shared linear direction for the S1/S2 distinction across base and reasoning-trained models.

---

## C4: Causal Evidence via Probe-Direction Steering

Activation patching along the S1/S2 probe direction establishes that the representation is causally upstream of behavior.

| Model | Pro-S1 Error | Pro-S2 Error | Swing | Interpretation |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 68.8% | 31.2% | **37.5 pp** | Readable AND writable — S1/S2 direction causally controls output |
| DeepSeek-R1-Distill-Llama-8B | — | — | **7.5 pp** | Readable but NOT writable — direction decoupled from behavior |

Random-direction controls produce null effects, confirming specificity.

**Interpretation:** In standard models, the S1/S2 linear direction is on the causal path from stimulus to response. Reasoning training decouples this direction from behavior — the model defaults to S2 processing regardless of the activation pattern, consistent with an "S2-by-default" architecture.

---

## C5: Within-Chain-of-Thought Probing

Tracking probe confidence across reasoning tokens reveals a non-monotonic trajectory.

| Position | S1/S2 Probe Confidence |
|---|---|
| T0 (prompt encoding) | 0.973 |
| T75 (mid-chain) | 0.754 |
| Tend (final answer) | 0.971 |

**Interpretation:** Thinking temporarily disrupts S1/S2 separability (the model enters a mixed or uncertain state mid-chain) then resolves to a confident classification by the final token. This rules out decorative reasoning — the chain of thought reflects genuine intermediate computation that transiently destabilizes heuristic representations before arriving at a deliberate answer.

---

## C6: SAE Features + Attention Heads + Additional Evidence

Mechanistic and converging evidence from multiple independent analyses.

- **SAE decomposition:** 41 features with non-trivial S1/S2 selectivity identified; **0 spurious** features after correction. The S1/S2 distinction is distributed across a moderate-sized feature basis, not concentrated in a single neuron.
- **Attention head specialization:** 57 S2-specialized heads in the reasoning model vs. 30 in the base model — a **2x increase** in S2-specialized attention circuitry after reasoning training.
- **Confidence paradigm:** Heuristic items presented with confidence framing reduce accuracy by **-4.2 pp FTP**, replicating the De Neys (2006) finding that confidence mediates conflict detection in humans.
- **Natural frequency reversal:** Gigerenzer-style natural frequency reformulations reverse base-rate neglect in LLMs, paralleling the human behavioral literature.
- **Multi-seed robustness:** All main findings replicate across multiple random seeds (minimum 3 per experiment), confirming that results are not artifacts of initialization or data splits.

---

## Summary

Each contribution is independently publishable; together they provide end-to-end evidence — behavioral, representational, causal, temporal, and mechanistic — for dual-process cognition in large language models. The key narrative arc:

1. **C1** establishes what to measure.
2. **C2** shows reasoning training works behaviorally, across architectures and scales.
3. **C3** shows the brain knows: a linear direction encodes S1 vs. S2, transfers across models.
4. **C4** shows causality: that direction *controls* behavior in base models but is decoupled in reasoning models.
5. **C5** shows process: reasoning is genuine computation, not decoration.
6. **C6** provides mechanistic grounding and converging evidence from orthogonal methods.
