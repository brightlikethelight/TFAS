# Advisor Summary: S1/S2 Mechanistic Signatures in LLMs

**Project**: HUSAI Semester Project — ICML MechInterp Workshop submission
**Date**: 2026-04-12 | **Deadline**: May 8 AOE (25 days)
**Student**: Bright Liu | **Status**: Code complete, paper drafted, awaiting advisor review

---

## One-sentence summary

Reasoning-distilled LLMs show *lower* linear separability of conflict vs. control items in their residual streams (AUC 0.974 -> 0.930, non-overlapping CIs), despite dramatically better behavioral performance — evidence that reasoning training integrates deliberation into default processing rather than sharpening a separate "System 2" pathway.

---

## Key numbers

**Behavioral lure rates (greedy decoding):**

| Model | Overall | base_rate | conjunction | syllogism |
|-------|---------|-----------|-------------|-----------|
| Llama-3.1-8B-Instruct | 27.3% | 84% | 55% | 52% |
| R1-Distill-Llama-8B | 2.4% | 4% | 0% | 0% |
| Qwen-3-8B NO_THINK | 21% | 56% | 95% | 0% |
| Qwen-3-8B THINK | 7% | 4% | 55% | 0% |
| OLMo-3-7B Instruct | 14.9% | 46% | 50% | -- |
| OLMo-3-7B Think | 0.9% | -- | -- | -- |

**Probe AUCs (vulnerable categories, bootstrap 95% CI):**

| Model | Peak AUC | 95% CI | Peak Layer |
|-------|----------|--------|------------|
| Llama-3.1-8B-Instruct | 0.974 | [0.952, 0.992] | L16 |
| R1-Distill-Llama-8B | 0.930 | [0.894, 0.960] | L31 |
| OLMo-3-7B Instruct | 0.996 | [0.988, 1.000] | L24 |
| OLMo-3-7B Think | 0.962 | [0.934, 0.982] | L22 |

**Cross-prediction**: Probe trained on vulnerable, tested on immune categories: AUC = 0.378 at L14 (below chance; captures processing mode, not surface features).

---

## What is novel

1. **Cross-prediction specificity test**: No prior work on LLM cognitive bias probing controls for the surface-feature confound this way. The 0.378 AUC directly addresses whether probes detect processing mode or task structure.
2. **Thinking toggle + internals**: Qwen's /think mode reduces lure rates (21% -> 7%) without changing pre-generation probe signatures (both AUC = 0.971). This dissociates training effects from inference-time effects on representations — a result not available from cross-model comparisons alone.
3. **SAE falsification**: 41 Goodfire L19 features survive Ma et al. spuriousness testing (0 spurious), with 74% explained variance. SAE does NOT transfer to R1-Distill (EV = 25%), consistent with the "reorganized representations" story.
4. **Three architecture families**: Llama, Qwen, and OLMo all show the same behavioral pattern (reasoning variants reduce lure rates). Llama and OLMo both show the probe AUC gap with non-overlapping CIs.

---

## Honest weaknesses

1. **OLMo probe gap is smaller**: 0.996 -> 0.962 (vs. Llama's 0.974 -> 0.930). CIs do not overlap, but the effect is compressed near ceiling. A skeptic could call this marginal.
2. **Natural frequency finding is underpowered**: N=10 items, non-significant by Fisher exact test. R1-Distill result was invalidated by a scoring bug and retracted. Currently a preliminary observation, not a finding.
3. **Generation strategy dependence**: Multi-seed sampled decoding flips category profiles vs. greedy (e.g., framing: 0% greedy -> 53% sampled). Probe results are unaffected (P0 representation), but the behavioral narrative is more nuanced than the greedy-only table suggests.
4. **No causal evidence**: We have representational correlates, not causal demonstrations. Steering/ablation experiments are descoped to a future ICLR paper.
5. **Cross-prediction does not fully replicate in R1-Distill**: High transfer at early layers (0.878), low at late layers (0.385). The specificity story is cleaner for Llama than for R1-Distill.

---

## What we need from you

1. **Framing feedback**: Is "reasoning training blurs S1/S2 boundary" the right headline, or should we lead with the cross-prediction specificity result? The critical self-review flags overclaiming risk with "mechanistic" in the title — should we use "representational" instead?
2. **Co-author list**: Please confirm who should be listed and in what order.
3. **Department affiliation**: Harvard SEAS? Harvard Psychology? HUSAI?
4. **Paper format**: ICML MechInterp Workshop accepts 4pp (short) or 8pp (long). Current draft is ~7.5pp in 2-column. Do you recommend submitting as 8pp long, or cutting to 4pp short for a tighter narrative?
5. **ICML template**: We need to switch from `article` class to official ICML 2026 style. Any preferred Overleaf setup?

---

## Timeline

25 days to May 8 deadline. Paper draft is written, all experiments complete except R1-Distill multi-seed (running now). Critical path: advisor feedback -> ICML template switch -> figure finalization -> submit May 7 (1-day buffer).
