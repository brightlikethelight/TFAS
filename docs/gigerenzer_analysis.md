# Gigerenzer Natural Frequency Analysis: Reverse Effect in LLMs

**Date**: 2026-04-12
**Data source**: `results_pod/behavioral/new_items_llama_31_8b_instruct.json`, `results_pod/behavioral/new_items_r1_distill_llama_8b.json`
**Comparison baseline**: `results_pod/behavioral/llama31_8b_ALL.json`, `results_pod/behavioral/r1_distill_llama_ALL.json`

---

## 1. Result Summary

We tested Gigerenzer and Hoffrage's (1995) ecological rationality hypothesis: does reformulating base rate problems using natural frequencies ("10 out of 1000") instead of probabilities ("1%") reduce base rate neglect?

**Natural frequency framing increased lure rates in both models -- the opposite of Gigerenzer's prediction.**

| Model | Probability format | Natural frequency format | Direction |
|-------|-------------------|------------------------|-----------|
| Llama-3.1-8B-Instruct | 84% (21/25) | **100% (10/10)** | +16pp WORSE |
| R1-Distill-Llama-8B | 4% (1/25) | **50% (5/10)** | +46pp MUCH WORSE |

> **CORRECTED (2026-04-13)**: Original run showed 40% due to a scoring bug (BPE artifacts + truncation). Re-run with fixed scoring yields **50% lure** (5/10 lured, 5/10 correct, 0 other). The reversal is even more dramatic than initially measured: R1-Distill goes from near-immunity (4%) to coin-flip performance (50%).

Sunk cost items (tested alongside) showed 0% lure rate for both models in both formats, confirming that the effect is specific to base rate problems and not a generic artifact of the new item set.

### Item-level breakdown (natural frequency format)

**Llama-3.1-8B-Instruct**: 10/10 conflict items answered with the lure. 10/10 controls answered correctly. The model uniformly fails.

**R1-Distill-Llama-8B**: 5/10 lure, 5/10 correct, 0/10 other. With fixed scoring (corrected BPE artifact and truncation bugs), the reasoning model splits exactly 50/50 between lure and correct answers. Unlike the base Llama model which uniformly falls for the lure, R1-Distill's reasoning sometimes overcomes the representativeness heuristic -- but only half the time. Controls: 10/10 correct. The clean 50/50 split (zero "other") confirms that the model always reaches a definitive answer; the earlier 40% "other" rate was entirely a scoring artifact.

---

## 2. Why This Matters

Gigerenzer's natural frequency hypothesis is one of the most influential critiques of the heuristics-and-biases program. The core claim: human base rate neglect is not a deep cognitive failure but an artifact of presenting statistical information in formats that are ecologically unnatural (single-event probabilities). When the same information is presented as natural frequencies -- the format in which humans encountered statistical regularities during evolutionary history -- the bias largely disappears.

This has been replicated extensively in humans (Gigerenzer & Hoffrage, 1995; Cosmides & Tooby, 1996; Hoffrage et al., 2000; Gigerenzer, 2015 meta-analysis). Typical human results show base rate neglect dropping from ~70-80% under probability framing to ~20-30% under frequency framing.

**Our LLM results show the exact reverse.** Both models perform worse with natural frequencies, and the reasoning-trained model shows the most dramatic regression. This is a genuine falsification of the Gigerenzer critique as applied to LLMs.

This matters for three reasons:

1. **It rules out a major alternative explanation.** If someone argues our base rate neglect findings merely reflect sensitivity to problem format (not a processing-mode distinction), this result shows that changing the format to the one that helps humans makes LLMs worse. The bias is not an artifact of format.

2. **It demonstrates mechanistic divergence between LLM and human cognition.** Gigerenzer's frequency facilitation effect depends on an evolutionary argument about the computational architecture of the mind being adapted to frequency-based input. LLMs have no such evolutionary history -- their "ecological niche" is the training corpus, where probability format is far more common than frequency format.

3. **It reveals fragility in reasoning training.** R1-Distill's leap from 4% to 50% means the reasoning distillation that nearly eliminated base rate neglect in probability format is highly sensitive to input framing. The learned strategy is partially format-dependent rather than principle-general.

---

## 3. Interpretations

### 3.1 Training distribution mismatch

The ecological rationality hypothesis rests on the argument that humans evolved to process frequencies because that is the format in which statistical information was encountered ancestrally. For LLMs, the relevant "ecology" is the training corpus. Probability-format base rate problems (textbook style, "the probability is 1%") are far more common in training data than frequency-format problems ("10 out of 1000 people"). The models may have learned heuristic-bypassing strategies that are specific to the probability format simply because that is where the training signal is concentrated. Frequency format is, for the model, the ecologically unusual presentation.

This is almost the mirror image of Gigerenzer's argument: if the bias-reduction effect depends on ecological familiarity with the input format, and the LLM's ecology is inverted relative to humans, we should expect the inverted result.

### 3.2 Token-level distraction

Natural frequency prompts are longer and more syntactically complex: "Out of every 1000 people in this city, 10 are carpenters" versus "1% of people in this city are carpenters." The additional tokens provide more surface-level content for the representational heuristic to latch onto. The frequency format also introduces explicit counts (10, 1000) that may trigger numerical comparison heuristics rather than Bayesian reasoning. This is a confound that should be controlled in follow-up work (see Section 5).

### 3.3 Fragility of reasoning training

R1-Distill's regression from 4% to 50% is the most striking result. This model nearly solved base rate neglect in probability format -- but that solution transfers to frequency format only half the time. This suggests the reasoning distillation taught a partially format-dependent strategy: the model can sometimes invoke Bayesian reasoning on frequency-format items (5/10 correct), but the surface-level cues that reliably trigger this strategy in probability format ("the probability is 1%") are absent in frequency format ("10 out of 1000"), causing the strategy to fire inconsistently.

This has direct implications for evaluating reasoning models: benchmark performance in one format may dramatically overestimate general reasoning capability.

### 3.4 R1-Distill's clean 50/50 split

With corrected scoring, R1-Distill produces 0% "other" responses on frequency-format items -- every response clearly maps to either the lure or correct answer. The earlier 40% "other" rate was entirely a scoring artifact (BPE artifacts breaking substring matching). The real pattern is more informative: the model always reasons to a definitive conclusion, but the conclusion is correct only half the time. Examining the 5 lured items vs. the 5 correct items may reveal which item features (base rate ratio, description vividness, occupation rarity) tip the balance between the representativeness heuristic and Bayesian reasoning.

---

## 4. Implications for the Paper

### Strengthens specificity claims

Our paper's central contribution is the finding that S1/S2-like processing signatures are category-specific and format-specific, not a monolithic "reasoning switch." The Gigerenzer result provides the sharpest evidence for format specificity: the same underlying statistical problem, in two different surface formats, produces dramatically different lure rates.

### Provides a genuine falsification

The preregistration (H8) committed to honestly reporting whichever direction this result went. The result is a falsification of the Gigerenzer prediction as applied to LLMs, which is a clean, directional finding rather than a null result. This is more informative than if the frequency format had simply failed to help.

### Supports the "training ecology" framing

The result supports a framing in the paper where we argue that LLMs' cognitive biases arise from training data statistics rather than from the kind of evolved computational architecture that Gigerenzer's framework presupposes. This distinguishes our findings from straightforward anthropomorphism: we are not claiming LLMs "think like humans" but rather that they exhibit structurally similar biases with mechanistically distinct origins.

### Sunk cost immunity holds

The 0% lure rate on sunk cost items across both models and both formats adds another data point to the category-specificity story. Sunk cost is immune regardless of format, just as CRT, arithmetic, framing, and anchoring are immune. The vulnerability pattern is consistent and replicable.

---

## 5. Suggested Follow-Up Experiments

### 5.1 Vary the population N

Test natural frequency prompts with different population sizes: N=100 ("1 out of 100"), N=1000 ("10 out of 1000"), N=10000 ("100 out of 10000"). If the lure rate varies with N despite identical base rates, this would confirm that numerical magnitude is a distractor. If it does not, the effect is genuinely about frequency format.

### 5.2 Chain-of-thought prompting on frequency items

Test whether explicit chain-of-thought ("Let's think step by step") rescues performance on frequency-format items. For R1-Distill, compare the thinking-mode outputs between probability and frequency format to identify where the reasoning chain diverges.

### 5.3 Explicit Bayesian hints

Add a line like "Use Bayes' theorem to calculate the probability" to frequency-format items. If this restores performance, it suggests the model knows the procedure but fails to activate it without the right surface cues -- reinforcing the format-specificity interpretation.

### 5.4 Prompt-length control

Create probability-format items that are artificially lengthened to match the token count of frequency-format items (e.g., by adding irrelevant but syntactically parallel filler). This controls for the token-length confound in Section 3.2.

### 5.5 Cross-model extension

Test the same frequency-format items on Qwen 3-8B (THINK and NO_THINK modes) and OLMo-2-7B-Instruct. If the reverse Gigerenzer effect is universal across architectures, it is a robust property of LLM training. If it is specific to certain models, the training data composition becomes the key variable.

### 5.6 Larger N for statistical power

The current result is based on 10 natural frequency conflict items per model. While the effect sizes are large (especially Llama's 100% lure rate), a larger item set (25-30 items) would provide more statistical power for McNemar's test and allow subgroup analysis by item content.
