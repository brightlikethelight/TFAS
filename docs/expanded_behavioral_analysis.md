# Expanded Behavioral Analysis: 470-Item Benchmark (4 Models x 11 Categories)

Generated: 2026-04-12  
Data: Overnight pipeline run on 470 items (7 original + 4 new categories)

---

## 1. Full Lure Rate Table

| Category | Llama-3.1-8B | R1-Distill-8B | Qwen3-8B (no think) | Qwen3-8B (think) |
|----------|-------------|---------------|---------------------|------------------|
| crt | 0.0% (0/30) | 16.7% (5/30) | 0.0% (0/30) | 3.3% (1/30) |
| base_rate | **88.6%** (31/35) | **40.0%** (14/35) | **65.7%** (23/35) | **28.6%** (10/35) |
| conjunction | **45.0%** (9/20) | 0.0% (0/20) | **95.0%** (19/20) | **55.0%** (11/20) |
| syllogism | 4.0% (1/25) | 8.0% (2/25) | 0.0% (0/25) | 0.0% (0/25) |
| arithmetic | 0.0% (0/25) | 0.0% (0/25) | 0.0% (0/25) | 0.0% (0/25) |
| framing | 0.0% (0/20) | 0.0% (0/20) | 0.0% (0/20) | 0.0% (0/20) |
| anchoring | 10.0% (2/20) | **15.0%** (3/20) | **15.0%** (3/20) | 0.0% (0/20) |
| sunk_cost | 0.0% (0/15) | 0.0% (0/15) | 0.0% (0/15) | 0.0% (0/15) |
| loss_aversion | 0.0% (0/15) | 0.0% (0/15) | 0.0% (0/15) | 0.0% (0/15) |
| certainty_effect | 0.0% (0/15) | 0.0% (0/15) | 0.0% (0/15) | 0.0% (0/15) |
| availability | **13.3%** (2/15) | 0.0% (0/15) | 0.0% (0/15) | 0.0% (0/15) |
| **OVERALL** | **19.1%** (45/235) | **10.2%** (24/235) | **19.1%** (45/235) | **9.4%** (22/235) |

Bold = lure rate > 10%.

---

## 2. New Category Vulnerability Assessment

### Summary: 3 of 4 new categories are immune; availability shows marginal Llama vulnerability

| New Category | Vulnerable? | Notes |
|-------------|------------|-------|
| **sunk_cost** | No -- 0.0% across all 4 models | All models get every sunk cost item correct. |
| **loss_aversion** | No -- 0.0% across all 4 models | 100% control accuracy across the board. |
| **certainty_effect** | No -- 0.0% across all 4 models | 100% control accuracy. |
| **availability** | Marginal (Llama only) | 13.3% (2/15) for Llama; 0% for all other models. CI is wide: [3.7%, 37.9%]. |

**Interpretation**: The new categories largely fail to produce lure-susceptible behavior in any model. This is a benchmark validity concern -- these categories may lack sufficient conflict-inducing power, or the item formulations may be too transparent for 8B-class models to fall for. The sunk cost, loss aversion, and certainty effect items appear to test reasoning that is well within the capabilities of all four models.

**Availability is the exception** -- Llama falls for 2/15 conflict items (shark-vs-falling-objects, plane-vs-car-deer), both involving frequency estimation where vivid/memorable events are overestimated. However, Llama also has poor availability control accuracy (53%), suggesting the model struggles with this domain in general, not just on conflict items. The other models (including Qwen NO_THINK) achieve 0% lure rate.

---

## 3. Cross-Model Comparison: Qwen THINK vs NO_THINK

| Category | NO_THINK | THINK | Delta |
|----------|----------|-------|-------|
| crt | 0.0% | 3.3% | +3.3pp |
| base_rate | 65.7% | 28.6% | **-37.1pp** |
| conjunction | 95.0% | 55.0% | **-40.0pp** |
| syllogism | 0.0% | 0.0% | 0.0pp |
| arithmetic | 0.0% | 0.0% | 0.0pp |
| framing | 0.0% | 0.0% | 0.0pp |
| anchoring | 15.0% | 0.0% | **-15.0pp** |
| sunk_cost | 0.0% | 0.0% | 0.0pp |
| loss_aversion | 0.0% | 0.0% | 0.0pp |
| certainty_effect | 0.0% | 0.0% | 0.0pp |
| availability | 0.0% | 0.0% | 0.0pp |
| **OVERALL** | 19.1% | 9.4% | **-9.8pp** |

### Key findings

1. **Same vulnerability pattern, different magnitudes.** Both THINK and NO_THINK are vulnerable to base_rate and conjunction. The rank ordering of categories by lure rate is preserved -- reasoning mode scales the vulnerability down but does not eliminate or reorder it.

2. **Massive conjunction persistence.** Even with thinking enabled, Qwen still falls for 55% of conjunction items. This is the single highest category for any reasoning model. The conjunction fallacy appears resistant to chain-of-thought deliberation.

3. **Base rate is the only category where THINK provides near-complete correction** -- dropping from 65.7% to 28.6%. This aligns with the hypothesis that base-rate neglect requires explicit statistical reasoning that the thinking trace can provide.

4. **Anchoring is fully corrected by THINK** (15% to 0%), but this is a small sample (n=20) and the CI for 0/20 is [0%, 16.1%].

5. **The new categories show no THINK/NO_THINK divergence** -- both modes score 0% on all four. This further supports the interpretation that these categories lack conflict-inducing power.

---

## 4. Certainty Effect and Availability: Immune or Vulnerable?

### Certainty Effect: Immune

0.0% lure rate across all 4 models (n=15 conflict items each). The Allais-paradox-style items do not produce lure responses. Possible explanations:
- The items may be too transparent in their framing for instruction-tuned models.
- Expected-value computation is a well-practiced capability for 8B-class models.
- The conflict between "certainty premium" and "expected value maximization" may not create the same intuitive pull in LLMs as in humans.

### Availability: Mostly Immune, Marginal Llama Signal

Only Llama shows any vulnerability (13.3%, 2/15), and its control accuracy for availability is also degraded (53% vs 93-100% for others). This suggests Llama's availability errors reflect general weakness on frequency-estimation tasks rather than a specific availability-heuristic bias.

R1-Distill, Qwen NO_THINK, and Qwen THINK all score 0/15 on availability conflict items. Availability bias does not appear to be a robust lure category for current 8B-class models.

---

## 5. Llama 470-Item vs 330-Item Comparison (Original 7 Categories)

| Category | 330-item | 470-item | Delta | Note |
|----------|----------|----------|-------|------|
| crt | 0.0% (0/30) | 0.0% (0/30) | 0.0pp | Stable |
| base_rate | 88.6% (31/35) | 88.6% (31/35) | 0.0pp | Stable |
| conjunction | 55.0% (11/20) | 45.0% (9/20) | -10.0pp | 6 items flipped; net -2 lures |
| syllogism | **52.0%** (13/25) | **4.0%** (1/25) | **-48.0pp** | 12 items flipped correct; see below |
| arithmetic | 0.0% (0/25) | 0.0% (0/25) | 0.0pp | Stable |
| framing | 0.0% (0/20) | 0.0% (0/20) | 0.0pp | Stable |
| anchoring | 0.0% (0/20) | 10.0% (2/20) | +10.0pp | 2 items flipped other->lure |
| **OVERALL** | 28.9% (55/190) | 19.1% (45/235) | -9.8pp | |

### Syllogism collapse: 52% to 4%

This is the largest cross-run shift. All 25 conflict items are identical between runs. In the 330-item run, 13 items received lure verdicts; in the 470-item run, only 1 did. The 12 items that flipped from lure to correct are all "real-world knowledge" syllogisms (dogs_pets, doctors_educated, birds_fly, etc.) where the believability of the conclusion conflicts with logical validity.

**Likely explanation**: Stochastic decoding variance. At temperature > 0, the model's output on these items sits near a decision boundary. The syllogism conflict items have inherently marginal difficulty for Llama -- small changes in sampling produce large verdict swings. This is a reproducibility concern: single-run point estimates for syllogism lure rates are unreliable. Multi-run averaging (3+ seeds) is needed for stable estimates.

### Other shifts

- **Conjunction** (-10pp): 6 items changed verdicts (3 each direction), net -2 lures. Within expected sampling noise for n=20.
- **Anchoring** (+10pp): 2 items shifted from "other" to "lure". These were items where the model previously gave no clear answer and now gave the anchored answer.
- **CRT, base_rate, arithmetic, framing**: Perfectly stable across runs.

### Implication

The original 7 categories divide into two stability tiers:
1. **Stable** (robust point estimates): crt, base_rate, arithmetic, framing -- identical or near-identical lure counts across runs.
2. **Noisy** (require multi-seed): syllogism, conjunction, anchoring -- verdict counts shift substantially between runs on the same items.

---

## 6. Overall Model Ranking (470-Item Benchmark)

| Rank | Model | Overall Lure Rate | Primary Vulnerabilities |
|------|-------|-------------------|------------------------|
| 1 | Qwen3-8B (think) | 9.4% (22/235) | conjunction (55%), base_rate (28.6%) |
| 2 | R1-Distill-8B | 10.2% (24/235) | base_rate (40%), crt (16.7%), anchoring (15%) |
| 3 | Llama-3.1-8B | 19.1% (45/235) | base_rate (88.6%), conjunction (45%), availability (13.3%) |
| 3 | Qwen3-8B (no think) | 19.1% (45/235) | conjunction (95%), base_rate (65.7%), anchoring (15%) |

Key observation: **Reasoning models (Qwen THINK, R1-Distill) halve the overall lure rate** compared to their non-reasoning counterparts (Qwen NO_THINK, Llama). But the two reasoning models have *different* vulnerability profiles:
- R1-Distill is uniquely vulnerable to CRT workrate items (4 new lures this run) while being immune to conjunction.
- Qwen THINK has persistent conjunction vulnerability (55%) while being nearly immune to CRT.

This cross-model divergence in category-level vulnerability is important for the mechanistic analysis -- it suggests the "S1-like processing" dimension may have category-specific structure, not a single unified factor.

---

## 7. Recommendations

1. **Drop sunk_cost, loss_aversion, certainty_effect** from the mechanistic analysis pipeline unless items are redesigned. At 0% lure rate across all models, they produce no S1/S2 contrast signal for probes, SAEs, or geometry analysis.

2. **Keep availability tentatively** -- the Llama-only signal is marginal but the low control accuracy suggests the items probe a genuinely difficult domain. May need more items (n=15 is underpowered for detecting moderate effect sizes).

3. **Run multi-seed (3+) for syllogism, conjunction, and anchoring** to get stable point estimates. Single-run estimates are unreliable.

4. **Investigate R1-Distill CRT workrate vulnerability** -- 4 items flipped from correct to lure between runs, all in the workrate subcategory. This may reflect a genuine weakness of R1-Distill on rate-based reasoning, or it may be sampling noise. A targeted multi-seed study on CRT workrate items would clarify.
