# R1-Distill Multi-Seed Robustness Analysis

## Setup

- **Model**: DeepSeek-R1-Distill-Llama-8B (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`)
- **Seeds**: 0, 42, 123
- **Generation**: `do_sample=True`, temperature=0.7, top_p=0.95, max_new_tokens=2048
- **Benchmark**: 470 items (235 conflict pairs), 11 categories
- **Source**: `results/robustness/deepseek-ai_DeepSeek-R1-Distill-Llama-8B_multiseed.json`
- **Compute**: B200 pod, 3h31m total (70.1 + 77.5 + 63.5 min per seed)

## Overall Result

| Metric | Value |
|--------|-------|
| Mean lure rate | 12.1% |
| Std (pp) | 1.0 |
| Per-seed | 11.5% / 11.5% / 13.2% |
| Stable? | **Yes** |

## Key Question: Is the 2.4% Greedy Lure Rate Stable Under Sampling?

**No -- it increases substantially.** The greedy-decoding lure rate of 2.4% rises to 12.1% under sampling (temperature=0.7). This is a 5x increase, though R1-Distill remains the most robust model in the comparison (Llama-3.1-8B goes from 27.3% greedy to 27.5% sampled).

The 12.1% rate is **stable across seeds** (std = 1.0pp, well below the 10pp instability threshold). Seeds 0 and 42 are nearly identical (11.5%), and seed 123 is slightly higher (13.2%) but within normal variance.

## Per-Category Breakdown

| Category | Mean | Std (pp) | Stable? | Seeds: 0 / 42 / 123 |
|----------|------|----------|---------|----------------------|
| crt | 35.6% | 5.1 | Yes | 30.0% / 40.0% / 36.7% |
| framing | 31.7% | 17.6 | **No** | 15.0% / 30.0% / 50.0% |
| loss_aversion | 22.2% | 10.2 | **No** | 33.3% / 13.3% / 20.0% |
| syllogism | 14.7% | 8.3 | Yes | 24.0% / 8.0% / 12.0% |
| conjunction | 11.7% | 2.9 | Yes | 10.0% / 10.0% / 15.0% |
| arithmetic | 5.3% | 2.3 | Yes | 8.0% / 4.0% / 4.0% |
| anchoring | 1.7% | 2.9 | Yes | 0.0% / 5.0% / 0.0% |
| base_rate | 1.0% | 1.6 | Yes | 0.0% / 2.9% / 0.0% |
| availability | 0.0% | 0.0 | Yes | 0.0% / 0.0% / 0.0% |
| certainty_effect | 0.0% | 0.0 | Yes | 0.0% / 0.0% / 0.0% |
| sunk_cost | 0.0% | 0.0 | Yes | 0.0% / 0.0% / 0.0% |

## Unstable Categories

Two categories exceed the 10pp instability threshold:

1. **Framing** (std = 17.6pp): Ranges from 15% to 50% across seeds. This is the same category that showed dramatic sensitivity in Llama-3.1-8B (0% greedy -> 53% sampled). Framing items are inherently sensitive to generation strategy because the "correct" and "lured" answers are semantically close (loss vs. gain frame).

2. **Loss aversion** (std = 10.2pp): Ranges from 13% to 33%. Marginal instability (just over the 10pp threshold). With only 15 conflict items per category, 1-2 item flips cause large percentage swings.

Both unstable categories have small item counts (20 and 15 items respectively), amplifying variance.

## Comparison: R1-Distill vs. Llama-3.1-8B Under Sampling

| Metric | R1-Distill | Llama-3.1-8B |
|--------|-----------|--------------|
| Greedy lure rate | 2.4% | 27.3% |
| Sampled lure rate (mean) | 12.1% | 27.5% |
| Sampled std (pp) | 1.0 | 1.3 |
| Delta (sampled - greedy) | +9.7pp | +0.2pp |
| Unstable categories | 2 | 1 |

R1-Distill's greedy-mode "near-immunity" is partially a decoding artifact. Under sampling, many more items become vulnerable. However, R1-Distill still shows 2.3x lower lure rates than Llama-3.1-8B under identical sampling conditions, confirming that reasoning distillation provides genuine robustness, not just greedy-mode surface immunity.

## Implications for the Paper

### The 2.4% is Not the Whole Story

The greedy lure rate (2.4%) is the model's maximum-probability behavior. The sampled rate (12.1%) reveals a broader vulnerability profile. Both numbers should be reported:
- Table 1 (primary behavioral results): keep greedy numbers as the primary comparison
- Robustness appendix: report sampled numbers with this analysis
- Key claim: "R1-Distill reduces lure rates by ~2.3x under sampling and ~11x under greedy decoding"

### CRT and Framing Are the Weak Spots

Even for R1-Distill, CRT (35.6%) and framing (31.7%) emerge as the highest-vulnerability categories under sampling. This is notable because both are **zero** under greedy decoding. The model's "thinking" traces may sometimes lead it to the correct answer under greedy (highest-probability path), but the probability mass on the lured answer is non-trivial.

### Probe Results Are Unaffected

As with the Llama analysis, probe results (AUC, cross-prediction, transfer) are computed on P0 representations (single forward pass, no generation) and are invariant to decoding strategy.

## Raw Data

Full per-item, per-seed results in the source JSON. Seed timings: 4209s (seed 0), 4653s (seed 42), 3809s (seed 123). Total wall time: 12671s (211.2 min).
