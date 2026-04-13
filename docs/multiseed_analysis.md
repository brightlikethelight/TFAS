# Multi-Seed Robustness Analysis

## Setup

- **Model**: Llama-3.1-8B-Instruct (unsloth/Meta-Llama-3.1-8B-Instruct)
- **Seeds**: 0, 42, 123
- **Generation**: `do_sample=True`, temperature=0.7, top_p=0.95, max_new_tokens=256
- **Benchmark**: 470 items (235 conflict pairs), 11 categories
- **Source**: `results/robustness/unsloth_Meta-Llama-3.1-8B-Instruct_multiseed.json`

## Overall Result

| Metric | Value |
|--------|-------|
| Mean lure rate | 27.5% |
| Std (pp) | 1.3 |
| Per-seed | 26.4% / 27.2% / 28.9% |
| Stable? | Yes |

The overall lure rate is stable across seeds and close to the greedy baseline (27.3%). Sampling does not meaningfully change aggregate vulnerability.

## Per-Category Breakdown

| Category | Greedy | Sampled (mean +/- std) | Stable? | Direction |
|----------|--------|------------------------|---------|-----------|
| base_rate | 84% | 81.0% +/- 3.3pp | Yes | Slight decrease |
| framing | 0% | 53.3% +/- 2.9pp | Yes | **0% -> 53%** |
| syllogism | 52% | 41.3% +/- 9.2pp | Yes | Moderate decrease, high variance |
| crt | 0% | 35.6% +/- 5.1pp | Yes | **0% -> 36%** |
| availability | 0%* | 8.9% +/- 10.2pp | **No** | Unstable |
| certainty_effect | 0%* | 6.7% +/- 6.7pp | Yes | Small emergence |
| conjunction | 55% | 3.3% +/- 2.9pp | Yes | **55% -> 3%** |
| sunk_cost | 0%* | 4.4% +/- 3.8pp | Yes | Small emergence |
| arithmetic | 0% | 4.0% +/- 0.0pp | Yes | Stable near-zero |
| anchoring | 0% | 0.0% +/- 0.0pp | Yes | No change |
| loss_aversion | 0% | 0.0% +/- 0.0pp | Yes | No change |

*Greedy values for expanded categories from supplementary results.

## Key Findings

### 1. Formerly "immune" categories become vulnerable under sampling

- **Framing**: 0% (greedy) -> 53% (sampled). This is the single largest shift. Under greedy decoding, the model always picks the normatively correct answer; under sampling, it falls for the gain/loss frame more than half the time. The low std (2.9pp) means this is a robust, reproducible effect.
- **CRT**: 0% (greedy) -> 36% (sampled). CRT items that were perfectly solved under greedy decoding are failed ~1/3 of the time with sampling.

### 2. Conjunction fallacy collapses

- **Conjunction**: 55% (greedy) -> 3% (sampled). The greedy mode consistently picks the conjunction (the "Linda is a bank teller AND feminist" answer); sampling almost never does. This is striking: the greedy maximum-probability answer is the lure, but the probability mass is overwhelmingly on the correct answer.

### 3. The vulnerability profile is inverted

Under greedy decoding, the vulnerable categories are {base_rate, conjunction, syllogism} and the immune categories include {CRT, framing}. Under sampling, conjunction becomes immune and {CRT, framing} become vulnerable. The profile is generation-strategy-dependent.

### 4. Overall stability masks category instability

The overall lure rate is nearly identical (27.3% greedy vs. 27.5% sampled), but this masks a complete reshuffling of which categories are vulnerable. Aggregate metrics are misleading.

### 5. Availability is the only unstable category

Availability shows std = 10.2pp across seeds, with per-seed rates of 0%, 7%, 20%. This category has only 15 conflict items, so sampling variance is expected. All other categories are stable (std < 10pp).

## Implications for the Paper

### What this means for behavioral results

Our primary behavioral results (Table 1) use greedy decoding. The multi-seed analysis shows these reflect the model's **maximum-probability behavior**, not its distributional behavior. The "immune" designation for CRT and framing is specific to greedy decoding.

### What this means for probe results

The probe results are **unaffected** by generation strategy. Probes measure the representation at P0 (the last prompt token before generation), which is computed in a single forward pass with no sampling involved. The probe AUC values, cross-prediction results, and transfer matrices are properties of the model's representation, not its generation behavior.

### Recommended paper language

The behavioral results should be qualified as "under greedy decoding" where the distinction matters. The probe results need no qualification.

## Raw Data Reference

Full per-item, per-seed results are in the source JSON file. Seed timings: 464s (seed 0), 494s (seed 42), 415s (seed 123).
