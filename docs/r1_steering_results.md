# R1-Distill Probe Steering Results

**Date**: 2026-04-14
**Runtime**: ~10.9 hours (39,167s steering sweep + overhead)
**Pod**: 198.13.252.84:44933

## Experiment Summary

Probe-direction steering at layer 14 of DeepSeek-R1-Distill-Llama-8B, compared against Llama-3.1-8B-Instruct baseline. 9 alpha values [-5, -3, -1, -0.5, 0, 0.5, 1, 3, 5] applied to the S1/S2 probe direction, with 5 random-direction controls per alpha.

- **R1 probe AUC**: 0.928 (80 conflict + 80 control items)
- **Llama probe AUC**: 0.960

## Dose-Response Comparison

### R1-Distill-Llama-8B (layer 14, max_new_tokens=2048)

| Alpha | Lure Rate | Correct Rate | Other Rate | n_lure |
|-------|-----------|-------------|------------|--------|
| -5.0  | 0.062     | 0.638       | 0.300      | 5      |
| -3.0  | 0.100     | 0.588       | 0.312      | 8      |
| -1.0  | 0.075     | 0.625       | 0.300      | 6      |
| -0.5  | 0.062     | 0.638       | 0.300      | 5      |
|  0.0  | 0.062     | 0.638       | 0.300      | 5      |
| +0.5  | 0.050     | 0.638       | 0.312      | 4      |
| +1.0  | 0.050     | 0.638       | 0.312      | 4      |
| +3.0  | 0.075     | 0.612       | 0.312      | 6      |
| +5.0  | 0.138     | 0.562       | 0.300      | 11     |

**Baseline lure rate**: 6.25% (alpha=0)
**Max lure rate**: 13.75% (alpha=+5)
**Min lure rate**: 5.0% (alpha=+0.5/+1.0)
**Delta (alpha=-5 to alpha=+5)**: -0.075 (7.5 pp)

### Llama-3.1-8B-Instruct (layer 14, max_new_tokens=128)

| Alpha | Lure Rate | Correct Rate | Other Rate | n_lure |
|-------|-----------|-------------|------------|--------|
| -5.0  | 0.688     | 0.312       | 0.000      | 55     |
| -3.0  | 0.612     | 0.388       | 0.000      | 49     |
| -1.0  | 0.562     | 0.438       | 0.000      | 45     |
| -0.5  | 0.538     | 0.462       | 0.000      | 43     |
|  0.0  | 0.525     | 0.475       | 0.000      | 42     |
| +0.5  | 0.488     | 0.512       | 0.000      | 39     |
| +1.0  | 0.488     | 0.512       | 0.000      | 39     |
| +3.0  | 0.412     | 0.588       | 0.000      | 33     |
| +5.0  | 0.312     | 0.688       | 0.000      | 25     |

**Baseline lure rate**: 52.5% (alpha=0)
**Max lure rate**: 68.75% (alpha=-5)
**Min lure rate**: 31.25% (alpha=+5)
**Delta (alpha=-5 to alpha=+5)**: 0.375 (37.5 pp)

## Random Direction Controls

### R1-Distill
- Random directions at alpha=+5: mean lure rate 0.103 +/- 0.019
- Probe direction at alpha=+5: lure rate 0.138
- **Probe vs random gap at alpha=+5**: 3.5 pp (marginal)
- Random directions at alpha=-5: mean lure rate 0.115 +/- 0.053
- Probe direction at alpha=-5: lure rate 0.062
- **Probe vs random gap at alpha=-5**: 5.3 pp (wrong direction -- random dirs increase lure more than probe)

### Llama
- Random directions at alpha=+5: mean lure rate 0.578 +/- 0.079
- Probe direction at alpha=+5: lure rate 0.312
- **Probe vs random gap at alpha=+5**: 26.5 pp (strong specificity)
- Random directions at alpha=-5: mean lure rate 0.555 +/- 0.082
- Probe direction at alpha=-5: lure rate 0.688
- **Probe vs random gap at alpha=-5**: 13.3 pp (clear specificity)

## Key Findings

### 1. R1 shows WEAK or NO causal effect from probe steering

The R1 lure rate swings only 7.5 pp across the full alpha range (-5 to +5), compared to Llama's 37.5 pp swing. The script itself classified this as "Weak or no causal effect."

### 2. R1's probe direction is barely distinguishable from random directions

At alpha=+5, the probe direction produces lure rate 0.138 while random directions average 0.103. This 3.5 pp gap is within 2 SD of the random distribution. At alpha=-5, random directions actually produce HIGHER lure rates (0.115) than the probe direction (0.062), meaning the probe direction is not behaving as expected.

### 3. R1 is dramatically more robust to activation steering than Llama

- R1 baseline lure rate: 6.25% (already very low -- the model rarely falls for cognitive biases)
- Llama baseline lure rate: 52.5% (highly susceptible)
- Even at maximal S1-promoting steering (alpha=-5), R1 only reaches 6.25% lure rate
- Llama at the same steering reaches 68.75%

### 4. R1's "other" rate (~30%) suggests CoT generation often fails to commit to an answer

About 30% of R1's outputs across all alpha values are classified as "other" (neither lure nor correct). This likely reflects the 2048-token truncation cutting off the reasoning chain before a final answer is produced. Llama at 128 tokens has 0% "other" -- it always commits.

### 5. The deliberation-intensity dimension is NOT causally operative in R1 at layer 14

Despite the probe achieving 0.928 AUC (the representation IS there), steering along the probe direction does not meaningfully change behavior. This is consistent with two hypotheses:
1. **Redundant encoding**: The deliberation signal is encoded across many layers/dimensions, and perturbing one layer is insufficient to override the others.
2. **Robustness from reasoning training**: The RLHF/reasoning training may have made R1's decision-making robust to single-direction perturbations, even though the S1/S2 representation exists.

## Interpretation for the Paper

This is an important negative result. While both Llama and R1-Distill encode S1/S2 information linearly (probe AUCs > 0.92), the causal role of this representation differs dramatically:
- In **Llama** (non-reasoning model): the probe direction is causally potent, producing a 37.5 pp swing in lure rate, well above random-direction controls.
- In **R1-Distill** (reasoning model): the probe direction is causally inert, producing only a 7.5 pp swing indistinguishable from random-direction perturbations.

This suggests that reasoning training may create **representational redundancy** or **robustness** that decouples any single linear direction from behavioral control, even though the representation is still linearly decodable. The "deliberation intensity" signal exists but is not causally necessary at this layer.

## Follow-up Jobs Queued

After R1 completion, the following were launched on the same pod:
1. **Qwen3-8B within-CoT extraction** (PID 534259) -- extracting activations at T25/T50/T75 within thinking traces
2. **OLMo-3-32B full pipeline** (PID 533419) -- already running from prior launch

## Files

- R1 results: `results/causal/probe_steering_r1_l14.json`
- Llama results: `results/causal/probe_steering_llama_l14.json`
- R1 figure: `figures/fig_probe_steering_deepseek_r1_distill_llama_8b_l14.pdf` (on pod)
