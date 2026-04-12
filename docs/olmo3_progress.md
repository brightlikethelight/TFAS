# OLMo-3-7B Experiment Progress

**Last updated**: 2026-04-12 20:20 UTC
**Pod**: B200 @ 198.13.252.84:44933

## Pipeline Status

| Stage | OLMo Instruct | OLMo Think |
|-------|--------------|------------|
| Behavioral validation | COMPLETE (381s) | IN PROGRESS (80/470, ~17%) |
| Activation extraction | COMPLETE (100 MB h5) | Pending |
| Probe training | COMPLETE | Pending |

**GPU**: B200, 80% utilization, 71.7 / 183.4 GB VRAM used
**ETA for full pipeline**: ~3.5 hours from now (~23:50 UTC)

Think model behavioral is slow: 2048 max_new_tokens generates ~29s/item vs Instruct's ~0.8s/item. This is expected -- reasoning tokens are the whole point.

## OLMo-3-7B-Instruct Behavioral Results

Overall lure rate: **35/235 = 14.9%** (235 conflict items)
Control accuracy: 197/235 = 83.8%

### Per-Category Lure Rates

| Category | Lured/Total | Rate | Status |
|----------|------------|------|--------|
| base_rate | 16/35 | 45.7% | VULNERABLE |
| conjunction | 10/20 | 50.0% | VULNERABLE |
| loss_aversion | 5/15 | 33.3% | VULNERABLE |
| availability | 2/15 | 13.3% | immune |
| anchoring | 1/20 | 5.0% | immune |
| crt | 1/30 | 3.3% | immune |
| arithmetic | 0/25 | 0.0% | immune |
| certainty_effect | 0/15 | 0.0% | immune |
| framing | 0/20 | 0.0% | immune |
| sunk_cost | 0/15 | 0.0% | immune |
| syllogism | 0/25 | 0.0% | immune |

## Cross-Architecture Comparison

| Category | Llama Instruct | R1-Distill | OLMo Instruct |
|----------|---------------|------------|---------------|
| base_rate | 84.0% (21/25) | 4.0% (1/25) | 45.7% (16/35) |
| conjunction | 55.0% (11/20) | 0.0% (0/20) | 50.0% (10/20) |
| syllogism | 52.0% (13/25) | 0.0% (0/25) | 0.0% (0/25) |
| loss_aversion | -- | -- | 33.3% (5/15) |
| anchoring | 0.0% (0/20) | 10.0% (2/20) | 5.0% (1/20) |
| crt | 0.0% (0/30) | 3.3% (1/30) | 3.3% (1/30) |
| framing | 0.0% (0/20) | 0.0% (0/20) | 0.0% (0/20) |
| arithmetic | 0.0% (0/25) | 0.0% (0/25) | 0.0% (0/25) |

### Key Observations

1. **base_rate and conjunction are universally vulnerable in Instruct models**: Both Llama Instruct and OLMo Instruct show high lure rates for these categories. This is cross-architecture replication of the core vulnerability pattern.

2. **syllogism diverges**: Llama Instruct was 52% vulnerable to syllogism, but OLMo Instruct is 0%. This may reflect OLMo's training data or instruction-tuning differences.

3. **loss_aversion is new**: OLMo Instruct shows 33.3% vulnerability to loss_aversion (new category not in the original Llama benchmark). This is a novel finding.

4. **R1-Distill near-zero across the board**: The reasoning model (R1-Distill) shows dramatically lower lure rates, consistent with the hypothesis that extended reasoning suppresses cognitive biases.

5. **Critical test pending**: If OLMo Think shows the same pattern (near-zero lure rates + lower probe AUC), this transforms the finding from "one architecture" to "general phenomenon."

## OLMo-3-7B-Instruct Probe Results

160 vulnerable items used for probe training (conflict items where model answered correctly).

### P0 (last prompt token) -- strong signal

| Layer | AUC (mean +/- std) |
|-------|-------------------|
| 0 | 0.861 +/- 0.005 |
| 4 | 0.907 +/- 0.004 |
| 8 | 0.956 +/- 0.004 |
| 12 | 0.991 +/- 0.001 |
| 16 | 0.998 +/- 0.002 |
| 20 | 0.998 +/- 0.001 |
| 24 | 0.998 +/- 0.001 |
| 28 | 0.994 +/- 0.006 |
| 31 | 0.986 +/- 0.006 |

Peak AUC: **0.998** at layers 16-24 (mid-to-late layers).
This matches the Llama pattern: conflict detection is nearly perfect in the residual stream.

### P2 (answer token) -- no signal

All layers: AUC = 0.500 (chance). Expected for Instruct model with no thinking tokens.

## What Remains

1. OLMo Think behavioral validation (~3 hr)
2. OLMo Think activation extraction (~15 min)
3. OLMo Think probe training (~5 min)
4. Cross-model comparison (Think vs Instruct probes)
