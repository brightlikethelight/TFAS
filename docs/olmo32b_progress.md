# OLMo-3.1-32B Experiment Progress

**Last updated**: 2026-04-14 13:10 UTC
**Pod**: B200 @ 198.13.252.84:44933
**Log**: `/workspace/olmo32b_log.txt`

## Pipeline Status

| Stage | OLMo-32B-Instruct | OLMo-32B-Think |
|-------|-------------------|----------------|
| Behavioral validation | COMPLETE (17 min, 470 items) | IN PROGRESS (1/470, ~0.2%) |
| Activation extraction | COMPLETE (18 min, 247 MB h5) | Pending |
| Probe training | COMPLETE (7 min) | Pending |

**GPU**: B200, 89% utilization, 136.5 / 191.5 GB VRAM used (OLMo-32B-Think loaded)
**Think model behavioral started**: 12:51 UTC
**Think model speed**: ~122s/item (2048 max_new_tokens)
**ETA for Think behavioral**: ~16 hours from start (~04:50 UTC Apr 15)
**ETA for full Think pipeline**: ~16.7 hours (~05:30 UTC Apr 15)

## OLMo-3.1-32B-Instruct Behavioral Results

Overall lure rate: **46/470 = 9.8%**

### Per-Category Lure Rates

| Category | Lured/Total | Rate | Status |
|----------|------------|------|--------|
| base_rate | 26/70 | 37.1% | VULNERABLE |
| conjunction | 10/40 | 25.0% | VULNERABLE |
| framing | 6/40 | 15.0% | moderate |
| anchoring | 2/40 | 5.0% | low |
| crt | 2/60 | 3.3% | low |
| arithmetic | 0/50 | 0.0% | immune |
| availability | 0/30 | 0.0% | immune |
| certainty_effect | 0/30 | 0.0% | immune |
| loss_aversion | 0/30 | 0.0% | immune |
| sunk_cost | 0/30 | 0.0% | immune |
| syllogism | 0/50 | 0.0% | immune |

## OLMo-3.1-32B-Instruct Probe Results (vulnerable categories)

160 vulnerable items (conflict items from base_rate + conjunction + syllogism).
128 layer-position combinations probed (64 layers x P0, P2).

### P0 (last prompt token) -- strong signal

| Layer | AUC (mean +/- std) |
|-------|-------------------|
| L00 | 0.862 +/- 0.010 |
| L10 | 0.953 +/- 0.004 |
| L20 | 1.000 +/- 0.000 |
| L21 | 1.000 +/- 0.000 |
| L22 | 1.000 +/- 0.000 |
| L24 | 0.997 +/- 0.001 |
| L32 | 0.993 +/- 0.002 |
| L50 | 0.997 +/- 0.001 |
| L55 | 0.997 +/- 0.001 |
| L63 | 0.996 +/- 0.002 |

Peak AUC: **1.000** at layers 20-22 (early-mid layers). Perfect conflict detection.
This is deeper than the 7B model (peak at L16-24), consistent with deeper architecture.

### P2 (answer token) -- no signal

All layers: AUC = 0.500 (chance). Expected for Instruct model with no thinking tokens.

## Scale Comparison: 7B vs 32B

| Metric | OLMo-7B-Inst | OLMo-7B-Think | OLMo-32B-Inst | OLMo-32B-Think |
|--------|-------------|---------------|---------------|----------------|
| Overall lure rate | 7.4% | 0.4% | 9.8% | TBD |
| base_rate | 22.9% | 0.0% | 37.1% | TBD |
| conjunction | 25.0% | 0.0% | 25.0% | TBD |
| framing | 0.0% | 0.0% | 15.0% | TBD |
| anchoring | 2.5% | 0.0% | 5.0% | TBD |
| syllogism | 0.0% | 2.0% | 0.0% | TBD |
| Peak probe AUC | 0.998 | TBD | 1.000 | TBD |

## KEY QUESTION: Does scale replicate?

**Correction on baseline**: The OLMo-32B-Instruct overall lure rate is **9.8%**, not 19.6%. The 19.6% figure may be from a different model or subset.

**Preliminary observations**:

1. **32B-Instruct is MORE lured on base_rate than 7B-Instruct** (37.1% vs 22.9%). Scale does not help with base rate neglect -- it may even hurt. This is consistent with the finding that larger models can be *more* susceptible to certain biases because they are better at pattern-matching surface statistics.

2. **conjunction stays flat** (25.0% for both 7B and 32B Instruct). This vulnerability appears scale-invariant.

3. **framing vulnerability emerges at 32B** (15.0% vs 0.0% at 7B). The larger model's greater language understanding may make it more susceptible to framing manipulations.

4. **The 7B Think model was 0.4% overall** -- near-zero. If the 32B Think model also shows near-zero lure rates, the reasoning-suppresses-bias finding replicates across both scale AND architecture.

5. **Probe results match 7B pattern**: perfect conflict detection in residual stream (AUC 1.000 at best layer) but no signal at answer position for Instruct. The 32B model's peak is slightly earlier in relative terms (L20-22 / 64 layers = 31-34% depth vs L16-24 / 32 layers = 50-75% depth in 7B).

## What Remains

1. Think behavioral validation (~16 hours, ETA ~05:00 UTC Apr 15)
2. Think activation extraction (~30 min for 32B)
3. Think probe training (~10 min)
4. Cross-model comparison (Think vs Instruct probes)
5. If Think lure rate is near-zero: **scale result replicates** -- write up for paper
