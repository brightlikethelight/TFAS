# OLMo-32B Results: Instruct vs Think

**Status**: Instruct complete. Think extraction in progress (ETA ~05:15 UTC Apr 15).
**Last updated**: 2026-04-14 21:43 UTC

## Pipeline Status

| Stage | Instruct | Think |
|-------|----------|-------|
| Behavioral validation | Done | Done |
| Activation extraction | Done (236 MB) | **IN PROGRESS** (20/470, ~7.25h remaining) |
| Linear probes | Done | Pending (extraction must finish first) |
| Instruct vs Think comparison | Pending | Pending |

## Behavioral Results

### OLMo-32B Instruct (no thinking)
- **Overall lure rate: 19.6%** (46/235 conflict items)
- Heavy vulnerability in base_rate (74%), conjunction (50%), framing (30%)
- Moderate: anchoring (10%), crt (7%)
- Immune: arithmetic, availability, certainty_effect, loss_aversion, sunk_cost, syllogism

### OLMo-32B Think (reasoning)
- **Overall lure rate: 0.4%** (1/235 conflict items)
- Only lure: sunk_cost 1/15 (7%)
- **All vulnerable categories zeroed out**: base_rate 0%, conjunction 0%, syllogism 0%
- Thinking nearly eliminates cognitive bias susceptibility at 32B scale

### Behavioral comparison: Think reduces lure rate from 19.6% to 0.4% (49x reduction)

## Probe Results (Instruct only so far)

### OLMo-32B Instruct P0 (pre-question position)
- **Peak AUC: 0.9999 at L20** (+/- 0.0001)
- L22 also at 0.9999
- Plateau at ~0.997 from L24 through L63
- Consistent with other models: near-perfect discrimination at early-mid layers

### OLMo-32B Instruct P2 (answer position)
- **All layers: AUC = 0.5000** (chance)
- P2 at chance confirms probe measures conflict detection, not answer memorization

## Cross-scale Comparison

| Model | Params | Instruct Peak P0 | Think Peak P0 | Instruct Lure | Think Lure |
|-------|--------|-------------------|---------------|---------------|------------|
| OLMo-3 7B | 7B | 0.9983 (L21) | 0.9928 (L28) | ~15%* | ~2%* |
| OLMo-32B | 32B | **0.9999 (L20)** | *pending* | **19.6%** | **0.4%** |

*7B lure rates from log data, approximate.

### Key observations so far
1. **32B Instruct peak AUC (0.9999) slightly exceeds 7B (0.9983)** -- larger model has marginally better internal conflict representation
2. **32B lure rate (19.6%) is higher than 7B (~15%)** -- bigger model, bigger bias susceptibility in Instruct mode (more "System 1" to override)
3. **32B Think lure rate (0.4%) is lower than 7B (~2%)** -- reasoning at scale is more effective
4. P0/P2 dissociation holds perfectly at 32B scale

## Pending: Critical Question

**Does Think blurring replicate at 32B?**

At 7B: Think peak AUC (0.9928) was lower than Instruct (0.9983), delta = 0.0055.
This "blurring" effect -- reasoning partially dissolving the internal conflict signal -- is the core finding.

If 32B Think peak AUC < 0.9999 (Instruct), the blurring effect replicates at scale.
If 32B Think peak AUC >= 0.9999, the effect may be scale-dependent.

**ETA for Think probes: ~8 hours (extraction ~7.25h + probing ~10min)**

## Technical Notes
- Pod: B200, 183 GB VRAM
- 32B Think uses 66.4 GB VRAM for extraction
- Think model generates 2048 tokens per item (vs 256 for Instruct) -- extraction is ~8x slower
- 64 layers, 5120 hidden dim, 40 Q-heads, 8 KV-heads (GQA)
- Background GRPO training (v11b_live_bow) running concurrently on same GPU
