# GPU Status Report -- 2026-04-12 21:22 EDT (2026-04-13 01:22 UTC)

## GPU Status
- **Pod**: `198.13.252.84:44933` (B200, CUDA 13.0, Driver 580.126.09)
- **GPU Util**: 78%, 334W / 1000W
- **VRAM**: 74,376 MiB / 183,359 MiB (40.6% used)
- **Processes**: 3 Python processes running (GRPO training + OLMo extraction)

## Pipeline State (`gpu_pipeline_state.json`)
All 4 completed stages:
| Stage | Elapsed | Finished (UTC) |
|---|---|---|
| reextract_activations | 78.4 min | 17:29 |
| attention_entropy | 3.4 min | 17:32 |
| bootstrap_cis | 70.9 min | 18:43 |
| new_items | 6.8 min | 19:19 |

**No failures.**

## OLMo Think Extraction Progress
- **Status**: RUNNING -- 230/470 samples complete (48.9%)
- **Elapsed**: 6721s (~112 min)
- **Estimated remaining**: ~7013s (~117 min)
- **Estimated completion**: ~03:15 UTC (roughly 23:15 EDT)
- **olmo3_think.h5**: Does NOT exist yet (extraction must finish first)
- **OLMo Think probes**: Cannot run until extraction completes

## SAE (Goodfire)
- **Status**: SAE results directory is EMPTY on pod
- **No SAE results have been produced**
- Unclear if SAE pipeline has started or is queued after OLMo extraction

## OLMo Instruct Probe Results (AVAILABLE)
- **Peak AUC**: 0.9983 +/- 0.0011 at **Layer 21** (position P0)
- **n_samples**: 160 vulnerable items
- **AUC progression**: Rises from 0.861 (L0) to peak at L21, then gradually declines to 0.986 (L31)
- **P2 position**: All layers = 0.500 (chance), as expected for non-diagnostic position

## Key Comparison: OLMo Think vs Instruct
- **OLMo Instruct**: Peak AUC = 0.9983 (L21) -- AVAILABLE
- **OLMo Think**: NOT YET AVAILABLE (extraction at 48.9%)
- **Verdict**: Cannot make the key comparison yet. ETA ~2 hours.

## Downloaded Results (Local Sync Status)
All 14 probe JSON files synced to `results_pod/probes/`. No new SAE results to download.

## Estimated Time to Completion

| Task | Status | ETA |
|---|---|---|
| OLMo Think extraction | 230/470 (49%) | ~2h (03:15 UTC) |
| OLMo Think probes | Blocked on extraction | ~2.5h (after extraction + probe run ~20-30 min) |
| SAE Goodfire | Not started / unclear | Unknown |

**Best case for OLMo Think vs Instruct comparison: ~2.5 hours from now (03:45 UTC / 23:45 EDT).**
