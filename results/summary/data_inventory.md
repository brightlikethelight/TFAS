# S1/S2 Results Data Inventory

Generated: 2026-04-13

## Summary

Total JSON result files: 38
Models covered: Llama-3.1-8B-Instruct, DeepSeek-R1-Distill-Llama-8B, Qwen-3-8B (think/no-think), OLMo-3-7B-Instruct, OLMo-3-7B-Think
Workstreams with data: behavioral, probes, cross-prediction, transfer matrix, geometry, attention, bootstrap CIs, lure susceptibility

---

## Aggregated Outputs

| File | Contents | Generated |
|------|----------|-----------|
| `summary/all_results.json` | Unified JSON with behavioral, probe, cross-prediction, transfer matrix, lure susceptibility, and geometry results for all models | 2026-04-13 |
| `summary/paper_tables.tex` | 5 LaTeX tables (behavioral lure rates, probe summary, cross-prediction, transfer matrix, lure susceptibility) | 2026-04-13 |
| `summary/behavioral_complete.json` | Earlier behavioral-only aggregation | 2026-04-12 |
| `summary/behavioral_tables.tex` | Earlier behavioral-only LaTeX tables | 2026-04-12 |

---

## Behavioral Results

Raw item-level trial data from benchmark evaluation.

| File | Model | Items | Format | Generated |
|------|-------|-------|--------|-----------|
| `behavioral/llama31_8b_ALL.json` | Llama-3.1-8B-Instruct | ~470 (all categories) | item-level | 2026-04-12 |
| `behavioral/llama31_8b_60items.json` | Llama-3.1-8B-Instruct | 60 items | item-level | 2026-04-12 |
| `behavioral/r1_distill_llama_ALL.json` | R1-Distill-Llama-8B | ~470 (all categories) | item-level | 2026-04-12 |
| `behavioral/r1_distill_llama_60items.json` | R1-Distill-Llama-8B | 60 items | item-level | 2026-04-12 |
| `behavioral/r1_distill_llama_20items.json` | R1-Distill-Llama-8B | 20 items (pilot) | item-level | 2026-04-12 |
| `behavioral/r1_distill_qwen_60items.json` | R1-Distill-Qwen-7B | 60 items | item-level | 2026-04-12 |
| `behavioral/qwen3_8b_no_think.json` | Qwen-3-8B (no think) | ~470 items | item-level | 2026-04-12 |
| `behavioral/qwen3_8b_think.json` | Qwen-3-8B (think) | ~470 items | item-level | 2026-04-12 |
| `behavioral/olmo3_instruct_ALL.json` | OLMo-3-7B-Instruct | 470 items (all categories) | item-level | 2026-04-12 |
| `behavioral/olmo3_think_ALL.json` | OLMo-3-7B-Think | 470 items (all categories) | item-level | 2026-04-12 |
| `behavioral/new_items_llama_31_8b_instruct.json` | Llama-3.1-8B-Instruct | new structural isomorphs | item-level | 2026-04-12 |
| `behavioral/new_items_r1_distill_llama_8b.json` | R1-Distill-Llama-8B | new structural isomorphs | item-level | 2026-04-12 |
| `behavioral/llama31_lure_susceptibility.json` | Llama-3.1-8B-Instruct | per-item lure susceptibility scores | continuous scores | 2026-04-12 |
| `behavioral/r1_distill_lure_susceptibility.json` | R1-Distill-Llama-8B | per-item lure susceptibility scores | continuous scores | 2026-04-12 |

---

## Probe Results

Linear probe AUCs by layer for conflict vs. control classification.

| File | Model(s) | Contents | Generated |
|------|----------|----------|-----------|
| `probes/olmo3_instruct_vulnerable.json` | OLMo-3-7B-Instruct | Per-layer AUCs (L0-L31, P0 and P2), 32 layers, peak AUC 0.998 at L21 | 2026-04-12 |
| `probes/llama31_ALL_layer_aucs.json` | Llama-3.1-8B-Instruct | Layer AUCs on all categories | 2026-04-12 |
| `probes/llama31_8b_layer_aucs.json` | Llama-3.1-8B-Instruct | Detailed layer AUCs (multiple category subsets) | 2026-04-12 |
| `probes/llama31_base_rate+conjunction+syllogism_layer_aucs.json` | Llama-3.1-8B-Instruct | AUCs on vulnerable categories only | 2026-04-12 |
| `probes/llama31_crt+arithmetic+framing+anchoring_layer_aucs.json` | Llama-3.1-8B-Instruct | AUCs on immune categories only | 2026-04-12 |
| `probes/r1_distill_ALL_layer_aucs.json` | R1-Distill-Llama-8B | Layer AUCs on all categories | 2026-04-12 |
| `probes/r1_distill_base_rate+conjunction+syllogism_layer_aucs.json` | R1-Distill-Llama-8B | AUCs on vulnerable categories only | 2026-04-12 |
| `probes/r1_distill_crt+arithmetic+framing+anchoring_layer_aucs.json` | R1-Distill-Llama-8B | AUCs on immune categories only | 2026-04-12 |
| `probes/qwen3_nothink_ALL_layer_aucs.json` | Qwen-3-8B (no think) | Layer AUCs on all categories | 2026-04-12 |
| `probes/qwen3_nothink_base_rate+conjunction+syllogism_layer_aucs.json` | Qwen-3-8B (no think) | AUCs on vulnerable categories only | 2026-04-12 |
| `probes/llama_vs_r1_layer_aucs.json` | Llama vs R1-Distill | Comparative layer AUCs | 2026-04-12 |
| `probes/probe_analysis_report.json` | Llama + R1-Distill | Combined probe analysis with basic stats | 2026-04-12 |

---

## Cross-Prediction Results

Probes trained on vulnerable categories, tested on immune categories.

| File | Model | Contents | Generated |
|------|-------|----------|-----------|
| `probes/llama_cross_prediction.json` | Llama-3.1-8B-Instruct | Within-vulnerable and transfer-to-immune AUCs at 10 layers. Mean transfer AUC = 0.444 (SPECIFIC) | 2026-04-12 |
| `probes/r1_distill_cross_prediction.json` | R1-Distill-Llama-8B | Cross-prediction AUCs. Mean transfer AUC = 0.654 (MIXED) | 2026-04-12 |

---

## Transfer Matrix

Category-to-category probe transfer at Llama L14.

| File | Contents | Generated |
|------|----------|-----------|
| `probes/transfer_matrix_l14_llama.json` | 7x7 category transfer matrix (all categories), Llama-3.1-8B Layer 14. Base_rate <-> conjunction near-perfect transfer (0.993/0.998). | 2026-04-12 |

---

## Attention Results

Raw per-head attention entropy and statistics.

| File | Model | Contents | Generated |
|------|-------|----------|-----------|
| `attention/llama31_attention.json` | Llama-3.1-8B-Instruct | Full attention extraction (~85 MB) | 2026-04-12 |
| `attention/r1_distill_attention.json` | R1-Distill-Llama-8B | Full attention extraction (~85 MB) | 2026-04-12 |
| `attention/attention_summary.json` | Both | Summarized attention entropy statistics | 2026-04-12 |

---

## Bootstrap Confidence Intervals

1000-resample bootstrap CIs for probe AUCs.

| File | Model | Contents | Generated |
|------|-------|----------|-----------|
| `bootstrap_cis/unsloth_Meta-Llama-3.1-8B-Instruct_bootstrap_cis.json` | Llama-3.1-8B-Instruct | Bootstrap CI per layer. Mean AUC 0.962 [0.950, 0.972] | 2026-04-12 |
| `bootstrap_cis/deepseek-ai_DeepSeek-R1-Distill-Llama-8B_bootstrap_cis.json` | R1-Distill-Llama-8B | Bootstrap CI per layer. Mean AUC 0.912 [0.901, 0.921] | 2026-04-12 |

---

## Geometry Results

Representational geometry analysis (silhouette scores, CKA).

| File | Contents | Generated |
|------|----------|-----------|
| `geometry/llama_vs_r1_geometry.json` | Silhouette scores (Llama: 0.079, R1: 0.059) and CKA similarity matrix (range 0.379-0.985) | 2026-04-12 |

---

## Other

| File | Contents | Generated |
|------|----------|-----------|
| `final_statistics.json` | Consolidated statistics with hypothesis test results (H1 linear decodability, cross-prediction specificity, lure susceptibility). Used as enrichment source by aggregate_results.py | 2026-04-12 |

---

## Empty Workstreams

The following workstream directories exist but contain no result files:
- `causal/` -- causal intervention experiments (not yet run)
- `metacog/` -- metacognitive monitoring (stretch goal, not yet run)
- `sae/` -- SAE feature analysis (not yet run)
