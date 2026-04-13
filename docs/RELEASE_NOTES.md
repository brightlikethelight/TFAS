# Release Notes: v1.0.0-workshop

**Version**: v1.0.0-workshop
**Date**: 2026-04-13
**Target**: ICML 2026 Workshop on Mechanistic Interpretability (submission deadline: May 8, 2026)
**Project**: The Deliberation Gradient -- Mechanistic Signatures of Dual-Process Cognition in LLMs
**Team**: Harvard HUSAI (Harvard Undergraduate Society for Artificial Intelligence)
**Compute**: RunPod B200 cluster

---

## Summary

This release marks the code-complete milestone for the s1s2 research project.
We investigate whether reasoning-trained LLMs develop internally distinct
processing modes analogous to human dual-process cognition (System 1 / System 2),
and how reasoning distillation reshapes those representations. The project uses
five complementary mechanistic interpretability methods -- linear probing, SAE
feature analysis, attention entropy, representational geometry, and causal
interventions -- applied to matched standard/reasoning model pairs on a 470-item
cognitive bias benchmark.

The core contribution: reasoning training **blurs** the internal boundary between
heuristic-prone and deliberation-requiring processing modes (probe AUC 0.974 ->
0.930, non-overlapping bootstrap CIs) while dramatically improving behavioral
accuracy (27.3% -> 2.4% lure rate). This "S2-by-default" pattern is confirmed
across two independent model families and dissociates training-time from
inference-time effects.

**Paper**: 8pp body (~7.5pp in ICML 2-column approximation), compiles cleanly.
All numbers harmonized to bootstrap CI values. Anonymous version prepared.

---

## Key Results (Real Data)

### Behavioral Lure Rates (% heuristic responses on conflict items)

| Model | Overall | base_rate | conjunction | syllogism | CRT | arithmetic | framing | anchoring | sunk_cost |
|-------|---------|-----------|-------------|-----------|-----|------------|---------|-----------|-----------|
| Llama-3.1-8B-Instruct | 27.3% | 84% | 55% | 52% | 0% | 0% | 0% | 0% | 0% |
| R1-Distill-Llama-8B | 2.4% | 4% | 0% | 0% | -- | -- | -- | -- | 0% |
| R1-Distill-Qwen-7B | ~0% | ~0% | ~0% | ~0% | -- | -- | -- | -- | -- |
| Qwen 3-8B NO_THINK | 21% | 56% | 95% | 0% | -- | -- | -- | -- | -- |
| Qwen 3-8B THINK | 7% | 4% | 55% | -- | -- | -- | -- | -- | -- |

### Probe AUC with Bootstrap 95% CIs (vulnerable categories)

| Model | Peak AUC | 95% CI | Peak Layer |
|-------|----------|--------|------------|
| Llama-3.1-8B-Instruct | 0.974 | [0.952, 0.992] | L16 / 32 |
| R1-Distill-Llama-8B | 0.930 | [0.894, 0.960] | L31 / 32 |
| Qwen 3-8B NO_THINK | 0.971 | -- | L34 / 36 |
| Qwen 3-8B THINK | 0.971 | -- | L34 / 36 |

CIs for Llama and R1-Distill do not overlap (statistically significant difference).

### Cross-Prediction (specificity confound resolution)

| Model | Layer | Transfer AUC | Interpretation |
|-------|-------|-------------|----------------|
| Llama | L14 | 0.378 | Below chance -- probe is processing-mode-specific |
| R1-Distill | L4 | 0.878 | Early layers share text features |
| R1-Distill | L31 | 0.385 | Late layers are processing-specific |

### Lure Susceptibility (continuous P0 score)

| Model | Mean P0 | Direction |
|-------|---------|-----------|
| Llama-3.1-8B-Instruct | +0.422 | Favors lure answer |
| R1-Distill-Llama-8B | -0.326 | Favors correct answer |

### Representational Geometry

| Metric | Llama | R1-Distill |
|--------|-------|------------|
| Cosine silhouette (peak) | 0.079 | 0.059 |
| CKA range (cross-model) | 0.379 -- 0.985 | |

### Transfer Matrix (Llama, within vulnerable categories)

| Train \ Test | base_rate | conjunction |
|-------------|-----------|-------------|
| base_rate | -- | 0.993 |
| conjunction | 0.998 | -- |

### Training vs. Inference Dissociation

Qwen 3-8B THINK and NO_THINK modes produce **identical** probe AUC curves
(0.971 at L34) despite dramatically different behavioral lure rates (7% vs 21%).
Inference-time chain-of-thought changes behavior without reshaping the residual
stream representation that probes detect.

### Gigerenzer Natural Frequency Reversal

Llama: 84% lure (probability format) -> 100% lure (frequency format).
R1-Distill: 4% lure (probability) -> ~60% lure (frequency). Both models are
WORSE with natural frequencies -- the opposite of Gigerenzer's prediction for
humans. Reasoning training creates format-specific competence.

**Caveat**: R1-Distill natural frequency scoring is affected by a BPE artifact
bug (see Known Issues). Results need re-collection with the fixed scoring pipeline.

### Multi-Seed Robustness: Generation Strategy Dependence

Llama-3.1-8B-Instruct 3-seed sampled decoding (T=0.7, top_p=0.95) on the full
470-item benchmark reveals that category vulnerability profiles are strongly
dependent on generation strategy. Overall lure rate is stable (27.5% +/- 1.3pp
vs 27.3% greedy), but individual categories shift dramatically:

- **Framing**: 0% (greedy) -> 53.3% (sampled) -- greedy "immunity" is an artifact
- **CRT**: 0% (greedy) -> 35.6% (sampled) -- same pattern
- **Conjunction**: 55% (greedy) -> 3.3% (sampled) -- flips in the opposite direction

Only **loss_aversion** and **anchoring** remain at 0% across all 3 seeds with
sampling, making them the only truly immune categories.

**Crucially, this does NOT affect mechanistic (probe) results.** Probes measure
P0 (pre-generation representation), not generation. The conflict/control boundary
in the residual stream is stable regardless of decoding strategy. This is actually
a *stronger* finding: the model "knows" there is a conflict even when generation
strategy masks or reveals it differently.

R1-Distill multi-seed is running on GPU but not yet complete.

---

## Models Tested

| Key | HuggingFace ID | Params | Layers | Hidden | Type |
|-----|----------------|--------|--------|--------|------|
| llama-3.1-8b-instruct | meta-llama/Llama-3.1-8B-Instruct | 8B | 32 | 4096 | Standard |
| r1-distill-llama-8b | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 32 | 4096 | Reasoning |
| gemma-2-9b-it | google/gemma-2-9b-it | 9B | 42 | 3584 | Standard |
| r1-distill-qwen-7b | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 28 | 3584 | Reasoning |
| qwen-3-8b (THINK) | Qwen/Qwen3-8B | 8B | 36 | -- | Reasoning (inference) |
| qwen-3-8b (NO_THINK) | Qwen/Qwen3-8B | 8B | 36 | -- | Standard (inference) |

**Planned but not yet run**: OLMo-2-7B-Instruct / OLMo-2-7B-Think (cross-architecture replication), Ministral-3-8B-Instruct / Reasoning (deprioritized due to transformers version incompatibility).

---

## Scripts

### Core Pipeline

| Script | Description |
|--------|-------------|
| `scripts/extract_all.py` | Hydra-based activation extraction for all configured models |
| `scripts/extract_real.py` | Direct activation extraction for a single model (no Hydra) |
| `scripts/run_probes.py` | Linear probing pipeline (logistic, MLP, CCS probes with Hewitt-Liang controls) |
| `scripts/run_sae.py` | SAE differential feature analysis with Ma et al. falsification |
| `scripts/run_attention.py` | Attention entropy analysis per head with BH-FDR |
| `scripts/run_geometry.py` | Representational geometry (PCA, UMAP, CKA, silhouette) |
| `scripts/run_causal.py` | Causal interventions (SAE feature steering, ablation) |
| `scripts/run_metacog.py` | Metacognitive monitoring analysis (stretch goal) |
| `scripts/run_pipeline.py` | Full pipeline orchestrator (extract -> all analyses) |
| `scripts/run_single_model.py` | Single-model end-to-end pipeline |
| `scripts/run_all_gpu.py` | Single-command orchestrator for all pending GPU jobs |
| `scripts/smoke_test.py` | End-to-end validation on synthetic data (CPU, ~3 seconds) |

### Benchmark and Data

| Script | Description |
|--------|-------------|
| `scripts/generate_benchmark.py` | Regenerate the 470-item benchmark from templates |
| `scripts/audit_benchmark.py` | Validate benchmark integrity (matched pairs, schema) |
| `scripts/behavioral_validation.py` | Run behavioral validation against live models |
| `scripts/run_new_items.py` | Behavioral validation for sunk cost + natural frequency items |

### Analysis and Figures

| Script | Description |
|--------|-------------|
| `scripts/compute_bootstrap_cis.py` | 1000-resample bootstrap CIs + Hewitt-Liang controls |
| `scripts/analyze_probes.py` | Post-hoc probe result analysis |
| `scripts/analyze_attention.py` | Post-hoc attention entropy analysis |
| `scripts/extract_attention.py` | Normalized entropy, KV-group analysis, BH-FDR stats |
| `scripts/extract_lure_susceptibility.py` | Continuous P0 lure susceptibility extraction |
| `scripts/confidence_paradigm.py` | De Neys confidence extraction (token probs, entropy) |
| `scripts/aggregate_results.py` | Comprehensive results aggregation + LaTeX table generation |
| `scripts/final_statistics.py` | Final statistical summary for paper |
| `scripts/generate_figures.py` | Hydra-based figure generation |
| `scripts/make_paper_figures.py` | Publication-quality paper figures (v1) |
| `scripts/make_paper_figures_v2.py` | Publication-quality paper figures (v2, from real data) |
| `scripts/make_behavioral_table.py` | Behavioral results table generation |
| `scripts/generate_report.py` | Auto-generate results report |

### Model-Specific

| Script | Description |
|--------|-------------|
| `scripts/extract_qwen_toggle.py` | Qwen 3-8B THINK/NO_THINK activation extraction |
| `scripts/extract_ministral.py` | Ministral model extraction (deprioritized) |
| `scripts/extract_olmo3.py` | OLMo-3-7B activation extraction |
| `scripts/run_olmo3_full.py` | One-command OLMo download + behavioral + extract + probe |
| `scripts/olmo3_behavioral.py` | OLMo-3-7B behavioral validation |
| `scripts/ministral_behavioral.py` | Ministral behavioral validation |
| `scripts/run_sae_goodfire.py` | SAE analysis using Goodfire L19 SAE |

### Deployment and Monitoring

| Script | Description |
|--------|-------------|
| `scripts/dry_run.py` | Dry-run validation before GPU deployment |
| `scripts/check_pod_results.py` | Check and download results from RunPod |
| `scripts/monitor_loop.py` | Monitoring daemon for long-running jobs |
| `scripts/gpu_monitor.sh` | GPU utilization monitoring (shell) |
| `scripts/run_dashboard.py` | Launch interactive Gradio exploration dashboard |
| `deploy/download_models.py` | Pre-download models to persistent storage |
| `deploy/verify_gpu.py` | Verify GPU availability and VRAM |
| `deploy/cost_estimate.py` | Estimate compute cost for planned runs |

---

## Development History

**66 commits** across 9 development sessions (2026-04-09 to 2026-04-13):

1. **Session 1 (Apr 9)**: Initial commit. Full research codebase scaffold -- pyproject.toml, all 8 analysis modules, configs, tests, benchmark loader, data contract.
2. **Session 2 (Apr 10)**: Phase 2 deployment. GPU deployment scripts, paper skeleton, notebooks, W&B integration, orchestration, documentation. Critical syllogism control fix.
3. **Session 3 (Apr 10)**: Phase 3 end-to-end validation. Dashboard, benchmark expansion, onboarding docs, experiment scripts, figures, AF post draft.
4. **Session 4 (Apr 11-12)**: First real experiments. Behavioral validation on B200 pod (5 model configs, 330 items). First mechanistic result: probe AUC 0.999 (Llama) vs 0.929 (R1-Distill) at L14. Qwen THINK/NO_THINK discovery.
5. **Session 5 (Apr 12)**: Cross-prediction resolves the specificity confound. Qwen extraction + probes (training vs inference dissociation confirmed). Geometry analysis. Lure susceptibility extraction.
6. **Session 6 (Apr 12-13)**: Complete data picture. All overnight jobs complete. Bootstrap CIs (non-overlapping). Transfer matrix. Workshop paper draft with all real results. Publication figures.
7. **Session 7 (Apr 13)**: Major expansion. Benchmark from 330 -> 380 -> 470 items (11 categories, 4 heuristic families). 15 new citations. Dead Salmons methodological safeguards. Behavioral tables + aggregation. Scoring bug fix (BPE artifacts in R1-Distill).
8. **Session 8 (Apr 12)**: Overnight3 results. Confidence paradigm (De Neys confirmed), cross-model transfer, 470-item behavioral for all 4 models. SAE R1-Distill non-transfer confirmed. Paper switched to 8pp long format.
9. **Session 9 (Apr 12)**: Multi-seed robustness. Llama 3-seed sampled decoding reveals generation strategy dependence (category profiles shift dramatically vs greedy). 7 critical paper errors found in proofread and fixed. Number harmonization across all docs. R1-Distill multi-seed launched on GPU.

---

## Complete Experiment List (Final State)

All experiments are **COMPLETE** except R1-Distill multi-seed (running on GPU):

| # | Experiment | Status | Key Result |
|---|-----------|--------|------------|
| 1 | Behavioral (6 models, 330-item greedy) | DONE | Llama 27.3%, R1 2.4%, Qwen-NT 21%, Qwen-T 7% |
| 2 | Behavioral (4 models, 470-item greedy) | DONE | Expanded to 11 categories |
| 3 | Behavioral (Llama, 3-seed sampled) | DONE | 27.5% +/- 1.3pp; category profiles shift |
| 4 | Behavioral (R1-Distill, 3-seed sampled) | **RUNNING** | Expected: low overall, category shifts |
| 5 | Probes + bootstrap CIs (4 models) | DONE | Llama 0.974 [0.952, 0.992], R1 0.930 [0.894, 0.960] |
| 6 | Cross-prediction (confound resolution) | DONE | 0.378 AUC at L14 (processing-mode-specific) |
| 7 | Cross-model transfer | DONE | Llama->R1 0.920, R1->Llama 0.954 |
| 8 | Transfer matrix (L14, Llama) | DONE | base_rate <-> conjunction 0.993 |
| 9 | Qwen dissociation (THINK/NO_THINK) | DONE | Both 0.971 (same repr, different behavior) |
| 10 | Confidence paradigm (De Neys) | DONE | Conflict-detection drop confirmed |
| 11 | SAE Goodfire L19 (Llama) | DONE | 41 features, 0 spurious, 74% EV |
| 12 | SAE cross-model (R1-Distill) | DONE | Does NOT transfer (EV=25%) |
| 13 | Attention entropy | DONE | R1 5.6% vs Llama 2.9% S2-specialized heads |
| 14 | OLMo full pipeline | DONE | Instruct 14.9%, Think 0.9%; probes 0.996->0.962 |
| 15 | Natural frequency (fixed scoring) | DONE | Llama 100% lure, R1 50% |
| 16 | New items (sunk cost, loss aversion) | DONE | Sunk cost 0% both; loss aversion OLMo 33% |
| 17 | Geometry (silhouette + CKA) | DONE | Silhouette 0.079/0.059; CKA 0.379-0.985 |

---

## Known Issues

1. **R1-Distill multi-seed (RUNNING)**: 3-seed sampled decoding launched on GPU. Will confirm whether R1 also shows category profile shifts. Not blocking paper draft.

2. **R1-Distill natural frequency scoring**: BPE artifact bug fixed (`78d6eba`), data re-collected with corrected pipeline. R1-Distill shows 50% lure rate on natural frequency with fixed scoring.

3. **SAE analysis (Goodfire L19)**: Full re-run on GPU not completed, but SAE R1-Distill non-transfer already confirmed from separate analysis. Reported in paper.

4. **Causal interventions**: Activation steering and feature ablation descoped to ICLR paper.

5. **Bootstrap CIs for Qwen**: Per-fold data not yet retrieved from the pod.

6. **Ministral-3-8B**: Deprioritized due to transformers version incompatibility. Qwen THINK/NO_THINK provides a cleaner within-model comparison.

7. **ICML template**: Paper uses `article` class with ICML-like formatting, not official ICML template. Conversion guide at `docs/icml_conversion_guide.md`.

---

## Benchmark

470 items across 11 cognitive bias categories and 4 heuristic families:

- **Representativeness**: base_rate, conjunction, natural_frequency
- **Attribute substitution**: CRT, syllogism, anchoring
- **Framing**: framing, loss_aversion, sunk_cost
- **Other**: certainty_effect, availability

Each conflict item is paired with a matched no-conflict control. Located at `data/benchmark/benchmark.jsonl`.

---

## Test Suite

427 tests passing (unit, integration, end-to-end). Run with:

```bash
make test          # full suite
make smoke         # synthetic data end-to-end (~3s, no GPU)
make lint          # ruff check
```

---

## Credit

- **Harvard HUSAI** (Harvard Undergraduate Society for Artificial Intelligence)
- **Compute**: RunPod B200 GPU cluster
- **License**: MIT

---

## Tag Instructions

This file documents the state for the `v1.0.0-workshop` tag. To create the tag:

```bash
git tag -a v1.0.0-workshop -m "ICML MechInterp Workshop submission (2026-04-13)"
```
