# s1s2 Session State

**Last updated**: 2026-04-16 (REVIEWER-FEEDBACK REVISION COMPLETE, SPECIFICITY EXPERIMENT RUNNING)
**Active focus**: Wait for Llama steering specificity to finish, then final polish.
**Target**: NeurIPS 2026. Abstract May 4, full paper May 6, ICML workshop May 8.

---

## MORNING BRIEFING (read this first)

**Day 2 of 19 until NeurIPS full paper submission (May 6).**  GPT-5.4 Pro + Gemini 3.1 Pro feedback received and largely addressed. All critical methodological fixes integrated into the paper. One GPU experiment running.

### Session status (2026-04-16)

- **R1 continuous steering (temporal washout test)**: COMPLETE. Result: NO coherent causal effect even under continuous intervention across full 2048-token trace. Lure at α=+5 (13.75%) is WITHIN 2σ of random-direction band (10.25%±1.9%) and in the OPPOSITE direction to Llama. Claim tightened from "7.5pp swing" to "readable but not writable, confirmed even under continuous intervention".
- **Length confound audit**: COMPLETE. Base rate d=+3.69, conjunction d=+6.36, syllogism d=+0.02. Syllogism's balanced length + high probe AUC rules out length as primary signal. Added as appendix E.
- **Full behavior table (GPT fix)**: INTEGRATED as appendix F with Wilson CIs + McNemar tests.
- **Architectural L16→L31 framing (Gemini)**: INTEGRATED in §5.2 and Discussion.
- **Inverse scaling citation (Gemini)**: INTEGRATED in §5.5.
- **ACC/Botvinick framing (Gemini)**: INTEGRATED in §5.1.

### GPU experiment in progress

**Llama steering specificity + mean-diff baseline** (PID 581131, 18min+ elapsed, 34% GPU util).
- Tests: probe direction × {conflict, control} items × 7 alphas × 2 directions (probe_coef, mean_diff)
- cos(probe, meandiff) = 0.3144 (orthogonal enough to be methodologically distinct)
- Expected total: ~2-3 hours
- Output: `results/causal/steering_specificity_llama_l14.json`

### Pre-revision checkpoint

**Previous ALL NINE EXPERIMENT LINES ARE COMPLETE.** Every result needed for both papers has data. OLMo-32B Think probes: 0.9978 AUC.

### Experiment completion summary

| # | Experiment | Status | Headline |
|---|-----------|--------|----------|
| 1 | Behavioral | DONE | 8 models (Llama, R1, Qwen x2, OLMo-7B x2, OLMo-32B x2) |
| 2 | Probes + CIs | DONE | 7 models, all with bootstrap CIs (incl. OLMo-32B Think: 0.9978) |
| 3 | Causal steering | DONE | Llama 37.5pp swing + R1 7.5pp swing |
| 4 | Within-CoT probing | DONE | 7-position trajectory, non-monotonic |
| 5 | SAE features | DONE | 41 features, 0 spurious (Ma et al. falsified) |
| 6 | Attention entropy | DONE | 57 vs 30 S2-specialized heads |
| 7 | Scale (OLMo-32B) | DONE | Instruct 19.6%, Think 0.4%; probes 0.9999/0.9978 |
| 8 | Multi-seed robustness | DONE | Both Llama and R1 stable across seeds |
| 9 | Natural frequency | DONE | Llama 100% lure (Gigerenzer hypothesis rejected) |

### Paper status

- **NeurIPS paper**: 9pp body + 8pp appendix, compiles clean, all numbers filled, official style file integrated.
- **ICML workshop paper**: ready for submission.
- **Anonymous version ready** at `submission/workshop_paper_anonymous.tex`.
- **OpenReview portal**: open today. Ready to submit.

### GPU status

- **Idle.** All compute jobs complete. No outstanding GPU work.

### What remains

1. **Submit on OpenReview** (portal open today).
2. **May 4**: submit NeurIPS abstract.
3. **May 6**: submit NeurIPS full paper.
4. **May 8**: submit ICML workshop paper.
5. Advisor review before each deadline.

---

## Complete experiment status table

| Experiment | Status | Key Result | Data Location |
|-----------|--------|------------|---------------|
| Behavioral (8 models, greedy) | DONE | Llama 27.3%, R1 2.4%, Qwen-NT 21%, Qwen-T 7%, OLMo-7B-I 14.9%, OLMo-7B-T 0.9%, OLMo-32B-I 19.6%, OLMo-32B-T 0.4% | `results/behavioral/` |
| Behavioral (470-item expanded) | DONE | 11 categories | `results/behavioral/*_470.json` |
| Behavioral (Llama, 3-seed sampled) | DONE | 27.5% +/- 1.3pp; category profiles shift | `results/robustness/unsloth_*_multiseed.json` |
| Behavioral (R1, 3-seed sampled) | DONE | Stable across seeds | `results/robustness/deepseek-ai_*_multiseed.json` |
| Probes + bootstrap CIs (8 models) | DONE | Llama 0.974, R1 0.930, OLMo-7B-I 0.996, OLMo-7B-T 0.962, Qwen-NT 0.971, Qwen-T 0.971, OLMo-32B-I 0.9999, OLMo-32B-T 0.9978 | `results/probes/`, `results/bootstrap_cis/` |
| Cross-prediction | DONE | 0.378 AUC at L14 (confound resolved) | `results/probes/llama_cross_prediction.json` |
| Cross-model transfer | DONE | Llama->R1 direction transfers | `results/probes/cross_model_transfer_llama_r1.json` |
| Transfer matrix | DONE | base_rate <-> conjunction 0.993 | `results/probes/transfer_matrix_l14_llama.json` |
| Qwen dissociation | DONE | THINK = NO_THINK = 0.971 (same repr, different behavior) | `results/probes/` |
| Confidence paradigm | DONE | De Neys confidence drop confirmed | `results/confidence/` |
| SAE Goodfire L19 (Llama) | DONE | 41 features, 0 spurious, 74% EV | `results/sae/` |
| SAE cross-model (R1) | DONE | Does NOT transfer (EV=25%) | `results/sae/` |
| Attention entropy | DONE | 57 vs 30 S2-specialized heads | `results/attention/` |
| OLMo-7B full pipeline | DONE | Instruct 14.9%, Think 0.9%; probes 0.996->0.962 | `results/probes/olmo3_*` |
| OLMo-32B behavioral | DONE | Instruct 19.6%, Think 0.4% | `results/behavioral/` |
| OLMo-32B-Think extraction + probes | DONE | Probe AUC 0.9978 | `data/activations/`, `results/probes/` |
| Natural frequency | DONE | Llama 100% lure, R1 50% (Gigerenzer rejected) | `results/behavioral/new_items_*` |
| New items (sunk cost, loss aversion) | DONE | Sunk cost 0% both; loss aversion OLMo 33% | `results/behavioral/new_items_*` |
| Geometry (silhouette + CKA) | DONE | Silhouette 0.079/0.059; CKA 0.379-0.985 | `results/geometry/` |
| Causal steering (Llama L14) | DONE | **37.5pp causal swing**, random controls flat | `results/causal/` |
| Causal steering (R1-Distill) | DONE | **7.5pp causal swing** | `results/causal/` |
| Within-CoT probing | DONE | 7-position trajectory, **non-monotonic** | `results/probes/` |
| Multi-seed robustness (both models) | DONE | Llama 27.5% +/- 1.3pp, R1 stable | `results/robustness/` |

---

## Ten key scientific findings (all confirmed)

1. **Category-specific vulnerability (updated by multi-seed)**: Under greedy decoding, 3 vulnerable (base_rate, conjunction, syllogism), 4 immune. Under sampled decoding, profiles shift: framing (53%), CRT (36%) become vulnerable; conjunction drops to 3%. Only loss_aversion and anchoring are truly immune across all conditions.

2. **Reasoning training blurs S1/S2 boundary**: AUC 0.974 (Llama) -> 0.930 (R1-Distill), peak layers L16/L31. CIs do not overlap. OLMo confirms: 0.996 -> 0.962.

3. **Training vs inference dissociation**: Qwen THINK and NO_THINK probes are identical (both 0.971) despite different behavior. Distillation rewires representations; inference-time reasoning acts downstream.

4. **Cross-prediction resolves the confound**: Llama probe trained on vulnerable, tested on immune: 0.378 AUC at L14. Probe detects processing mode, not task structure.

5. **Lure susceptibility is graded**: Llama +0.42 (favors lure) vs R1-Distill -0.33 (favors correct). Continuous, not binary.

6. **Shared bias representations**: base_rate and conjunction transfer at 0.993. Syllogism is more distinct (0.594-0.627 transfer from others to syllogism).

7. **OLMo cross-architecture replication**: Behavioral (14.9% vs 0.9%) AND mechanistic (0.996 [0.988, 1.000] -> 0.962 [0.934, 0.982]). Three architecture families, same story.

8. **SAE features survive falsification**: 41 features at L19, 0 spurious (Ma et al. test), 74% EV. Goodfire SAE does NOT transfer to R1-Distill (EV=25%).

9. **CAUSAL EVIDENCE — Probe direction steers bias behavior** (BIGGEST FINDING):
   - Llama L14: **37.5pp causal swing**. alpha=+5 (toward S2): lure drops ~21pp. alpha=-5 (toward S1): lure rises ~16pp. Random directions flat.
   - R1-Distill: **7.5pp causal swing**. Smaller effect consistent with already-low baseline lure rate.
   - Multi-model causal replication: the probe direction is not merely diagnostic — it is a functional axis.

10. **Within-CoT probing — non-monotonic trajectory**: 7-position trajectory through the thinking trace shows deliberation intensity is NOT monotonically increasing. The model's internal state follows a non-monotonic path, with potential "backtracking" in representation space.

---

## Scale replication (NEW — OLMo-32B)

| Model | Overall Lure Rate |
|-------|-------------------|
| OLMo-32B Instruct | **19.6%** |
| OLMo-32B Think | **0.4%** |

The 32B scale replicates the pattern seen at 7B/8B: reasoning training dramatically reduces lure susceptibility. The Think model at 32B achieves near-zero lure rate (0.4%), the lowest of any model tested.

---

## Key numbers (REAL DATA)

### Behavioral lure rates (greedy decoding)

| Model | Overall | base_rate | conjunction | syllogism | CRT | arithmetic | framing | anchoring |
|-------|---------|-----------|-------------|-----------|-----|------------|---------|-----------|
| Llama-3.1-8B-Instruct | **27.3%** | 84% | 55% | 52% | 0% | 0% | 0% | 0% |
| R1-Distill-Llama-8B | **2.4%** | 4% | 0% | 0% | --- | --- | --- | --- |
| R1-Distill-Qwen-7B | **~0%** | ~0% | ~0% | ~0% | --- | --- | --- | --- |
| Qwen 3-8B NO_THINK | **21%** | 56% | 95% | 0% | --- | --- | --- | --- |
| Qwen 3-8B THINK | **7%** | 4% | 55% | --- | --- | --- | --- | --- |
| OLMo-3-7B Instruct | **14.9%** | --- | --- | --- | --- | --- | --- | --- |
| OLMo-3-7B Think | **0.9%** | --- | --- | --- | --- | --- | --- | --- |
| OLMo-32B Instruct | **19.6%** | --- | --- | --- | --- | --- | --- | --- |
| OLMo-32B Think | **0.4%** | --- | --- | --- | --- | --- | --- | --- |

### Behavioral lure rates (sampled, Llama, mean of 3 seeds)

| Category | Mean LR | Std (pp) | Per-seed (0 / 42 / 123) |
|----------|---------|----------|-------------------------|
| base_rate | 81.0% | 3.3 | 83 / 83 / 77 |
| framing | 53.3% | 2.9 | 50 / 55 / 55 |
| syllogism | 41.3% | 9.2 | 36 / 36 / 52 |
| crt | 35.6% | 5.1 | 30 / 37 / 40 |
| availability | 8.9% | 10.2 | 0 / 7 / 20 |
| certainty_effect | 6.7% | 6.7 | 13 / 7 / 0 |
| sunk_cost | 4.4% | 3.8 | 7 / 0 / 7 |
| arithmetic | 4.0% | 0.0 | 4 / 4 / 4 |
| conjunction | 3.3% | 2.9 | 5 / 5 / 0 |
| loss_aversion | 0.0% | 0.0 | 0 / 0 / 0 |
| anchoring | 0.0% | 0.0 | 0 / 0 / 0 |

### Lure susceptibility (continuous)

- Llama-3.1-8B-Instruct: mean **+0.422** (favors lure)
- R1-Distill-Llama-8B: mean **-0.326** (favors correct)

### New item behavioral results

- Natural frequency (Llama): **100% lure** (Gigerenzer format does NOT help)
- Natural frequency (R1-Distill): **50% lure** (fixed scoring)
- Sunk cost: **0% lure** both models (immune)
- Loss aversion: OLMo-specific vulnerability (**33%**)

### Mechanistic: linear probes (vulnerable categories)

| Model | Peak AUC | Bootstrap 95% CI | Peak Layer |
|-------|----------|-------------------|------------|
| Llama-3.1-8B-Instruct | **0.974** | [0.952, 0.992] | L16 |
| R1-Distill-Llama-8B | **0.930** | [0.894, 0.960] | L31 |
| OLMo-3-7B Instruct | **0.996** | [0.988, 1.000] | L24 |
| OLMo-3-7B Think | **0.962** | [0.934, 0.982] | L22 |
| Qwen 3-8B NO_THINK | **0.971** | -- | L34 |
| Qwen 3-8B THINK | **0.971** | -- | L34 |
| OLMo-32B Instruct | **0.9999** | -- | -- |
| OLMo-32B Think | **0.9978** | -- | -- |

### Cross-prediction (confound resolution)

- Llama probe trained on vulnerable, tested on immune: **0.378 AUC at L14**
- L16 cross-prediction: 0.569 (different "peak" definition)
- R1-Distill cross-prediction: mixed results

### Transfer matrix (L14, Llama)

- base_rate <-> conjunction: **0.993 / 0.998** (nearly identical representations)
- syllogism -> base_rate: **0.950** (transfers out)
- base_rate -> syllogism: **0.594** (does NOT transfer in)
- conjunction -> syllogism: **0.627** (does NOT transfer in)

### Cross-model transfer

- Llama -> R1-Distill at L23: AUC = 0.920 (within-model = 0.963)
- R1-Distill -> Llama at L15: AUC = 0.954 (within-model = 0.919)

### SAE (Goodfire L19)

- **41 features** survive Ma et al. falsification (24 S1-binary + 17 S2-graded)
- **0 spurious** features
- **74% explained variance**
- Does NOT transfer to R1-Distill (EV=25%)

### Attention entropy

- 57 S2-specialized heads (one model)
- 30 S2-specialized heads (other model)

### Geometry

- Silhouette scores: 0.079 (Llama), 0.059 (R1-Distill)
- CKA range: 0.379-0.985

### Causal steering

| Model | Condition | Lure Rate | Delta from baseline | Total Swing |
|-------|-----------|-----------|---------------------|-------------|
| Llama L14 | Baseline | 52.5% | -- | -- |
| Llama L14 | alpha=+5 (toward S2) | ~31% | ~-21pp | **37.5pp** |
| Llama L14 | alpha=-5 (toward S1) | ~69% | ~+16pp | -- |
| Llama L14 | Random directions | 52-58% | flat | -- |
| R1-Distill | Full sweep | -- | -- | **7.5pp** |

### Within-CoT probing (NEW)

- 7-position trajectory through the thinking trace
- **Non-monotonic**: deliberation intensity does NOT increase linearly through CoT
- Data supports "backtracking in representation space" interpretation

### Confidence paradigm (De Neys)

- Llama shows predicted confidence drop on conflict items
- R1-Distill confidence data extracted
- Results at `results/confidence/llama_confidence.json`, `r1_distill_confidence.json`

---

## Paper status

### Current state
- **NeurIPS paper**: 9pp body + 8pp appendix, compiles clean. All numbers filled. Official NeurIPS 2026 style file integrated. Checklist complete. OpenReview metadata prepared.
- **ICML workshop paper**: ready for submission.
- **All 3 new contributions have data**: causal steering (Llama 37.5pp + R1 7.5pp), within-CoT probing (7-position non-monotonic), scale (OLMo-32B 19.6%/0.4%, probes 0.9999/0.9978).
- **Anonymous version** at `submission/workshop_paper_anonymous.tex`.
- **Supplementary** with all real data at `paper/supplementary.tex` and `submission/supplementary.tex`.

### Files
- Main paper: `paper/workshop_paper.tex`
- ICML version: `paper/workshop_paper_icml.tex`
- NeurIPS version: `paper/neurips_paper.tex`
- Anonymous NeurIPS: `paper/neurips_paper_anonymous.tex`
- Anonymous workshop: `submission/workshop_paper_anonymous.tex`
- Supplementary: `paper/supplementary.tex`
- References: `paper/references.bib`
- Figures: `paper/figures/` and `submission/figures/`

### What the paper still needs
1. Advisor review before submission.
2. Submit on OpenReview (portal open today).

---

## Submission deadlines

| Date | Milestone |
|------|-----------|
| **May 4** | NeurIPS abstract submission |
| **May 6** | NeurIPS full paper submission |
| **May 8** | ICML workshop paper submission |

---

## GitHub

- **Repo**: https://github.com/brightlikethelight/TFAS.git
- **Branch**: main

---

## Activation data on disk

| Model | File size | Dimensions | Items |
|-------|-----------|------------|-------|
| Llama-3.1-8B-Instruct | ~140 MB HDF5 | 32 layers x 4096 hidden | 470 |
| R1-Distill-Llama-8B | ~140 MB HDF5 | 32 layers x 4096 hidden | 470 |
| Qwen 3-8B NO_THINK | 157.2 MB HDF5 | 36 layers x hidden | 330 |
| Qwen 3-8B THINK | 157 MB HDF5 | 36 layers x hidden | 330 |
| OLMo-3-7B Instruct | on pod | -- | 470 |
| OLMo-3-7B Think | on pod | -- | 470 |
| OLMo-32B Instruct | on pod | -- | -- |
| OLMo-32B Think | on pod (complete) | -- | -- |

---

## What's NOT done

All experiments and paper preparation are complete. Remaining items are submission logistics only:

1. **Submit on OpenReview** (portal open today).
2. **Advisor review** before each deadline.

## Active blockers

- **None.** All experiments complete, all papers ready, GPU idle.

---

## Infrastructure

- **Tests**: 365/365 passing
- **Benchmark**: 470 items, 11 categories, 4 heuristic families
- **Codebase**: extract, probes, sae, attention, geometry, causal, metacog, viz
- **Scripts**: `scripts/run_all_gpu.py` orchestrator + individual scripts
- **Multi-seed script**: `scripts/multiseed_behavioral.py`

### Test commands

```bash
make install   # pip install -e ".[dev]" + pre-commit hooks
make lint      # ruff check
make test      # pytest tests/
make smoke     # all 4 workstreams on synthetic data (~3s)
```

---

## Session history

### Session 12 (April 12, 2026) — FINAL STATE FOR SUBMISSION

Project state finalized for OpenReview submission. All changes:
1. **OLMo-32B Think probes**: 0.9978 AUC (last compute job complete).
2. **SESSION_STATE.md**: updated to reflect all experiments complete, GPU idle, both papers ready.
3. **All uncommitted changes staged and committed**: AF post edits, compiled paper PDFs, OLMo-32B Think behavioral results.
4. **Pushed to GitHub**. Repo ready for submission tag at user's discretion.

### Session 11 (April 12, 2026) — ALL EXPERIMENTS COMPLETE

All nine experiment lines now have data. Key completions since last session:
1. **R1-Distill causal steering**: 7.5pp swing (smaller than Llama's 37.5pp, consistent with low baseline).
2. **Within-CoT probing**: 7-position trajectory through thinking trace shows **non-monotonic** deliberation intensity — model "backtracks" in representation space.
3. **OLMo-32B scale**: Instruct 19.6%, Think 0.4% — pattern replicates at 4x scale.
4. **Multi-seed R1**: Both models stable, robustness confirmed.
5. **Natural frequency**: Llama 100% lure rate — Gigerenzer hypothesis that natural frequencies eliminate base rate neglect is **rejected** for this model.
6. **Attention**: 57 vs 30 S2-specialized heads.
7. **SAE**: 41 features, 0 spurious confirmed.
8. **NeurIPS paper**: 9pp body + 8pp appendix, anonymous version ready.
9. **Status**: purely editorial work remains (fill numbers, style file, advisor review, submit).

### Session 10 (April 12, 2026) — CAUSAL STEERING (biggest finding)

1. **Causal steering experiment (Llama L14)**: Probe `coef_` used as steering direction. alpha=+5 (S2): lure 52.5% -> 31.2% (-21.3pp). alpha=-5 (S1): lure -> 68.8% (+16.3pp). Random directions flat. Total causal swing: 37.6pp. Probe CV AUC 0.960.
2. **This transforms the entire paper**: all prior correlational findings (probes, SAE, transfer, geometry) are now backed by causal evidence. The probe direction is not just diagnostic — it is a functional axis the model uses.
3. **NeurIPS sprint**: Day 2 of 23. Paper scaffold exists, causal section (SS5.3) being written.
4. **GPU queue**: R1-Distill steering running, Qwen + OLMo-32B queued.

### Session 9 (April 12, 2026) — Multi-seed robustness + paper polish

1. **Multi-seed behavioral (Llama)**: 3-seed sampled decoding on full 470-item benchmark. Overall 27.5% +/- 1.3pp (stable). Category profiles shift dramatically vs greedy: framing 0%->53%, CRT 0%->36%, conjunction 55%->3%.
2. **Paper proofread**: Found 7 critical factual errors (wrong table numbers, overclaims, layer mismatches). All fixed in commit `733f83e`.
3. **Number harmonization**: All docs updated to use bootstrap CI values consistently (commit `3215d73`).
4. **ICML 2-column approximation**: ~7.5pp body confirmed to fit 8pp limit.
5. **Anonymous version**: Prepared at `submission/workshop_paper_anonymous.tex`.
6. **R1-Distill multi-seed**: Launched on GPU.

### Session 8 (April 12, 2026) — Overnight3 results + 8pp format switch

1. Overnight3 pipeline: 5/6 jobs succeeded. Confidence paradigm (De Neys confirmed), cross-model transfer (direction shared), 470-item behavioral for all 4 models. SAE R1 failed (dependency issue).
2. Paper switched from 4pp workshop to 8pp long format.
3. SAE R1-Distill analysis: Goodfire SAE does NOT transfer (EV=25%).

### Session 7 (April 13, 2026) — Benchmark expansion + theory

1. Benchmark expanded from 330 to 380 to 470 items (sunk cost, natural frequency, loss aversion, certainty effect, availability).
2. Theoretical grounding added: De Neys conflict detection, Botvinick ACC mapping, Evans Type 2 autonomy, Stanovich dysrationalia.
3. New scripts: bootstrap CIs, confidence paradigm, run_all_gpu orchestrator, new items behavioral, aggregate results, attention extraction.

### Session 6 (April 12-13, 2026) — Overnight2 results

1. Cross-prediction: confound RESOLVED (0.378 AUC at L16/L14).
2. Transfer matrix: base_rate <-> conjunction share representations (0.993).
3. Qwen THINK probes: 0.971 = identical to NO_THINK. Training vs inference dissociation confirmed.
4. Lure susceptibility computed: Llama +0.422, R1 -0.326.

### Session 5 (April 12, 2026) — Overnight1 results + specificity confound

1. Qwen THINK behavioral: 7% overall lure.
2. Expanded probes: revealed specificity confound (AUC 1.0 on immune at L0-1).
3. Geometry analysis: silhouette + CKA.
4. Qwen NO_THINK extraction + probes: 0.971 at L34.

### Session 4 (April 11-12, 2026) — Behavioral validation + first probes

1. Behavioral validation on B200 pod: 330-item benchmark, 5 models.
2. Activation extraction: Llama + R1-Distill, ~140 MB each.
3. First probe result: Llama 0.974 at L16, R1 0.930 at L31.
4. Discovered Goodfire L19 SAE and Qwen /think toggle.

---

## Key W&B / artifact pointers

- Llama activations: `data/activations/` on B200 pod, ~140 MB HDF5, 470 items
- R1-Distill activations: `data/activations/` on B200 pod, ~140 MB HDF5, 470 items
- Qwen NO_THINK/THINK activations: `data/activations/` on B200 pod, 157 MB HDF5 each
- OLMo activations: `data/activations/` on B200 pod
- OLMo-32B activations: `data/activations/` on B200 pod
- All behavioral results: `results/behavioral/`
- Multi-seed robustness: `results/robustness/`
- Probe results: `results/probes/`
- SAE results: `results/sae/`
- Bootstrap CIs: `results/bootstrap_cis/`
- Attention entropy: `results/attention/`
- Confidence paradigm: `results/confidence/`
- Geometry: `results/geometry/`
- Causal steering: `results/causal/`
