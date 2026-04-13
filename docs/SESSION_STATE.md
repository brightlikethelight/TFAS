# s1s2 Session State

**Last updated**: 2026-04-12 (multi-seed robustness complete for Llama, R1-Distill running, paper proofread, submission package ready)
**Active focus**: R1-Distill multi-seed (running on GPU), switch to official ICML template, advisor review.
**Target**: ICML MechInterp Workshop, May 8 AOE (~26 days).

---

## MORNING BRIEFING (read this first)

**ALL experiments are COMPLETE except R1-Distill multi-seed (running now on GPU).** The pipeline is done. Paper is 8pp long format, compiles cleanly. Seven critical factual errors were found in proofread and fixed (commit `733f83e`). Anonymous version prepared. Submission package at `submission/`.

**CRITICAL NEW FINDING**: Multi-seed robustness testing revealed that category vulnerability profiles shift dramatically between greedy and sampled decoding. The greedy "immune" categories are NOT immune under sampling. This does NOT affect probe results (P0 representation, not generation), but it changes the behavioral narrative.

### What happened this session (session 9, April 12)

1. **Multi-seed robustness (Llama, 3-seed, sampled)**: DONE. Results in `results/robustness/unsloth_Meta-Llama-3.1-8B-Instruct_multiseed.json`.
2. **7 critical paper errors fixed** (commit `733f83e`): Table 1 wrong numbers, transfer matrix overclaim, cross-prediction layer mismatch, OLMo probe numbers, cross-model transfer baselines.
3. **Anonymous paper version** prepared at `submission/workshop_paper_anonymous.tex`.
4. **ICML 2-column approximation** shows ~7.5pp body, fits 8pp limit.
5. **R1-Distill multi-seed**: RUNNING on GPU now. Script: `scripts/multiseed_behavioral.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --seeds 0,42,123 --max-new-tokens 4096`.

---

## Multi-seed robustness results (KEY NEW DATA)

### Llama-3.1-8B-Instruct, 3 seeds (0, 42, 123), T=0.7, top_p=0.95, 470 items

**Overall**: 27.5% +/- 1.3pp (stable, not flagged unstable)

| Category | Greedy LR | Sampled Mean LR | Sampled Std (pp) | Stable? | Interpretation |
|----------|-----------|-----------------|-------------------|---------|----------------|
| base_rate | 84% | **81.0%** | 3.3pp | YES | Still highly vulnerable |
| framing | 0% | **53.3%** | 2.9pp | YES | **FLIPPED**: immune -> majority-lured |
| syllogism | 52% | **41.3%** | 9.2pp | YES (barely) | Reduced, borderline unstable |
| crt | 0% | **35.6%** | 5.1pp | YES | **FLIPPED**: immune -> 36% lured |
| availability | 0% | **8.9%** | 10.2pp | **NO** | Flagged unstable (0-20% range) |
| certainty_effect | 0% | **6.7%** | 6.7pp | YES | Small but nonzero |
| sunk_cost | 0% | **4.4%** | 3.8pp | YES | Near-zero |
| arithmetic | 0% | **4.0%** | 0.0pp | YES | Rock-stable low |
| conjunction | 55% | **3.3%** | 2.9pp | YES | **FLIPPED**: majority-lured -> near-immune |
| loss_aversion | 0% | **0.0%** | 0.0pp | YES | Truly immune |
| anchoring | 0% | **0.0%** | 0.0pp | YES | Truly immune |

### Implications for the paper

- **The "3 vulnerable, 4 immune" narrative from greedy decoding is WRONG under sampling.** With sampling: framing (53%), CRT (36%), and syllogism (41%) are all substantial. Conjunction drops from 55% to 3%.
- **Probe results are UNAFFECTED**: Probes measure P0 (representation), not generation. The conflict/control boundary in the residual stream is the same regardless of decoding strategy. This is actually a *stronger* finding -- the model "knows" there's a conflict even when its generation doesn't reveal it.
- **Paper framing needs updating**: Cannot claim CRT/framing/anchoring are "immune." Must frame as "greedy decoding yields X, sampled decoding yields Y, but internal representations are stable."
- **Only loss_aversion and anchoring are truly immune** (0% across all 3 seeds with sampling).

### R1-Distill multi-seed: RUNNING

- Script: `scripts/multiseed_behavioral.py`
- Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- Seeds: 0, 42, 123
- Max new tokens: 4096
- Output will be at: `results/robustness/deepseek-ai_DeepSeek-R1-Distill-Llama-8B_multiseed.json`
- Expected: R1 overall lure rate should remain low (~2-5%), but category profile may shift similarly.

---

## Complete experiment status table

| Experiment | Status | Key Result | Data Location |
|-----------|--------|------------|---------------|
| Behavioral (6 models, 330-item greedy) | DONE | Llama 27.3%, R1 2.4%, Qwen-NT 21%, Qwen-T 7% | `results/behavioral/` |
| Behavioral (4 models, 470-item greedy) | DONE | Expanded to 11 categories | `results/behavioral/*_470.json` |
| Behavioral (Llama, 3-seed sampled) | DONE | 27.5% +/- 1.3pp; category profiles shift | `results/robustness/unsloth_*_multiseed.json` |
| Behavioral (R1, 3-seed sampled) | **RUNNING** | -- | `results/robustness/deepseek-ai_*_multiseed.json` |
| Probes + bootstrap CIs (4 models) | DONE | Llama 0.974 [0.952, 0.992], R1 0.930 [0.894, 0.960] | `results/probes/`, `results/bootstrap_cis/` |
| Cross-prediction | DONE | 0.378 AUC at L14 (confound resolved) | `results/probes/llama_cross_prediction.json` |
| Cross-model transfer | DONE | Llama->R1 direction transfers | `results/probes/cross_model_transfer_llama_r1.json` |
| Transfer matrix | DONE | base_rate <-> conjunction 0.993 | `results/probes/transfer_matrix_l14_llama.json` |
| Qwen dissociation | DONE | THINK = NO_THINK = 0.971 (same repr, different behavior) | `results/probes/` |
| Confidence paradigm | DONE | De Neys confidence drop confirmed | `results/confidence/` |
| SAE Goodfire L19 (Llama) | DONE | 41 features, 0 spurious, 74% EV | `results/sae/` |
| SAE cross-model (R1) | DONE | Does NOT transfer (EV=25%) | `results/sae/` |
| Attention entropy | DONE | R1 5.6% vs Llama 2.9% S2-specialized heads | `results/attention/` |
| OLMo full pipeline | DONE | Instruct 14.9%, Think 0.9%; probes 0.996->0.962 | `results/probes/olmo3_*` |
| Natural frequency (fixed scoring) | DONE | Llama 100% lure, R1 50% | `results/behavioral/new_items_*` |
| New items (sunk cost, loss aversion) | DONE | Sunk cost 0% both; loss aversion OLMo 33% | `results/behavioral/new_items_*` |
| Geometry (silhouette + CKA) | DONE | Silhouette 0.079/0.059; CKA 0.379-0.985 | `results/geometry/` |

---

## Eight key scientific findings (all confirmed)

1. **Category-specific vulnerability (updated by multi-seed)**: Under greedy decoding, 3 vulnerable (base_rate, conjunction, syllogism), 4 immune. Under sampled decoding, profiles shift: framing (53%), CRT (36%) become vulnerable; conjunction drops to 3%. Only loss_aversion and anchoring are truly immune across all conditions.

2. **Reasoning training blurs S1/S2 boundary**: AUC 0.974 (Llama) -> 0.930 (R1-Distill), peak layers L16/L31. CIs do not overlap. OLMo confirms: 0.996 -> 0.962.

3. **Training vs inference dissociation**: Qwen THINK and NO_THINK probes are identical (both 0.971) despite different behavior. Distillation rewires representations; inference-time reasoning acts downstream.

4. **Cross-prediction resolves the confound**: Llama probe trained on vulnerable, tested on immune: 0.378 AUC at L14. Probe detects processing mode, not task structure.

5. **Lure susceptibility is graded**: Llama +0.42 (favors lure) vs R1-Distill -0.33 (favors correct). Continuous, not binary.

6. **Shared bias representations**: base_rate and conjunction transfer at 0.993. Syllogism is more distinct (0.594-0.627 transfer from others to syllogism).

7. **OLMo cross-architecture replication**: Behavioral (14.9% vs 0.9%) AND mechanistic (0.996 [0.988, 1.000] -> 0.962 [0.934, 0.982]). Three architecture families, same story.

8. **SAE features survive falsification**: 41 features at L19, 0 spurious (Ma et al. test), 74% EV. Goodfire SAE does NOT transfer to R1-Distill (EV=25%).

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

- R1-Distill: **5.6%** S2-specialized heads
- Llama: **2.9%** S2-specialized heads

### Geometry

- Silhouette scores: 0.079 (Llama), 0.059 (R1-Distill)
- CKA range: 0.379-0.985

### Confidence paradigm (De Neys)

- Llama shows predicted confidence drop on conflict items
- R1-Distill confidence data extracted
- Results at `results/confidence/llama_confidence.json`, `r1_distill_confidence.json`

---

## Paper status

### Current state
- **8pp long format**, compiles cleanly (`pdflatex` + `bibtex`)
- **7 critical factual errors** found in proofread and fixed (commit `733f83e`)
  - C1-C3: Table 1 wrong numbers for R1-Distill, OLMo Think, Qwen
  - C4: Transfer matrix overclaim ("AUC >= 0.950" false for syllogism)
  - C5: Cross-prediction layer mismatch (L14 not L16)
  - C6: OLMo probe numbers mismatch (wrong layers, wrong AUCs)
  - C7: Cross-model transfer within-model baselines wrong
- **All numbers harmonized** to bootstrap CI values (commit `3215d73`)
- **ICML 2-column approximation** shows ~7.5pp body (fits 8pp limit)
- **Anonymous version** at `submission/workshop_paper_anonymous.tex`
- **Supplementary** with all real data at `paper/supplementary.tex` and `submission/supplementary.tex`

### Files
- Main paper: `paper/workshop_paper.tex`
- ICML version: `paper/workshop_paper_icml.tex`
- Anonymous: `submission/workshop_paper_anonymous.tex`
- Supplementary: `paper/supplementary.tex`
- References: `paper/references.bib`
- Figures: `paper/figures/` and `submission/figures/`

### What the paper still needs
1. **Switch to official ICML template** (currently using `article` class with ICML-like formatting). Guide at `docs/icml_conversion_guide.md`.
2. **Update behavioral narrative** to account for multi-seed findings (greedy vs sampled profiles).
3. **Integrate R1-Distill multi-seed** results once GPU job completes.
4. **Advisor review** before submission.

---

## GitHub

- **Repo**: https://github.com/brightlikethelight/TFAS.git
- **Commits**: 64 (as of session 9)
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

---

## What's NOT done

1. **R1-Distill multi-seed** (RUNNING on GPU) -- will confirm whether R1 also shows category profile shifts with sampling. Not blocking paper draft but should be integrated before submission.
2. **Switch to official ICML template** -- currently using `article` class. Guide written at `docs/icml_conversion_guide.md`.
3. **Advisor review** -- paper needs signoff before submission.
4. **SAE R1 re-run** -- failed in overnight3 (dependency issue). Not blocking paper (SAE R1 non-transfer already reported from separate analysis).
5. **Certainty effect + availability behavioral** -- new categories in expanded benchmark. Stretch goal.
6. **Causal interventions** -- steering/ablation descoped to ICLR paper.

## Active blockers

- **NONE for GPU work** (all experiments complete except R1-Distill multi-seed which is running).
- **Paper**: needs ICML template switch, multi-seed narrative update, advisor review.
- **R1 multi-seed**: running, not blocking paper draft.

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

### Session 9 (April 12, 2026) — Multi-seed robustness + paper polish

1. **Multi-seed behavioral (Llama)**: 3-seed sampled decoding on full 470-item benchmark. Overall 27.5% +/- 1.3pp (stable). Category profiles shift dramatically vs greedy: framing 0%->53%, CRT 0%->36%, conjunction 55%->3%.
2. **Paper proofread**: Found 7 critical factual errors (wrong table numbers, overclaims, layer mismatches). All fixed in commit `733f83e`.
3. **Number harmonization**: All docs updated to use bootstrap CI values consistently (commit `3215d73`).
4. **ICML 2-column approximation**: ~7.5pp body confirmed to fit 8pp limit.
5. **Anonymous version**: Prepared at `submission/workshop_paper_anonymous.tex`.
6. **R1-Distill multi-seed**: Launched on GPU, running now.

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

## Timeline to submission

- **Now -> Apr 18**: Integrate R1 multi-seed results, update behavioral narrative for greedy-vs-sampled, switch to ICML template
- **Apr 19-25**: Figures finalized, discussion polished, internal review
- **Apr 26-May 2**: Advisor review, external feedback, revisions
- **May 3-8**: Final polish, submit by May 8 AOE

---

## Key W&B / artifact pointers

- Llama activations: `data/activations/` on B200 pod, ~140 MB HDF5, 470 items
- R1-Distill activations: `data/activations/` on B200 pod, ~140 MB HDF5, 470 items
- Qwen NO_THINK/THINK activations: `data/activations/` on B200 pod, 157 MB HDF5 each
- OLMo activations: `data/activations/` on B200 pod
- All behavioral results: `results/behavioral/`
- Multi-seed robustness: `results/robustness/`
- Probe results: `results/probes/`
- SAE results: `results/sae/`
- Bootstrap CIs: `results/bootstrap_cis/`
- Attention entropy: `results/attention/`
- Confidence paradigm: `results/confidence/`
- Geometry: `results/geometry/`
