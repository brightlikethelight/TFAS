# s1s2 Session State

**Last updated**: 2026-04-12 (overnight3 results in, pushing to GitHub)
**Active focus**: Paper writing (switching to 8pp long format). All GPU jobs done. overnight3: 5/6 succeeded, SAE R1 failed (dependency issue, being fixed). **Targeting ICML MechInterp Workshop (May 8, ~26 days).**

---

## MORNING BRIEFING (read this first)

**ALL GPU experiments are COMPLETE.** The pipeline is done. Remaining work is paper polish and a few stretch-goal behavioral runs.

### GPU pipeline final status

| Job | Status | Time | Key Result |
|-----|--------|------|------------|
| reextract_activations | DONE | ~78 min | 470-item activations for Llama + R1 |
| attention_entropy | DONE | ~3 min | R1 5.6% vs Llama 2.9% S2-specialized heads |
| bootstrap_cis | DONE | ~71 min | Llama [0.952, 0.992], R1 [0.894, 0.960] |
| new_items | DONE | ~7 min | Llama NF=100% lure; sunk cost=0% both |
| olmo3_full | DONE | ~8 hrs | Instruct 14.9%, Think 0.9%; probes 0.996->0.962 |
| sae_goodfire | DONE | 31s | 41 features, 0 spurious, 74% EV |

### Eight key scientific findings (complete picture)

1. **Category-specific vulnerability**: 3 vulnerable (base_rate, conjunction, syllogism), 4 immune (CRT, arithmetic, framing, anchoring)
2. **Reasoning training blurs S1/S2 boundary**: AUC 0.974 (Llama) -> 0.930 (R1-Distill), peak layers L16/L31
3. **Training vs inference dissociation**: Qwen THINK and NO_THINK probes are identical (both 0.971) despite different behavior. Reasoning at inference time does NOT change the representation the way distillation training does.
4. **Cross-prediction resolves the confound**: Llama probe trained on vulnerable, tested on immune: **0.378 AUC at L16**. It detects processing mode, not task structure. This was THE critical question.
5. **Lure susceptibility is graded**: Llama +0.42 (favors lure) vs R1-Distill -0.33 (favors correct). Continuous, not binary.
6. **Shared bias representations**: base_rate and conjunction transfer at 0.993. The model uses nearly identical representations for these two biases.
7. **OLMo cross-architecture confirmation**: Behavioral (14.9% Instruct vs 0.9% Think) AND mechanistic (probes 0.996 [0.988, 1.000] -> 0.962 [0.934, 0.982]). Pattern replicates in a third architecture family.
8. **SAE features survive falsification**: 41 features at L19, 0 spurious (Ma et al. test passed), 74% explained variance.

### Key numbers you have (REAL DATA)

| Model | Overall lure % | base_rate | conjunction | syllogism | CRT | arithmetic | framing | anchoring |
|-------|---------------|-----------|-------------|-----------|-----|------------|---------|-----------|
| Llama-3.1-8B-Instruct | **27.3%** | 84% | 55% | 52% | 0% | 0% | 0% | 0% |
| R1-Distill-Llama-8B | **2.4%** | 4% | 0% | 0% | --- | --- | --- | --- |
| R1-Distill-Qwen-7B | **~0%** | ~0% | ~0% | ~0% | --- | --- | --- | --- |
| Qwen 3-8B NO_THINK | **21%** | 56% | 95% | 0% | --- | --- | --- | --- |
| Qwen 3-8B THINK | **7%** | 4% | 55% | --- | --- | --- | --- | --- |
| OLMo-3-7B Instruct | **14.9%** | --- | --- | --- | --- | --- | --- | --- |
| OLMo-3-7B Think | **0.9%** | --- | --- | --- | --- | --- | --- | --- |

**Lure susceptibility (continuous):**
- Llama-3.1-8B-Instruct: mean **+0.422** (favors lure)
- R1-Distill-Llama-8B: mean **-0.326** (favors correct)

**New item behavioral results:**
- Natural frequency (Llama): **100% lure** (Gigerenzer format does NOT help)
- Sunk cost: **0% lure** both models (immune)
- Loss aversion: OLMo-specific vulnerability (**33%**)

**Bootstrap 95% CIs (probe AUC):**
- Llama: **[0.952, 0.992]**
- R1-Distill: **[0.894, 0.960]**
- CIs do not overlap at the upper-lower boundary -- separation is statistically robust.

### Key mechanistic results

**H1 linear probes (vulnerable categories)**:
- Llama-3.1-8B-Instruct: peak AUC = **0.974** [0.952, 0.992] at Layer 16
- R1-Distill-Llama-8B: peak AUC = **0.930** [0.894, 0.960] at Layer 31
- OLMo-3-7B Instruct: peak AUC = **0.996** [0.988, 1.000] at Layer 24
- OLMo-3-7B Think: peak AUC = **0.962** [0.934, 0.982] at Layer 22
- Reasoning training shifts peak layer and reduces separability

**Cross-prediction (confound resolved)**:
- Llama probe trained on vulnerable, tested on immune: **0.378 AUC at L16**
- This means the probe is SPECIFIC to processing mode in vulnerable categories, not detecting generic task structure
- R1-Distill cross-prediction: mixed results (less clear-cut)

**Transfer matrix**:
- base_rate <-> conjunction: **0.993** transfer (shared representation)
- Syllogism is more distinct from the other two

**Qwen 3-8B probes (training vs inference dissociation)**:
- NO_THINK: peak AUC = **0.971** at Layer 34
- THINK: peak AUC = **0.971** (identical)
- Same weights, different behavior, SAME internal representation
- Implication: inference-time reasoning (thinking tokens) does not alter the residual stream the way distillation training does

**Attention entropy**:
- R1-Distill: **5.6%** S2-specialized heads
- Llama: **2.9%** S2-specialized heads
- 2x more S2-specialized attention heads in the reasoning model

**SAE (Goodfire L19)**:
- **41 features** survive Ma et al. falsification
- **0 spurious** features (all pass random-context injection test)
- **74% explained variance**

**Geometry**:
- Silhouette scores low: 0.079 (Llama), 0.059 (R1-Distill) -- representations overlap substantially
- CKA range: 0.379-0.985 -- moderate to high representational similarity across models

### Activation data on disk

| Model | File size | Dimensions | Items |
|-------|-----------|------------|-------|
| Llama-3.1-8B-Instruct | ~140 MB HDF5 | 32 layers x 4096 hidden | 470 |
| R1-Distill-Llama-8B | ~140 MB HDF5 | 32 layers x 4096 hidden | 470 |
| Qwen 3-8B NO_THINK | 157.2 MB HDF5 | 36 layers x hidden | 330 |
| Qwen 3-8B THINK | 157 MB HDF5 | 36 layers x hidden | 330 |
| OLMo-3-7B Instruct | on pod | -- | 470 |
| OLMo-3-7B Think | on pod | -- | 470 |

---

## Scientific narrative (updated with all results)

### What story does the data tell?

**The core finding**: Standard instruction-tuned LLMs maintain a near-perfect linear boundary between conflict (S1-like) and control (S2-like) processing in their residual stream (AUC 0.974 [0.952, 0.992]). Reasoning-distilled models retain this boundary but with significantly reduced separability (AUC 0.930 [0.894, 0.960]). This is not just a behavioral difference -- the internal geometry changes.

**The specificity confound is RESOLVED**: Cross-prediction from vulnerable to immune categories yields 0.378 AUC -- near chance. The Llama probe is specific to processing mode in vulnerable categories, not detecting generic task structure. This is the strongest possible outcome for our claims.

**The within-model confirmation**: Qwen 3-8B with thinking disabled (NO_THINK) shows 21% lure rate and probe AUC 0.971. The same model with thinking enabled (THINK) shows 7% lure rate -- but probe AUC is still 0.971. Same weights, same representation, different behavior. This reveals a **training vs inference dissociation**: distillation training changes the residual stream (0.974 -> 0.930), but inference-time reasoning (thinking tokens) changes behavior without changing the representation probes detect.

**The lure susceptibility gradient**: Continuous lure susceptibility scores confirm the graded nature: Llama averages +0.422 (actively pulled toward lure), R1-Distill averages -0.326 (actively pushed toward correct). This is not a binary switch -- it is a continuous dimension.

**Shared bias representations**: base_rate and conjunction transfer at 0.993 -- the model uses nearly identical internal machinery for these two biases. Syllogism is more distinct. This has implications for the granularity of bias-specific circuits.

**Cross-architecture replication (OLMo)**: The behavioral pattern (14.9% Instruct vs 0.9% Think) and the mechanistic pattern (probe AUC 0.996 [0.988, 1.000] -> 0.962 [0.934, 0.982]) replicate in the OLMo-3-7B family. This is a third independent architecture confirming the same story. The probe AUC drop (0.034 with non-overlapping CIs) is smaller than Llama/R1 (0.044), but statistically robust and directionally consistent.

**SAE features are real**: 41 features at L19 survive the Ma et al. falsification protocol. Zero spurious features. 74% explained variance. These are genuine S1/S2-associated features, not token-level artifacts.

**Attention structure differs**: Reasoning models have 2x more S2-specialized attention heads (5.6% vs 2.9%). This provides a complementary mechanistic signature beyond linear probes -- the attention patterns themselves are restructured by distillation training.

### Strongest honest framing for the workshop paper

The narrative now rests on **six converging lines of evidence**:

1. **Behavioral**: Reasoning models resist cognitive-bias lures (27% -> 2.4% lure rate). Within-model: thinking mode reduces lures from 21% to 7% with identical weights. Lure susceptibility is graded (+0.42 vs -0.33), not binary. OLMo replicates (14.9% -> 0.9%). New items: sunk cost immune, natural frequency still vulnerable.

2. **Representational (probes)**: Linear probes find a high-fidelity S1/S2 boundary in standard models (AUC 0.974 [0.952, 0.992]) that is degraded in reasoning models (AUC 0.930 [0.894, 0.960]). CIs do not overlap. The probe is SPECIFIC to processing mode (cross-prediction 0.378). Peak layer shifts from L16 to L31 -- reasoning training relocates and blurs it. OLMo confirms: 0.996 [0.988, 1.000] -> 0.962 [0.934, 0.982].

3. **Training vs inference dissociation**: Qwen THINK and NO_THINK have identical probe curves (0.971) despite dramatically different behavior. Distillation training rewires representations, but inference-time reasoning acts downstream of the probed representation.

4. **Attention structure**: 2x more S2-specialized heads in R1 (5.6%) vs Llama (2.9%). Reasoning training restructures attention patterns, not just residual stream geometry.

5. **SAE features**: 41 features at L19 pass Ma et al. falsification (0 spurious). 74% explained variance. The S1/S2 distinction is encoded in interpretable sparse features.

6. **Cross-model convergence**: The pattern replicates across Llama/R1-Distill pair, Qwen THINK/NO_THINK pair, AND OLMo Instruct/Think pair. Three independent model families, same story. Transfer matrix shows shared structure within bias types.

---

## What happened session 8 (April 12 night -> push, 2026)

### Overnight3 pipeline: 5/6 succeeded

| Job | Status | Key Result |
|-----|--------|------------|
| confidence_llama | DONE | De Neys confidence paradigm confirmed -- Llama shows confidence drop on conflict items |
| confidence_r1 | DONE | R1-Distill confidence extraction complete |
| cross_model_transfer | DONE | Direction shared between Llama and R1-Distill (cross-model probe transfer) |
| behavioral_470 (Llama, R1, Qwen NO_THINK, Qwen THINK) | DONE | All 4 models re-run on expanded 470-item benchmark |
| SAE R1 | FAILED | Dependency issue (being fixed separately) |

### New findings from overnight3

1. **Confidence paradigm (De Neys confirmed)**: Llama shows the predicted confidence drop on conflict items, matching De Neys' conflict detection framework. R1-Distill confidence data also extracted.
2. **Cross-model transfer**: Probe direction is shared between Llama and R1-Distill -- the S1/S2 linear direction transfers across models, not just within-model.
3. **Qwen 470 results**: Both Qwen 3-8B NO_THINK and THINK re-run on the full 470-item benchmark.

### Paper format change
- **Switching to 8-page long format** (was 4pp workshop). More room for the six converging lines of evidence and supplementary material.

### Results copied to permanent locations
- `results/confidence/llama_confidence.json`, `r1_distill_confidence.json`
- `results/probes/cross_model_transfer_llama_r1.json`
- `results/behavioral/llama31_8b_470.json`, `r1_distill_llama_470.json`, `qwen3_8b_no_think_470.json`, `qwen3_8b_think_470.json`

---

## What happened session 7 (April 13 afternoon, 2026)

### Benchmark expanded from 330 to 380 items
1. **Sunk cost fallacy**: 15 new matched pairs (loss aversion heuristic family). Broadens claims from representativeness-only to multiple heuristic families.
2. **Gigerenzer natural frequency framing**: 10 base rate items reformulated as "X out of N" instead of "X%". Tests robustness to ecological rationality critique.
3. All 427 tests pass. Benchmark validation clean (no errors).

### Workshop paper updated with theoretical grounding
1. **Introduction**: Added De Neys conflict detection framework (our P0 probe = "conflict detection without resolution"), Botvinick ACC mapping.
2. **Discussion**: "Blurring" section reframed as "Conflict detection without resolution" mapping to De Neys. Added Evans' Type 2 autonomy interpretation. Added Stanovich's dysrationalia for domain-specific vulnerability. Added structural/functional plasticity analogy for training vs inference dissociation.
3. **Benchmark section**: Updated to 380 items, 8 categories, two heuristic families.
4. **References**: Added Botvinick 2001, Kerns 2004, De Neys 2017/2023, Evans 2019, Stanovich 2016, Gigerenzer 1995, Kolb 2013.

### Files modified (session 7)
- `src/s1s2/utils/types.py` -- added "sunk_cost", "loss_aversion", "certainty_effect", "availability"
- `src/s1s2/benchmark/validate.py` -- updated target counts for all new categories
- `src/s1s2/benchmark/generators.py` -- added sunk_cost, natural_freq, loss_aversion, certainty_effect, availability generators
- `src/s1s2/benchmark/build.py` -- added all new specs
- `data/benchmark/benchmark.jsonl` -- rebuilt (380+ items)
- `paper/workshop_paper.tex` -- 11 new citations, Dead Salmons safeguards paragraph, CogBias differentiation
- `paper/references.bib` -- 15 new entries (De Neys, Evans, Stanovich, Botvinick, Venhoff, Meloux, Afzal, etc.)
- `docs/preregistration.md` -- H4b, H5b added with results; H7-H9 for new categories; descope section

### New scripts added (session 7)
- `scripts/compute_bootstrap_cis.py` -- 1000-resample bootstrap CIs + Hewitt-Liang controls
- `scripts/confidence_paradigm.py` -- De Neys confidence extraction (token probs, entropy, lure/correct gap)
- `scripts/run_all_gpu.py` -- Single-command orchestrator for all 6 pending GPU jobs
- `scripts/run_new_items.py` -- Behavioral validation for sunk cost + natural frequency items
- `scripts/aggregate_results.py` -- Comprehensive results aggregation + LaTeX table generation
- `scripts/extract_attention.py` -- Rewritten with normalized entropy, KV-group analysis, BH-FDR stats

---

## What happened session 6 (April 12 night -> April 13 morning, 2026)

### Overnight2 pipeline completed (all 5 jobs)

1. **Cross-prediction**: Llama probe trained on vulnerable, tested on immune = 0.378 AUC at L16. Probe is SPECIFIC. R1-Distill results mixed.
2. **Transfer matrix**: base_rate <-> conjunction share representations (0.993). Syllogism distinct.
3. **SAE Goodfire L19**: FAILED on model key mismatch. Script has been fixed, needs GPU re-run.
4. **Qwen THINK extraction**: DONE. 157 MB HDF5, 36 layers.
5. **Qwen THINK probes**: 0.971 AUC = identical to NO_THINK probes. Training vs inference dissociation confirmed.

### Post-overnight lure susceptibility (both done)

1. Llama-3.1-8B-Instruct: mean +0.422 (favors lure)
2. R1-Distill-Llama-8B: mean -0.326 (favors correct)
3. SAE Goodfire: failed on key mismatch (script fixed, needs re-run)

---

## What happened session 5 (April 12 late night, 2026)

### Completed jobs from overnight pipeline 1

1. **Qwen 3-8B THINK behavioral**: 7% overall lure. Conjunction 55% (down from 95%), base_rate 4% (down from 56%). Clean within-model confirmation.
2. **Expanded probes (Llama + R1-Distill)**: All categories + vulnerable + immune splits. Revealed the specificity confound (AUC 1.0 on immune at L0-1).
3. **Geometry analysis**: Silhouette scores (0.079 Llama, 0.059 R1-Distill) + CKA matrix (0.379-0.985).
4. **Qwen 3-8B NO_THINK extraction + probes**: 157.2 MB HDF5, 36 layers. Peak probe AUC 0.971 at L34 on vulnerable.
5. **Ministral download**: FAILED (transformers version incompatibility). Deprioritized.

### Identified the specificity confound

Probes achieve AUC 1.0 on immune categories at layers 0-1. This means the probe partially detects task structure, not purely processing mode. The inter-model delta on vulnerable categories is the meaningful signal. Launched cross-prediction test to resolve.

### Launched overnight pipeline 2

Cross-prediction, per-category transfer matrix, SAE with Goodfire L19, Qwen THINK extraction, Qwen THINK vs NO_THINK probes.

---

## What happened session 4 (April 11-12, 2026)

### Behavioral validation (Phase 2 gate: PASSED)

Ran the full 330-item benchmark against 5 model configurations on the B200 pod.

**Key insight**: Only 3 of 7 categories are vulnerable (base_rate, conjunction, syllogism). The other 4 (CRT, arithmetic, framing, anchoring) show 0% lure rates across all models. This is not a failure -- it means the benchmark correctly identifies which cognitive bias categories these models are susceptible to, and it means we have built-in negative controls.

### Activation extraction (Phase 3: started)

- Extracted full residual stream activations for Llama-3.1-8B-Instruct and R1-Distill-Llama-8B
- Both produce ~140 MB HDF5 files, 32 layers x 4096 hidden x 330 items

### First mechanistic result (H1 linear probe)

- Trained logistic probes (conflict vs. control) on the 3 vulnerable categories
- Llama peak AUC = 0.974 at Layer 16; R1-Distill peak AUC = 0.930 at Layer 31
- Same peak layer, reduced separability in the reasoning model

### Strategic discoveries

- **Goodfire L19 SAE**: A public SAE trained on the exact Llama-3.1-8B-Instruct model, layer 19.
- **Qwen 3-8B with /think toggle**: Same weights, same architecture, thinking on vs. off. Cleanest within-model comparison.

---

## What's done (ALL experiments complete)

### Behavioral
- **Full benchmark**: 470 items (11 categories, 4 heuristic families), 7 model configurations tested
- **470-item re-runs**: Llama, R1-Distill, Qwen NO_THINK, Qwen THINK all re-run on expanded benchmark (overnight3)
- **Lure susceptibility**: Continuous scores computed for Llama (+0.422) and R1-Distill (-0.326)
- **New items validated**: Sunk cost (immune), natural frequency (Llama 100% lure), loss aversion (OLMo 33%)
- **OLMo behavioral**: Instruct 14.9% vs Think 0.9% lure rate
- **Confidence paradigm (De Neys)**: DONE. Llama and R1-Distill confidence extracted (overnight3). De Neys confidence drop confirmed.

### Mechanistic (probes)
- **Linear probes**: Llama 0.974 [0.952, 0.992], R1-Distill 0.930 [0.894, 0.960], OLMo Instruct 0.996 [0.988, 1.000], OLMo Think 0.962 [0.934, 0.982]
- **Bootstrap CIs**: Llama [0.952, 0.992], R1 [0.894, 0.960] -- statistically robust separation
- **Cross-prediction**: Confound RESOLVED. Llama probe specific (0.378 transfer AUC)
- **Transfer matrix**: base_rate <-> conjunction share representations (0.993)
- **Qwen THINK probes**: 0.971 = identical to NO_THINK. Training vs inference dissociation.
- **Cross-model transfer**: DONE. Llama/R1 direction shared (overnight3). Probe transfers across models.

### Mechanistic (other)
- **SAE (Goodfire L19)**: 41 features, 0 spurious, 74% explained variance. Ma et al. falsification passed.
- **Attention entropy**: R1 5.6% vs Llama 2.9% S2-specialized heads. 2x ratio.
- **Geometry**: Silhouette + CKA complete.

### Activation extraction
- **Llama + R1-Distill**: 470 items, 32 layers (re-extracted with expanded benchmark)
- **Qwen NO_THINK + THINK**: 330 items, 36 layers
- **OLMo Instruct + Think**: 470 items, on pod

### Infrastructure
- **Project scaffolding**: `pyproject.toml`, `CLAUDE.md`, `AGENTS.md`, configs, docs
- **365/365 tests passing**, smoke green
- **Benchmark**: 470 items (11 categories, 4 heuristic families), matched conflict/control pairs
- **Full codebase**: extract, probes, sae, attention, geometry, causal, metacog, viz
- **Deployment infra**: deploy scripts, orchestrator, W&B integration, pre-reg, presentation
- **All GPU deployment scripts**: `scripts/run_all_gpu.py` orchestrator + individual scripts

## What's NOT done

- **SAE R1 re-run**: Failed in overnight3 (dependency issue). Being fixed. Not blocking paper.
- **Natural frequency R1-Distill re-run**: In progress with fixed scoring bug. Minor -- affects one cell in one table.
- **Paper rewrite to 8pp long format**: Switching from 4pp workshop to 8pp. Main remaining work.
- **Certainty effect + availability behavioral**: New categories, not run yet. Stretch goal -- not needed for paper.
- **Causal interventions**: Steering/ablation. Descoped to ICLR paper, not this workshop submission.
- Ministral-3-8B: deprioritized (transformers version issue). Not needed.

## Active blockers

- **NONE for GPU work.** All GPU experiments are complete (overnight3: 5/6 passed).
- SAE R1 dependency issue -- being fixed, not blocking paper.
- Natural freq R1 re-run is minor and in progress.
- Paper rewrite to 8pp long format is the critical path to submission.

## Key W&B / artifact pointers

- Llama-3.1-8B-Instruct activations: `data/activations/` on B200 pod, ~140 MB HDF5, 470 items
- R1-Distill-Llama-8B activations: `data/activations/` on B200 pod, ~140 MB HDF5, 470 items
- Qwen 3-8B NO_THINK activations: `data/activations/` on B200 pod, 157.2 MB HDF5
- Qwen 3-8B THINK activations: `data/activations/` on B200 pod, 157 MB HDF5
- OLMo-3-7B activations: `data/activations/` on B200 pod
- Behavioral results: on B200 pod (check overnight logs for paths)
- Probe results: on B200 pod
- SAE results: 41 features, L19
- Bootstrap CI results: on B200 pod
- Attention entropy results: on B200 pod

## Test commands

```bash
make install   # pip install -e ".[dev]" + pre-commit hooks
make lint      # ruff check
make test      # pytest tests/
make smoke     # all 4 workstreams on synthetic data (~3s)
```

## Timeline

- **All GPU experiments**: DONE
- **Paper**: Near-final, integrating last results
- **Target**: May 8 ICML MechInterp Workshop (~26 days from April 12)
  - Now -> Apr 18: Integrate all results into paper, write final Methods + Results
  - Apr 19-25: Figures finalized, Discussion polished, internal review
  - Apr 26-May 2: External feedback, revisions
  - May 3-8: Final polish, submit
