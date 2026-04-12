# s1s2 Session State

**Last updated**: 2026-04-12 late night (session 5 — Bright going to sleep, overnight pipeline 2)
**Active focus**: Phase 4 — probes show strong results but specificity confound identified. Cross-prediction test running overnight to resolve. **Targeting ICML MechInterp Workshop (May 8, 26 days).**

---

## MORNING BRIEFING (read this first)

You went to sleep April 12, late night. The SECOND overnight pipeline is running. Here is what matters.

### The specificity confound (CRITICAL — shapes all claims)

Probes achieve AUC 1.0 on **immune** categories (CRT, arithmetic, framing, anchoring) at layers 0-1. This means the probe partially detects **task structure** (presence of lure text), not purely processing mode. The meaningful signal is the **inter-model DELTA**: 0.999 (Llama) vs 0.929 (R1-Distill) on **vulnerable** categories. The confound resolution hinges on the cross-prediction test running overnight.

### Overnight pipeline 2 (B200 pod)

Check `/workspace/overnight2_log.txt` on the pod.

| Job | What it tests |
|-----|---------------|
| Cross-prediction (train vulnerable -> test immune) | **THE** confound resolution. If transfer AUC >> 0.5, probe detects task structure. If transfer ~ 0.5, vulnerable-category probe is specific to processing mode. |
| Per-category transfer matrix | Full transfer grid between all 7 categories. |
| SAE analysis with Goodfire L19 | Real SAE features on exact Llama-3.1-8B-Instruct model. Ma et al. falsification included. |
| Qwen 3-8B THINK extraction | Activations for the thinking-mode model (counterpart to existing NO_THINK data). |
| Qwen THINK vs NO_THINK probe comparison | Same weights, different processing. Cleanest within-model test. |

### Morning checklist (in priority order)

1. SSH into B200 pod, read `/workspace/overnight2_log.txt`
2. **Cross-prediction AUC**: This is the single most important number. If transfer to immune >> 0.5, the confound is real and claims must be restructured around the delta, not the absolute probe accuracy. If transfer ~ 0.5, the vulnerable-category probe is specific and claims can be stronger.
3. **SAE results**: Did Goodfire L19 produce significant features after Ma et al. falsification? Token-trivial features eliminated?
4. **Qwen THINK probes**: Does thinking-mode show different probe curves than no-think? This is the clean within-model comparison (same weights, different mode).
5. **Per-category transfer matrix**: Which categories transfer to which? This tells you the fine structure of what probes learn.
6. Run **OLMo-3-7B pair** experiments (scripts ready)
7. Run **continuous lure_susceptibility extraction**
8. Run **attention entropy extraction**
9. Continue workshop paper writing

### Key numbers you have (REAL DATA)

| Model | Overall lure % | base_rate | conjunction | syllogism | CRT | arithmetic | framing | anchoring |
|-------|---------------|-----------|-------------|-----------|-----|------------|---------|-----------|
| Llama-3.1-8B-Instruct | **27.3%** | 84% | 55% | 52% | 0% | 0% | 0% | 0% |
| R1-Distill-Llama-8B | **2.4%** | 4% | 0% | 0% | --- | --- | --- | --- |
| R1-Distill-Qwen-7B | **~0%** | ~0% | ~0% | ~0% | --- | --- | --- | --- |
| Qwen 3-8B NO_THINK | **21%** | 56% | 95% | 0% | --- | --- | --- | --- |
| Qwen 3-8B THINK | **7%** | 4% | 55% | --- | --- | --- | --- | --- |

**Qwen 3-8B THINK** (newly completed): 7% overall lure (vs 21% NO_THINK). Conjunction drops from 95% to 55%. Base rate drops from 56% to 4%. This is the within-model confirmation of H1 with identical weights.

### Key mechanistic results

**H1 linear probes (vulnerable categories)**:
- Llama-3.1-8B-Instruct: peak AUC = **0.999** at Layer 14
- R1-Distill-Llama-8B: peak AUC = **0.929** at Layer 14
- Both peak at the same layer — the "where" doesn't change, the "how much" does

**Expanded probes (newly completed)**:
- Llama + R1-Distill probed on all categories + vulnerable + immune splits
- Immune categories show AUC 1.0 at L0-1 (the confound)

**Geometry (newly completed)**:
- Silhouette scores low: 0.079 (Llama), 0.059 (R1-Distill) — representations overlap substantially
- CKA range: 0.379-0.985 — moderate to high representational similarity across models

**Qwen 3-8B NO_THINK probes (newly completed)**:
- 157.2 MB HDF5, 36 layers extracted
- Peak AUC = **0.971** at Layer 34 on vulnerable categories

### Activation data on disk

| Model | File size | Dimensions | Items |
|-------|-----------|------------|-------|
| Llama-3.1-8B-Instruct | 139.7 MB HDF5 | 32 layers x 4096 hidden | 330 |
| R1-Distill-Llama-8B | 140.0 MB HDF5 | 32 layers x 4096 hidden | 330 |
| Qwen 3-8B NO_THINK | 157.2 MB HDF5 | 36 layers x hidden | 330 |
| Qwen 3-8B THINK | EXTRACTING (overnight) | 36 layers x hidden | 330 |

---

## Scientific narrative so far

### What story does the data tell?

**The core finding**: Standard instruction-tuned LLMs maintain a near-perfect linear boundary between conflict (S1-like) and control (S2-like) processing in their residual stream (AUC 0.999). Reasoning-distilled models retain this boundary but with significantly reduced separability (AUC 0.929). This is not just a behavioral difference — the internal geometry changes.

**The within-model confirmation**: Qwen 3-8B with thinking disabled (NO_THINK) shows 21% lure rate and probe AUC 0.971. The same model with thinking enabled (THINK) shows 7% lure rate. Same weights, same architecture, same parameters — only the processing mode differs. This is the cleanest evidence that reasoning mode changes internal processing, not just output formatting.

**The specificity problem**: Probes also achieve high accuracy on immune categories at early layers, suggesting they partially detect task-level features (lure text presence) rather than purely processing-mode features. The honest framing is: probes detect a composite of task structure AND processing mode. The inter-model delta (0.999 vs 0.929) is the signal that survives this confound, because task structure is identical across model pairs.

**The geometry story**: Low silhouette scores (0.059-0.079) mean conflict and control representations are not cleanly clustered — they overlap substantially in activation space. But a linear probe can still separate them with near-perfect accuracy. This means the signal is a narrow linear direction, not a broad geometric separation. Reasoning training compresses even this narrow direction.

### Strongest honest framing for the workshop paper

The cleanest narrative centers on **three converging lines of evidence**:

1. **Behavioral**: Reasoning models resist cognitive-bias lures (27% -> 2.4% lure rate). Within-model: thinking mode reduces lures from 21% to 7% with identical weights.

2. **Representational**: Linear probes find a high-fidelity S1/S2 boundary in standard models (AUC 0.999) that is degraded in reasoning models (AUC 0.929). The direction exists at the same layer (L14) in both — reasoning training doesn't relocate it, it blurs it.

3. **Cross-model convergence**: The pattern replicates across Llama/R1-Distill pair AND Qwen THINK/NO_THINK pair. Two independent model families, same story.

**What to be careful about**: Do NOT claim probes detect "pure S1/S2 processing mode." The specificity confound means they detect a mixture. Frame it as: "The residual stream encodes a direction that correlates with cognitive-bias susceptibility, and this direction is modulated by reasoning training." The cross-prediction results (overnight) will determine how strongly you can frame the specificity.

**What would make the paper much stronger**: If cross-prediction AUC from vulnerable to immune is ~0.5, you can claim the probe is specific to processing mode in vulnerable categories. If SAE features survive Ma et al. falsification, you have interpretable mechanistic evidence beyond probes. These are both running overnight.

---

## What happened this session (April 12 late night, 2026)

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

**Key insight**: Only 3 of 7 categories are vulnerable (base_rate, conjunction, syllogism). The other 4 (CRT, arithmetic, framing, anchoring) show 0% lure rates across all models. This is not a failure — it means the benchmark correctly identifies which cognitive bias categories these models are susceptible to, and it means we have built-in negative controls.

### Activation extraction (Phase 3: started)

- Extracted full residual stream activations for Llama-3.1-8B-Instruct and R1-Distill-Llama-8B
- Both produce ~140 MB HDF5 files, 32 layers x 4096 hidden x 330 items

### First mechanistic result (H1 linear probe)

- Trained logistic probes (conflict vs. control) on the 3 vulnerable categories
- Llama peak AUC = 0.999 at Layer 14; R1-Distill peak AUC = 0.929 at Layer 14
- Same peak layer, reduced separability in the reasoning model

### Strategic discoveries

- **Goodfire L19 SAE**: A public SAE trained on the exact Llama-3.1-8B-Instruct model, layer 19.
- **Qwen 3-8B with /think toggle**: Same weights, same architecture, thinking on vs. off. Cleanest within-model comparison.

---

## What's done (cumulative)

- **Project scaffolding**: `pyproject.toml`, `CLAUDE.md`, `AGENTS.md`, configs, docs
- **365/365 tests passing**, smoke green
- **Benchmark**: 330 items (expanded from 284), 7 categories, matched conflict/control pairs
- **Full codebase**: extract, probes, sae, attention, geometry, causal, metacog, viz
- **Deployment infra**: deploy scripts, orchestrator, W&B integration, pre-reg, presentation
- **Behavioral validation**: 5 model configs tested on full benchmark (REAL DATA)
- **Activation extraction**: 3 models complete (Llama, R1-Distill-Llama, Qwen NO_THINK), 1 running (Qwen THINK)
- **Mechanistic results**: Linear probes on vulnerable categories (H1 confirmed), expanded probes (all categories), geometry (silhouette + CKA)
- **Qwen 3-8B THINK behavioral**: Within-model confirmation (7% vs 21% lure)
- **Specificity confound identified**: Probes detect task structure at L0-1, inter-model delta is the real signal

## What's still NOT done

- Cross-prediction confound resolution (overnight pipeline 2)
- SAE analysis with Goodfire L19 (overnight pipeline 2)
- Qwen THINK extraction + probes (overnight pipeline 2)
- Per-category transfer matrix (overnight pipeline 2)
- Attention entropy analysis on real data
- Causal interventions (steering, ablation)
- OLMo-3-7B pair experiments (scripts ready)
- Continuous lure_susceptibility extraction
- Workshop paper writing (figures ready, narrative drafted above, need text)
- Ministral-3-8B: deprioritized (transformers version issue)
- FASRC access (status unknown, B200 pod is working fine)

## Active blockers

- **Cross-prediction result**: Determines how strongly we can frame the specificity claims. Not blocking work but blocking the paper framing.
- **Ministral**: transformers version incompatibility. Deprioritized — Qwen THINK/NO_THINK is a cleaner within-model pair anyway.
- FASRC access still pending but not blocking progress (RunPod B200 is sufficient)

## Overnight pipeline 2 (B200 pod)

All jobs launched before sleep on April 12, late night. Check `/workspace/overnight2_log.txt`.

```
1. Cross-prediction test (train vulnerable -> test immune)
2. Per-category transfer matrix
3. SAE analysis with Goodfire L19 (Ma et al. falsification included)
4. Qwen 3-8B THINK activation extraction
5. Qwen THINK vs NO_THINK probe comparison
```

## Key W&B / artifact pointers

- Llama-3.1-8B-Instruct activations: `data/activations/` on B200 pod, 139.7 MB HDF5
- R1-Distill-Llama-8B activations: `data/activations/` on B200 pod, 140.0 MB HDF5
- Qwen 3-8B NO_THINK activations: `data/activations/` on B200 pod, 157.2 MB HDF5
- Behavioral results: on B200 pod (check overnight logs for paths)
- Probe results: on B200 pod

## Test commands

```bash
make install   # pip install -e ".[dev]" + pre-commit hooks
make lint      # ruff check
make test      # pytest tests/
make smoke     # all 4 workstreams on synthetic data (~3s)
```

## Timeline

- **Now -> May 8**: ICML MechInterp Workshop paper (26 days)
  - Week 1 (Apr 12-18): Complete all extractions, run full probe/geometry/attention analysis, resolve specificity confound, start SAE
  - Week 2 (Apr 19-25): Causal interventions, SAE falsification, write Methods + Results
  - Week 3 (Apr 26-May 2): Figures, writing, internal review
  - Week 4 (May 3-8): Polish, submit
