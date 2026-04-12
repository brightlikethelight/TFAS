# s1s2 Session State

**Last updated**: 2026-04-12 01:30 AM (session 4 — Bright going to sleep)
**Active focus**: Phase 4 — behavioral validation DONE, first mechanistic results in hand, overnight jobs running on B200 pod. **Targeting ICML MechInterp Workshop (May 8, 26 days).**

---

## MORNING BRIEFING (read this first)

You went to sleep April 12. Here is what matters right now.

### Overnight jobs running on B200 pod

All launched before sleep. Check `/workspace/overnight_log.txt` on the pod.

| Job | Status at sleep | What to check |
|-----|-----------------|---------------|
| Qwen 3-8B THINK behavioral (330 items) | RUNNING | Lure rate. If it drops dramatically from 21% (NO_THINK), that is the within-model confirmation of H1. |
| Expanded probes — Llama (all categories + specificity) | RUNNING | Do immune categories (CRT, arithmetic, framing, anchoring) show the same probe signal? They should NOT. |
| Expanded probes — R1-Distill (same) | RUNNING | Same check. |
| Geometry analysis (silhouette + CKA) | RUNNING | Cosine silhouette scores + cross-model CKA. |
| Qwen 3-8B NO_THINK activation extraction + probes | RUNNING | HDF5 should be ~140 MB like the others. |
| Ministral-3-8B pair download | RUNNING | Model weights should be cached on the pod. |

### Your morning checklist (in order)

1. SSH into B200 pod, read `/workspace/overnight_log.txt`
2. **Qwen 3-8B THINK lure rate** — THE critical number. If THINK lure << NO_THINK lure (especially on conjunction/base_rate), you have within-model confirmation with identical weights. This is the cleanest result in the paper.
3. **Specificity control probes** — immune categories should show chance-level probe accuracy. If they don't, the probe is picking up something other than S1/S2 processing mode.
4. **Geometry results** — silhouette scores and CKA matrices.
5. Start **SAE analysis** with Goodfire L19 SAE (exact match for Llama-3.1-8B-Instruct, layer 19). This is the new discovery — a public SAE trained on the exact model we use.
6. **Begin writing** the ICML MechInterp Workshop paper. Figure code is ready. 26 days to deadline.

### Key numbers you have (REAL DATA, not synthetic)

| Model | Overall lure % | base_rate | conjunction | syllogism | CRT | arithmetic | framing | anchoring |
|-------|---------------|-----------|-------------|-----------|-----|------------|---------|-----------|
| Llama-3.1-8B-Instruct | **27.3%** | 84% | 55% | 52% | 0% | 0% | 0% | 0% |
| R1-Distill-Llama-8B | **2.4%** | 4% | 0% | 0% | — | — | — | — |
| R1-Distill-Qwen-7B | **~0%** | ~0% | ~0% | ~0% | — | — | — | — |
| Qwen 3-8B NO_THINK | **21%** | 56% | 95% | 0% | — | — | — | — |
| Qwen 3-8B THINK | **PENDING** (overnight) | | | | | | | |

### Key mechanistic result (H1 PASS — first real finding)

Linear probes on the 3 vulnerable categories (base_rate, conjunction, syllogism):
- **Llama-3.1-8B-Instruct**: peak AUC = **0.999** at Layer 14
- **R1-Distill-Llama-8B**: peak AUC = **0.929** at Layer 14
- Both peak at the **same layer** — the "where" doesn't change, the "how much" does
- Standard model has near-perfect S1/S2 separability; reasoning distillation has blurred it
- This is consistent with H1: reasoning training integrates S2-like processing into the residual stream, reducing the distinctiveness of the S1/S2 boundary

### Activation data on disk

| Model | File size | Dimensions | Items |
|-------|-----------|------------|-------|
| Llama-3.1-8B-Instruct | 139.7 MB HDF5 | 32 layers × 4096 hidden | 330 |
| R1-Distill-Llama-8B | 140.0 MB HDF5 | 32 layers × 4096 hidden | 330 |

---

## What happened this session (April 11-12, 2026)

### Behavioral validation (Phase 2 gate: PASSED)

Ran the full 330-item benchmark against 5 model configurations on the B200 pod.

**Key insight**: Only 3 of 7 categories are vulnerable (base_rate, conjunction, syllogism). The other 4 (CRT, arithmetic, framing, anchoring) show 0% lure rates across all models. This is not a failure — it means the benchmark correctly identifies which cognitive bias categories these models are susceptible to, and it means we have built-in negative controls.

**Strategic pivot**: Focus primary analysis on the 3 vulnerable categories. Use the 4 immune categories as specificity controls for probes (probes should NOT be able to distinguish conflict/control in immune categories).

### Activation extraction (Phase 3: started)

- Extracted full residual stream activations for Llama-3.1-8B-Instruct and R1-Distill-Llama-8B
- Both produce ~140 MB HDF5 files, 32 layers × 4096 hidden × 330 items
- Qwen 3-8B NO_THINK extraction running overnight

### First mechanistic result (H1 linear probe)

- Trained logistic probes (conflict vs. control) on the 3 vulnerable categories
- Llama peak AUC = 0.999 at Layer 14; R1-Distill peak AUC = 0.929 at Layer 14
- Same peak layer, reduced separability in the reasoning model
- Interpretation: reasoning training doesn't relocate S1/S2 representations — it blurs them, consistent with integrating S2-like processing throughout the forward pass

### Strategic discoveries

- **Goodfire L19 SAE**: A public SAE trained on the exact Llama-3.1-8B-Instruct model, layer 19. Previously we only had Llama Scope (trained on Base, not Instruct). This is a direct match — no reconstruction fidelity concerns.
- **Ministral-3-8B pair**: Discovered as potentially the cleanest matched pair (same architecture, same weights, thinking toggle). Download started overnight.
- **Qwen 3-8B with /think toggle**: Same weights, same architecture, thinking on vs. off. This is the cleanest possible within-model comparison. NO_THINK shows 21% lure; THINK result pending overnight.
- **ICML MechInterp Workshop**: May 8 deadline, 26 days out. This is the target venue. Workshop paper, not full paper — more tractable.

---

## What's done (cumulative)

- **Project scaffolding**: `pyproject.toml`, `CLAUDE.md`, `AGENTS.md`, configs, docs
- **365/365 tests passing**, smoke green
- **Benchmark**: 330 items (expanded from 284), 7 categories, matched conflict/control pairs
- **Full codebase**: extract, probes, sae, attention, geometry, causal, metacog, viz
- **Deployment infra**: deploy scripts, orchestrator, W&B integration, pre-reg, presentation
- **Behavioral validation**: 5 model configs tested on full benchmark (REAL DATA)
- **Activation extraction**: 2 models complete (Llama, R1-Distill-Llama), 1 running (Qwen NO_THINK)
- **First mechanistic result**: Linear probes on vulnerable categories, H1 confirmed

## What's still NOT done

- Qwen 3-8B THINK behavioral + activation extraction (overnight)
- Expanded probes (all categories + specificity controls) — overnight
- Geometry analysis (silhouette, CKA) — overnight
- SAE analysis with Goodfire L19 (ready to start morning)
- Attention entropy analysis on real data
- Causal interventions (steering, ablation)
- Ministral-3-8B pair extraction + analysis
- Workshop paper writing (figures ready, need text)
- FASRC access (Kempner deadline was Apr 14 — status unknown, B200 pod is working fine for now)

## Active blockers

- **None critical** — B200 pod is running, overnight jobs launched, data is flowing
- FASRC access still pending but not blocking progress (RunPod B200 is sufficient)
- Goodfire L19 SAE needs to be loaded and verified before SAE workstream can produce real results

## Overnight jobs (B200 pod)

All jobs launched before sleep on April 12 ~01:00 AM. Check `/workspace/overnight_log.txt`.

```
1. Qwen 3-8B THINK behavioral (330 items)
2. Expanded probes — Llama (all categories + specificity)
3. Expanded probes — R1-Distill (same)
4. Geometry analysis (silhouette + CKA)
5. Qwen 3-8B NO_THINK activation extraction + probes
6. Ministral-3-8B pair model download
```

## Key W&B / artifact pointers

- Llama-3.1-8B-Instruct activations: `data/activations/` on B200 pod, 139.7 MB HDF5
- R1-Distill-Llama-8B activations: `data/activations/` on B200 pod, 140.0 MB HDF5
- Behavioral results: on B200 pod (check overnight log for paths)
- Probe results (Layer 14 peak): on B200 pod

## Test commands

```bash
make install   # pip install -e ".[dev]" + pre-commit hooks
make lint      # ruff check
make test      # pytest tests/
make smoke     # all 4 workstreams on synthetic data (~3s)
```

## Timeline

- **Now → May 8**: ICML MechInterp Workshop paper (26 days)
  - Week 1 (Apr 12-18): Complete all extractions, run full probe/geometry/attention analysis, start SAE
  - Week 2 (Apr 19-25): Causal interventions, SAE falsification, write Methods + Results
  - Week 3 (Apr 26-May 2): Figures, writing, internal review
  - Week 4 (May 3-8): Polish, submit
