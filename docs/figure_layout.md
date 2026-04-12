# Figure Layout: ICML MechInterp Workshop Paper (4 pages)

**Constraint**: ICML two-column format, 4 pages. 2-3 figures max in main body. Appendix unlimited.

---

## Design Rationale

The paper's central claim is three-fold convergence: (1) reasoning training suppresses heuristic behavior, (2) this is visible as reduced linear separability in the residual stream, and (3) the effect replicates across model families. Every main-body visual must directly support one of these three prongs. Anything that elaborates, provides controls, or adds nuance goes in the appendix.

The key visual priority: **the probe curve comparison (Llama vs R1-Distill) is the most important finding** because it is the mechanistic core of the paper. It must be Figure 1.

---

## Main Paper

### Figure 1: Two-panel — Layer-wise Probe Curves + Cross-Prediction Specificity

**Left panel (a)**: Layer-wise probe AUC for Llama (0.999 peak) vs R1-Distill (0.929 peak) on vulnerable categories, 32 layers. Both curves peak at layer 14. Include the Hewitt-Liang random-label control baseline (should be near 0.5). Optionally add Qwen NO_THINK curve (peaks at L34, different architecture depth — normalize x-axis to fractional depth if including).

**Right panel (b)**: Cross-prediction bar chart or paired bars. Four conditions: (i) train vulnerable / test vulnerable, (ii) train vulnerable / test immune, (iii) train immune / test immune, (iv) train immune / test vulnerable. This is the confound resolution figure. If transfer to immune is near 0.5, it proves specificity. If high, it proves the probe detects task structure — either way, it must be shown because reviewers will ask.

**Why this is Figure 1**: The probe delta (0.999 vs 0.929) at the same layer is the paper's mechanistic headline. No other figure communicates the core finding as efficiently. The cross-prediction panel preempts the most obvious reviewer objection (specificity confound) in the same visual. Together they answer: "Is the signal real?" and "Is the signal specific?" in one glance.

**Size**: Full column width (two-panel). Approximately 1/3 of a column height.

---

### Figure 2: Qwen Think vs No-Think Probe Overlay

**Single panel**: Layer-wise probe AUC for Qwen 3-8B in THINK mode vs NO_THINK mode, same plot style as Figure 1a. Same weights, same architecture — only the processing mode differs. If the curves are nearly identical (as preliminary data suggests), this is a striking result: the internal representation is the same regardless of whether the model "thinks," meaning the probe direction encodes something about the input/model pair, not about the reasoning trace.

If the curves ARE different, it shows thinking mode changes internal geometry even with identical weights, which supports the deliberation-intensity interpretation.

**Why this earns main-body space**: This is the cleanest within-model comparison in the paper. No confound from different training data, different architectures, or different parameter counts. It directly tests whether the representational signature is a property of the model's weights (invariant to thinking mode) or the model's processing (modulated by thinking mode). Reviewers at a mech-interp workshop will find this more compelling than behavioral bars — everyone already knows reasoning models do better on benchmarks. The representational question is novel.

**Size**: Single column width. Approximately 1/4 of a column height.

---

### Table 1: Behavioral Results (replaces a bar chart)

| Model | Overall | base_rate | conjunction | syllogism |
|-------|---------|-----------|-------------|-----------|
| Llama-3.1-8B-Instruct | 27.3% | 84% | 55% | 52% |
| R1-Distill-Llama-8B | 2.4% | 4% | 0% | 0% |
| Qwen 3-8B (no-think) | 21% | 56% | 95% | 0% |
| Qwen 3-8B (think) | 7% | 4% | 55% | 0% |

Only show the 3 vulnerable categories. Note in the caption that the remaining 4 categories (CRT, arithmetic, framing, anchoring) show 0% lure rates across all models and serve as negative controls (full table in appendix).

**Why a table instead of bars**: The behavioral story is simple — reasoning models resist lures. A 4-row, 5-column table communicates this in ~4 lines of vertical space. A bar chart with 4 models x 3 categories would consume 1/3 of a column for the same information with less precision (readers can't read exact percentages off bars). The space savings matter enormously in a 4-page paper.

**Why R1-Distill-Qwen is dropped from the main table**: It shows ~0% across the board, which is redundant with R1-Distill-Llama. Include in appendix.

---

### Table 2: Probe Summary Statistics (optional, only if space permits)

| Model | Vulnerable peak AUC | Layer | Selectivity |
|-------|---------------------|-------|-------------|
| Llama-3.1-8B-Instruct | 0.999 | 14 | (real - control) |
| R1-Distill-Llama-8B | 0.929 | 14 | (real - control) |
| Qwen 3-8B (no-think) | 0.971 | 34 | (real - control) |
| Qwen 3-8B (think) | TBD | TBD | TBD |

This may be redundant with Figure 1. Include only if the selectivity numbers need a home and Figure 1 caption is already dense. Otherwise fold into Figure 1 caption or the text.

---

## What Does NOT Make the Main Paper

### Behavioral bars (item 1) — replaced by Table 1
Bar charts are visually appealing but space-inefficient for 4x3 data. The table is strictly better here.

### Lure susceptibility distribution (item 5) — appendix
The continuous lure-susceptibility distribution (Llama vs R1-Distill) is an interesting elaboration but not load-bearing for the main argument. The behavioral table already shows the discrete version. The distribution adds nuance (e.g., bimodal vs unimodal) but costs a figure slot.

### Transfer matrix heatmap (item 6) — appendix
The 7x7 transfer matrix is important for understanding the fine structure of what probes learn, but it is a detailed analysis that supports the cross-prediction panel in Figure 1b. Reference it from the main text ("full transfer matrix in Appendix Figure X") but do not spend main-body space on it.

### Category-specific heatmap (item 7) — appendix
Models x 7 categories is too detailed for the main argument, which hinges on the vulnerable vs immune split, not individual categories. Appendix.

---

## Appendix Layout

### Appendix A: Full Behavioral Results
- **Table A1**: Full 5-model x 7-category lure rate table (including R1-Distill-Qwen and all immune categories at 0%)
- **Figure A1**: Behavioral bar chart (item 1) — all 4 models x all 7 categories, for visual readers who prefer it

### Appendix B: Probe Details
- **Figure A2**: Layer-wise probe curves with confidence bands (bootstrap CIs), one subplot per model
- **Figure A3**: Category-specific heatmap (item 7) — models x all 7 categories, probe AUC at peak layer
- **Table A2**: Full probe results table (all models, all category splits, all layers)
- **Figure A4**: Hewitt-Liang control selectivity plots

### Appendix C: Transfer and Specificity
- **Figure A5**: Full 7x7 transfer matrix heatmap (item 6) — train on row category, test on column category
- **Table A3**: Transfer AUC values with CIs

### Appendix D: Geometry
- **Figure A6**: Silhouette score curves across layers (Llama vs R1-Distill)
- **Figure A7**: CKA cross-model similarity matrix
- **Table A4**: PCA variance explained, silhouette peaks, CKA ranges

### Appendix E: Continuous Metrics (if data ready)
- **Figure A8**: Lure susceptibility distribution (item 5)

---

## Column-Inch Budget (approximate)

ICML two-column format gives roughly 9.5 inches of usable text height per page, so ~38 column-inches across 4 pages (but two columns = ~76 column-inches total). Title/abstract/references consume ~20 column-inches. That leaves ~56 column-inches for content.

| Element | Est. column-inches | Notes |
|---------|-------------------|-------|
| Table 1 (behavioral) | 2.5 | 4 rows + header + caption |
| Figure 1 (2-panel probes) | 7-8 | Full column width = spans both columns, ~3.5" tall |
| Figure 2 (Qwen overlay) | 5-6 | Single column, ~2.5" tall + caption |
| Table 2 (probe stats) | 2 | Only if space permits |
| Running text | ~38-40 | Intro, methods, results, discussion |
| **Total** | ~55-58 | Tight but feasible |

If space is critically tight, Table 2 is the first cut. Figure 2 is the second cut (fold Qwen result into one sentence and move the figure to appendix). Figure 1 and Table 1 are non-negotiable.

---

## Priority Stack (if forced to cut)

1. **Keep**: Figure 1 (probe curves + cross-prediction) — the paper's reason for existing
2. **Keep**: Table 1 (behavioral) — establishes the behavioral phenomenon
3. **Keep if possible**: Figure 2 (Qwen overlay) — within-model replication, mech-interp novelty
4. **Cut first**: Table 2 — redundant with Figure 1
5. **Already cut**: Everything else is appendix

---

## Production Notes

- All figures: PDF vector output, matplotlib with consistent style
- Color scheme: use colorblind-safe palette (e.g., Okabe-Ito). Llama = blue, R1-Distill = orange, Qwen no-think = green, Qwen think = green-dashed
- Font size in figures: minimum 8pt for ICML readability
- Figure 1 should use `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))` for full-width two-panel
- Figure 2 should use `fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5))` for single-column
- X-axis labels: "Layer" for absolute, "Relative depth (layer / total)" if comparing architectures with different layer counts
- All probe AUC plots: include shaded bootstrap 95% CI bands
