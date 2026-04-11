# s1s2 — Mechanistic Signatures of Dual-Process Cognition in LLMs

> The Deliberation Gradient: testing whether LLMs exhibit internally distinct processing modes analogous to human "System 1" (fast/heuristic) and "System 2" (slow/deliberative) cognition, and whether reasoning training amplifies this distinction.

**HUSAI** (Harvard Undergraduate AI Safety) research project.

## TL;DR

We run a battery of cognitive bias tasks (CRT, base-rate neglect, syllogisms, anchoring, framing, conjunction fallacy, multi-step arithmetic) through **standard** models (Llama-3.1-8B-Instruct, Gemma-2-9B-IT) and **reasoning** models (DeepSeek-R1-Distill-Llama-8B, R1-Distill-Qwen-7B), then characterize their internal computation with five complementary mechanistic interpretability methods:

1. **Linear probes** — what is linearly decodable about processing mode at each layer?
2. **SAE feature analysis** — which sparse features activate differentially on conflict vs no-conflict tasks?
3. **Attention entropy** — does attention focus differently in S1-eliciting vs S2-requiring conditions?
4. **Representational geometry** — do S1 and S2 activations occupy distinguishable subspaces?
5. **Causal interventions** — do SAE feature ablations / activation steering causally shift behavior?

The headline comparison is **Llama-3.1-8B-Instruct vs DeepSeek-R1-Distill-Llama-8B**: same architecture, different training. Any internal differences are attributable to reasoning distillation.

## Critical Framing — read before coding

We adopt a **graded "deliberation intensity" framework**, not a binary S1/S2 categorization. Reasons:
- Even psychologists have abandoned the binary (Melnikoff & Bargh 2018)
- Coda-Forno et al. (2025) found overlapping subspaces in dual architectures
- "Reasoning on a Spectrum" (Ziabari et al. 2025) showed monotonic interpolation

**Language rules**:
- NEVER write "LLMs have System 1 and System 2"
- PREFER "S1-like and S2-like processing signatures"
- BEST: "deliberation-intensity gradient"

We use S1/S2 vocabulary as an operationalized distinction, not a cognitive claim.

## Project layout

```
s1s2/
├── pyproject.toml          # package definition
├── README.md               # this file
├── CLAUDE.md               # coding conventions for Claude Code
├── AGENTS.md               # equivalent for Codex CLI
├── docs/
│   ├── data_contract.md    # HDF5 schema everyone agrees on
│   ├── architecture.md     # system architecture
│   ├── SESSION_STATE.md    # live experiment state (per Bright's session protocol)
│   └── LESSONS_LEARNED_COMPACT.md
├── configs/                # Hydra configs
│   ├── extract.yaml
│   ├── probe.yaml
│   ├── sae.yaml
│   ├── attention.yaml
│   ├── geometry.yaml
│   └── causal.yaml
├── data/
│   ├── benchmark/          # cognitive bias problems
│   ├── activations/        # cached activation HDF5 files
│   └── raw_outputs/        # model generations
├── src/s1s2/
│   ├── benchmark/          # benchmark loading + validation
│   ├── extract/            # activation extraction with hooks
│   ├── probes/             # linear/MLP/CCS probes + Hewitt-Liang controls
│   ├── sae/                # SAE feature analysis + Ma et al. falsification
│   ├── attention/          # attention entropy + head classification
│   ├── geometry/           # PCA/UMAP/CKA/silhouette
│   ├── causal/             # SAE feature steering + ablation
│   ├── metacog/            # surprise correlation, difficulty detector
│   ├── viz/                # plotting utilities
│   └── utils/              # shared utilities
├── scripts/                # entry-point scripts
│   ├── extract_all.py
│   ├── run_probes.py
│   ├── run_sae.py
│   ├── run_attention.py
│   ├── run_geometry.py
│   ├── run_causal.py
│   └── smoke_test.py
├── tests/                  # pytest unit tests
├── results/                # analysis outputs (per workstream)
├── figures/                # publication figures
└── notebooks/              # exploratory analysis
```

## Quickstart

```bash
# install
pip install -e ".[sae,dev]"

# smoke test (5-step end-to-end)
python scripts/smoke_test.py

# extract activations (one model)
python scripts/extract_all.py model=llama-3.1-8b-instruct

# run probes
python scripts/run_probes.py

# run SAE analysis
python scripts/run_sae.py
```

## Headline comparison

**Llama-3.1-8B-Instruct vs DeepSeek-R1-Distill-Llama-8B** — same architecture (32 layers, 32 heads, 4096 hidden), different training.

| Model | Type | Params | Layers | Heads | Hidden | SAEs |
|-------|------|--------|--------|-------|--------|------|
| Llama-3.1-8B-Instruct | Standard | 8B | 32 | 32 | 4096 | Llama Scope |
| Gemma-2-9B-IT | Standard | 9B | 42 | 16 | 3584 | Gemma Scope |
| DeepSeek-R1-Distill-Llama-8B | Reasoning | 8B | 32 | 32 | 4096 | Llama Scope (test fit) |
| R1-Distill-Qwen-7B | Reasoning | 7B | 28 | 28 | 3584 | None — train if needed |

## Go/no-go decision points

| Week | Decision | Pass criterion |
|------|----------|----------------|
| 2 | Behavioral validation | Models show >30% lure responses on conflict items |
| 3 | Infrastructure | Activations cached for 2+ models |
| 5 | Probes work | Probe ROC-AUC > 0.6 for S1/S2 classification at some layer |
| 7 | SAE features | Significant differential features after Ma et al. falsification |
| 9 | Full assessment | Commit to final paper scope |

## Key references

- Zhang et al. 2025 — Probing R1 hidden states (arxiv 2504.05419)
- Fartale et al. 2025 — Recall vs reasoning attention (arxiv 2510.03366)
- Ji-An et al. 2025 — Metacognitive monitoring (arxiv 2505.13763)
- Ma et al. 2026 — SAE reasoning feature falsification (arxiv 2601.05679) ⚠️ CRITICAL
- Hagendorff et al. 2023 — CRT for LLMs (Nature Comp Sci) — OSF dataset our benchmark builds on
- Goodfire 2025 — Under the hood of R1 (R1 SAE: backtracking #15204, self-ref #24186)

## License

MIT. See LICENSE.
