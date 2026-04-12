# The Deliberation Gradient: Mechanistic Signatures of Dual-Process Cognition in LLMs

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-427%20passed-brightgreen.svg)]()

Code, data, and analysis pipeline for the paper *"The Deliberation Gradient: Mechanistic Signatures of Dual-Process Cognition in LLMs."*
We investigate whether reasoning-trained LLMs develop internal representations that distinguish fast, heuristic-driven (S1-like) processing from slow, deliberative (S2-like) processing, and how reasoning training reshapes these representations.
The key finding: reasoning training **blurs** the internal boundary between processing modes (AUC 0.999 to 0.929) while dramatically improving behavioral accuracy, revealing a dissociation between representation and behavior.

**[Paper (PDF)](paper/workshop_paper.pdf)**

## Key Results

| Finding | Metric | Value |
|---------|--------|-------|
| Behavioral improvement | Lure rate (conflict items) | 27.3% &rarr; 2.4% |
| Representational blurring | Probe AUC (S1/S2 classification) | 0.999 &rarr; 0.929 |
| Processing-mode specificity | Cross-prediction accuracy | 0.378 (low &rarr; probes track processing mode, not surface features) |
| Training vs. inference dissociation | Training changes encoding; inference-time reasoning changes behavior | Confirmed across model pairs |

## Quick Start

```bash
# Clone and install
git clone https://github.com/HUSAI/s1s2.git
cd s1s2
pip install -e ".[dev]"

# Run the test suite (427 tests)
make test

# Quick validation on synthetic data (no GPU required)
make smoke
```

Optional extras for SAE analysis and the interactive dashboard:

```bash
pip install -e ".[sae]"       # SAE-Lens, TransformerLens, nnsight
pip install -e ".[dashboard]" # Gradio dashboard
```

## Benchmark

The **S1/S2 Cognitive Bias Benchmark** is a 470-item evaluation suite spanning 11 cognitive-bias categories across 4 heuristic families (anchoring, framing, base-rate neglect, attribute substitution).
Each conflict item is paired with a no-conflict control matched on surface form and difficulty, enabling clean S1/S2 contrasts.

- **Location:** `data/benchmark/benchmark.jsonl`
- **Format:** One JSON object per line with fields for `item_id`, `category`, `heuristic_family`, `conflict`, `prompt`, `correct_answer`, and `lure_answer`.

```python
from s1s2.benchmark.loader import load_benchmark

items = load_benchmark()                          # all 470 items
conflict_items = load_benchmark(conflict_only=True)  # S1-triggering items only
```

Regenerate from templates:

```bash
make benchmark
```

## Reproducing Results

### 1. Extract activations

Cache hidden states and attention patterns into HDF5 for all models:

```bash
# Single model
python scripts/extract_real.py --model meta-llama/Llama-3.1-8B-Instruct

# All configured models
make extract
```

Activations are saved to `data/activations/` (gitignored; ~2-8 GB per model depending on sequence length).

### 2. Run probes with bootstrap confidence intervals

```bash
python scripts/compute_bootstrap_cis.py --h5-path data/activations/<run>.h5
```

Or run all five analysis workstreams:

```bash
make all-analyses   # probes, sae, attention, geometry, causal, metacog
```

### 3. Generate paper figures

```bash
python scripts/make_paper_figures.py
# Or:
make figures
```

Figures are written to `figures/` as PDF and PNG.

## Project Structure

```
s1s2/
├── src/s1s2/            # Core library
│   ├── benchmark/       # 470-item cognitive bias benchmark
│   ├── probes/          # Linear probing with Hewitt-Liang control tasks
│   ├── sae/             # SAE feature analysis with Ma et al. falsification
│   ├── attention/       # Per-head attention entropy analysis
│   ├── geometry/        # Representational geometry (CKA, silhouette, UMAP)
│   ├── causal/          # Causal interventions (activation patching)
│   ├── extract/         # Activation extraction and HDF5 caching
│   ├── metacog/         # Metacognitive monitoring (stretch goal)
│   ├── viz/             # Shared visualization utilities
│   ├── dashboard/       # Interactive Gradio exploration dashboard
│   └── utils/           # Seeding, config, W&B helpers
├── scripts/             # CLI entry points for extraction, analysis, figures
├── configs/             # Hydra YAML configs (models, workstreams)
├── paper/               # Workshop paper LaTeX source
├── data/                # Benchmark JSONL and activation caches (gitignored)
├── results/             # Per-workstream JSON results (gitignored)
├── figures/             # Generated figures (gitignored)
└── tests/               # 427 tests (unit, integration, end-to-end)
```

Each analysis workstream follows a uniform module layout:

```
<workstream>/
├── __init__.py    # Public API
├── core.py        # Main analysis logic
├── data.py        # HDF5 I/O
├── stats.py       # Statistical tests (BH-FDR, permutation, bootstrap)
├── viz.py         # Plotting
└── cli.py         # Hydra CLI entry point
```

## Models

All models are 7-8B parameters. Matched pairs share the same base architecture, isolating the effect of reasoning training.

| Key | HuggingFace ID | Layers | Hidden | Reasoning | Matched With |
|-----|----------------|--------|--------|-----------|--------------|
| `llama-3.1-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | 32 | 4096 | No | `r1-distill-llama-8b` |
| `r1-distill-llama-8b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 32 | 4096 | Yes | `llama-3.1-8b-instruct` |
| `gemma-2-9b-it` | `google/gemma-2-9b-it` | 42 | 3584 | No | -- |
| `r1-distill-qwen-7b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 28 | 3584 | Yes | -- |
| `ministral-3-8b-instruct` | `mistralai/Ministral-3-8B-Instruct-2512` | 32 | 4096 | No | `ministral-3-8b-reasoning` |
| `ministral-3-8b-reasoning` | `mistralai/Ministral-3-8B-Reasoning-2512` | 32 | 4096 | Yes | `ministral-3-8b-instruct` |
| `olmo-3-7b-instruct` | `allenai/OLMo-3-7B-Instruct` | -- | -- | No | `olmo-3-7b-think` |
| `olmo-3-7b-think` | `allenai/OLMo-3-7B-Think` | -- | -- | Yes | `olmo-3-7b-instruct` |

Model configs live in `configs/models.yaml`. Scripts reference models by key, never by hardcoded HuggingFace ID.

## Statistical Methodology

All results follow strict statistical standards:

- **Multiple comparisons:** Benjamini-Hochberg FDR at q = 0.05
- **Confidence intervals:** Bootstrap with 1,000 resamples
- **Permutation tests:** 1,000-10,000 shuffles with North et al. (2002) correction
- **Cross-validation:** 5-fold stratified by category and label
- **Probe controls:** Hewitt & Liang (2019) selectivity; real - random-label accuracy must exceed 5 pp
- **SAE controls:** Ma et al. (2026) falsification of spurious features
- **Reproducibility:** All random seeds fixed via `s1s2.utils.seed.set_global_seed()`; Hydra configs saved alongside results

## Citation

```bibtex
@inproceedings{liu2026deliberation,
  title     = {The Deliberation Gradient: Mechanistic Signatures of
               Dual-Process Cognition in {LLMs}},
  author    = {Liu, Bright and {HUSAI S1/S2 Team}},
  booktitle = {ICML 2026 Workshop on Mechanistic Interpretability},
  year      = {2026},
}
```

## License

MIT License. Copyright (c) 2026 HUSAI (Harvard Undergraduate AI Safety).

See [LICENSE](LICENSE) for full text.
