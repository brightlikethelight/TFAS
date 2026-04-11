# Team Roles -- The Deliberation Gradient

**HUSAI Research Project -- Spring/Summer 2026**

7-9 people across 7 roles. Some roles can be shared. All roles contribute to the final paper as co-authors.

---

## 1. Project Lead

**People**: 1

**Responsibilities**
- Overall project coordination and timeline management
- Advisor communication and meeting scheduling
- Scoping decisions at go/no-go gates (weeks 2, 3, 5, 7, 9)
- Cross-workstream integration -- ensuring methods produce compatible results
- Final paper scope decisions and submission logistics
- Running weekly full-team syncs

**Time commitment**: 8-10 hrs/week (higher in weeks 1-2 and 9-14)

**Skills needed**: ML research experience, project management, clear communication. Must be comfortable making scoping calls under uncertainty.

**Skills you'll gain**: Research leadership, cross-functional coordination, publication strategy, stakeholder management.

**Difficulty**: 3/3 stars

---

## 2. Benchmark Lead

**People**: 1 (can pair with a second person for item generation)

**Responsibilities**
- Curating and validating 284+ cognitive bias problems across 7 categories
- Sourcing from Hagendorff et al. OSF dataset, generating novel structural isomorphs
- Implementing conflict/no-conflict matched pair design
- Prompt engineering for constrained free-response and multiple-choice formats
- Running behavioral validation (week 2 go/no-go gate)
- Contamination testing: checking whether models have memorized classic problems
- Maintaining the three-tier scoring system (binary, categorical, continuous)

**Time commitment**: 8-10 hrs/week in weeks 1-3, then 4-6 hrs/week

**Skills needed**: Careful attention to detail, basic statistics (for Bayesian verification of base-rate problems), willingness to read cognitive science literature.

**Skills you'll gain**: Experimental design, benchmark construction, prompt engineering, understanding of cognitive biases and dual-process theory.

**Difficulty**: 1/3 stars (labor-intensive but conceptually accessible)

---

## 3. Infrastructure Lead

**People**: 1

**Responsibilities**
- Building and maintaining the activation extraction pipeline (`src/s1s2/extract/`)
- HDF5 caching system per the data contract (`docs/data_contract.md`)
- SAE loading and reconstruction fidelity verification
- HPC setup (FASRC or RunPod) -- job scheduling, checkpointing, data transfer
- Ensuring all workstreams can read from the shared activation cache
- Smoke testing end-to-end pipeline on small subsets before full runs

**Time commitment**: 10 hrs/week in weeks 1-4 (critical path), then 6 hrs/week

**Skills needed**: Strong Python/PyTorch engineering. Comfortable with GPU workflows, HuggingFace Transformers, and debugging CUDA/memory issues. This is the highest-leverage engineering role.

**Skills you'll gain**: Large-scale ML infrastructure, HPC workflows, activation caching architecture, SAE internals, data pipeline design.

**Difficulty**: 3/3 stars (everyone is blocked without this working)

---

## 4. Probes + Geometry Team

**People**: 2

**Responsibilities**
- Implementing linear probes (logistic regression, MLP, CCS) across all layers and token positions
- Hewitt & Liang control tasks (random-label baselines) -- non-negotiable
- Leave-one-category-out cross-validation
- Difficulty-matched subsetting
- Self-correction trajectory probing (for reasoning models)
- Representational geometry: PCA, UMAP, random projection baselines
- Layer-wise silhouette scores with bootstrap CIs
- CKA analysis between matched model pairs

**Time commitment**: 6-8 hrs/week

**Skills needed**: Comfort with scikit-learn, basic linear algebra (PCA, logistic regression). One person should be comfortable with statistical testing.

**Skills you'll gain**: Linear probing methodology, representational similarity analysis, CKA, rigorous evaluation with proper controls, scientific figure generation.

**Difficulty**: 2/3 stars

**Suggested split**: one person focuses on probes (`src/s1s2/probes/`), the other on geometry (`src/s1s2/geometry/`). Both contribute to each other's code review.

---

## 5. SAE + Causal Team

**People**: 2

**Responsibilities**
- Loading pre-trained SAEs (Llama Scope, Gemma Scope) and verifying reconstruction fidelity
- Differential feature activation analysis (conflict vs. no-conflict)
- Volcano plot generation (log fold change vs. -log10 p-value)
- Ma et al. falsification protocol on all candidate features -- mandatory
- Causal interventions: SAE feature ablation and amplification
- Random feature ablation controls
- Dose-response curves (steering coefficient 0.5 to 5.0)
- Side-effect monitoring (MMLU/HellaSwag capability checks)
- If needed: training custom SAEs via SAELens for R1-Distill models

**Time commitment**: 6-10 hrs/week (higher if custom SAE training needed)

**Skills needed**: Comfort with PyTorch, willingness to learn SAE internals. One person should be comfortable with statistical testing (Mann-Whitney U, BH-FDR correction).

**Skills you'll gain**: Sparse autoencoder analysis, causal intervention methodology, feature falsification, mechanistic interpretability research skills, SAELens tooling.

**Difficulty**: 3/3 stars (most technically novel component)

**Suggested split**: one person focuses on feature analysis (`src/s1s2/sae/`), the other on causal interventions (`src/s1s2/causal/`).

---

## 6. Attention + Metacognition

**People**: 1-2

**Responsibilities**
- Computing attention entropy metrics (Shannon, normalized, Gini) per head per position
- Handling GQA non-independence (report at query-head and KV-group granularity)
- Handling Gemma-2 sliding window layers (odd vs. even layer separation)
- Incremental entropy computation inside forward hooks (never materializing full attention matrices)
- Head classification following Fartale et al. (S1-specialized vs. S2-specialized)
- Temporal entropy dynamics over generation
- Stretch goal -- metacognitive monitoring: surprise correlation, difficulty-sensitive SAE features, "confidently wrong" test

**Time commitment**: 6-8 hrs/week

**Skills needed**: Comfortable with PyTorch hooks and tensor operations. Understanding of attention mechanism internals helpful but not required.

**Skills you'll gain**: Attention mechanism analysis, information-theoretic metrics, incremental computation in hooks, GQA architecture understanding, metacognitive monitoring methods.

**Difficulty**: 2/3 stars (attention) to 3/3 stars (metacognition stretch goal)

---

## 7. Writing Lead

**People**: 1

**Responsibilities**
- Drafting Introduction, Related Work, and Benchmark sections (weeks 1-4, before results)
- Methods sections for each workstream (in coordination with workstream leads)
- Results and Discussion drafting once data comes in (weeks 7-9)
- Figure design and polishing (weeks 10-13)
- Managing the supplement (geometry, metacog, per-task breakdowns, robustness checks)
- Internal review coordination (circulating drafts, collecting feedback)
- Alignment Forum post drafting (June 2026)
- Conference submission formatting and logistics (ICML workshop July, ICLR October)

**Time commitment**: 4-6 hrs/week in weeks 1-8, then 8-10 hrs/week in weeks 9-14

**Skills needed**: Strong scientific writing. Ability to synthesize technical results into coherent narrative. Must be someone other than the Project Lead (to distribute workload).

**Skills you'll gain**: Academic paper writing, LaTeX/Overleaf, figure design, conference submission process, scientific storytelling.

**Difficulty**: 2/3 stars

---

## Role Summary

| Role | People | Peak Load | Difficulty | Key Codebase Area |
|------|--------|-----------|------------|-------------------|
| Project Lead | 1 | Weeks 1-2, 9-14 | 3/3 | All (coordination) |
| Benchmark Lead | 1 | Weeks 1-3 | 1/3 | `src/s1s2/benchmark/`, `data/benchmark/` |
| Infrastructure Lead | 1 | Weeks 1-4 | 3/3 | `src/s1s2/extract/`, `data/activations/` |
| Probes + Geometry | 2 | Weeks 3-8 | 2/3 | `src/s1s2/probes/`, `src/s1s2/geometry/` |
| SAE + Causal | 2 | Weeks 5-9 | 3/3 | `src/s1s2/sae/`, `src/s1s2/causal/` |
| Attention + Metacog | 1-2 | Weeks 3-8 | 2-3/3 | `src/s1s2/attention/`, `src/s1s2/metacog/` |
| Writing Lead | 1 | Weeks 9-14 | 2/3 | `docs/`, `figures/` |

**Total**: 8-10 people. Some individuals may hold multiple roles if the team is smaller (e.g., Project Lead + Writing Lead, though this is discouraged).
