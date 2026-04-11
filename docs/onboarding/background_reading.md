# Background Reading -- The Deliberation Gradient

Curated reading list organized by priority. Start at the top and work
down. The "must-read" tier covers what you need to understand before
writing your first line of code.

---

## Tier 1: Must-read (before your first contribution)

These three readings give you the conceptual frame (dual-process
theory), the empirical foundation (our benchmark), and the
methodological toolkit (mechanistic interpretability).

### 1. Kahneman -- Thinking, Fast and Slow (2011), Chapters 1-3

The origin of the S1/S2 framework. Chapter 1 introduces the
two-systems metaphor, Chapter 2 covers attention and effort, Chapter
3 covers the lazy controller. If you can't read the book, a 10-minute
summary of the two-systems idea is sufficient -- but know that
**even psychologists have moved past the strict binary** (Melnikoff &
Bargh 2018). Our project treats S1/S2 as a graded operational
distinction, not a literal claim about LLM cognition.

- **Book**: Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar,
  Straus and Giroux.

### 2. Hagendorff et al. 2023 -- Human-like intuitive behavior and reasoning biases emerged in large language models but disappeared in ChatGPT

This is the paper our benchmark builds on. Hagendorff et al. tested
GPT-3.5 and GPT-4 on classic cognitive bias tasks (CRT, base-rate
neglect, conjunction fallacy, etc.) and found that larger models
exhibit more human-like biases. Their OSF dataset is the seed for our
benchmark; we extend it with novel structural isomorphs to reduce
contamination.

- **Paper**: Hagendorff, T., Fabi, S., & Kosinski, M. (2023).
  Human-like intuitive behavior and reasoning biases emerged in large
  language models but disappeared in ChatGPT. *Nature Computational
  Science*, 3(10), 833-838.
  https://doi.org/10.1038/s43588-023-00527-x
- **Data**: https://osf.io/s7a4b/

### 3. One mechanistic interpretability primer (pick one)

You need a working understanding of what it means to "look inside" a
transformer. Pick whichever format suits you:

- **Option A**: Anthropic. (2023). Towards Monosemanticity:
  Decomposing Language Models With Dictionary Learning. Anthropic
  Research Blog.
  https://transformer-circuits.pub/2023/monosemantic-features
  - The foundational SAE paper. Explains why neurons are polysemantic,
    how sparse autoencoders decompose them into interpretable
    features, and what "feature" means in this context.

- **Option B**: Nanda, N. (2023). A Comprehensive Mechanistic
  Interpretability Explainer & Glossary. ARENA.
  https://www.neelnanda.io/mechanistic-interpretability/glossary
  - A practical walkthrough of residual streams, attention heads,
    superposition, and the basic toolkit. Good if you prefer
    a "how to actually do MI" angle over "why SAEs work."

---

## Tier 2: Should-read (within first 2 weeks)

These papers cover the specific methodologies we implement.

### 4. Zhang et al. 2025 -- Probing the hidden reasoning processes of DeepSeek-R1

Probes the hidden states of DeepSeek-R1 across its thinking trace to
show that the model's internal confidence about the answer evolves
during chain-of-thought. Directly motivates our trajectory probing
(T0 -> T25 -> T50 -> T75 -> Tend positions).

- **Paper**: Zhang, S., Zhao, C., & Gao, J. (2025). Probing the
  hidden reasoning processes of DeepSeek-R1. arXiv:2504.05419.
  https://arxiv.org/abs/2504.05419

### 5. Fartale et al. 2025 -- Disentangling Recall and Reasoning in LLMs

Uses attention analysis to separate "recall" heads (retrieve from
training data) from "reasoning" heads (compose novel answers).
Directly motivates our attention entropy workstream and the
per-head classification into S1-specialized vs S2-specialized heads.

- **Paper**: Fartale, S., Neagu, R., Abrudan, T., & Leordeanu, M.
  (2025). Disentangling Recall and Reasoning in Transformers.
  arXiv:2510.03366.
  https://arxiv.org/abs/2510.03366

### 6. Ma et al. 2026 -- Reassessing SAE Reasoning Features in Large Language Models

**THE most critical methodology paper for this project.** Ma et al.
demonstrated that 45-90% of SAE features previously claimed to encode
"reasoning" are spurious -- they fire on specific tokens ("Let",
"First", "wait") rather than on reasoning processes. We implement
their falsification test as a mandatory filter on all candidate
features. Read the abstract and Section 3 (the falsification
protocol) at minimum.

- **Paper**: Ma, Z., Liu, Y., & Chen, X. (2026). Reassessing SAE
  Reasoning Features in Large Language Models. arXiv:2601.05679.
  https://arxiv.org/abs/2601.05679
- **Code in our repo**: `src/s1s2/sae/falsification.py`

### 7. Hewitt & Liang 2019 -- Designing and Interpreting Probes with Control Tasks

The standard methodology for testing whether probe accuracy reflects
genuine information in the representation vs. the probe's own
expressiveness. Our probing workstream implements their random-label
control task, and selectivity (real accuracy - control accuracy) is
the metric we report, not raw accuracy.

- **Paper**: Hewitt, J., & Liang, P. (2019). Designing and
  Interpreting Probes with Control Tasks. *EMNLP*.
  https://arxiv.org/abs/1909.03368

---

## Tier 3: Nice-to-read (ongoing reference)

These papers provide deeper context for specific workstreams. Read
them as you need them.

### 8. Burns et al. 2022 -- Discovering Latent Knowledge in Language Models Without Supervision (CCS)

Contrast-consistent search -- an unsupervised probe that finds
directions in activation space corresponding to truth/falsehood
without labeled data. We implement CCS as one of our probe types.

- **Paper**: Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2022).
  Discovering Latent Knowledge in Language Models Without Supervision.
  arXiv:2212.03827.
  https://arxiv.org/abs/2212.03827

### 9. Kornblith et al. 2019 -- Similarity of Neural Network Representations Revisited (CKA)

Centered Kernel Alignment -- the metric we use to compare
representational geometry across model pairs (e.g., Llama-Instruct
vs. DeepSeek-R1-Distill-Llama). High CKA = similar internal
representations despite different training.

- **Paper**: Kornblith, S., Norouzi, M., Lee, H., & Hinton, G.
  (2019). Similarity of Neural Network Representations Revisited.
  *ICML*. arXiv:1905.00414.
  https://arxiv.org/abs/1905.00414

### 10. Ji-An et al. 2025 -- Metacognitive Monitoring in Large Language Models

Measures whether LLMs have calibrated confidence -- do they "know what
they know"? Motivates our metacognitive monitoring stretch goal
(surprise correlation, "confidently wrong" test).

- **Paper**: Ji-An, L., Liu, Y., & Ranganath, R. (2025).
  Metacognitive Monitoring in Large Language Models.
  arXiv:2505.13763.
  https://arxiv.org/abs/2505.13763

### 11. Goodfire 2025 -- Under the Hood of R1

Blog post applying Goodfire's SAE to DeepSeek-R1, identifying a
"backtracking" feature (#15204) and a "self-reference" feature
(#24186). Motivates our use of the Goodfire R1 SAE and the causal
intervention experiments.

- **Blog**: Goodfire. (2025). Under the Hood of R1.
  https://goodfire.ai/blog/under-the-hood-of-r1

### 12. Coda-Forno et al. 2025 -- Representational overlap between dual architectures

Found overlapping (not cleanly separable) subspaces in dual-process
neural architectures. This is why we use a graded framework rather
than binary S1/S2.

- **Paper**: Coda-Forno, J., Binz, M., & Schulz, E. (2025). On the
  role of attention in next-token prediction with transformers.

### 13. Ziabari et al. 2025 -- Reasoning on a Spectrum

Showed monotonic interpolation between System 1 and System 2
endpoints across model scales and training regimes. Further
motivation for the continuous deliberation-intensity framing.

---

## Reading strategy

If you have limited time, the minimum viable reading list is:

1. A summary of S1/S2 dual-process theory (15 min)
2. Hagendorff et al. 2023 abstract + results (10 min)
3. Ma et al. 2026 abstract + Section 3 (15 min)
4. `CLAUDE.md` in the repo (15 min) -- yes, this counts as required reading

That gives you the experimental question, the benchmark, the critical
methodological guardrail, and the coding conventions.
