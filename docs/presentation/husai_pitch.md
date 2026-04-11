---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
  }
  h1 {
    color: #1a1a2e;
  }
  h2 {
    color: #16213e;
  }
  blockquote {
    border-left: 4px solid #e94560;
    padding-left: 1em;
    color: #555;
  }
---

# The Deliberation Gradient

### Mechanistic Signatures of Dual-Process Cognition in LLMs

**HUSAI Research Project -- Spring/Summer 2026**

> When you solve "2 + 2," your brain barely activates.
> When you solve "347 x 28," something different happens inside.
> Does the same thing happen inside an LLM?

---

# The Question

## When an LLM "thinks carefully," does something change inside?

Reasoning models (DeepSeek-R1, o1) produce long chains of thought -- "wait, let me reconsider..."

But is that just **theater**, or is there a real **mechanistic shift** in how the model processes information?

- Only ~2.3% of chain-of-thought reasoning steps causally influence the final answer
- Models trained to "reason" score higher on hard problems
- **Nobody has checked whether the internal computation actually changes**

We will.

---

# Dual-Process Theory in 60 Seconds

| | S1-like Processing | S2-like Processing |
|---|---|---|
| **Speed** | Fast, automatic | Slow, effortful |
| **Mode** | Pattern matching, heuristics | Step-by-step deliberation |
| **Errors** | Systematic biases | Correct but costly |
| **Human example** | "A bat and ball cost $1.10..." -> "$0.10!" | "Wait... let me do the algebra." |

**Critical framing**: we treat this as a *continuous gradient of deliberation intensity*, not a binary switch. Even psychologists have abandoned the binary (Melnikoff & Bargh 2018).

We use S1/S2 as an operational shorthand, not a cognitive claim about LLMs.

---

# What We're Testing

> **Can we see S1-like and S2-like processing mechanistically -- not just in model outputs, but in internal representations?**

Three specific hypotheses:

1. There exist **linearly decodable directions** in activation space that distinguish heuristic from deliberative processing

2. **Sparse autoencoder features** activate differentially on bias-triggering vs. control problems

3. Reasoning-trained models show **amplified internal distinctions** compared to standard models with the same architecture

---

# The Approach: Five Complementary Methods

| Method | What it measures | Priority |
|--------|-----------------|----------|
| **Linear Probes** | Is the S1/S2 distinction linearly decodable at each layer? | Must-do |
| **SAE Feature Analysis** | Which sparse features activate differentially? | Must-do (most novel) |
| **Attention Entropy** | Does attention focus shift between conditions? | Should-do |
| **Representational Geometry** | Do S1/S2 activations occupy separable subspaces? | Should-do |
| **Causal Interventions** | Can we steer models from S1-like to S2-like behavior? | Validation |

Each method has built-in controls and falsification tests. No hand-waving.

---

# The Headline Comparison

## Same brain, different training

**Llama-3.1-8B-Instruct** vs. **DeepSeek-R1-Distill-Llama-8B**

- Same architecture: 32 layers, 32 heads, 4096 hidden dim
- Different training: standard instruction-tuning vs. reasoning distillation
- Any internal difference is attributable to reasoning training alone

Plus cross-architecture validation on Gemma-2-9B-IT and R1-Distill-Qwen-7B.

Pre-trained SAEs available for both architecture families (Llama Scope, Gemma Scope) -- no need to train our own.

---

# The Benchmark

## 284 cognitive bias problems across 7 categories

| Category | What it tests |
|----------|--------------|
| CRT Variants | Reflexive vs. reflective math (novel items -- classics are memorized) |
| Base Rate Neglect | Ignoring prior probabilities |
| Belief-Bias Syllogisms | Logic vs. believability |
| Anchoring | Numerical anchoring effects |
| Framing Effects | Gain vs. loss framing |
| Conjunction Fallacy | "Linda problem" structural isomorphs |
| Multi-Step Arithmetic | Graded difficulty with carrying traps |

Every conflict item has a **matched no-conflict control** (same surface form, no S1/S2 divergence). This is how we rule out the difficulty confound.

---

# Why This Matters for AI Safety

**If models can pattern-match their way to correct answers without genuine deliberation, we need to know.**

- **Trust calibration**: Should we trust a model's answer on a novel problem if it's just retrieving patterns?
- **Oversight**: Knowing *when* models actually reason tells us *when* oversight matters most
- **Deceptive alignment**: A model that appears to reason but doesn't is harder to evaluate
- **Reasoning training evaluation**: Did distillation actually change computation, or just output format?

This is one of the few projects that can produce mechanistic evidence -- not just behavioral observations -- about how reasoning training changes model internals.

---

# Timeline and Roles

**14 weeks, 8-10 people, 6-10 hrs/week**

| Phase | Weeks | Focus |
|-------|-------|-------|
| Foundation | 1-2 | Benchmark, activation caching, behavioral validation |
| Core Analysis | 3-6 | Probes, attention entropy, SAE features |
| Validation | 7-8 | Causal interventions, robustness checks |
| Writing | 9-13 | Paper drafting, figures, revisions |
| Submission | 14 | Camera-ready |

**Roles**: Project Lead, Benchmark Lead, Infrastructure Lead, Probes + Geometry (2), SAE + Causal (2), Attention + Metacog (1-2), Writing Lead

Go/no-go gates at weeks 2, 3, 5, 7, and 9 -- we scope to what works.

---

# Join Us

## What you'll learn
- Mechanistic interpretability (probing, SAEs, activation analysis)
- Research-grade statistical methods (permutation tests, FDR correction, bootstrap CIs)
- PyTorch internals (hooks, activation caching, HPC workflows)
- Academic paper writing and publication process

## What we'll produce
- **June 2026**: Alignment Forum post (establishes priority)
- **July 2026**: ICML MechInterp Workshop submission
- **October 2026**: ICLR 2027 full conference submission

## Prerequisites
Python + basic ML knowledge. No mech interp experience required -- we'll teach you.

**Interested? Talk to us after the meeting or reach out: [husai-s1s2@harvard.edu]**
