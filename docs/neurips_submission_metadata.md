# NeurIPS 2026 OpenReview Submission Metadata

Prepared: 2026-05-03 (updated)
Abstract deadline: May 4, 2026 AOE
Full paper deadline: May 6, 2026 AOE

---

## Title

Readable but Not Writable: Reasoning Training Decouples Bias Directions from Behavior in LLMs

## Abstract

Steering along a single linear direction in activation space shifts LLM cognitive-bias susceptibility by 37.5 percentage points, but only in standard models. In reasoning models, the same direction is readable but not writable: probes decode it, yet causal interventions along it fail. This dissociation is our central finding. We compare matched-architecture pairs---Llama-3.1-8B-Instruct vs. R1-Distill-Llama-8B, OLMo 7B/32B Instruct vs. Think---and Qwen-3-8B's thinking toggle, across 470 contrastive items spanning 11 bias categories. Reasoning training collapses heuristic lure rates: 27.3% to 2.4% (Llama), 19.6% to 0.4% at 32B. Activation probes far exceed text baselines (AUC 0.974/0.930 vs. 0.820--0.863 across three text methods), confirming the signal lives in geometry, not surface tokens. Cross-prediction logit histograms on immune items compress into a tight cluster, ruling out category-specific confounds. Probe-direction steering produces a 37.5pp dose-response in Llama (lure to correct, 100% converting to correct), while R1---steered continuously across 2048 tokens at four layers including its probe peak---shows no coherent dose-response, ruling out temporal washout and a deeper writable locus. Within-chain-of-thought probing reveals a non-monotonic trajectory (T0=0.973, T75=0.754, Tend=0.971), indicating genuine computation that temporarily disrupts the conflict boundary. Qwen's thinking toggle yields a further dissociation: both modes achieve high separability (~0.97) yet cross-mode probe transfer is at chance (AUC 0.496). Scaling to 32B amplifies standard-model vulnerability (14.9% to 19.6%) while reasoning training remains effective (0.4%). These results show that reasoning training reorganizes bias representations into a read-only substrate that resists external steering.

**Word count: 243**

## TL;DR

Probe-direction steering shifts cognitive bias susceptibility by 37.5pp in standard LLMs but fails in reasoning models—the bias direction is readable but not writable after reasoning training.

**Character count: 193**

## Keywords

1. mechanistic interpretability
2. dual-process cognition
3. cognitive bias
4. reasoning models
5. linear probes
6. activation steering
7. sparse autoencoders
8. representational geometry

## Track

Main Track

## Primary Area

Interpretability and explainability

**Backup options (if "Interpretability and explainability" is not available):**
- Representation learning
- Cognitive science and AI

## Supplementary Materials

- Paper + appendix in a single PDF
- Code repository link (to be released upon acceptance)

## Ethics Statement

### Human Subjects

None. All experiments use publicly available language models and synthetic benchmark items. No human data was collected.

### AI Assistance Disclosure

We used AI coding assistants (Claude Code) for implementation, experiment orchestration, and draft editing. All scientific hypotheses, experimental designs, and interpretations are by the authors.

### Compute Resources

- Primary: RunPod B200 GPU instances
- Estimated total compute: ~120 GPU-hours (B200), ~50 CPU-hours (analysis/probing)

## Conflicts of Interest

### Institutional Conflicts

- Harvard University (author affiliation)
- [ADD any advisor/co-author institution conflicts before submission]

### Other Conflicts

- None known at this time

## Author Information

### Author 1

- **Name:** Bright Liu
- **Email:** brightliu@college.harvard.edu
- **Affiliation:** Harvard University
- **Role:** Lead author

### Co-authors

- **Status:** Pending advisor confirmation (add before May 4 abstract deadline)

---

## Pre-Submission Checklist

- [ ] Finalize co-author list and get all OpenReview profile URLs
- [ ] Fill in email addresses
- [ ] Estimate and record total compute hours
- [ ] Verify abstract is within 250-word limit (currently 239)
- [ ] Verify TL;DR is within 250-character limit (currently 159)
- [ ] Confirm primary area selection matches NeurIPS 2026 dropdown options
- [ ] Prepare single PDF with paper + appendix
- [ ] Prepare code repository for release (anonymized if needed for review)
- [ ] Review NeurIPS 2026 style guidelines and formatting requirements
- [ ] Complete OpenReview author registration for all authors
