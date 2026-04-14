# NeurIPS 2026 OpenReview Submission Metadata

Prepared: 2026-04-12
Submission deadline: TBD (typically mid-May)

---

## Title

What Changes When LLMs Learn to Reason? Causal Mechanistic Signatures of Cognitive Bias Processing

## Abstract

Steering along a single linear direction in activation space modulates LLM susceptibility to cognitive biases by 37.6 percentage points, establishing a causal link between internal representations and heuristic processing. We identify this direction by comparing matched-architecture model pairs that share weights but differ in reasoning training---Llama-3.1-8B-Instruct vs. R1-Distill-Llama-8B, OLMo-3-7B-Instruct vs. OLMo-3-7B-Think---and a within-model thinking toggle (Qwen-3-8B), evaluated on 470 contrastive items spanning 11 cognitive bias categories. Reasoning training dramatically reduces heuristic lure rates: 27.3% to 2.4% (Llama) and 14.9% to 0.9% (OLMo). Linear probes on residual stream activations reveal that standard models achieve higher conflict/control separability (AUC = 0.974 [0.952, 0.992]) than reasoning models (AUC = 0.930 [0.894, 0.960])---the S1-like/S2-like boundary is blurred, not sharpened, by training. Cross-prediction on immune categories (AUC = 0.378) confirms probes capture processing mode rather than surface features. Crucially, causal interventions via probe-direction steering reduce the lure rate by 21 percentage points (31.2% at alpha=+5 vs. 52.5% baseline); random directions produce no effect. Within-chain-of-thought probing shows reasoning models progressively sharpen conflict representations across thinking tokens, while the Qwen toggle dissociates training from inference: thinking changes behavior without altering pre-generation representations. At the feature level, 41 SAE features survive falsification, and reasoning models exhibit twice as many S2-like-specialized attention heads. We evaluate at scales up to 32B parameters, finding the same training-induced blurring pattern. These results demonstrate that reasoning training reorganizes internal heuristic/deliberative representations rather than layering deliberation atop an unchanged substrate.

**Word count: 239**

## TL;DR

Probe-direction steering causally modulates cognitive bias susceptibility in LLMs (37.5pp swing), and reasoning training decouples this direction from behavioral control.

**Character count: 159**

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
- Estimated total compute: [TO BE FILLED before submission]

## Conflicts of Interest

### Institutional Conflicts

- Harvard University (author affiliation)
- [ADD any advisor/co-author institution conflicts before submission]

### Other Conflicts

- None known at this time

## Author Information

### Author 1

- **Name:** Bright Liu
- **Email:** [TO BE FILLED]
- **Affiliation:** Harvard University
- **Role:** Lead author

### Co-authors

- **Status:** TO BE FINALIZED before submission
- [ADD co-authors here with name, email, affiliation, and OpenReview profile URL]

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
