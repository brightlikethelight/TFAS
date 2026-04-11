# Agent Conventions (Codex CLI / general)

Codex CLI sessions: see `CLAUDE.md` for the canonical project conventions. They apply equally to Codex.

When invoking Codex as a code reviewer (Bright's writer-reviewer pattern), focus on:

1. **Statistical correctness** — BH-FDR application, permutation test framing, bootstrap sample reuse
2. **Hewitt & Liang controls** — actually run, not just claimed
3. **Ma et al. SAE falsification** — actually applied to candidate features
4. **Confound controls** — matched-difficulty subsetting, GQA non-independence
5. **Off-by-one in probe layer indexing** — TransformerLens vs HuggingFace conventions differ
6. **Gemma-2 sliding window separation** — odd vs even layers
7. **Memory leaks in activation extraction** — forgotten hooks, retained graphs, ungated `output_attentions`
8. **Reasoning model thinking trace handling** — `<think>...</think>` parsing must be robust to truncation

Run `codex exec review --uncommitted` after any non-trivial change.
