# Data Contract: HDF5 Activation Cache Schema

This is the **single source of truth** for the activation file format. Every workstream reads from this format. Do not invent variants.

## File Layout

One HDF5 file per benchmark run: `data/activations/{run_name}.h5`

The `run_name` is typically `main` (full benchmark, all 4 models), `smoke` (5 problems, 1 model), or descriptive like `crt_only_llama`.

## Top-Level Groups

```
{run_name}.h5
├── /metadata                                  # Run-level info (HDF5 attributes)
│   ├── benchmark_path: str                    # Path to benchmark JSONL
│   ├── benchmark_sha256: str                  # Hash of benchmark file
│   ├── created_at: str                        # ISO 8601 timestamp
│   ├── git_sha: str                           # Code revision
│   ├── seed: int                              # Master seed
│   ├── schema_version: int                    # Currently 1
│   └── config: str                            # JSON-serialized Hydra config
│
├── /problems                                  # Per-problem metadata
│   ├── id: (n_problems,) S64                  # Problem ID string
│   ├── category: (n_problems,) S32            # crt | base_rate | syllogism | anchoring | framing | conjunction | arithmetic
│   ├── conflict: (n_problems,) bool           # True = S1 lure present, False = no-conflict control
│   ├── difficulty: (n_problems,) int8         # 1-5
│   ├── prompt_text: (n_problems,) S2048
│   ├── correct_answer: (n_problems,) S128
│   ├── lure_answer: (n_problems,) S128        # Empty for non-conflict items
│   ├── matched_pair_id: (n_problems,) S64     # Links conflict↔control pairs
│   └── prompt_token_count: (n_problems,) int32
│
└── /models                                    # One subgroup per model
    ├── /meta-llama_Llama-3.1-8B-Instruct/
    ├── /google_gemma-2-9b-it/
    ├── /deepseek-ai_DeepSeek-R1-Distill-Llama-8B/
    └── /deepseek-ai_DeepSeek-R1-Distill-Qwen-7B/
```

The model key is the HuggingFace model ID with `/` replaced by `_`.

## Per-Model Subgroup

```
/models/{model_key}/
├── /metadata                                  # Model-level info (HDF5 attributes)
│   ├── hf_model_id: str
│   ├── n_layers: int                          # 32 (Llama), 42 (Gemma), 28 (Qwen)
│   ├── n_heads: int                           # Query heads
│   ├── n_kv_heads: int                        # KV heads (for GQA)
│   ├── hidden_dim: int
│   ├── head_dim: int
│   ├── dtype: str                             # "bfloat16" or "float16"
│   ├── extracted_at: str
│   └── is_reasoning_model: bool
│
├── /residual                                  # Residual stream activations
│   ├── /layer_00: (n_problems, n_positions, hidden_dim) bf16
│   ├── /layer_01: (n_problems, n_positions, hidden_dim) bf16
│   ├── ...
│   └── /layer_NN: (n_problems, n_positions, hidden_dim) bf16   # NN = n_layers - 1
│
├── /position_index                            # Token position labels
│   ├── /labels: (n_positions,) S16            # ["P0", "P2", "T0", "T25", "T50", "T75", "Tend"]
│   ├── /token_indices: (n_problems, n_positions) int32   # Absolute token indices in the full generation
│   └── /valid: (n_problems, n_positions) bool            # False if position not applicable
│
├── /attention                                 # Per-head attention metrics (computed incrementally during extraction)
│   ├── /entropy: (n_problems, n_layers, n_heads, n_positions) float32       # Shannon entropy in bits
│   ├── /entropy_normalized: (n_problems, n_layers, n_heads, n_positions) float32
│   ├── /gini: (n_problems, n_layers, n_heads, n_positions) float32
│   ├── /max_attn: (n_problems, n_layers, n_heads, n_positions) float32
│   ├── /focus_5: (n_problems, n_layers, n_heads, n_positions) float32        # sum of top-5 attention weights
│   └── /effective_rank: (n_problems, n_layers, n_heads, n_positions) float32 # 2^entropy
│
├── /token_surprises                           # Per-token negative log2-prob (in bits)
│   ├── /by_position: (n_problems, n_positions) float32
│   └── /full_trace_offsets: (n_problems + 1,) int64    # CSR-style offsets into full_trace_values
│   └── /full_trace_values: (total_tokens,) float32     # Concatenated per-token surprises for all problems
│
├── /generations                               # Decoded generation strings
│   ├── /full_text: (n_problems,) S8192
│   ├── /thinking_text: (n_problems,) S8192     # Empty for non-reasoning models
│   ├── /answer_text: (n_problems,) S512
│   ├── /thinking_token_count: (n_problems,) int32
│   └── /answer_token_count: (n_problems,) int32
│
└── /behavior                                  # Per-problem behavioral outcome
    ├── /predicted_answer: (n_problems,) S128
    ├── /correct: (n_problems,) bool
    ├── /matches_lure: (n_problems,) bool                      # True if model gave the S1 lure answer
    └── /response_category: (n_problems,) S16                  # "correct" | "lure" | "other_wrong" | "refusal"
```

## Position Labels

| Label | Description | Applicable to |
|-------|-------------|---------------|
| `P0` | Last token of the prompt (pre-generation) | All models |
| `P2` | Final answer token | All models |
| `T0` | First token after `<think>` | Reasoning only |
| `T25` | 25% through thinking trace | Reasoning only |
| `T50` | 50% through thinking trace | Reasoning only |
| `T75` | 75% through thinking trace | Reasoning only |
| `Tend` | Last token before `</think>` | Reasoning only |
| `Tswitch` | First token after `</think>` | Reasoning only |

For non-reasoning models, T-positions exist but `valid=False`. This avoids ragged datasets.

## Layer Indexing

- `layer_00` = output of transformer block 0 (post-MLP, post-residual)
- `layer_NN` = output of the last transformer block (n_layers - 1)
- This matches TransformerLens's `blocks.{i}.hook_resid_post`
- Llama-3.1-8B has 32 layers (0-31). Gemma-2-9B has 42 (0-41). R1-Distill-Qwen-7B has 28 (0-27).

## Memory Footprint

For 400 problems × 7 positions × 4096 hidden dim × bf16:
- Per layer: 400 × 7 × 4096 × 2 = 22.9 MB
- Per model (32 layers): ~735 MB
- All 4 models: ~3 GB residuals
- Plus attention metrics (~800 MB), surprises (~50 MB), generations (~10 MB)
- **Total ~4 GB per run** — trivially storable

## Reading the Format (Python)

```python
import h5py
from s1s2.utils.io import open_activations

with open_activations("data/activations/main.h5") as f:
    # Get residual stream at layer 16 for Llama
    resid = f["/models/meta-llama_Llama-3.1-8B-Instruct/residual/layer_16"][:]
    # shape: (n_problems, n_positions, 4096)

    # Get position labels
    labels = f["/models/meta-llama_Llama-3.1-8B-Instruct/position_index/labels"][:]

    # P0 only
    p0_idx = list(labels).index(b'P0')
    p0_resid = resid[:, p0_idx, :]  # (n_problems, 4096)

    # Per-problem metadata
    is_conflict = f["/problems/conflict"][:]
    behavioral_correct = f["/models/meta-llama_Llama-3.1-8B-Instruct/behavior/correct"][:]
```

A helper module `src/s1s2/utils/io.py` provides typed accessors so workstreams don't need to remember string keys.

## Writing the Format (Python)

The `extract` workstream owns writing. Other workstreams **must not** write back to the activation file. They produce derived results in `results/{workstream}/`.

## Versioning

`schema_version` is currently **1**. To change the schema:
1. Bump the version
2. Add migration logic in `src/s1s2/utils/io.py::migrate()`
3. Update this doc
4. Notify other workstream owners
