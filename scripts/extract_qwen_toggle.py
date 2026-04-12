#!/usr/bin/env python3
"""Extract activations from Qwen3-8B in both thinking modes.

The CLEANEST S1/S2 comparison: same model, same weights, just
enable_thinking=True vs enable_thinking=False. No distillation confound,
no architecture confound, no weight difference. Pure mode toggle.

For each mode we extract residual stream at P0 (last prompt token) for all layers.
For thinking ON, we additionally extract at T0 (first token after <think>) and
Tend (last token before </think>).

Outputs:
    data/activations/qwen3_8b_nothink.h5  (positions: P0, P2)
    data/activations/qwen3_8b_think.h5    (positions: P0, P2, T0, Tend)

Usage (B200 pod):
    python scripts/extract_qwen_toggle.py --mode nothink
    python scripts/extract_qwen_toggle.py --mode think
    python scripts/extract_qwen_toggle.py --mode both
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"

# Qwen3-8B architecture constants for reference
# n_layers=36, hidden_dim=4096, n_heads=32, n_kv_heads=8


def parse_answer(text: str, correct: str, lure: str) -> str:
    """Classify the generated answer as correct, lure, or other."""
    if correct and re.search(re.escape(correct), text):
        return "correct"
    if lure and re.search(re.escape(lure), text):
        return "lure"
    return "other"


def split_thinking(text: str) -> tuple[str, str]:
    """Split response into (thinking_trace, answer) around <think>...</think> tags."""
    if "<think>" in text and "</think>" in text:
        ts = text.index("<think>") + len("<think>")
        te = text.index("</think>")
        return text[ts:te].strip(), text[te + len("</think>") :].strip()
    return "", text.strip()


def find_think_token_positions(
    gen_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    prompt_len: int,
) -> tuple[int | None, int | None]:
    """Find T0 and Tend positions in the generated sequence.

    T0: index of the first token AFTER <think> (the start of the thinking trace)
    Tend: index of the last token BEFORE </think> (the end of the thinking trace)

    Returns absolute positions in the full sequence (prompt + generated).
    """
    # Encode the think tags to find their token IDs
    think_open_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)

    gen_list = gen_ids.tolist()

    # Find <think> tag in generated tokens
    t0_pos = None
    for i in range(prompt_len, len(gen_list) - len(think_open_ids) + 1):
        if gen_list[i : i + len(think_open_ids)] == think_open_ids:
            # T0 is the first token after the <think> tag
            t0_pos = i + len(think_open_ids)
            break

    # Find </think> tag in generated tokens (search from end for robustness)
    tend_pos = None
    for i in range(len(gen_list) - len(think_close_ids), prompt_len - 1, -1):
        if gen_list[i : i + len(think_close_ids)] == think_close_ids:
            # Tend is the last token before </think>
            tend_pos = i - 1
            break

    # Validate: T0 must come before Tend
    if t0_pos is not None and tend_pos is not None and t0_pos > tend_pos:
        return None, None

    return t0_pos, tend_pos


def extract_hidden_state_at_position(
    hidden_states: tuple,
    position: int,
    prompt_len: int,
    layer_idx: int,
) -> np.ndarray | None:
    """Extract a single hidden state vector from generate()'s output_hidden_states.

    hidden_states is structured as hidden_states[step][layer_in_step], where:
      - step 0 processes the full prompt (seq_len = prompt_len)
      - step k>0 processes 1 new token (seq_len = 1)
    layer indices within each step: 0=embedding, 1..N=transformer layers.

    Args:
        hidden_states: The hidden_states output from generate().
        position: Absolute position in the full sequence.
        prompt_len: Length of the prompt in tokens.
        layer_idx: Transformer layer index (0-based, maps to layer_idx+1 in HF).

    Returns:
        float16 numpy array of shape (hidden_dim,), or None if position is invalid.
    """
    hf_layer = layer_idx + 1  # skip embedding layer

    if position < prompt_len:
        # Position is within the prompt -- use step 0
        step = 0
        pos_in_step = position
    else:
        # Position is in the generated tokens
        # step k (for k>=1) corresponds to the (k-1)th generated token,
        # i.e., absolute position prompt_len + (k - 1)
        step = position - prompt_len + 1

    if step >= len(hidden_states):
        return None
    if hf_layer >= len(hidden_states[step]):
        return None

    hs = hidden_states[step][hf_layer]  # (1, seq_len_at_step, hidden_dim)

    if position < prompt_len:
        if pos_in_step >= hs.shape[1]:
            return None
        vec = hs[0, pos_in_step, :]
    else:
        # For generation steps k>=1, seq_len=1, so index 0
        vec = hs[0, -1, :]

    return vec.float().cpu().numpy().astype(np.float16)


def run_extraction(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    items: list[dict],
    enable_thinking: bool,
    max_new_tokens: int,
    extract_layers: list[int],
    output_path: str,
    args: argparse.Namespace,
) -> None:
    """Run extraction for one mode (thinking ON or OFF) and write HDF5."""
    mode_label = "THINK" if enable_thinking else "NO_THINK"
    n_problems = len(items)
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)

    if enable_thinking:
        position_labels = ["P0", "P2", "T0", "Tend"]
    else:
        position_labels = ["P0", "P2"]
    n_positions = len(position_labels)

    print(f"\n{'='*60}")
    print(f"MODE: {mode_label}")
    print(f"enable_thinking={enable_thinking}, max_new_tokens={max_new_tokens}")
    print(f"Positions: {position_labels}")
    print(f"Items: {n_problems}, Layers: {len(extract_layers)}")
    print(f"{'='*60}\n")

    # Pre-allocate storage
    residuals = {
        layer: np.zeros((n_problems, n_positions, hidden_dim), dtype=np.float16)
        for layer in extract_layers
    }
    position_token_indices = np.zeros((n_problems, n_positions), dtype=np.int32)
    position_valid = np.zeros((n_problems, n_positions), dtype=bool)

    behavior = {
        "predicted_answer": [],
        "correct": np.zeros(n_problems, dtype=bool),
        "matches_lure": np.zeros(n_problems, dtype=bool),
        "response_category": [],
        "thinking_text": [],
        "answer_text": [],
        "n_gen_tokens": np.zeros(n_problems, dtype=np.int32),
        "prompt_token_count": np.zeros(n_problems, dtype=np.int32),
    }

    t_total = time.time()
    for i, item in enumerate(items):
        # Apply chat template with thinking toggle
        messages = [{"role": "user", "content": item["prompt"]}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]
        behavior["prompt_token_count"][i] = prompt_len

        # Generate with hidden states
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        gen_ids = outputs.sequences[0]
        n_gen = gen_ids.shape[0] - prompt_len
        response = tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=False)

        # --- Extract P0 (last prompt token) ---
        p0_pos = prompt_len - 1
        position_token_indices[i, 0] = p0_pos
        position_valid[i, 0] = True

        for layer_idx in extract_layers:
            vec = extract_hidden_state_at_position(
                outputs.hidden_states, p0_pos, prompt_len, layer_idx
            )
            if vec is not None:
                residuals[layer_idx][i, 0, :] = vec

        # --- Extract P2 (last generated token) ---
        p2_pos = gen_ids.shape[0] - 1
        position_token_indices[i, 1] = p2_pos
        position_valid[i, 1] = True

        last_step = len(outputs.hidden_states) - 1
        for layer_idx in extract_layers:
            if last_step > 0:
                vec = extract_hidden_state_at_position(
                    outputs.hidden_states, p2_pos, prompt_len, layer_idx
                )
                if vec is not None:
                    residuals[layer_idx][i, 1, :] = vec

        # --- For thinking mode: extract T0 and Tend ---
        if enable_thinking:
            t0_pos, tend_pos = find_think_token_positions(gen_ids, tokenizer, prompt_len)

            # T0
            if t0_pos is not None and t0_pos < gen_ids.shape[0]:
                position_token_indices[i, 2] = t0_pos
                position_valid[i, 2] = True
                for layer_idx in extract_layers:
                    vec = extract_hidden_state_at_position(
                        outputs.hidden_states, t0_pos, prompt_len, layer_idx
                    )
                    if vec is not None:
                        residuals[layer_idx][i, 2, :] = vec

            # Tend
            if tend_pos is not None and tend_pos < gen_ids.shape[0]:
                position_token_indices[i, 3] = tend_pos
                position_valid[i, 3] = True
                for layer_idx in extract_layers:
                    vec = extract_hidden_state_at_position(
                        outputs.hidden_states, tend_pos, prompt_len, layer_idx
                    )
                    if vec is not None:
                        residuals[layer_idx][i, 3, :] = vec

        # Parse behavior
        thinking, answer = split_thinking(response)
        verdict = parse_answer(answer, item["correct_answer"], item.get("lure_answer", ""))

        behavior["predicted_answer"].append(answer[:200])
        behavior["correct"][i] = verdict == "correct"
        behavior["matches_lure"][i] = verdict == "lure"
        behavior["response_category"].append(verdict)
        behavior["thinking_text"].append(thinking[:2000])
        behavior["answer_text"].append(answer[:500])
        behavior["n_gen_tokens"][i] = n_gen

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_total
            eta = elapsed / (i + 1) * (n_problems - i - 1)
            t0_ok = "Y" if (enable_thinking and position_valid[i, 2]) else "-"
            tend_ok = "Y" if (enable_thinking and position_valid[i, 3]) else "-"
            print(
                f"  [{i+1}/{n_problems}] {verdict:8s} {n_gen:4d} tok  "
                f"T0={t0_ok} Tend={tend_ok}  "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )

    # --- Write HDF5 ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model_key = MODEL_ID.replace("/", "_")

    with h5py.File(output_path, "w") as f:
        # Metadata
        meta = f.create_group("/metadata")
        meta.attrs["schema_version"] = 1
        meta.attrs["benchmark_path"] = args.benchmark
        bm_hash = hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest()
        meta.attrs["benchmark_sha256"] = bm_hash
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["git_sha"] = "unknown"
        meta.attrs["seed"] = 0
        meta.attrs["config"] = json.dumps(
            {
                "model": MODEL_ID,
                "enable_thinking": enable_thinking,
                "max_new_tokens": max_new_tokens,
                "mode": mode_label,
                "n_items": n_problems,
                "extract_layers": extract_layers,
            }
        )

        # Problems
        prob = f.create_group("/problems")
        prob.create_dataset(
            "id", data=np.array([it["id"].encode()[:64] for it in items], dtype="S64")
        )
        prob.create_dataset(
            "category",
            data=np.array([it["category"].encode()[:32] for it in items], dtype="S32"),
        )
        prob.create_dataset(
            "conflict", data=np.array([it["conflict"] for it in items], dtype=bool)
        )
        prob.create_dataset(
            "difficulty",
            data=np.array([it.get("difficulty", 2) for it in items], dtype=np.int8),
        )
        prob.create_dataset(
            "prompt_text",
            data=np.array([it["prompt"].encode()[:2048] for it in items], dtype="S2048"),
        )
        prob.create_dataset(
            "correct_answer",
            data=np.array(
                [it["correct_answer"].encode()[:128] for it in items], dtype="S128"
            ),
        )
        prob.create_dataset(
            "lure_answer",
            data=np.array(
                [it.get("lure_answer", "").encode()[:128] for it in items], dtype="S128"
            ),
        )
        prob.create_dataset(
            "matched_pair_id",
            data=np.array(
                [it["matched_pair_id"].encode()[:64] for it in items], dtype="S64"
            ),
        )
        prob.create_dataset("prompt_token_count", data=behavior["prompt_token_count"])

        # Model group
        mgrp = f.create_group(f"/models/{model_key}")
        mmeta = mgrp.create_group("metadata")
        mmeta.attrs["hf_model_id"] = MODEL_ID
        mmeta.attrs["n_layers"] = n_layers
        mmeta.attrs["n_heads"] = n_heads
        mmeta.attrs["n_kv_heads"] = n_kv_heads
        mmeta.attrs["hidden_dim"] = hidden_dim
        mmeta.attrs["head_dim"] = hidden_dim // n_heads
        mmeta.attrs["dtype"] = "float16"
        mmeta.attrs["extracted_at"] = datetime.now(timezone.utc).isoformat()
        mmeta.attrs["is_reasoning_model"] = enable_thinking
        mmeta.attrs["enable_thinking"] = enable_thinking

        # Residual streams
        resid = mgrp.create_group("residual")
        for layer_idx in extract_layers:
            resid.create_dataset(
                f"layer_{layer_idx:02d}",
                data=residuals[layer_idx],
                compression="gzip",
                compression_opts=4,
            )

        # Position index
        pos = mgrp.create_group("position_index")
        pos.create_dataset(
            "labels",
            data=np.array([s.encode() for s in position_labels], dtype="S16"),
        )
        pos.create_dataset("token_indices", data=position_token_indices)
        pos.create_dataset("valid", data=position_valid)

        # Behavior
        beh = mgrp.create_group("behavior")
        beh.create_dataset(
            "predicted_answer",
            data=np.array(
                [s.encode()[:128] for s in behavior["predicted_answer"]], dtype="S128"
            ),
        )
        beh.create_dataset("correct", data=behavior["correct"])
        beh.create_dataset("matches_lure", data=behavior["matches_lure"])
        beh.create_dataset(
            "response_category",
            data=np.array(
                [s.encode()[:16] for s in behavior["response_category"]], dtype="S16"
            ),
        )

        # Generations
        gen = mgrp.create_group("generations")
        gen.create_dataset(
            "full_text",
            data=np.array(["".encode()[:8192]] * n_problems, dtype="S8192"),
        )
        gen.create_dataset(
            "thinking_text",
            data=np.array(
                [s.encode()[:8192] for s in behavior["thinking_text"]], dtype="S8192"
            ),
        )
        gen.create_dataset(
            "answer_text",
            data=np.array(
                [s.encode()[:512] for s in behavior["answer_text"]], dtype="S512"
            ),
        )
        gen.create_dataset(
            "thinking_token_count", data=np.zeros(n_problems, dtype=np.int32)
        )
        gen.create_dataset("answer_token_count", data=behavior["n_gen_tokens"])

    elapsed_total = time.time() - t_total
    n_correct = int(behavior["correct"].sum())
    n_lured = int(behavior["matches_lure"].sum())
    n_conflict = sum(1 for it in items if it["conflict"])
    file_size_mb = Path(output_path).stat().st_size / 1e6

    if enable_thinking:
        n_t0_valid = int(position_valid[:, 2].sum())
        n_tend_valid = int(position_valid[:, 3].sum())
        think_stats = f"T0 valid: {n_t0_valid}/{n_problems}, Tend valid: {n_tend_valid}/{n_problems}"
    else:
        think_stats = "N/A (no-think mode)"

    print(f"\n{'='*60}")
    print(f"MODE: {mode_label}")
    print(f"Model: {MODEL_ID}")
    print(f"Items: {n_problems}, Layers extracted: {len(extract_layers)}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Behavioral: {n_correct}/{n_problems} correct, {n_lured}/{n_conflict} lured ({n_lured / max(n_conflict, 1):.1%})")
    print(f"Think positions: {think_stats}")
    print(f"HDF5: {output_path} ({file_size_mb:.1f} MB)")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total / 60:.1f} min)")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Qwen3-8B activations with thinking toggle"
    )
    parser.add_argument(
        "--mode",
        choices=["nothink", "think", "both"],
        default="both",
        help="Which mode(s) to run",
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmark/benchmark.jsonl",
        help="Path to benchmark JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="data/activations",
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--cache-dir",
        default="/workspace/hf_cache",
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="Limit number of items (None=all 330)",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices to extract (None=all 36)",
    )
    parser.add_argument(
        "--nothink-max-tokens",
        type=int,
        default=256,
        help="max_new_tokens for no-think mode",
    )
    parser.add_argument(
        "--think-max-tokens",
        type=int,
        default=2048,
        help="max_new_tokens for think mode",
    )
    args = parser.parse_args()

    # Load model once, use for both modes
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=args.cache_dir)

    # Suppress pad_token_id warning
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Propagate pad_token_id to model config to silence generate() warning
    model.config.pad_token_id = tokenizer.pad_token_id

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)

    print(f"Architecture: {n_layers} layers, {hidden_dim} hidden, {n_heads} Q-heads, {n_kv_heads} KV-heads")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    if args.layers:
        extract_layers = [int(x) for x in args.layers.split(",")]
    else:
        extract_layers = list(range(n_layers))
    print(f"Extracting layers: {extract_layers}")

    # Load benchmark
    items = []
    with open(args.benchmark) as f:
        for line in f:
            items.append(json.loads(line))
    if args.n_items:
        items = items[: args.n_items]
    print(f"Benchmark: {len(items)} items loaded")

    # Run extraction(s)
    modes_to_run = []
    if args.mode in ("nothink", "both"):
        modes_to_run.append(("nothink", False, args.nothink_max_tokens))
    if args.mode in ("think", "both"):
        modes_to_run.append(("think", True, args.think_max_tokens))

    for mode_name, enable_thinking, max_new_tokens in modes_to_run:
        output_path = str(Path(args.output_dir) / f"qwen3_8b_{mode_name}.h5")
        run_extraction(
            model=model,
            tokenizer=tokenizer,
            items=items,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            extract_layers=extract_layers,
            output_path=output_path,
            args=args,
        )

    print("\nDone. All requested modes extracted.")


if __name__ == "__main__":
    main()
