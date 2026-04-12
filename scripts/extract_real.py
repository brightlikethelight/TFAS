#!/usr/bin/env python3
"""Extract residual stream activations from a real model on the full benchmark.

This is a focused script for the B200 pod that bypasses the full Hydra pipeline
and does exactly what we need for probing: residual stream at P0 for all layers.

Usage:
    python scripts/extract_real.py \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --output data/activations/llama31_8b.h5 \
        --max-new-tokens 256

    python scripts/extract_real.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --output data/activations/r1_distill_llama.h5 \
        --max-new-tokens 2048
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


def parse_answer(text: str, correct: str, lure: str) -> str:
    if correct and re.search(re.escape(correct), text):
        return "correct"
    if lure and re.search(re.escape(lure), text):
        return "lure"
    return "other"


def split_thinking(text: str) -> tuple[str, str]:
    if "<think>" in text and "</think>" in text:
        ts = text.index("<think>") + len("<think>")
        te = text.index("</think>")
        return text[ts:te].strip(), text[te + len("</think>"):].strip()
    return "", text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cache-dir", default="/workspace/hf_cache")
    parser.add_argument("--n-items", type=int, default=None, help="Limit items (None=all)")
    parser.add_argument("--layers", default=None, help="Comma-separated layers (None=all)")
    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir,
        dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    is_reasoning = "r1" in args.model.lower() or "reason" in args.model.lower()

    if args.layers:
        extract_layers = [int(x) for x in args.layers.split(",")]
    else:
        extract_layers = list(range(n_layers))

    print(f"Model: {n_layers} layers, {hidden_dim} hidden, {n_heads} heads, {n_kv_heads} KV heads")
    print(f"Reasoning model: {is_reasoning}")
    print(f"Extracting layers: {extract_layers}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Load benchmark
    items = []
    with open(args.benchmark) as f:
        for line in f:
            items.append(json.loads(line))
    if args.n_items:
        items = items[:args.n_items]
    n_problems = len(items)
    print(f"Benchmark: {n_problems} items")

    # Position labels
    position_labels = ["P0", "P2"]
    n_positions = len(position_labels)

    # Pre-allocate storage
    residuals = {
        layer: np.zeros((n_problems, n_positions, hidden_dim), dtype=np.float16)
        for layer in extract_layers
    }
    behavior = {
        "predicted_answer": [],
        "correct": np.zeros(n_problems, dtype=bool),
        "matches_lure": np.zeros(n_problems, dtype=bool),
        "response_category": [],
        "thinking_text": [],
        "answer_text": [],
        "n_gen_tokens": np.zeros(n_problems, dtype=np.int32),
    }

    # Register hooks
    hooks = []
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is (hidden_states, ...) for decoder layers
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook_fn

    for layer_idx in extract_layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Process items
    t_total = time.time()
    for i, item in enumerate(items):
        # Tokenize
        messages = [{"role": "user", "content": item["prompt"]}]
        input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(input_text, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        # Generate
        captured.clear()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        gen_ids = outputs.sequences[0]
        n_gen = gen_ids.shape[0] - prompt_len
        response = tok.decode(gen_ids[prompt_len:], skip_special_tokens=False)

        # Extract P0 (last prompt token) from the first forward pass hidden states
        # output_hidden_states gives us hidden states for each generation step
        # The first step includes the full prompt processing
        if outputs.hidden_states and len(outputs.hidden_states) > 0:
            # hidden_states[step][layer] has shape (batch, seq_len_at_step, hidden)
            # For step 0 (prompt processing), seq_len = prompt_len
            for layer_idx in extract_layers:
                # Step 0, layer+1 (layer 0 is embeddings)
                if layer_idx + 1 < len(outputs.hidden_states[0]):
                    hs = outputs.hidden_states[0][layer_idx + 1]  # (1, prompt_len, hidden)
                    # P0 = last prompt token
                    residuals[layer_idx][i, 0, :] = hs[0, -1, :].float().cpu().numpy().astype(np.float16)

                # P2 = last generated token (approximate: use last step)
                last_step = len(outputs.hidden_states) - 1
                if last_step > 0 and layer_idx + 1 < len(outputs.hidden_states[last_step]):
                    hs_last = outputs.hidden_states[last_step][layer_idx + 1]
                    residuals[layer_idx][i, 1, :] = hs_last[0, -1, :].float().cpu().numpy().astype(np.float16)

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
            print(f"  [{i+1}/{n_problems}] {verdict:8s} {n_gen:4d} tok  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Write HDF5
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model_key = args.model.replace("/", "_")

    with h5py.File(args.output, "w") as f:
        # Metadata
        meta = f.create_group("/metadata")
        meta.attrs["schema_version"] = 1
        meta.attrs["benchmark_path"] = args.benchmark
        bm_hash = hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest()
        meta.attrs["benchmark_sha256"] = bm_hash
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["git_sha"] = "unknown"
        meta.attrs["seed"] = 0
        meta.attrs["config"] = json.dumps(vars(args))

        # Problems
        prob = f.create_group("/problems")
        prob.create_dataset("id", data=np.array([it["id"].encode()[:64] for it in items], dtype="S64"))
        prob.create_dataset("category", data=np.array([it["category"].encode()[:32] for it in items], dtype="S32"))
        prob.create_dataset("conflict", data=np.array([it["conflict"] for it in items], dtype=bool))
        prob.create_dataset("difficulty", data=np.array([it.get("difficulty", 2) for it in items], dtype=np.int8))
        prob.create_dataset("prompt_text", data=np.array([it["prompt"].encode()[:2048] for it in items], dtype="S2048"))
        prob.create_dataset("correct_answer", data=np.array([it["correct_answer"].encode()[:128] for it in items], dtype="S128"))
        prob.create_dataset("lure_answer", data=np.array([it.get("lure_answer", "").encode()[:128] for it in items], dtype="S128"))
        prob.create_dataset("matched_pair_id", data=np.array([it["matched_pair_id"].encode()[:64] for it in items], dtype="S64"))
        prob.create_dataset("prompt_token_count", data=np.full(n_problems, 0, dtype=np.int32))

        # Model group
        mgrp = f.create_group(f"/models/{model_key}")
        mmeta = mgrp.create_group("metadata")
        mmeta.attrs["hf_model_id"] = args.model
        mmeta.attrs["n_layers"] = n_layers
        mmeta.attrs["n_heads"] = n_heads
        mmeta.attrs["n_kv_heads"] = n_kv_heads
        mmeta.attrs["hidden_dim"] = hidden_dim
        mmeta.attrs["head_dim"] = hidden_dim // n_heads
        mmeta.attrs["dtype"] = "float16"
        mmeta.attrs["extracted_at"] = datetime.now(timezone.utc).isoformat()
        mmeta.attrs["is_reasoning_model"] = is_reasoning

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
        pos.create_dataset("labels", data=np.array([s.encode() for s in position_labels], dtype="S16"))
        pos.create_dataset("token_indices", data=np.zeros((n_problems, n_positions), dtype=np.int32))
        pos.create_dataset("valid", data=np.ones((n_problems, n_positions), dtype=bool))

        # Behavior
        beh = mgrp.create_group("behavior")
        beh.create_dataset("predicted_answer", data=np.array([s.encode()[:128] for s in behavior["predicted_answer"]], dtype="S128"))
        beh.create_dataset("correct", data=behavior["correct"])
        beh.create_dataset("matches_lure", data=behavior["matches_lure"])
        beh.create_dataset("response_category", data=np.array([s.encode()[:16] for s in behavior["response_category"]], dtype="S16"))

        # Generations
        gen = mgrp.create_group("generations")
        gen.create_dataset("full_text", data=np.array(["".encode()[:8192]] * n_problems, dtype="S8192"))
        gen.create_dataset("thinking_text", data=np.array([s.encode()[:8192] for s in behavior["thinking_text"]], dtype="S8192"))
        gen.create_dataset("answer_text", data=np.array([s.encode()[:512] for s in behavior["answer_text"]], dtype="S512"))
        gen.create_dataset("thinking_token_count", data=np.zeros(n_problems, dtype=np.int32))
        gen.create_dataset("answer_token_count", data=behavior["n_gen_tokens"])

    elapsed_total = time.time() - t_total
    n_correct = int(behavior["correct"].sum())
    n_lured = int(behavior["matches_lure"].sum())
    n_conflict = sum(1 for it in items if it["conflict"])

    print(f"\n{'='*60}")
    print(f"MODEL: {args.model}")
    print(f"Items: {n_problems}, Layers extracted: {len(extract_layers)}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Behavioral: {n_correct}/{n_problems} correct, {n_lured}/{n_conflict} lured ({n_lured/max(n_conflict,1):.1%})")
    print(f"HDF5: {args.output} ({Path(args.output).stat().st_size/1e6:.1f} MB)")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
