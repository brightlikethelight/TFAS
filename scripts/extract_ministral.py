#!/usr/bin/env python3
"""Extract residual stream activations from the Ministral-3-8B matched pair.

Same approach as extract_real.py but adapted for the Ministral architecture.
Extracts P0 (last prompt token) and P2 (last generated token) residual stream
activations at all layers, plus behavioral data.

Ministral-3-8B-2512 uses Mistral's architecture (MistralForCausalLM) with
model.model.layers[i] for decoder layer hooks — same hook path as Llama.

Usage (on B200 pod):
    # Instruct
    python scripts/extract_ministral.py \
        --model mistralai/Ministral-3-8B-Instruct-2512 \
        --output data/activations/ministral_3b_instruct.h5 \
        --max-new-tokens 512

    # Reasoning
    python scripts/extract_ministral.py \
        --model mistralai/Ministral-3-8B-Reasoning-2512 \
        --output data/activations/ministral_3b_reasoning.h5 \
        --max-new-tokens 4096

    # Both back-to-back (shared benchmark metadata, separate model groups)
    python scripts/extract_ministral.py --run-pair
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -- Ministral pair specs -------------------------------------------------
PAIR = {
    "instruct": {
        "hf_id": "mistralai/Ministral-3-8B-Instruct-2512",
        "max_new_tokens": 512,
        "output": "data/activations/ministral_3b_instruct.h5",
    },
    "reasoning": {
        "hf_id": "mistralai/Ministral-3-8B-Reasoning-2512",
        "max_new_tokens": 4096,
        "output": "data/activations/ministral_3b_reasoning.h5",
    },
}


def parse_answer(text: str, correct: str, lure: str) -> str:
    if correct and re.search(re.escape(correct), text):
        return "correct"
    if lure and re.search(re.escape(lure), text):
        return "lure"
    return "other"


def split_thinking(text: str) -> tuple[str, str]:
    """Split response into thinking trace and final answer.

    Ministral-Reasoning uses <think>...</think> blocks.
    """
    if "<think>" in text and "</think>" in text:
        ts = text.index("<think>") + len("<think>")
        te = text.index("</think>")
        return text[ts:te].strip(), text[te + len("</think>"):].strip()
    return "", text.strip()


def extract_model(
    model_id: str,
    benchmark_path: str,
    output_path: str,
    max_new_tokens: int,
    cache_dir: str,
    n_items: int | None,
    layers_str: str | None,
    hf_token: str | None,
) -> dict:
    """Extract activations from a single Ministral model."""
    print(f"\n{'='*60}")
    print(f"EXTRACTING: {model_id}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    tok_kwargs: dict = {"cache_dir": cache_dir}
    model_kwargs: dict = {
        "cache_dir": cache_dir,
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda",
    }
    if hf_token:
        tok_kwargs["token"] = hf_token
        model_kwargs["token"] = hf_token

    tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    # Verify architecture and print config
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    hidden_dim = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    is_reasoning = "reason" in model_id.lower()

    print(f"Architecture: {cfg.architectures}")
    print(f"Layers: {n_layers}, Hidden: {hidden_dim}")
    print(f"Heads: {n_heads}, KV heads: {n_kv_heads}")
    print(f"Head dim: {hidden_dim // n_heads}")
    print(f"Vocab: {cfg.vocab_size}")
    print(f"Reasoning model: {is_reasoning}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Verify the hook target path exists. Mistral uses model.model.layers[i],
    # same as Llama. If the architecture changes, this will error loudly.
    assert hasattr(model, "model") and hasattr(model.model, "layers"), (
        f"Unexpected architecture for {model_id}: missing model.model.layers. "
        f"Got {type(model)}. Check that this is MistralForCausalLM."
    )
    print(f"Hook target: model.model.layers (n={len(model.model.layers)})")
    assert len(model.model.layers) == n_layers

    if layers_str:
        extract_layers = [int(x) for x in layers_str.split(",")]
    else:
        extract_layers = list(range(n_layers))

    print(f"Extracting layers: {extract_layers}")

    # Validate chat template
    test_msg = [{"role": "user", "content": "Hello"}]
    formatted = tok.apply_chat_template(
        test_msg, tokenize=False, add_generation_prompt=True
    )
    print(f"Chat template sample: {formatted[:120]}...")

    # Load benchmark
    items = []
    with open(benchmark_path) as f:
        for line in f:
            items.append(json.loads(line))
    if n_items:
        items = items[:n_items]
    n_problems = len(items)
    print(f"Benchmark: {n_problems} items")

    # Position labels
    position_labels = ["P0", "P2"]
    n_positions = len(position_labels)

    # Pre-allocate
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

    # Register hooks on decoder layers
    hooks = []
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook_fn

    for layer_idx in extract_layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Process items
    t_total = time.time()
    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item["prompt"]}]
        input_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(input_text, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        captured.clear()
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
        response = tok.decode(gen_ids[prompt_len:], skip_special_tokens=False)

        # Extract P0 (last prompt token) and P2 (last generated token) from
        # output_hidden_states. hidden_states[step][layer+1] where layer 0 =
        # embeddings. Step 0 = prompt processing (seq_len = prompt_len).
        if outputs.hidden_states and len(outputs.hidden_states) > 0:
            for layer_idx in extract_layers:
                # P0: last prompt token from step 0
                if layer_idx + 1 < len(outputs.hidden_states[0]):
                    hs = outputs.hidden_states[0][layer_idx + 1]
                    residuals[layer_idx][i, 0, :] = (
                        hs[0, -1, :].float().cpu().numpy().astype(np.float16)
                    )

                # P2: last generated token from final step
                last_step = len(outputs.hidden_states) - 1
                if last_step > 0 and layer_idx + 1 < len(outputs.hidden_states[last_step]):
                    hs_last = outputs.hidden_states[last_step][layer_idx + 1]
                    residuals[layer_idx][i, 1, :] = (
                        hs_last[0, -1, :].float().cpu().numpy().astype(np.float16)
                    )

        # Parse behavior
        thinking, answer = split_thinking(response)
        verdict = parse_answer(
            answer, item["correct_answer"], item.get("lure_answer", "")
        )

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
            print(
                f"  [{i+1}/{n_problems}] {verdict:8s} {n_gen:4d} tok  "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )

    # Remove hooks
    for h in hooks:
        h.remove()

    # Write HDF5
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model_key = model_id.replace("/", "_")

    with h5py.File(output_path, "w") as f:
        # Metadata
        meta = f.create_group("/metadata")
        meta.attrs["schema_version"] = 1
        meta.attrs["benchmark_path"] = benchmark_path
        bm_hash = hashlib.sha256(Path(benchmark_path).read_bytes()).hexdigest()
        meta.attrs["benchmark_sha256"] = bm_hash
        meta.attrs["created_at"] = datetime.now(UTC).isoformat()
        meta.attrs["git_sha"] = "unknown"
        meta.attrs["seed"] = 0
        meta.attrs["config"] = json.dumps({
            "model": model_id,
            "benchmark": benchmark_path,
            "output": output_path,
            "max_new_tokens": max_new_tokens,
            "n_items": n_items,
            "layers": layers_str,
        })

        # Problems
        prob = f.create_group("/problems")
        prob.create_dataset(
            "id",
            data=np.array([it["id"].encode()[:64] for it in items], dtype="S64"),
        )
        prob.create_dataset(
            "category",
            data=np.array([it["category"].encode()[:32] for it in items], dtype="S32"),
        )
        prob.create_dataset(
            "conflict",
            data=np.array([it["conflict"] for it in items], dtype=bool),
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
        prob.create_dataset(
            "prompt_token_count",
            data=np.full(n_problems, 0, dtype=np.int32),
        )

        # Model group
        mgrp = f.create_group(f"/models/{model_key}")
        mmeta = mgrp.create_group("metadata")
        mmeta.attrs["hf_model_id"] = model_id
        mmeta.attrs["n_layers"] = n_layers
        mmeta.attrs["n_heads"] = n_heads
        mmeta.attrs["n_kv_heads"] = n_kv_heads
        mmeta.attrs["hidden_dim"] = hidden_dim
        mmeta.attrs["head_dim"] = hidden_dim // n_heads
        mmeta.attrs["dtype"] = "float16"
        mmeta.attrs["extracted_at"] = datetime.now(UTC).isoformat()
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
        pos.create_dataset(
            "labels",
            data=np.array([s.encode() for s in position_labels], dtype="S16"),
        )
        pos.create_dataset(
            "token_indices",
            data=np.zeros((n_problems, n_positions), dtype=np.int32),
        )
        pos.create_dataset(
            "valid",
            data=np.ones((n_problems, n_positions), dtype=bool),
        )

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
            data=np.array([b""[:8192]] * n_problems, dtype="S8192"),
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
            "thinking_token_count",
            data=np.zeros(n_problems, dtype=np.int32),
        )
        gen.create_dataset(
            "answer_token_count",
            data=behavior["n_gen_tokens"],
        )

    elapsed_total = time.time() - t_total
    n_correct = int(behavior["correct"].sum())
    n_lured = int(behavior["matches_lure"].sum())
    n_conflict = sum(1 for it in items if it["conflict"])

    summary = {
        "model": model_id,
        "n_items": n_problems,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_correct": n_correct,
        "n_lured": n_lured,
        "n_conflict": n_conflict,
        "lure_rate": n_lured / max(n_conflict, 1),
        "output": output_path,
        "size_mb": Path(output_path).stat().st_size / 1e6,
        "elapsed_s": elapsed_total,
    }

    print(f"\n{'='*60}")
    print(f"MODEL: {model_id}")
    print(f"Items: {n_problems}, Layers extracted: {len(extract_layers)}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Behavioral: {n_correct}/{n_problems} correct, "
          f"{n_lured}/{n_conflict} lured ({n_lured/max(n_conflict,1):.1%})")
    print(f"HDF5: {output_path} ({summary['size_mb']:.1f} MB)")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"{'='*60}")

    # Free GPU memory
    del model
    del tok
    torch.cuda.empty_cache()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract activations from Ministral-3-8B pair"
    )
    parser.add_argument("--model", default=None, help="Single model HF ID")
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--cache-dir", default="/workspace/hf_cache")
    parser.add_argument("--n-items", type=int, default=None, help="Limit items (None=all)")
    parser.add_argument("--layers", default=None, help="Comma-separated layers (None=all)")
    parser.add_argument("--hf-token", default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument(
        "--run-pair", action="store_true",
        help="Extract both Instruct and Reasoning back-to-back"
    )
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    if args.run_pair:
        print("Extracting Ministral-3-8B matched pair")
        print(f"Benchmark: {args.benchmark}")
        summaries = {}
        for variant, spec in PAIR.items():
            summaries[variant] = extract_model(
                model_id=spec["hf_id"],
                benchmark_path=args.benchmark,
                output_path=spec["output"],
                max_new_tokens=args.max_new_tokens or spec["max_new_tokens"],
                cache_dir=args.cache_dir,
                n_items=args.n_items,
                layers_str=args.layers,
                hf_token=hf_token,
            )

        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE — Ministral-3-8B pair")
        print(f"{'='*60}")
        for variant, s in summaries.items():
            print(f"  {variant:12s}  lure={s['lure_rate']:.1%}  "
                  f"{s['output']} ({s['size_mb']:.1f} MB)  "
                  f"{s['elapsed_s']:.0f}s")

    elif args.model:
        if not args.output:
            parser.error("--output required when using --model")
        is_reasoning = "reason" in args.model.lower()
        default_tokens = 4096 if is_reasoning else 512
        extract_model(
            model_id=args.model,
            benchmark_path=args.benchmark,
            output_path=args.output,
            max_new_tokens=args.max_new_tokens or default_tokens,
            cache_dir=args.cache_dir,
            n_items=args.n_items,
            layers_str=args.layers,
            hf_token=hf_token,
        )
    else:
        parser.error("Provide --model or --run-pair")


if __name__ == "__main__":
    main()
