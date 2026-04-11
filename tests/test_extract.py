"""Tests for the extraction workstream.

These tests cover the schema writer, the thinking-trace parser, position
computation, incremental attention metrics, behavioral scoring, and an
end-to-end round-trip that exercises the HDF5 read API.

The end-to-end test uses ``sshleifer/tiny-gpt2`` -- a 2-layer GPT-2 with
hidden_size=2 that runs on CPU in under a second. It is the minimum test
fixture that verifies the real HuggingFace generate-with-hooks path without
needing a GPU.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

# Make sure OpenMP doesn't crash on macOS when torch + numpy share the runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Package is not pip-installed in the dev env; prepend src/ to sys.path.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from s1s2.extract import (
    ActivationWriter,
    AttentionMetricsCollector,
    ExtractionConfig,
    GenerationConfig,
    ModelSpec,
    ProblemMetadata,
    RunMetadata,
    SingleModelExtractor,
    compute_positions,
    find_thinking_span,
    score_response,
    score_response_detailed,
    split_thinking_answer,
    validate_file,
)
from s1s2.extract.hooks import _row_metrics, metrics_at_positions
from s1s2.utils.io import get_attention_metric, get_residual, open_activations, position_labels
from s1s2.utils.types import ALL_POSITIONS

# --------------------------------------------------------------------------- #
# test_thinking_trace_parsing                                                  #
# --------------------------------------------------------------------------- #


class TestThinkingTraceParsing:
    def test_complete_trace(self):
        gen = "<think>Let me think step by step. 2+2=4.</think>The answer is 4."
        span = find_thinking_span(gen)
        assert span.present is True
        assert span.truncated is False
        assert gen[span.start_char : span.end_char] == "Let me think step by step. 2+2=4."

        thinking, answer, _ = split_thinking_answer(gen)
        assert thinking == "Let me think step by step. 2+2=4."
        assert answer == "The answer is 4."

    def test_truncated_trace_no_close_tag(self):
        gen = "<think>I need to compute this slowly and carefully, let me"
        span = find_thinking_span(gen)
        assert span.present is True
        assert span.truncated is True
        assert span.end_char == len(gen)

        thinking, answer, _ = split_thinking_answer(gen)
        assert thinking.startswith("I need to compute")
        # On truncation we return the full generation as the answer candidate
        assert answer == gen

    def test_absent_trace(self):
        gen = "The answer is 42."
        span = find_thinking_span(gen)
        assert span.present is False
        assert span.truncated is False
        thinking, answer, _ = split_thinking_answer(gen)
        assert thinking == ""
        assert answer == gen

    def test_nested_or_repeated_tags(self):
        # Two <think> blocks: we take the first <think> and the LAST </think>
        gen = "<think>one</think>answer<think>two</think>tail"
        span = find_thinking_span(gen)
        assert span.present is True
        assert span.truncated is False
        # start_char is right after the FIRST <think>
        assert gen[span.start_char : span.end_char].startswith("one")
        # end_char is right before the LAST </think>
        assert gen[span.start_char : span.end_char].endswith("two")

    def test_empty_trace(self):
        gen = "<think></think>final"
        span = find_thinking_span(gen)
        assert span.present is True
        assert span.truncated is False
        assert span.start_char == len("<think>")
        assert span.end_char == len("<think>")


# --------------------------------------------------------------------------- #
# test_position_computation                                                    #
# --------------------------------------------------------------------------- #


def _synthetic_char_spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """Build char spans by greedily matching each token in order."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for tok in tokens:
        start = text.find(tok, cursor)
        if start == -1:
            raise AssertionError(f"token {tok!r} not found after offset {cursor}")
        end = start + len(tok)
        spans.append((start, end))
        cursor = end
    return spans


class TestPositionComputation:
    def test_reasoning_percentiles_exact(self):
        # Thinking trace with 8 "word" tokens. 25/50/75% = indices 2/4/6.
        text = "<think>one two three four five six seven eight</think>final"
        token_strs = [
            "<think>",
            "one ",
            "two ",
            "three ",
            "four ",
            "five ",
            "six ",
            "seven ",
            "eight",
            "</think>",
            "final",
        ]
        spans = _synthetic_char_spans(text, token_strs)
        positions = compute_positions(
            prompt_len=10,
            generation_text=text,
            token_char_spans=spans,
            is_reasoning_model=True,
            eos_token_in_generation=False,
        )
        labels = {p.label: p for p in positions}
        assert labels["T0"].valid
        assert labels["T25"].valid
        assert labels["T50"].valid
        assert labels["T75"].valid
        assert labels["Tend"].valid
        assert labels["Tswitch"].valid

        # Token "one" is generation index 1 -> absolute index 10 + 1 = 11
        assert labels["T0"].token_index == 10 + 1
        # 25% through 8 tokens -> index 1 + 2 = 3 -> abs 13
        assert labels["T25"].token_index == 10 + 3
        # 50% through 8 tokens -> index 1 + 4 = 5 -> abs 15
        assert labels["T50"].token_index == 10 + 5
        # 75% through 8 tokens -> index 1 + 6 = 7 -> abs 17
        assert labels["T75"].token_index == 10 + 7
        # Tend = last thinking token ("eight" at gen idx 8) -> abs 18
        assert labels["Tend"].token_index == 10 + 8
        # Tswitch = "final" at gen idx 10 -> abs 20
        assert labels["Tswitch"].token_index == 10 + 10

    def test_non_reasoning_model_invalidates_t_positions(self):
        text = "just an answer"
        spans = _synthetic_char_spans(text, ["just ", "an ", "answer"])
        positions = compute_positions(
            prompt_len=5,
            generation_text=text,
            token_char_spans=spans,
            is_reasoning_model=False,
            eos_token_in_generation=False,
        )
        labels = {p.label: p for p in positions}
        assert labels["P0"].valid
        assert labels["P2"].valid
        for t in ("T0", "T25", "T50", "T75", "Tend", "Tswitch"):
            assert labels[t].valid is False, f"expected {t} to be invalid for non-reasoning model"

    def test_truncated_reasoning_invalidates_tend_tswitch(self):
        # Thinking block with no closing tag
        text = "<think>I am still thinking"
        spans = _synthetic_char_spans(text, ["<think>", "I ", "am ", "still ", "thinking"])
        positions = compute_positions(
            prompt_len=3,
            generation_text=text,
            token_char_spans=spans,
            is_reasoning_model=True,
            eos_token_in_generation=False,
        )
        labels = {p.label: p for p in positions}
        assert labels["T0"].valid
        assert labels["Tend"].valid is False
        assert labels["Tswitch"].valid is False

    def test_p0_is_last_prompt_token(self):
        positions = compute_positions(
            prompt_len=5,
            generation_text="x",
            token_char_spans=[(0, 1)],
            is_reasoning_model=False,
            eos_token_in_generation=False,
        )
        p0 = next(p for p in positions if p.label == "P0")
        assert p0.token_index == 4
        p2 = next(p for p in positions if p.label == "P2")
        assert p2.token_index == 5  # prompt_len + 0
        assert p2.valid

    def test_eos_excluded_from_p2(self):
        positions = compute_positions(
            prompt_len=5,
            generation_text="ab",
            token_char_spans=[(0, 1), (1, 2)],
            is_reasoning_model=False,
            eos_token_in_generation=True,
        )
        # With EOS in generation, P2 is at n_gen - 2 -> gen idx 0 -> abs 5
        p2 = next(p for p in positions if p.label == "P2")
        assert p2.token_index == 5


# --------------------------------------------------------------------------- #
# test_hdf5_schema_conformance                                                 #
# --------------------------------------------------------------------------- #


def _make_dummy_problem(idx: int = 0, conflict: bool = True) -> ProblemMetadata:
    return ProblemMetadata(
        id=f"p_{idx}",
        category="crt",
        conflict=conflict,
        difficulty=2,
        prompt_text=f"prompt {idx}",
        correct_answer="5",
        lure_answer="10" if conflict else "",
        matched_pair_id=f"pair_{idx // 2}",
        prompt_token_count=7,
    )


class TestHDF5SchemaConformance:
    def test_writer_produces_conformant_file(self, tmp_path: Path):
        n_problems = 3
        n_layers = 2
        n_positions = len(ALL_POSITIONS)
        n_heads = 4
        hidden_dim = 8

        path = tmp_path / "test.h5"
        with ActivationWriter(path) as w:
            w.write_run_metadata(
                RunMetadata.build(
                    benchmark_path=str(tmp_path / "benchmark.jsonl"),
                    seed=0,
                    config_json="{}",
                )
            )
            w.write_problems([_make_dummy_problem(i) for i in range(n_problems)])

            w.create_model_group(
                model_key="tiny_model",
                hf_model_id="tiny/test",
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                hidden_dim=hidden_dim,
                head_dim=hidden_dim // n_heads,
                dtype="float16",
                is_reasoning_model=False,
            )
            w.allocate_residual(
                "tiny_model", n_layers, n_problems, n_positions, hidden_dim, "float16"
            )
            # Write dummy residual rows
            for i in range(n_problems):
                rows = np.random.randn(n_layers, n_positions, hidden_dim).astype(np.float16)
                w.write_residual_row("tiny_model", i, rows)

            w.write_position_index(
                "tiny_model",
                np.zeros((n_problems, n_positions), dtype=np.int32),
                np.ones((n_problems, n_positions), dtype=bool),
            )

            attn = {
                name: np.zeros((n_problems, n_layers, n_heads, n_positions), dtype=np.float32)
                for name in (
                    "entropy",
                    "entropy_normalized",
                    "gini",
                    "max_attn",
                    "focus_5",
                    "effective_rank",
                )
            }
            w.write_attention_metrics("tiny_model", attn)

            w.write_token_surprises(
                "tiny_model",
                by_position=np.zeros((n_problems, n_positions), dtype=np.float32),
                full_trace_offsets=np.zeros((n_problems + 1,), dtype=np.int64),
                full_trace_values=np.zeros((0,), dtype=np.float32),
            )

            w.write_generations(
                "tiny_model",
                full_text=["gen " + str(i) for i in range(n_problems)],
                thinking_text=["" for _ in range(n_problems)],
                answer_text=["5" for _ in range(n_problems)],
                thinking_token_count=np.zeros((n_problems,), dtype=np.int32),
                answer_token_count=np.ones((n_problems,), dtype=np.int32),
            )

            w.write_behavior(
                "tiny_model",
                predicted_answer=["5" for _ in range(n_problems)],
                correct=np.ones((n_problems,), dtype=bool),
                matches_lure=np.zeros((n_problems,), dtype=bool),
                response_category=["correct"] * n_problems,
            )

        errors = validate_file(path)
        assert errors == [], f"schema validation errors: {errors}"

        # Spot-check datasets exist + shapes
        with h5py.File(path, "r") as f:
            assert f["/metadata"].attrs["schema_version"] == 1
            assert f["/problems/id"].shape[0] == n_problems
            assert f["/models/tiny_model/residual/layer_00"].shape == (
                n_problems,
                n_positions,
                hidden_dim,
            )
            assert f["/models/tiny_model/attention/entropy"].shape == (
                n_problems,
                n_layers,
                n_heads,
                n_positions,
            )
            # labels match canonical order
            labels = [s.decode("utf-8") for s in f["/models/tiny_model/position_index/labels"][:]]
            assert labels == list(ALL_POSITIONS)

    def test_validator_catches_missing_datasets(self, tmp_path: Path):
        path = tmp_path / "broken.h5"
        with h5py.File(path, "w") as f:
            g = f.create_group("/metadata")
            g.attrs["schema_version"] = 1
            # intentionally incomplete
        errors = validate_file(path)
        assert any("problems" in e for e in errors), f"expected problems error, got {errors}"


# --------------------------------------------------------------------------- #
# test_attention_metrics_incremental                                           #
# --------------------------------------------------------------------------- #


class TestAttentionMetricsIncremental:
    def test_row_metrics_vs_numpy_reference(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal(size=32)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        entropy, entropy_n, gini, mx, focus5, eff = _row_metrics(probs)

        # Reference computations
        ref_entropy = float(-np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0))))
        assert pytest.approx(entropy, abs=1e-5) == ref_entropy
        assert pytest.approx(entropy_n, abs=1e-5) == ref_entropy / np.log2(32)
        assert pytest.approx(mx, abs=1e-6) == float(probs.max())
        # top-5 sum
        top5 = float(np.sort(probs)[-5:].sum())
        assert pytest.approx(focus5, abs=1e-6) == top5
        # effective rank
        assert pytest.approx(eff, rel=1e-4) == 2 ** ref_entropy
        # Gini is in [0, 1]
        assert 0.0 <= gini <= 1.0

    def test_uniform_row_has_max_entropy_zero_gini(self):
        probs = np.ones(16) / 16
        entropy, entropy_n, gini, mx, focus5, eff = _row_metrics(probs)
        assert pytest.approx(entropy, abs=1e-6) == 4.0
        assert pytest.approx(entropy_n, abs=1e-6) == 1.0
        assert pytest.approx(gini, abs=1e-6) == 0.0
        assert pytest.approx(mx, abs=1e-6) == 1 / 16

    def test_point_mass_has_zero_entropy_high_gini(self):
        probs = np.zeros(16)
        probs[3] = 1.0
        entropy, entropy_n, gini, mx, focus5, eff = _row_metrics(probs)
        assert pytest.approx(entropy, abs=1e-6) == 0.0
        assert pytest.approx(entropy_n, abs=1e-6) == 0.0
        assert gini > 0.9  # point mass has near-max Gini
        assert pytest.approx(mx, abs=1e-6) == 1.0

    def test_incremental_matches_batch_on_a_sequence(self):
        """Process a sequence of rows incrementally and verify the stacked
        output matches what you'd get from a single batched call.
        """
        import torch

        n_layers = 3
        n_heads = 2
        n_steps = 5
        rng = np.random.default_rng(42)
        # Build per-step attention of shape (1, n_heads, 1, prompt+step)
        step_attentions_list = []
        for s in range(n_steps):
            k_len = 4 + s
            logits = rng.standard_normal(size=(n_heads, k_len))
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            layer_tensors = []
            for _layer in range(n_layers):
                layer_tensors.append(torch.as_tensor(probs[None, :, None, :], dtype=torch.float32))
            step_attentions_list.append(tuple(layer_tensors))

        collector = AttentionMetricsCollector(n_layers=n_layers, n_heads=n_heads)
        for step in step_attentions_list:
            collector.process_step(step)
        metrics = collector.finalize()

        # shape checks
        assert metrics["entropy"].shape == (n_steps, n_layers, n_heads)
        # Reference: all layers get the same probs in the fixture, so layer 0 == layer 1.
        for layer_idx in range(1, n_layers):
            np.testing.assert_allclose(
                metrics["entropy"][:, 0, :], metrics["entropy"][:, layer_idx, :], rtol=1e-6
            )

        # Reference: compute metrics manually from the original probs
        for s, step in enumerate(step_attentions_list):
            probs_ref = step[0][0, :, 0, :].numpy()
            for h in range(n_heads):
                row = probs_ref[h]
                ref_ent = float(-np.sum(row * np.log2(np.clip(row, 1e-12, 1.0))))
                assert pytest.approx(metrics["entropy"][s, 0, h], abs=1e-5) == ref_ent

    def test_metrics_at_positions_mapping(self):
        metrics = {
            "entropy": np.array(
                [
                    [[1.0, 2.0]],
                    [[3.0, 4.0]],
                    [[5.0, 6.0]],
                ],
                dtype=np.float32,
            ),
            "entropy_normalized": np.zeros((3, 1, 2), dtype=np.float32),
            "gini": np.zeros((3, 1, 2), dtype=np.float32),
            "max_attn": np.zeros((3, 1, 2), dtype=np.float32),
            "focus_5": np.zeros((3, 1, 2), dtype=np.float32),
            "effective_rank": np.zeros((3, 1, 2), dtype=np.float32),
        }
        # prompt_len=4, steps correspond to: step0=prompt (abs tokens 0..3), step1=abs 4, step2=abs 5
        position_token_indices = np.array([3, 5, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        position_valid = np.array([True, True, False, False, False, False, False, False], dtype=bool)
        at_pos = metrics_at_positions(metrics, position_token_indices, position_valid, prompt_len=4)
        # Position 0 (P0 = abs 3 = step 0)
        assert at_pos["entropy"][0, 0, 0] == 1.0
        assert at_pos["entropy"][0, 1, 0] == 2.0
        # Position 1 (P2 = abs 5 = step 2)
        assert at_pos["entropy"][0, 0, 1] == 5.0
        assert at_pos["entropy"][0, 1, 1] == 6.0
        # Invalid positions are zero
        assert at_pos["entropy"][0, 0, 2] == 0.0


# --------------------------------------------------------------------------- #
# test_scoring                                                                 #
# --------------------------------------------------------------------------- #


def _item(
    conflict: bool = True,
    answer_pattern: str = r"\b5\b",
    lure_pattern: str = r"\b10\b",
    correct: str = "5",
    lure: str = "10",
):
    return SimpleNamespace(
        conflict=conflict,
        answer_pattern=answer_pattern,
        lure_pattern=lure_pattern,
        correct_answer=correct,
        lure_answer=lure,
    )


class TestScoring:
    def test_correct(self):
        cat, ans = score_response("The answer is 5 cents.", _item())
        assert cat == "correct"
        assert ans == "5"

    def test_lure(self):
        cat, ans = score_response("The answer is 10 cents.", _item())
        assert cat == "lure"
        assert ans == "10"

    def test_refusal_beats_correct(self):
        # Even if "5" appears in the response, refusal wins
        cat, ans = score_response("I cannot help with that. It would be 5.", _item())
        assert cat == "refusal"

    def test_other_wrong(self):
        cat, ans = score_response("I think it's 42 dollars.", _item())
        assert cat == "other_wrong"

    def test_non_conflict_item_never_marks_lure(self):
        # No conflict: even if "10" appears, cannot classify as lure
        cat, _ = score_response("The total is 10 and I give 5.", _item(conflict=False))
        assert cat == "correct"

    def test_lure_after_correct_prefers_final_answer(self):
        # Model gives correct first, then mentions lure later -> the later
        # match wins (model's "final answer" convention).
        res = score_response_detailed("The answer is 5, actually wait 10 is wrong.", _item())
        # "5" appears at pos ~14, "10" at pos ~33 -> lure wins
        assert res.category == "lure"

    def test_malformed_regex_falls_back_to_literal(self):
        # When the item's regex is malformed, the scorer falls back to a
        # literal-string match of the pattern text. We pass a pattern that
        # re.compile rejects (``5[``) but whose literal form is present in
        # the response.
        cat, _ = score_response("the result is 5[ please", _item(answer_pattern="5["))
        assert cat == "correct"


# --------------------------------------------------------------------------- #
# test_io_round_trip (end-to-end with tiny-gpt2)                               #
# --------------------------------------------------------------------------- #


class TestRoundTripTinyModel:
    """Drive the full extraction pipeline with sshleifer/tiny-gpt2 on CPU.

    This is the single integration test that exercises every moving part:
    tokenization, generate-with-outputs, hidden state selection, attention
    streaming, surprise computation, HDF5 writing, and reading via utils.io.
    """

    def test_full_pipeline(self, tmp_path: Path):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            pytest.skip("transformers/torch not available")

        # Prepare a tiny benchmark JSONL
        bench_path = tmp_path / "benchmark.jsonl"
        items_json = [
            {
                "id": "p_0",
                "category": "crt",
                "subcategory": "ratio",
                "conflict": True,
                "difficulty": 2,
                "prompt": "Hello",
                "system_prompt": None,
                "correct_answer": "5",
                "lure_answer": "10",
                "answer_pattern": r"\b5\b",
                "lure_pattern": r"\b10\b",
                "matched_pair_id": "pair_0",
                "source": "novel",
                "provenance_note": "test",
                "paraphrases": [],
            },
            {
                "id": "p_1",
                "category": "crt",
                "subcategory": "ratio",
                "conflict": False,
                "difficulty": 2,
                "prompt": "World",
                "system_prompt": None,
                "correct_answer": "7",
                "lure_answer": "",
                "answer_pattern": r"\b7\b",
                "lure_pattern": "",
                "matched_pair_id": "pair_0",
                "source": "novel",
                "provenance_note": "test",
                "paraphrases": [],
            },
        ]
        with bench_path.open("w") as f:
            for it in items_json:
                f.write(json.dumps(it) + "\n")

        # Build SimpleNamespace items (the shim format the CLI uses when the
        # real loader is missing).
        items = [SimpleNamespace(**d) for d in items_json]

        # Model spec for tiny-gpt2
        spec = ModelSpec(
            key="tiny",
            hdf5_key="tiny-gpt2",
            hf_id="sshleifer/tiny-gpt2",
            family="gpt2",
            n_layers=2,
            n_heads=2,
            n_kv_heads=2,
            hidden_dim=2,
            head_dim=1,
            is_reasoning=False,
        )
        gen_cfg = GenerationConfig(
            max_new_tokens=4,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            seed=0,
        )
        extr_cfg = ExtractionConfig(
            dtype="float16",
            attn_implementation="eager",
            log_every=1,
        )

        extractor = SingleModelExtractor(
            spec=spec,
            generation_cfg=gen_cfg,
            extraction_cfg=extr_cfg,
            device="cpu",
            torch_dtype=torch.float32,
        )
        extractor.load()

        n_layers = spec.n_layers  # confirmed from config
        n_heads = spec.n_heads
        hidden_dim = extractor.model.config.hidden_size

        out_path = tmp_path / "activations.h5"
        with ActivationWriter(out_path) as writer:
            writer.write_run_metadata(
                RunMetadata.build(
                    benchmark_path=str(bench_path),
                    seed=0,
                    config_json="{}",
                )
            )

            # Prompt token counts with this tokenizer
            prompt_counts = []
            for item in items:
                ids = extractor.tokenizer(item.prompt, return_tensors="pt").input_ids
                prompt_counts.append(int(ids.shape[1]))

            from s1s2.extract.core import build_problem_metadata_from_items

            writer.write_problems(build_problem_metadata_from_items(items, prompt_counts))

            writer.create_model_group(
                model_key=spec.hdf5_key,
                hf_model_id=spec.hf_id,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=spec.n_kv_heads,
                hidden_dim=hidden_dim,
                head_dim=spec.head_dim,
                dtype="float16",
                is_reasoning_model=False,
            )

            extractor.run(
                items=items,
                writer=writer,
                effective_n_layers=n_layers,
                effective_n_heads=n_heads,
                effective_hidden_dim=hidden_dim,
            )

        extractor.unload()

        # Validate schema
        errors = validate_file(out_path)
        assert errors == [], f"schema errors: {errors}"

        # Read back via the public IO API
        with open_activations(out_path) as f:
            assert f["/metadata"].attrs["schema_version"] == 1
            assert f["/problems/id"].shape[0] == 2
            labels = position_labels(f, "tiny-gpt2")
            assert labels == list(ALL_POSITIONS)
            resid = get_residual(f, "tiny-gpt2", layer=0)
            assert resid.shape == (2, len(ALL_POSITIONS), hidden_dim)
            # Residual dtype on disk is float16
            assert resid.dtype == np.float16

            ent = get_attention_metric(f, "tiny-gpt2", "entropy")
            assert ent.shape == (2, n_layers, n_heads, len(ALL_POSITIONS))
            # Entropy values must be finite and non-negative at valid positions
            assert np.all(np.isfinite(ent))
            assert np.all(ent >= -1e-6)

            # Surprises shape checks
            surprises = f["/models/tiny-gpt2/token_surprises/by_position"][:]
            assert surprises.shape == (2, len(ALL_POSITIONS))
            offsets = f["/models/tiny-gpt2/token_surprises/full_trace_offsets"][:]
            assert offsets.shape == (3,)  # n_problems + 1
            assert offsets[0] == 0
            assert offsets[-1] >= 0

            # Generations written
            full = f["/models/tiny-gpt2/generations/full_text"][:]
            assert full.shape == (2,)
            # Behavior written
            cats = f["/models/tiny-gpt2/behavior/response_category"][:]
            assert cats.shape == (2,)
            # All four allowed category labels
            for c in cats:
                assert c.decode("utf-8") in ("correct", "lure", "other_wrong", "refusal")
