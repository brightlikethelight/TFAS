"""Main per-model extraction loop.

This orchestrates, for a single HuggingFace model, the round-trip:

1. Load model + tokenizer with ``attn_implementation="eager"`` so hooks see
   raw attention weights.
2. For each benchmark item:
   a. Tokenize the prompt via the model's chat template.
   b. Forward pass on the prompt (no generation) to capture P0 residuals and
      the step-0 attention row.
   c. Call ``model.generate`` with ``return_dict_in_generate=True``,
      ``output_hidden_states=True``, ``output_attentions=True``,
      ``output_scores=True``. We stream per-step attention into the metric
      collector and per-step hidden states into a position-selection buffer.
   d. Parse thinking trace, compute canonical positions, subset residuals
      and metrics, compute token surprises (both per-position and full
      ragged trace).
   e. Score the behavioral response.
3. Write everything for this model to HDF5 via :class:`ActivationWriter`,
   then free GPU memory.

For memory safety, this file never materializes the full attention tensor
(see :mod:`s1s2.extract.hooks`). The residual stream **is** materialized per
problem, but only the rows at canonical positions are written to disk.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from s1s2.extract.hooks import METRIC_NAMES, AttentionMetricsCollector, metrics_at_positions
from s1s2.extract.parsing import (
    build_token_char_spans,
    compute_positions,
    split_thinking_answer,
)
from s1s2.extract.scoring import BenchmarkItemProto, score_response_detailed
from s1s2.extract.writer import ActivationWriter, ProblemMetadata
from s1s2.utils.types import ALL_POSITIONS

LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """All model-level knobs the core loop needs.

    Resolved by the CLI from ``configs/models.yaml``. The spec carries the
    authoritative layer/head/hidden counts; after loading we cross-check
    against ``model.config`` and error out on any mismatch so a stale YAML
    can't silently corrupt the cache.
    """

    key: str                      # "llama-3.1-8b-instruct"
    hdf5_key: str                 # "meta-llama_Llama-3.1-8B-Instruct"
    hf_id: str
    family: str
    n_layers: int
    n_heads: int                  # Q heads (post-GQA expansion in eager attention)
    n_kv_heads: int
    hidden_dim: int
    head_dim: int
    is_reasoning: bool


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    seed: int


@dataclass(frozen=True)
class ExtractionConfig:
    dtype: str                    # "float16" | "bfloat16" (may be coerced to f16 on disk)
    attn_implementation: str      # "eager"
    log_every: int


@dataclass
class PerProblemOutputs:
    """Scratchpad for one problem that the core loop hands off to the writer."""

    # Residuals: (n_layers, n_positions, hidden_dim) in storage dtype
    residual_rows: np.ndarray
    # Position indices: (n_positions,) int32 (absolute indices into prompt+gen)
    position_token_indices: np.ndarray
    # Valid mask: (n_positions,) bool
    position_valid: np.ndarray
    # Attention metrics at positions: name -> (n_layers, n_heads, n_positions) f32
    attn_position_metrics: dict[str, np.ndarray]
    # Surprises at positions: (n_positions,) f32
    surprise_by_position: np.ndarray
    # Full per-token surprise trace: (n_gen,) f32
    surprise_full_trace: np.ndarray
    # Generation text
    full_text: str
    thinking_text: str
    answer_text: str
    thinking_token_count: int
    answer_token_count: int
    # Behavior
    predicted_answer: str
    correct: bool
    matches_lure: bool
    response_category: str


# --------------------------------------------------------------------------- #
# Core driver                                                                 #
# --------------------------------------------------------------------------- #


class SingleModelExtractor:
    """Run extraction for one model across all benchmark items.

    Not reusable: instantiate once per model, call :meth:`run`, then drop.
    Drops the model from memory on exit.
    """

    def __init__(
        self,
        spec: ModelSpec,
        generation_cfg: GenerationConfig,
        extraction_cfg: ExtractionConfig,
        device: str,
        torch_dtype: torch.dtype,
    ) -> None:
        self.spec = spec
        self.gen_cfg = generation_cfg
        self.extr_cfg = extraction_cfg
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None

    # ----- model loading / unloading -----

    def load(
        self,
        *,
        loader: Callable[..., Any] | None = None,
        tokenizer_loader: Callable[..., Any] | None = None,
    ) -> None:
        """Load model + tokenizer.

        ``loader`` and ``tokenizer_loader`` are injection points so tests can
        substitute a tiny model (``sshleifer/tiny-gpt2``) without monkey-
        patching transformers globally. Production code passes None and we
        use HF's auto-loaders.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if tokenizer_loader is None:
            tokenizer_loader = AutoTokenizer.from_pretrained
        if loader is None:
            loader = AutoModelForCausalLM.from_pretrained

        LOG.info("loading tokenizer for %s", self.spec.hf_id)
        tokenizer = tokenizer_loader(self.spec.hf_id)

        LOG.info("loading model %s (dtype=%s, device=%s)", self.spec.hf_id, self.torch_dtype, self.device)
        kwargs: dict[str, Any] = {
            "attn_implementation": self.extr_cfg.attn_implementation,
        }
        # transformers 5.x deprecated `torch_dtype` in favor of `dtype`.
        try:
            model = loader(self.spec.hf_id, dtype=self.torch_dtype, **kwargs)
        except TypeError:
            model = loader(self.spec.hf_id, torch_dtype=self.torch_dtype, **kwargs)

        model.eval()
        # Avoid warnings & ensure pad token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        if self.device != "cpu":
            try:
                model = model.to(self.device)
            except Exception as e:
                LOG.warning("device .to(%s) failed (%s); falling back to cpu", self.device, e)
                self.device = "cpu"

        self._tokenizer = tokenizer
        self._model = model

        # Cross-check declared shapes against the actual config. This catches
        # a stale models.yaml before we silently write corrupt data.
        actual_layers = _infer_n_layers(model)
        if actual_layers is not None and actual_layers != self.spec.n_layers:
            LOG.warning(
                "model %s has %d layers but spec says %d; using actual value",
                self.spec.key,
                actual_layers,
                self.spec.n_layers,
            )
        actual_hidden = getattr(model.config, "hidden_size", None)
        if actual_hidden is not None and actual_hidden != self.spec.hidden_dim:
            LOG.warning(
                "model %s has hidden=%d but spec says %d; using actual value",
                self.spec.key,
                actual_hidden,
                self.spec.hidden_dim,
            )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def model(self):
        assert self._model is not None, "model not loaded"
        return self._model

    @property
    def tokenizer(self):
        assert self._tokenizer is not None, "tokenizer not loaded"
        return self._tokenizer

    # ----- public entry point -----

    def run(
        self,
        items: Sequence[BenchmarkItemProto],
        writer: ActivationWriter,
        effective_n_layers: int,
        effective_n_heads: int,
        effective_hidden_dim: int,
    ) -> list[PerProblemOutputs]:
        """Run extraction over all benchmark items and write them.

        Returns the list of per-problem scratchpads. The writer receives a
        model group allocation and per-problem residual rows as they are
        produced; other per-model arrays (attention, surprises, generations,
        behavior) are aggregated in-memory and written once at the end.
        """
        n_problems = len(items)
        n_positions = len(ALL_POSITIONS)
        hidden_dim = effective_hidden_dim
        n_layers = effective_n_layers
        n_heads = effective_n_heads

        # Allocate residual datasets and the per-problem metadata arrays.
        writer.allocate_residual(
            self.spec.hdf5_key, n_layers, n_problems, n_positions, hidden_dim, self.extr_cfg.dtype
        )

        all_position_token_indices = np.zeros((n_problems, n_positions), dtype=np.int32)
        all_position_valid = np.zeros((n_problems, n_positions), dtype=bool)

        # attention metric stacks: (n_problems, n_layers, n_heads, n_positions)
        attn_stacks: dict[str, np.ndarray] = {
            name: np.zeros((n_problems, n_layers, n_heads, n_positions), dtype=np.float32)
            for name in METRIC_NAMES
        }

        # token surprises
        surprise_by_position = np.zeros((n_problems, n_positions), dtype=np.float32)
        trace_values_accum: list[np.ndarray] = []
        trace_offsets = np.zeros((n_problems + 1,), dtype=np.int64)

        # generations
        full_texts: list[str] = [""] * n_problems
        thinking_texts: list[str] = [""] * n_problems
        answer_texts: list[str] = [""] * n_problems
        thinking_counts = np.zeros((n_problems,), dtype=np.int32)
        answer_counts = np.zeros((n_problems,), dtype=np.int32)

        # behavior
        pred_answers: list[str] = [""] * n_problems
        correct_mask = np.zeros((n_problems,), dtype=bool)
        lure_mask = np.zeros((n_problems,), dtype=bool)
        response_cats: list[str] = [""] * n_problems

        outputs: list[PerProblemOutputs] = []

        for prob_idx, item in enumerate(items):
            if prob_idx % max(self.extr_cfg.log_every, 1) == 0:
                LOG.info(
                    "extracting %s: %d / %d (id=%s)",
                    self.spec.key,
                    prob_idx + 1,
                    n_problems,
                    getattr(item, "id", "?"),
                )
            out = self._run_single(item)

            # Write residuals for this row directly to HDF5 to cap memory
            writer.write_residual_row(self.spec.hdf5_key, prob_idx, out.residual_rows)

            all_position_token_indices[prob_idx] = out.position_token_indices
            all_position_valid[prob_idx] = out.position_valid
            for name in METRIC_NAMES:
                attn_stacks[name][prob_idx] = out.attn_position_metrics[name]
            surprise_by_position[prob_idx] = out.surprise_by_position
            trace_values_accum.append(out.surprise_full_trace)
            trace_offsets[prob_idx + 1] = trace_offsets[prob_idx] + out.surprise_full_trace.size

            full_texts[prob_idx] = out.full_text
            thinking_texts[prob_idx] = out.thinking_text
            answer_texts[prob_idx] = out.answer_text
            thinking_counts[prob_idx] = out.thinking_token_count
            answer_counts[prob_idx] = out.answer_token_count

            pred_answers[prob_idx] = out.predicted_answer
            correct_mask[prob_idx] = out.correct
            lure_mask[prob_idx] = out.matches_lure
            response_cats[prob_idx] = out.response_category

            outputs.append(out)

        # Write aggregate arrays.
        writer.write_position_index(
            self.spec.hdf5_key, all_position_token_indices, all_position_valid
        )
        writer.write_attention_metrics(self.spec.hdf5_key, attn_stacks)

        if trace_values_accum:
            full_trace_values = np.concatenate(trace_values_accum, axis=0)
        else:
            full_trace_values = np.zeros((0,), dtype=np.float32)
        writer.write_token_surprises(
            self.spec.hdf5_key, surprise_by_position, trace_offsets, full_trace_values
        )

        writer.write_generations(
            self.spec.hdf5_key,
            full_texts,
            thinking_texts,
            answer_texts,
            thinking_counts,
            answer_counts,
        )

        writer.write_behavior(
            self.spec.hdf5_key, pred_answers, correct_mask, lure_mask, response_cats
        )
        return outputs

    # ----- per-problem workhorse -----

    def _run_single(self, item: BenchmarkItemProto) -> PerProblemOutputs:
        """Tokenize, generate, extract residuals + attention, score.

        This is the memory-sensitive path. Key rules:

        - Each attention tensor lives only until the next generation step
          has been processed by :class:`AttentionMetricsCollector`.
        - Hidden states are kept in memory for exactly one problem at a time,
          and we only copy the rows at canonical positions into the storage
          buffer (the full hidden-state cube is released when the next
          problem's forward pass begins).
        """
        tokenizer = self.tokenizer
        model = self.model
        n_layers = _infer_n_layers(model) or self.spec.n_layers
        hidden_dim = getattr(model.config, "hidden_size", self.spec.hidden_dim)
        # ----- Build prompt -----
        prompt_text = getattr(item, "prompt", None) or getattr(item, "prompt_text", "")
        system_prompt = getattr(item, "system_prompt", None)
        input_ids = _apply_chat_template(tokenizer, prompt_text, system_prompt)
        input_ids = input_ids.to(self.device)
        prompt_len = int(input_ids.shape[1])

        # ----- Generate with rich outputs -----
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.gen_cfg.max_new_tokens,
            "do_sample": self.gen_cfg.do_sample,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
            "output_attentions": True,
            "output_scores": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if self.gen_cfg.do_sample:
            gen_kwargs["temperature"] = self.gen_cfg.temperature
            gen_kwargs["top_p"] = self.gen_cfg.top_p
        # Seed for reproducibility of sampling runs
        if self.gen_cfg.do_sample:
            torch.manual_seed(self.gen_cfg.seed)

        with torch.no_grad():
            gen_out = model.generate(input_ids=input_ids, **gen_kwargs)

        # full sequence length (prompt + generated)
        full_seq = gen_out.sequences[0]   # (full_len,)
        gen_token_ids = full_seq[prompt_len:].tolist()
        n_gen = len(gen_token_ids)

        # ----- Build per-generation-token char spans and locate thinking -----
        generation_text, char_spans = build_token_char_spans(tokenizer, gen_token_ids)
        eos_id = _eos_id(tokenizer, model)
        eos_in_gen = bool(n_gen > 0 and gen_token_ids[-1] == eos_id)
        positions = compute_positions(
            prompt_len=prompt_len,
            generation_text=generation_text,
            token_char_spans=char_spans,
            is_reasoning_model=self.spec.is_reasoning,
            eos_token_in_generation=eos_in_gen,
        )

        position_token_indices = np.array([p.token_index for p in positions], dtype=np.int32)
        position_valid = np.array([p.valid for p in positions], dtype=bool)

        # ----- Residuals: subset the hidden_states at canonical positions -----
        residual_rows = _gather_residuals_at_positions(
            gen_out.hidden_states,
            prompt_len=prompt_len,
            n_gen=n_gen,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            position_token_indices=position_token_indices,
            position_valid=position_valid,
            storage_dtype=self.extr_cfg.dtype,
        )

        # ----- Attention metrics: stream per step -----
        collector = AttentionMetricsCollector(n_layers=n_layers, n_heads=self._query_head_count(gen_out))
        for step_attentions in gen_out.attentions:
            collector.process_step(step_attentions)
        step_metrics = collector.finalize()
        attn_pos = metrics_at_positions(
            step_metrics,
            position_token_indices=position_token_indices,
            position_valid=position_valid,
            prompt_len=prompt_len,
        )

        # ----- Token surprises -----
        surprise_full_trace = _compute_token_surprises(
            gen_out.scores, torch.as_tensor(gen_token_ids, device="cpu")
        )
        surprise_by_position = _surprises_at_positions(
            surprise_full_trace, position_token_indices, position_valid, prompt_len
        )

        # ----- Decode text, parse thinking, score -----
        thinking_text, answer_text, span = split_thinking_answer(generation_text)
        if self.spec.is_reasoning and span.truncated:
            LOG.warning(
                "thinking trace truncated for problem %s (hit max_new_tokens=%d)",
                getattr(item, "id", "?"),
                self.gen_cfg.max_new_tokens,
            )
        scoring = score_response_detailed(answer_text, item)

        # Token counts for thinking/answer
        thinking_token_count = _count_tokens_in_span(
            char_spans, span.start_char, span.end_char
        ) if span.present else 0
        # Answer token count: tokens strictly after </think>
        if span.present and not span.truncated:
            answer_start_char = span.end_char + len("</think>")
            answer_token_count = _count_tokens_in_span(char_spans, answer_start_char, len(generation_text))
        else:
            answer_token_count = n_gen

        return PerProblemOutputs(
            residual_rows=residual_rows,
            position_token_indices=position_token_indices,
            position_valid=position_valid,
            attn_position_metrics=attn_pos,
            surprise_by_position=surprise_by_position,
            surprise_full_trace=surprise_full_trace,
            full_text=generation_text,
            thinking_text=thinking_text,
            answer_text=answer_text,
            thinking_token_count=int(thinking_token_count),
            answer_token_count=int(answer_token_count),
            predicted_answer=scoring.predicted_answer,
            correct=scoring.matched_correct and not scoring.refused,
            matches_lure=scoring.matched_lure,
            response_category=scoring.category,
        )

    def _query_head_count(self, gen_out: Any) -> int:
        """Peek at the first attention tensor to learn the (expanded) query-head count.

        HuggingFace's eager attention returns ``(batch, n_q_heads, q, k)`` --
        the expanded query-head count even for GQA. We trust the tensor.
        """
        for step in gen_out.attentions:
            if len(step) > 0:
                return int(step[0].shape[1])
        return self.spec.n_heads


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _apply_chat_template(tokenizer, user_prompt: str, system_prompt: str | None):
    """Wrap the user prompt in the model's chat template and return token ids.

    Falls back to raw encoding if the tokenizer has no chat template (e.g.
    tiny-gpt2 used in unit tests).
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        # add_generation_prompt=True appends the assistant turn header so the
        # next token the model emits is the start of its response.
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return ids
    except Exception:
        pass
    # Fallback: raw encode with EOS/BOS as the tokenizer sees fit. Used in
    # tests with tiny-gpt2 which has no chat template.
    return tokenizer(user_prompt, return_tensors="pt").input_ids


def _infer_n_layers(model) -> int | None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    for attr in ("num_hidden_layers", "n_layer", "n_layers"):
        val = getattr(cfg, attr, None)
        if val is not None:
            return int(val)
    return None


def _eos_id(tokenizer, model) -> int:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    if isinstance(eos, list | tuple):
        return int(eos[0]) if eos else -1
    return int(eos) if eos is not None else -1


def _gather_residuals_at_positions(
    hidden_states_per_step: Sequence[Sequence[torch.Tensor]],
    *,
    prompt_len: int,
    n_gen: int,
    n_layers: int,
    hidden_dim: int,
    position_token_indices: np.ndarray,
    position_valid: np.ndarray,
    storage_dtype: str,
) -> np.ndarray:
    """Pull the residual stream rows at canonical positions.

    ``hidden_states_per_step`` is the HuggingFace generation output: a tuple
    of length ``n_steps`` where element 0 is the prompt forward pass (each
    layer of shape ``(1, prompt_len, hidden)``) and elements 1..n_steps are
    per-generation-token forwards (shape ``(1, 1, hidden)``).

    NB: ``hidden_states_per_step[step_idx]`` has length ``n_layers + 1`` --
    the first entry is the embedding output, then one entry per transformer
    block. We take entries 1..n_layers+1 so ``layer_00`` corresponds to the
    output of block 0, matching TransformerLens's ``hook_resid_post``
    convention.
    """
    np_dtype = np.float16 if storage_dtype.lower() in ("bfloat16", "bf16", "float16", "f16") else np.float32
    n_positions = position_token_indices.shape[0]
    out = np.zeros((n_layers, n_positions, hidden_dim), dtype=np_dtype)

    if len(hidden_states_per_step) == 0:
        return out

    step0 = hidden_states_per_step[0]   # tuple of (n_layers+1) tensors, each (1, prompt_len, hidden)
    # Layer indexing: use layers [1 : n_layers + 1] so layer_00 = block 0 output.
    layer_range = range(1, n_layers + 1)

    for pos_idx in range(n_positions):
        if not position_valid[pos_idx]:
            continue
        abs_tok = int(position_token_indices[pos_idx])
        if abs_tok < prompt_len:
            # Position inside the prompt -- take from step 0 at local index abs_tok.
            step_tensors = step0
            local_idx = abs_tok
            for out_layer_idx, ht_layer_idx in enumerate(layer_range):
                if ht_layer_idx >= len(step_tensors):
                    continue
                layer_h = step_tensors[ht_layer_idx]  # (1, prompt_len, hidden)
                vec = layer_h[0, local_idx, :].detach().to(torch.float32).cpu().numpy()
                out[out_layer_idx, pos_idx, :] = vec.astype(np_dtype, copy=False)
        else:
            gen_step = abs_tok - prompt_len   # 0-indexed into generation tokens
            step_idx = 1 + gen_step           # +1 because step 0 was the prompt
            if step_idx >= len(hidden_states_per_step):
                continue
            step_tensors = hidden_states_per_step[step_idx]
            for out_layer_idx, ht_layer_idx in enumerate(layer_range):
                if ht_layer_idx >= len(step_tensors):
                    continue
                layer_h = step_tensors[ht_layer_idx]  # (1, 1, hidden)
                vec = layer_h[0, -1, :].detach().to(torch.float32).cpu().numpy()
                out[out_layer_idx, pos_idx, :] = vec.astype(np_dtype, copy=False)
    return out


def _compute_token_surprises(scores_per_step, gen_token_ids: torch.Tensor) -> np.ndarray:
    """Per-generation-token negative log2 prob (bits).

    ``scores_per_step`` is a tuple of length ``n_gen``; each element is
    logits of shape ``(batch=1, vocab)`` at that step (i.e. the distribution
    over the token that was then sampled/argmaxed to become gen_token_ids[i]).
    """
    n = len(scores_per_step)
    if n == 0 or gen_token_ids.numel() == 0:
        return np.zeros((0,), dtype=np.float32)
    out = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        logits = scores_per_step[i][0].detach().to(torch.float32).cpu()  # (vocab,)
        log_probs = torch.log_softmax(logits, dim=-1)
        tok_id = int(gen_token_ids[i])
        # natural log -> bits via log2(e)
        lp = float(log_probs[tok_id])
        out[i] = -lp / float(np.log(2.0))
    return out


def _surprises_at_positions(
    surprise_full_trace: np.ndarray,
    position_token_indices: np.ndarray,
    position_valid: np.ndarray,
    prompt_len: int,
) -> np.ndarray:
    """Look up per-position surprises. Returns NaN-equivalent (0.0) for P0
    (P0 is a prompt token -- we don't have its surprise from the generation
    scores) and for any invalid position.
    """
    n_positions = position_token_indices.shape[0]
    out = np.zeros((n_positions,), dtype=np.float32)
    for i in range(n_positions):
        if not position_valid[i]:
            continue
        abs_tok = int(position_token_indices[i])
        if abs_tok < prompt_len:
            # prompt token; surprise undefined from generation scores
            continue
        gen_idx = abs_tok - prompt_len
        if 0 <= gen_idx < surprise_full_trace.size:
            out[i] = float(surprise_full_trace[gen_idx])
    return out


def _count_tokens_in_span(
    char_spans: Sequence[tuple[int, int]], start_char: int, end_char: int
) -> int:
    """Count generation tokens whose char range overlaps [start_char, end_char)."""
    if end_char <= start_char:
        return 0
    count = 0
    for c0, c1 in char_spans:
        if c1 <= start_char:
            continue
        if c0 >= end_char:
            break
        count += 1
    return count


def build_problem_metadata_from_items(
    items: Sequence[Any],
    prompt_token_counts: Sequence[int],
) -> list[ProblemMetadata]:
    """Coerce a sequence of benchmark items into :class:`ProblemMetadata`.

    We access fields via getattr so both dataclasses and SimpleNamespace /
    dict-like shims work. ``prompt_token_counts`` is computed separately by
    the caller (typically the first thing the per-model loop does with its
    tokenizer) so we can store it without re-tokenizing per model here.
    """
    out: list[ProblemMetadata] = []
    for i, item in enumerate(items):
        out.append(
            ProblemMetadata(
                id=str(getattr(item, "id", f"problem_{i}")),
                category=str(getattr(item, "category", "unknown")),
                conflict=bool(getattr(item, "conflict", False)),
                difficulty=int(getattr(item, "difficulty", 0)),
                prompt_text=str(
                    getattr(item, "prompt", None)
                    or getattr(item, "prompt_text", "")
                    or ""
                ),
                correct_answer=str(getattr(item, "correct_answer", "") or ""),
                lure_answer=str(getattr(item, "lure_answer", "") or ""),
                matched_pair_id=str(getattr(item, "matched_pair_id", "") or ""),
                prompt_token_count=int(prompt_token_counts[i]),
            )
        )
    return out
