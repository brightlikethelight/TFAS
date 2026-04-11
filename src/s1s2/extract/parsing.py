"""Thinking-trace parsing and canonical-position computation.

For reasoning models we need to locate the ``<think>...</think>`` span in the
generated text, and within it compute percentile landmarks (T0, T25, T50, T75,
Tend). Parsing must be robust to:

- Absent thinking block (non-reasoning models, or a reasoning model that
  skipped its trace). All T-positions become invalid.
- Truncated thinking block (generation ran out of budget before ``</think>``).
  We still label T0/T25/T50/T75 from the partial trace and mark Tend / Tswitch
  invalid so downstream analysis can filter.

The core function :func:`parse_and_locate_positions` takes the generated token
sequence and returns, for every label in :data:`s1s2.utils.types.ALL_POSITIONS`,
the absolute token index in the full sequence (prompt tokens + generation
tokens) and a ``valid`` bit. Indexing is absolute in the full sequence so
downstream code can cross-reference with logits / hidden states.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from s1s2.utils.types import ALL_POSITIONS, PositionLabel

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class ThinkingSpan:
    """Character-level span of the thinking block inside the generation string.

    Attributes
    ----------
    start_char, end_char: character indices in the *generation* string (NOT the
        full prompt+generation). ``start_char`` points at the character AFTER
        ``<think>``. ``end_char`` points at the character BEFORE ``</think>``.
    truncated: True iff ``</think>`` was never found. In that case ``end_char``
        is ``len(generation)`` and the answer text is empty.
    present: True iff a ``<think>`` tag was found at all.
    """

    start_char: int
    end_char: int
    truncated: bool
    present: bool

    @property
    def thinking_text(self) -> str:  # populated lazily by the caller
        raise NotImplementedError


def find_thinking_span(generation: str) -> ThinkingSpan:
    """Find the first ``<think>`` and last ``</think>`` via index_of.

    Avoids regex backtracking on multi-kilobyte traces. Handles the three
    degenerate cases:

    1. No ``<think>`` -> ``present=False, truncated=False``.
    2. ``<think>`` but no ``</think>`` -> ``present=True, truncated=True``,
       ``end_char = len(generation)``.
    3. Well-formed ``<think>...</think>`` -> both flags False.
    """
    open_idx = generation.find(THINK_OPEN)
    if open_idx == -1:
        return ThinkingSpan(start_char=0, end_char=0, truncated=False, present=False)
    start_char = open_idx + len(THINK_OPEN)
    close_idx = generation.rfind(THINK_CLOSE)
    if close_idx == -1 or close_idx < start_char:
        # Truncated: generation ran out before </think>, or the only </think> is
        # stray text inside the prompt echo (shouldn't happen with batch=1, but
        # defend against it).
        return ThinkingSpan(
            start_char=start_char,
            end_char=len(generation),
            truncated=True,
            present=True,
        )
    return ThinkingSpan(
        start_char=start_char,
        end_char=close_idx,
        truncated=False,
        present=True,
    )


def split_thinking_answer(generation: str) -> tuple[str, str, ThinkingSpan]:
    """Return ``(thinking_text, answer_text, span)``.

    For non-reasoning models (or truncated traces with no closing tag), the
    answer is the full generation -- scoring needs *something* to match.
    """
    span = find_thinking_span(generation)
    if not span.present:
        return "", generation, span
    thinking = generation[span.start_char : span.end_char]
    if span.truncated:
        # No closing tag: there is no real "answer" segment, but we still store
        # the full generation as the answer candidate for scoring so that a
        # half-emitted answer inside the thinking block isn't lost.
        answer = generation
    else:
        answer = generation[span.end_char + len(THINK_CLOSE) :]
    return thinking, answer, span


# --------------------------------------------------------------------------- #
# Token-level position computation                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PositionInfo:
    """Absolute token index + validity for one canonical position label."""

    label: PositionLabel
    token_index: int
    valid: bool


def _percentile_index(start: int, end: int, frac: float) -> int:
    """Compute the token index ``frac`` of the way from start (inclusive) to
    end (exclusive). ``frac`` in [0, 1]. Returns ``start`` if the span is
    empty, ``end - 1`` if ``frac >= 1``.
    """
    length = end - start
    if length <= 0:
        return start
    # round-down to be conservative and ensure result is strictly within
    # [start, end - 1]
    offset = int(frac * length)
    if offset >= length:
        offset = length - 1
    return start + offset


def compute_positions(
    prompt_len: int,
    generation_text: str,
    token_char_spans: Sequence[tuple[int, int]],
    is_reasoning_model: bool,
    eos_token_in_generation: bool,
) -> list[PositionInfo]:
    """Compute token indices + validity for every canonical position.

    Parameters
    ----------
    prompt_len:
        Number of prompt tokens. P0 = prompt_len - 1 (last prompt token,
        captured before generation begins).
    generation_text:
        Decoded generation string (everything AFTER the prompt). Used to find
        the ``<think>`` span.
    token_char_spans:
        Per-generation-token character offsets into ``generation_text``. Element
        ``i`` is ``(char_start, char_end)`` of generation token ``i``. Length
        must equal the number of generation tokens. Produced by
        :func:`build_token_char_spans`.
    is_reasoning_model:
        If False, only P0 / P2 are valid and T-positions get valid=False.
    eos_token_in_generation:
        If the generation includes an EOS/terminator token, P2 points to the
        token JUST before it. Otherwise P2 is the last generation token.

    Returns
    -------
    List of :class:`PositionInfo`, one per label in
    :data:`s1s2.utils.types.ALL_POSITIONS`, in the canonical order.
    """
    n_gen = len(token_char_spans)
    # absolute indices for generation tokens relative to the full sequence
    # (prompt + generation)
    gen_abs_start = prompt_len

    # ----- P0: last token of the prompt -----
    p0_abs = max(prompt_len - 1, 0)

    # ----- P2: final answer token -----
    if n_gen == 0:
        p2_abs = p0_abs
        p2_valid = False
    else:
        if eos_token_in_generation and n_gen >= 2:
            p2_abs = gen_abs_start + n_gen - 2
        else:
            p2_abs = gen_abs_start + n_gen - 1
        p2_valid = True

    # ----- T-positions -----
    # Locate the thinking span in generation_text (character indices),
    # then translate to token indices via token_char_spans.
    span = find_thinking_span(generation_text)
    t0 = t25 = t50 = t75 = tend = tswitch = -1
    t0_v = t25_v = t50_v = t75_v = tend_v = tswitch_v = False

    if is_reasoning_model and span.present and n_gen > 0:
        # Translate character span -> generation-token span.
        # A token "belongs" to the thinking block if its character span
        # overlaps strictly inside [span.start_char, span.end_char).
        first_tok = None
        last_tok = None
        for i, (c0, c1) in enumerate(token_char_spans):
            if c1 <= span.start_char:
                continue
            if c0 >= span.end_char:
                break
            if first_tok is None:
                first_tok = i
            last_tok = i
        # tokenizers can emit whitespace-only tokens at the boundaries; as
        # long as we got at least one thinking token, proceed.
        if first_tok is not None and last_tok is not None and last_tok >= first_tok:
            # Percentiles are computed over the half-open range [first_tok, last_tok+1).
            t0 = gen_abs_start + first_tok
            t0_v = True
            t25 = gen_abs_start + _percentile_index(first_tok, last_tok + 1, 0.25)
            t25_v = True
            t50 = gen_abs_start + _percentile_index(first_tok, last_tok + 1, 0.50)
            t50_v = True
            t75 = gen_abs_start + _percentile_index(first_tok, last_tok + 1, 0.75)
            t75_v = True
            if not span.truncated:
                tend = gen_abs_start + last_tok
                tend_v = True
                # Tswitch: first token whose char_start is >= end_char + len("</think>")
                after_close_char = span.end_char + len(THINK_CLOSE)
                for j, (c0, _c1) in enumerate(token_char_spans[last_tok + 1 :], start=last_tok + 1):
                    if c0 >= after_close_char:
                        tswitch = gen_abs_start + j
                        tswitch_v = True
                        break
                # If the model emitted </think> but then stopped, Tswitch is
                # invalid (there is no token after the close tag).

    # Pack in canonical order
    label_values: dict[str, tuple[int, bool]] = {
        "P0": (p0_abs, True),
        "P2": (p2_abs, p2_valid),
        "T0": (t0 if t0 >= 0 else 0, t0_v),
        "T25": (t25 if t25 >= 0 else 0, t25_v),
        "T50": (t50 if t50 >= 0 else 0, t50_v),
        "T75": (t75 if t75 >= 0 else 0, t75_v),
        "Tend": (tend if tend >= 0 else 0, tend_v),
        "Tswitch": (tswitch if tswitch >= 0 else 0, tswitch_v),
    }
    return [PositionInfo(label=lab, token_index=label_values[lab][0], valid=label_values[lab][1]) for lab in ALL_POSITIONS]


def build_token_char_spans(
    tokenizer, generated_token_ids: Sequence[int]
) -> tuple[str, list[tuple[int, int]]]:
    """Return ``(generation_text, spans)`` for a sequence of generation tokens.

    ``spans[i]`` is the half-open ``[char_start, char_end)`` range in
    ``generation_text`` occupied by token ``generated_token_ids[i]``. Computed
    by incremental decode: we decode prefixes of increasing length and diff
    the lengths. This is the only decoder-agnostic way to get per-token offsets
    that works with SentencePiece, BPE, and tiktoken tokenizers alike.

    The alternative (``tokenizer(text, return_offsets_mapping=True)``) only
    works for *fast* tokenizers and only on already-tokenized text; it does
    not give reliable offsets for ids that were produced by ``generate()`` and
    then round-tripped.
    """
    ids = list(generated_token_ids)
    spans: list[tuple[int, int]] = []
    prev_len = 0
    # We decode cumulative prefixes. For most tokenizers this is O(n) in the
    # total character count because each decode is O(current_len). Acceptable
    # for <=32K tokens.
    full_text = tokenizer.decode(ids, skip_special_tokens=False)
    # Fast path: single-token-at-a-time decode produces spans that reconstruct
    # the full text.
    running = ""
    for i in range(len(ids)):
        partial = tokenizer.decode(ids[: i + 1], skip_special_tokens=False)
        start = prev_len
        end = len(partial)
        # Defensive: some tokenizers normalize whitespace so len(partial) can
        # briefly shrink (e.g. trailing whitespace trimmed). Clamp end >= start.
        if end < start:
            end = start
        spans.append((start, end))
        prev_len = end
        running = partial
    # If cumulative decode drifts from the full decode (rare: BPE
    # normalization), override full_text with the cumulative one so spans
    # stay consistent.
    if running != full_text:
        full_text = running
    return full_text, spans
