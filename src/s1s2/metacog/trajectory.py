"""Self-correction trajectory analysis on reasoning model thinking traces.

For reasoning models (R1-distill family) we have access to the
``<think>...</think>`` block — the chain-of-thought trace the model
produced before its final answer. The metacognitive monitoring
hypothesis predicts that, on problems where the model self-corrects,
the difficulty detector should activate *before* the correction
("hold on, this doesn't add up") and quiet down afterwards.

This module parses traces for self-correction markers ("wait",
"actually", "let me reconsider", ...), locates them by character
offset, and exposes a small data structure that downstream
visualizations and analyses can hang activations off of.

A self-correction event is operationalized as:

1. The model **stated a candidate answer** (numeric, multiple choice,
   or yes/no) earlier in the trace.
2. **Then** emitted a marker token from the configured set.
3. **Then** the trace continues for at least ``min_post_chars`` more
   characters (so the marker isn't merely the last word of an aborted
   trace).

The "marker followed by more reasoning" requirement filters out the
purely decorative "Hmm." cases that the literature warns about
(only ~2.3% of "aha moments" actually drive the answer per Lin et al.).

Caveats
-------
This is a regex-level heuristic. It will overcount on certain trace
styles ("but" appears in many non-correction contexts: "I'll consider
A *but* also B"). The TrueThinking-Score filter that the wider project
uses is the proper way to keep only causally-relevant corrections; here
we expose all candidate events and let the consumer filter.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from beartype import beartype

from s1s2.utils.logging import get_logger

logger = get_logger("s1s2.metacog")


# Sensible default markers; CLI / config can override.
DEFAULT_MARKERS: tuple[str, ...] = (
    "wait",
    "actually",
    "let me reconsider",
    "but",
    "on second thought",
    "hold on",
    "hmm",
    "no, ",
    "scratch that",
    "let me try again",
    "rethink",
)


# Candidate-answer detection patterns. We don't need to recognize the
# correct vocabulary of every benchmark: a coarse-but-correct rule for
# "the model emitted something that looks like an answer commitment" is
# enough.
_ANSWER_COMMITMENT_PATTERNS = (
    re.compile(r"\b(?:the\s+)?answer\s+is\s+(\S+)", re.IGNORECASE),
    re.compile(r"\bI\s+(?:think|believe|guess)\s+(?:the\s+answer\s+is\s+)?(\S+)", re.IGNORECASE),
    re.compile(r"\bso\s+(?:it\s+is|the\s+answer\s+is)\s+(\S+)", re.IGNORECASE),
    re.compile(r"=\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),  # arithmetic
    re.compile(r"\b(?:option\s+)?\(?([abcdABCD])\)?\b"),
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SelfCorrectionEvent:
    """One marker hit inside one trace.

    Offsets are in characters within the supplied trace string.
    ``has_prior_commitment`` is the "an answer was claimed before this
    point" flag, the only thing that distinguishes a real self-correction
    from a decorative reasoning marker.
    """

    marker: str
    char_start: int
    char_end: int
    has_prior_commitment: bool
    prior_answer: str
    post_chars: int
    relative_position: float  # 0..1 along the trace


@dataclass
class TraceParseResult:
    """Per-trace summary suitable for indexing into activation arrays.

    ``self_corrects`` is the headline boolean: at least one marker was
    found AND followed at least ``min_post_chars`` characters of new
    reasoning AND was preceded by a candidate answer.

    The ``events`` list carries every match for downstream filtering.
    """

    problem_idx: int
    trace_chars: int
    events: list[SelfCorrectionEvent] = field(default_factory=list)
    self_corrects: bool = False
    n_markers_total: int = 0
    n_markers_with_prior: int = 0
    first_correction_char: int | None = None
    first_correction_relative: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@beartype
def _find_prior_commitment(trace: str, up_to: int) -> str | None:
    """Return the most recent answer commitment in ``trace[:up_to]`` (if any)."""
    head = trace[: max(0, up_to)]
    last_match: re.Match | None = None
    last_pos = -1
    for pat in _ANSWER_COMMITMENT_PATTERNS:
        for m in pat.finditer(head):
            if m.start() > last_pos:
                last_pos = m.start()
                last_match = m
    if last_match is None:
        return None
    try:
        return last_match.group(1)
    except (IndexError, ValueError):
        return last_match.group(0)


@beartype
def _compile_marker_patterns(markers: Iterable[str]) -> list[tuple[str, re.Pattern]]:
    """Compile each marker as a case-insensitive regex with word boundaries.

    Multi-word markers ("on second thought") get a literal-with-spaces
    pattern; single-word markers get word-boundary anchors so we don't
    match "waiting" when looking for "wait".
    """
    out = []
    for raw in markers:
        m = raw.strip().lower()
        if not m:
            continue
        if " " in m or m.endswith(",") or m.endswith(":"):
            patt = re.compile(r"(?i)" + re.escape(m))
        else:
            patt = re.compile(r"(?i)\b" + re.escape(m) + r"\b")
        out.append((m, patt))
    return out


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


@beartype
def parse_self_correction(
    trace: str,
    *,
    markers: Sequence[str] = DEFAULT_MARKERS,
    min_post_chars: int = 30,
    problem_idx: int = -1,
) -> TraceParseResult:
    """Find self-correction markers in one thinking trace.

    Parameters
    ----------
    trace
        The thinking text — typically the body of a ``<think>...</think>``
        block. Pass an empty string for non-reasoning models; the
        function will return a result with ``self_corrects=False``.
    markers
        Iterable of marker phrases. Case-insensitive. Single-word
        markers are matched with word boundaries; multi-word markers
        match literally.
    min_post_chars
        Required number of trace characters AFTER the marker for the
        match to count. This filters out markers that are part of an
        aborted final sentence rather than a real "back up and
        reconsider" event.
    problem_idx
        Stored on the returned :class:`TraceParseResult`; useful when
        iterating over the corpus.
    """

    if not trace:
        return TraceParseResult(problem_idx=int(problem_idx), trace_chars=0)

    n = len(trace)
    patterns = _compile_marker_patterns(markers)
    events: list[SelfCorrectionEvent] = []
    n_with_prior = 0

    for marker_text, patt in patterns:
        for m in patt.finditer(trace):
            char_start, char_end = m.span()
            post_chars = n - char_end
            if post_chars < min_post_chars:
                continue
            prior = _find_prior_commitment(trace, char_start)
            has_prior = prior is not None and len(prior) >= 1
            if has_prior:
                n_with_prior += 1
            events.append(
                SelfCorrectionEvent(
                    marker=marker_text,
                    char_start=int(char_start),
                    char_end=int(char_end),
                    has_prior_commitment=bool(has_prior),
                    prior_answer=str(prior or ""),
                    post_chars=int(post_chars),
                    relative_position=float(char_start / n) if n > 0 else 0.0,
                )
            )

    events.sort(key=lambda e: e.char_start)
    self_corrects = any(e.has_prior_commitment for e in events)

    first_with_prior = next((e for e in events if e.has_prior_commitment), None)
    first_char = first_with_prior.char_start if first_with_prior else None
    first_rel = first_with_prior.relative_position if first_with_prior else None

    return TraceParseResult(
        problem_idx=int(problem_idx),
        trace_chars=int(n),
        events=events,
        self_corrects=bool(self_corrects),
        n_markers_total=len(events),
        n_markers_with_prior=int(n_with_prior),
        first_correction_char=first_char,
        first_correction_relative=first_rel,
    )


@beartype
def parse_trace_corpus(
    thinking_texts: Sequence[str],
    *,
    markers: Sequence[str] = DEFAULT_MARKERS,
    min_post_chars: int = 30,
) -> list[TraceParseResult]:
    """Parse a list of traces, returning a parallel list of results.

    Convenience wrapper that simply maps :func:`parse_self_correction`
    over the corpus and tags each result with its problem index. Logs
    the corpus-level self-correction rate at INFO so we can spot
    obvious overcounting (>50% would be alarming for the CRT/syllogism
    benchmarks).
    """
    out = [
        parse_self_correction(
            t,
            markers=markers,
            min_post_chars=min_post_chars,
            problem_idx=i,
        )
        for i, t in enumerate(thinking_texts)
    ]
    if out:
        rate = sum(int(r.self_corrects) for r in out) / len(out)
        logger.info(
            "self-correction parser: %d/%d traces self-correct (%.1f%%)",
            sum(int(r.self_corrects) for r in out),
            len(out),
            100.0 * rate,
        )
    return out


# ---------------------------------------------------------------------------
# Activation alignment for trajectories
# ---------------------------------------------------------------------------


@beartype
def difficulty_trajectory_means(
    feature_activations: np.ndarray,
    parse_results: Sequence[TraceParseResult],
    *,
    pre_window: tuple[float, float] = (-0.20, 0.0),
    post_window: tuple[float, float] = (0.0, 0.20),
) -> dict[str, float | int]:
    """Average a feature's activation in pre/post windows around the first marker.

    The activation cache stores per-problem activations at canonical
    positions (T0, T25, T50, T75, Tend), so the most we can do without
    re-extracting full traces is bin the canonical positions by where
    the first self-correction event landed.

    Specifically: we use ``feature_activations`` of shape
    ``(n_problems, n_positions)`` and a list of position labels along
    the second axis where each label maps to a relative position in
    ``[0, 1]`` (T0=0, T25=0.25, T50=0.5, T75=0.75, Tend=1.0).

    Returns the mean pre-marker and mean post-marker activations across
    all self-correcting problems, and the count.

    For non-self-correcting problems we report the mean over the same
    canonical positions across the corpus, as a baseline. The headline
    statistic is the *difference* between pre and post on the
    self-correcting subset.
    """

    if feature_activations.ndim != 2:
        raise ValueError(
            f"feature_activations must be 2D (n_problems, n_positions); "
            f"got shape {feature_activations.shape}"
        )

    # Use 5 evenly spaced canonical positions; matches T0, T25, T50, T75, Tend.
    n_positions = feature_activations.shape[1]
    if n_positions <= 1:
        return {
            "n_self_correcting": 0,
            "mean_pre": 0.0,
            "mean_post": 0.0,
            "delta_post_minus_pre": 0.0,
            "n_baseline": 0,
            "baseline_mean": 0.0,
        }
    positions_rel = np.linspace(0.0, 1.0, n_positions)

    pre_low, pre_high = pre_window
    post_low, post_high = post_window

    pre_means: list[float] = []
    post_means: list[float] = []
    baseline_means: list[float] = []
    n_sc = 0
    for r in parse_results:
        rel = r.first_correction_relative
        row = feature_activations[r.problem_idx]
        if rel is None:
            baseline_means.append(float(row.mean()))
            continue
        pre_mask = (positions_rel >= rel + pre_low) & (positions_rel <= rel + pre_high)
        post_mask = (positions_rel >= rel + post_low) & (positions_rel <= rel + post_high)
        if not pre_mask.any() or not post_mask.any():
            # Pad: pick the closest single position to each window center.
            pre_idx = int(np.argmin(np.abs(positions_rel - (rel + pre_low / 2))))
            post_idx = int(np.argmin(np.abs(positions_rel - (rel + post_high / 2))))
            pre_val = float(row[pre_idx])
            post_val = float(row[post_idx])
        else:
            pre_val = float(row[pre_mask].mean())
            post_val = float(row[post_mask].mean())
        pre_means.append(pre_val)
        post_means.append(post_val)
        n_sc += 1

    return {
        "n_self_correcting": int(n_sc),
        "mean_pre": float(np.mean(pre_means)) if pre_means else 0.0,
        "mean_post": float(np.mean(post_means)) if post_means else 0.0,
        "delta_post_minus_pre": (
            float(np.mean(post_means) - np.mean(pre_means)) if (pre_means and post_means) else 0.0
        ),
        "n_baseline": len(baseline_means),
        "baseline_mean": (float(np.mean(baseline_means)) if baseline_means else 0.0),
    }


__all__ = [
    "DEFAULT_MARKERS",
    "SelfCorrectionEvent",
    "TraceParseResult",
    "difficulty_trajectory_means",
    "parse_self_correction",
    "parse_trace_corpus",
]
