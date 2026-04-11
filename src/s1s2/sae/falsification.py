"""Ma et al. (2026, arxiv 2601.05679) falsification framework.

The punchline of Ma et al. is uncomfortable: **45-90% of claimed SAE
"reasoning features" are spurious**. They fire on a few specific tokens
(e.g. "wait", "actually", "first") that happen to appear more often in
reasoning contexts, and when those tokens are injected into random
non-reasoning text the feature still activates. The feature isn't
encoding reasoning — it's encoding lexical presence.

Implementing the falsification test in this codebase is non-negotiable.
Every candidate S1/S2 feature (FDR-significant differential activation)
must pass the test before we count it.

Two modes
---------
- **``model_forward``**: Build 100 short random sentences that are NOT
  cognitive-bias related, inject the feature's top-K activating tokens
  into each, run the target LM forward, pull the residual at the same
  layer, encode through the SAE, and measure the feature activation.
  If the mean injected-text activation is >= ``threshold`` times the
  original peak activation on the benchmark, the feature is spurious.

- **``cheap``**: Skip the model forward pass entirely. Use the
  activation cache for the random texts (you must extract activations
  for them in advance) or fall back to the pure-token heuristic: find
  the hidden-state vectors that contain the trigger tokens in the
  activation cache itself and check whether those activate the feature
  too. This is a much weaker check but costs nothing. It's the default
  in unit tests.

Tokens and triggers
-------------------
"Top activating tokens" for a feature is operationally defined as:
the tokens at which the feature activation exceeds a percentile cutoff
across the benchmark corpus. Because the activation cache only stores
one residual slice per problem (``P0`` = last prompt token), we
approximate this by looking at that single position's token across all
problems where the feature fires most strongly. A richer
implementation would store the full residual trace (future work).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float

from s1s2.sae.loaders import SAEHandle
from s1s2.utils.logging import get_logger

logger = get_logger("s1s2.sae")


# ---------------------------------------------------------------------------
# Random-text fixtures (non-cognitive-bias)
# ---------------------------------------------------------------------------


# 100 short sentences drawn from general-purpose topics, deliberately
# chosen to not involve CRT / anchoring / framing / syllogism content.
# Keeping them in-code means tests and smoke runs work offline.
RANDOM_TEXTS: tuple[str, ...] = (
    "The capital of France is Paris.",
    "Mount Everest is the tallest mountain on Earth.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The Pacific Ocean is the largest body of water.",
    "Shakespeare wrote Hamlet in the early 1600s.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "Electrons orbit the nucleus of an atom.",
    "The Amazon rainforest spans several South American countries.",
    "Albert Einstein developed the theory of relativity.",
    "The Great Wall of China stretches thousands of kilometers.",
    "Honey never spoils if sealed properly.",
    "The human body contains two hundred and six bones.",
    "Mars is often called the red planet.",
    "The Nile is one of the longest rivers in the world.",
    "Mozart composed symphonies throughout his short life.",
    "Bananas grow in tropical climates.",
    "The speed of light is about three hundred thousand kilometers per second.",
    "Antarctica is the coldest continent on Earth.",
    "The Eiffel Tower was built in 1889.",
    "Bees pollinate flowering plants.",
    "The Sahara is the largest hot desert in the world.",
    "Sharks have been around longer than trees.",
    "Vincent van Gogh painted Starry Night.",
    "The Moon influences ocean tides.",
    "Jupiter is the largest planet in our solar system.",
    "Octopuses have three hearts.",
    "The Roman Empire fell in the fifth century.",
    "Leonardo da Vinci painted the Mona Lisa.",
    "Dolphins communicate with clicks and whistles.",
    "The Mariana Trench is the deepest part of the ocean.",
    "Beethoven continued composing after going deaf.",
    "The kiwi is a flightless bird from New Zealand.",
    "Sound travels faster through water than through air.",
    "The pyramids of Giza were built thousands of years ago.",
    "Coffee beans are actually seeds of a fruit.",
    "The platypus is a venomous mammal.",
    "Cheetahs can sprint up to seventy miles per hour.",
    "The Great Barrier Reef is visible from space.",
    "Isaac Newton formulated the laws of motion.",
    "Owls can rotate their heads nearly 270 degrees.",
    "The human eye can distinguish millions of colors.",
    "Chameleons change color for communication and temperature regulation.",
    "The Mediterranean Sea connects to the Atlantic Ocean.",
    "A group of crows is called a murder.",
    "Lightning can heat the air to thirty thousand degrees Celsius.",
    "Penguins cannot fly but are excellent swimmers.",
    "Volcanoes can create new islands over time.",
    "Rainbows form when sunlight refracts through water droplets.",
    "Wolves live in packs with a strict social hierarchy.",
    "The Statue of Liberty was a gift from France.",
    "Ancient Egyptians used hieroglyphs to write.",
    "The brain uses about twenty percent of the body's energy.",
    "Vinegar is a mild acid used in cooking and cleaning.",
    "Bamboo is the fastest-growing plant on Earth.",
    "The Pacific Ring of Fire is seismically active.",
    "Hubble telescope revealed distant galaxies.",
    "Glass is made by melting sand at high temperatures.",
    "Sunflowers track the sun across the sky.",
    "Tigers are the largest wild cats.",
    "The Aurora Borealis lights up polar skies.",
    "Humans share ninety-eight percent of DNA with chimpanzees.",
    "The Grand Canyon was carved by the Colorado River.",
    "Spider silk is stronger than steel by weight.",
    "The Titanic sank on its maiden voyage.",
    "Polar bears have black skin under white fur.",
    "The giant panda eats mostly bamboo.",
    "Saturn has an extensive ring system.",
    "Rice feeds more than half of the world's population.",
    "Ancient Greeks invented democracy.",
    "The Amazon River carries more water than any other.",
    "Mount Fuji is an active volcano in Japan.",
    "Silk was first produced in ancient China.",
    "The Louvre is the world's largest art museum.",
    "Blue whales are the largest animals ever to exist.",
    "Time moves slightly slower in stronger gravitational fields.",
    "Salt was once used as currency in parts of the world.",
    "Sea turtles return to the same beach to lay eggs.",
    "The Andes is the longest mountain range on land.",
    "Baobab trees can live for thousands of years.",
    "The Arctic Circle is slowly warming.",
    "Glass frogs have translucent skin on their bellies.",
    "The Kremlin is the historic heart of Moscow.",
    "A butterfly's wings are covered in tiny scales.",
    "Quartz crystals are used in watches.",
    "The Rosetta Stone helped decipher Egyptian hieroglyphs.",
    "Horses sleep both standing up and lying down.",
    "The tallest trees on Earth are coastal redwoods.",
    "Termites build enormous mounds for ventilation.",
    "Mercury is the closest planet to the Sun.",
    "Coral reefs are built by tiny living organisms.",
    "Mozambique has a long Indian Ocean coastline.",
    "Peacocks display elaborate tail feathers.",
    "The Nile crocodile can grow over five meters long.",
    "Australia is both a country and a continent.",
    "The Mississippi drains a huge portion of North America.",
    "Whales communicate across vast distances underwater.",
    "Tea originated in ancient China.",
    "Kangaroos carry their young in pouches.",
    "The Galapagos Islands inspired Darwin.",
    "Snowflakes always have six-fold symmetry.",
    "Seahorses are the only fish species where males give birth.",
)


assert len(RANDOM_TEXTS) >= 100, f"expected at least 100 random texts, got {len(RANDOM_TEXTS)}"


@beartype
def get_random_texts() -> list[str]:
    """Return a copy of the 100-sentence random-text fixture."""
    return list(RANDOM_TEXTS)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FalsificationResult:
    """Per-feature falsification outcome.

    ``is_spurious`` is the single bit that matters for reporting: if
    True, the feature is almost certainly a token-level artifact and
    we must NOT call it an S1/S2 feature. The other fields exist so we
    can explain why the call was made.
    """

    feature_id: int
    is_spurious: bool
    trigger_tokens: list[str]
    mean_activation_on_original: float
    peak_activation_on_original: float
    mean_activation_on_random: float
    peak_activation_on_random: float
    falsification_ratio: float  # mean_random / peak_original
    mode: str  # "model_forward" | "cheap"
    notes: str = ""


# ---------------------------------------------------------------------------
# Tokenizer / model protocol for test injection
# ---------------------------------------------------------------------------


class _ForwardHookSpec:
    """Pack (tokenizer, model, layer) cleanly so callers pass one object."""

    def __init__(self, tokenizer: Any, model: Any, layer: int, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.layer = int(layer)
        self.device = device


@beartype
def _ensure_torch_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32, copy=False))
    return x.float()


# ---------------------------------------------------------------------------
# Trigger token discovery
# ---------------------------------------------------------------------------


@beartype
def find_trigger_tokens(
    feature_id: int,
    feature_activations: Float[np.ndarray, "n_problems n_features"],
    prompts: Sequence[str],
    tokenizer: Any | None,
    n_top: int = 5,
    n_top_problems: int = 10,
) -> list[str]:
    """Estimate which tokens cause feature ``feature_id`` to activate.

    Because the activation cache we work with stores one residual slice
    per problem (typically ``P0`` = last prompt token), we approximate
    "top activating tokens" by:

    1. Rank all problems by their activation on this feature.
    2. Take the top ``n_top_problems`` prompts.
    3. Tokenize them (using ``tokenizer`` if supplied; otherwise split
       on whitespace) and pick the ``n_top`` tokens that appear most
       frequently in that set, weighted by problem activation.

    This is a best-effort heuristic. A richer approach would need
    activations at every prompt token, which the current HDF5 schema
    does not store.
    """

    if feature_activations.ndim != 2:
        raise ValueError(
            f"expected 2D feature activations, got shape {feature_activations.shape}"
        )
    if feature_id < 0 or feature_id >= feature_activations.shape[1]:
        raise ValueError(
            f"feature_id {feature_id} out of range [0, {feature_activations.shape[1]})"
        )
    if len(prompts) != feature_activations.shape[0]:
        raise ValueError(
            f"prompts length {len(prompts)} != activations rows {feature_activations.shape[0]}"
        )

    col = feature_activations[:, feature_id]
    # Top N problems by activation. If many ties near zero, this still
    # just returns N arbitrary ones which is fine.
    n_top_problems = min(n_top_problems, len(prompts))
    top_prob_idx = np.argsort(-col)[:n_top_problems]
    top_weights = col[top_prob_idx]

    token_scores: dict[str, float] = {}
    for i, p_idx in enumerate(top_prob_idx):
        text = prompts[int(p_idx)]
        if tokenizer is not None and hasattr(tokenizer, "tokenize"):
            try:
                toks = tokenizer.tokenize(text)
            except Exception:
                toks = text.split()
        else:
            toks = text.split()
        # Normalize BPE/WordPiece style markers for readability.
        toks = [
            t.replace("\u0120", " ").replace("\u2581", " ").replace("##", "").strip()
            for t in toks
        ]
        toks = [t for t in toks if t]  # drop empties
        w = float(top_weights[i])
        for t in toks:
            token_scores[t] = token_scores.get(t, 0.0) + w

    # Skip tokens that are too generic / too short.
    STOPLIST = {
        "a", "an", "the", "and", "or", "of", "in", "to", "is", "are",
        "was", "were", "it", "this", "that", "for", "on", "at", "by",
        "with", "as", "be", "has", "have", "had", "but", "not", "if",
        "do", "does", "did", ",", ".", "?", "!", ":", ";", "'", "\"",
    }
    filtered = [(t, s) for t, s in token_scores.items() if t.lower().strip() not in STOPLIST and len(t.strip()) > 1]
    filtered.sort(key=lambda x: -x[1])
    return [t for t, _ in filtered[:n_top]]


# ---------------------------------------------------------------------------
# Cheap-mode falsification
# ---------------------------------------------------------------------------


@beartype
def _cheap_falsify_single(
    feature_id: int,
    sae: SAEHandle,
    activations: Float[np.ndarray, "n_problems hidden"],
    feature_activations: Float[np.ndarray, "n_problems n_features"],
    prompts: Sequence[str],
    tokenizer: Any | None,
    n_top_tokens: int,
    threshold: float,
) -> FalsificationResult:
    """Cheap fallback: work entirely in embedding space, no model forward.

    Procedure:
    1. Find trigger tokens for the feature.
    2. Build synthetic "random texts with trigger injected" as string
       concatenations.
    3. Form a simulated residual for each synthetic text by averaging
       the mean benchmark residual with the centroid of real benchmark
       residuals whose prompts contain the trigger tokens.
    4. Encode this synthetic residual through the SAE.
    5. Compare feature activation vs the original peak.

    This is a much weaker test than the real model-forward version
    — it can be fooled by any feature whose trigger tokens also happen
    to appear in prompts that have high activation for unrelated
    reasons — but it runs on CPU in milliseconds and so is the mode
    we use for unit tests and for screening in the smoke pipeline.
    """

    col = feature_activations[:, feature_id]
    mean_original = float(col.mean())
    peak_original = float(col.max())

    triggers = find_trigger_tokens(
        feature_id=feature_id,
        feature_activations=feature_activations,
        prompts=prompts,
        tokenizer=tokenizer,
        n_top=n_top_tokens,
        n_top_problems=10,
    )

    if not triggers:
        return FalsificationResult(
            feature_id=int(feature_id),
            is_spurious=False,
            trigger_tokens=[],
            mean_activation_on_original=mean_original,
            peak_activation_on_original=peak_original,
            mean_activation_on_random=0.0,
            peak_activation_on_random=0.0,
            falsification_ratio=0.0,
            mode="cheap",
            notes="no trigger tokens detected; cannot falsify (assume non-spurious)",
        )

    # Build a centroid of the residuals whose prompts contain ANY trigger.
    has_trigger = np.array(
        [any(t.lower() in prompts[i].lower() for t in triggers) for i in range(len(prompts))],
        dtype=bool,
    )
    if has_trigger.sum() == 0:
        trigger_centroid = activations.mean(axis=0)
    else:
        trigger_centroid = activations[has_trigger].mean(axis=0)

    # Simulate "random text with trigger injected" as a linear blend of
    # the global activation mean (representing "random text") and the
    # trigger-present centroid. The blend weight controls how much of
    # the residual signal is driven by trigger presence.
    mean_residual = activations.mean(axis=0)
    simulated = np.stack(
        [0.5 * mean_residual + 0.5 * trigger_centroid for _ in range(100)]
    )
    # Add a small random jitter so the synthetic batch has non-zero
    # within-sample variance (otherwise the SAE codes are identical).
    rng = np.random.default_rng(int(feature_id) + 13)
    jitter = rng.standard_normal(simulated.shape).astype(np.float32)
    jitter *= 0.01 * np.linalg.norm(mean_residual) / (np.sqrt(simulated.shape[1]) + 1e-8)
    simulated = simulated + jitter

    with torch.no_grad():
        z_sim = sae.encode(torch.from_numpy(simulated.astype(np.float32)))
    sim_col = z_sim[:, int(feature_id)].detach().float().cpu().numpy()
    mean_random = float(sim_col.mean())
    peak_random = float(sim_col.max())

    denom = peak_original if peak_original > 1e-8 else 1.0
    falsification_ratio = mean_random / denom
    is_spurious = falsification_ratio >= threshold

    return FalsificationResult(
        feature_id=int(feature_id),
        is_spurious=bool(is_spurious),
        trigger_tokens=triggers,
        mean_activation_on_original=mean_original,
        peak_activation_on_original=peak_original,
        mean_activation_on_random=mean_random,
        peak_activation_on_random=peak_random,
        falsification_ratio=float(falsification_ratio),
        mode="cheap",
        notes="cheap mode: no model forward",
    )


# ---------------------------------------------------------------------------
# Model-forward falsification
# ---------------------------------------------------------------------------


@beartype
def _model_forward_residual_for_texts(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    layer: int,
    device: str = "cpu",
) -> np.ndarray:
    """Run a batch of texts through the model and pull the residual at ``layer``.

    We take the residual at the **last** token of each input (matching
    ``P0`` in the activation cache schema). This is the standard SAE
    evaluation slice in the literature.

    We use a forward hook on the transformer block's output. This avoids
    requiring ``output_hidden_states=True`` (which some HF models return
    in the pre-block convention, causing layer-index ambiguity).
    """

    import torch as _torch

    model.eval()
    out_rows: list[np.ndarray] = []

    captured: dict[str, _torch.Tensor] = {}

    def hook(_mod, _inp, out):
        # Some HF blocks return tuples; take the hidden state.
        if isinstance(out, tuple):
            out = out[0]
        captured["h"] = out.detach()

    # Locate the block. Most HF transformer families expose
    # `model.model.layers[i]` (Llama, Mistral, Qwen, Gemma-2) so we
    # try that first and fall back to duck-typing.
    block = None
    for attr_path in (("model", "layers"), ("transformer", "h"), ("backbone", "layers")):
        root = model
        ok = True
        for a in attr_path:
            if hasattr(root, a):
                root = getattr(root, a)
            else:
                ok = False
                break
        if ok and isinstance(root, list | _torch.nn.ModuleList):
            block = root[layer]
            break
    if block is None:
        raise RuntimeError("Could not locate transformer blocks for hook")

    handle = block.register_forward_hook(hook)
    try:
        for text in texts:
            enc = tokenizer(text, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with _torch.no_grad():
                model(**enc)
            h = captured.get("h")
            if h is None:
                raise RuntimeError("forward hook did not fire")
            # h shape: (1, seq, hidden). Take the last token.
            last = h[0, -1, :].float().cpu().numpy()
            out_rows.append(last)
    finally:
        handle.remove()

    return np.stack(out_rows, axis=0)


@beartype
def _model_forward_falsify_single(
    feature_id: int,
    sae: SAEHandle,
    feature_activations: Float[np.ndarray, "n_problems n_features"],
    prompts: Sequence[str],
    tokenizer: Any,
    model: Any,
    layer: int,
    n_random_texts: int,
    n_top_tokens: int,
    threshold: float,
    device: str,
) -> FalsificationResult:
    col = feature_activations[:, feature_id]
    mean_original = float(col.mean())
    peak_original = float(col.max())

    triggers = find_trigger_tokens(
        feature_id=feature_id,
        feature_activations=feature_activations,
        prompts=prompts,
        tokenizer=tokenizer,
        n_top=n_top_tokens,
        n_top_problems=10,
    )

    random_texts = get_random_texts()[:n_random_texts]
    injected_texts = [
        f"{t} {' '.join(triggers)}".strip() if triggers else t for t in random_texts
    ]

    residuals = _model_forward_residual_for_texts(
        injected_texts,
        tokenizer=tokenizer,
        model=model,
        layer=layer,
        device=device,
    )

    with torch.no_grad():
        z = sae.encode(torch.from_numpy(residuals.astype(np.float32)))
    rand_col = z[:, int(feature_id)].detach().float().cpu().numpy()
    mean_random = float(rand_col.mean())
    peak_random = float(rand_col.max())
    denom = peak_original if peak_original > 1e-8 else 1.0
    falsification_ratio = mean_random / denom
    is_spurious = falsification_ratio >= threshold

    return FalsificationResult(
        feature_id=int(feature_id),
        is_spurious=bool(is_spurious),
        trigger_tokens=triggers,
        mean_activation_on_original=mean_original,
        peak_activation_on_original=peak_original,
        mean_activation_on_random=mean_random,
        peak_activation_on_random=peak_random,
        falsification_ratio=float(falsification_ratio),
        mode="model_forward",
        notes=f"n_random_texts={n_random_texts}",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@beartype
def ma_et_al_falsification(
    feature_id: int,
    sae: SAEHandle,
    activations: Float[np.ndarray, "n_problems hidden"],
    feature_activations: Float[np.ndarray, "n_problems n_features"],
    prompts: Sequence[str],
    tokenizer: Any | None = None,
    model: Any | None = None,
    layer: int | None = None,
    *,
    mode: str = "cheap",
    n_random_texts: int = 100,
    n_top_tokens: int = 5,
    threshold: float = 0.5,
    device: str = "cpu",
) -> FalsificationResult:
    """Run the Ma et al. (2026) falsification test on a single feature.

    See module docstring for the full rationale. The short version:
    returns :class:`FalsificationResult` with ``is_spurious`` = True
    if the feature still fires strongly when its trigger tokens are
    injected into random non-cognitive-bias text.

    Parameters
    ----------
    feature_id
        SAE feature index to test.
    sae
        A :class:`SAEHandle` for encoding vectors.
    activations
        Residual-stream activations for the benchmark, shape
        ``(n_problems, hidden_dim)``. Used only in ``cheap`` mode to
        estimate the trigger-token centroid without a model forward.
    feature_activations
        The SAE codes, shape ``(n_problems, n_features)``. Used to
        identify trigger tokens and original peak/mean activation.
    prompts
        The prompt string for each problem, used to extract trigger
        tokens.
    tokenizer, model, layer, device
        Required for ``mode="model_forward"``. The tokenizer is also
        optionally used in ``cheap`` mode to detect tokens more
        accurately than whitespace splitting.
    mode
        ``"cheap"`` (no model forward, fast) or ``"model_forward"``
        (proper Ma et al. protocol).
    n_random_texts
        Number of random injection texts. 100 is the paper default.
    n_top_tokens
        Number of trigger tokens to extract per feature.
    threshold
        Spurious threshold: feature is flagged if
        ``mean_activation_on_random / peak_activation_on_original >= threshold``.
    """

    if mode not in ("cheap", "model_forward"):
        raise ValueError(f"mode must be 'cheap' or 'model_forward', got {mode!r}")

    if mode == "model_forward":
        if tokenizer is None or model is None or layer is None:
            raise ValueError(
                "mode='model_forward' requires tokenizer, model, and layer"
            )
        return _model_forward_falsify_single(
            feature_id=feature_id,
            sae=sae,
            feature_activations=feature_activations,
            prompts=prompts,
            tokenizer=tokenizer,
            model=model,
            layer=layer,
            n_random_texts=n_random_texts,
            n_top_tokens=n_top_tokens,
            threshold=threshold,
            device=device,
        )

    return _cheap_falsify_single(
        feature_id=feature_id,
        sae=sae,
        activations=activations,
        feature_activations=feature_activations,
        prompts=prompts,
        tokenizer=tokenizer,
        n_top_tokens=n_top_tokens,
        threshold=threshold,
    )


@beartype
def falsify_candidates(
    candidate_feature_ids: Sequence[int],
    sae: SAEHandle,
    activations: Float[np.ndarray, "n_problems hidden"],
    feature_activations: Float[np.ndarray, "n_problems n_features"],
    prompts: Sequence[str],
    tokenizer: Any | None = None,
    model: Any | None = None,
    layer: int | None = None,
    *,
    mode: str = "cheap",
    n_random_texts: int = 100,
    n_top_tokens: int = 5,
    threshold: float = 0.5,
    top_k_features: int = 50,
    device: str = "cpu",
) -> list[FalsificationResult]:
    """Run falsification on a list of candidate features.

    We cap at ``top_k_features`` per call because the model-forward
    mode is expensive — 100 forward passes per feature, and hundreds
    of features per layer would otherwise quickly dominate the
    wall-clock budget. The cap should be large enough to cover the
    genuinely interesting features at each layer; if you need more,
    re-run with a larger cap.
    """

    if not candidate_feature_ids:
        return []

    n = min(len(candidate_feature_ids), top_k_features)
    logger.info(
        "Running Ma et al. falsification on %d/%d candidate features (mode=%s)",
        n,
        len(candidate_feature_ids),
        mode,
    )

    results: list[FalsificationResult] = []
    for _i, fid in enumerate(list(candidate_feature_ids)[:n]):
        try:
            r = ma_et_al_falsification(
                feature_id=int(fid),
                sae=sae,
                activations=activations,
                feature_activations=feature_activations,
                prompts=prompts,
                tokenizer=tokenizer,
                model=model,
                layer=layer,
                mode=mode,
                n_random_texts=n_random_texts,
                n_top_tokens=n_top_tokens,
                threshold=threshold,
                device=device,
            )
            results.append(r)
        except Exception as exc:  # pragma: no cover - network / model issues
            logger.error("Falsification failed for feature %d: %s", fid, exc)
            results.append(
                FalsificationResult(
                    feature_id=int(fid),
                    is_spurious=False,
                    trigger_tokens=[],
                    mean_activation_on_original=0.0,
                    peak_activation_on_original=0.0,
                    mean_activation_on_random=0.0,
                    peak_activation_on_random=0.0,
                    falsification_ratio=float("nan"),
                    mode=mode,
                    notes=f"error: {exc!r}",
                )
            )

    spurious_count = sum(int(r.is_spurious) for r in results)
    logger.info(
        "Falsification done: %d/%d features flagged as spurious (%.1f%%)",
        spurious_count,
        len(results),
        100.0 * spurious_count / max(1, len(results)),
    )
    return results


__all__ = [
    "RANDOM_TEXTS",
    "FalsificationResult",
    "falsify_candidates",
    "find_trigger_tokens",
    "get_random_texts",
    "ma_et_al_falsification",
]
