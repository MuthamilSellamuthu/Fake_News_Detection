"""Module 8 — deterministic token attribution by occlusion.

WHY OCCLUSION IS THE PRIMARY METHOD. Module 4's classifier takes a 768-d
sentence embedding, not tokens. There is no token-level object inside it to
attribute to. Token attribution therefore has to go through the whole path --
text -> XLM-R -> mean-pool -> CE-MLP -- and occlusion is the honest way to do
that without gradients: mask one token, re-encode, measure how P(fake) moves.

    attribution(token_i) = P(fake | full text) - P(fake | text with token_i masked)

Positive attribution means the token pushed the prediction TOWARDS fake.

Cost is one forward pass per token, which is why the config caps tokens per
record and samples the split rather than explaining everything.

Determinism: no sampling, no background distribution, no random baseline. The
same text yields bit-identical attributions, which is what makes the stability
check meaningful rather than decorative.

`predict_fn` is injected -- a callable mapping a list of strings to P(fake) --
so every function here is testable without XLM-R.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

OCCLUSION = "occlusion"


@dataclass
class TokenAttribution:
    tokens: list[str]
    attributions: list[float]
    method: str
    base_probability: float
    n_tokens: int = 0
    truncated: bool = False
    seconds: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        self.n_tokens = len(self.tokens)

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens, "attributions": self.attributions,
                "method": self.method, "base_probability": self.base_probability,
                "n_tokens": self.n_tokens, "truncated": self.truncated,
                "seconds": round(self.seconds, 4), "error": self.error}

    def top_tokens(self, k: int = 10) -> dict[str, list[dict[str, Any]]]:
        """Highest positive and most negative contributors, ties by position."""
        idx = sorted(range(len(self.attributions)),
                     key=lambda i: (-self.attributions[i], i))
        pos = [{"token": self.tokens[i], "attribution": float(self.attributions[i]),
                "position": i} for i in idx[:k] if self.attributions[i] > 0]
        neg = [{"token": self.tokens[i], "attribution": float(self.attributions[i]),
                "position": i} for i in reversed(idx[-k:]) if self.attributions[i] < 0]
        return {"top_positive_tokens": pos, "top_negative_tokens": neg}


def simple_tokenize(text: str) -> list[str]:
    """Whitespace tokenisation, used only when no tokenizer is supplied.

    Deliberately NOT a subword tokenizer: occlusion masks whole words so the
    explanation is readable. Tamil is whitespace-separated at word level, so
    this preserves whole Tamil words rather than shattering them the way a
    naive regex would (a bug this project already hit in Module 6's BM25).
    """
    return text.split()


def occlusion_attribution(
    text: str,
    predict_fn: Callable[[Sequence[str]], np.ndarray],
    tokenize_fn: Callable[[str], list[str]] | None = None,
    mask_token: str = "",
    max_tokens: int = 128,
    batch_size: int = 16,
) -> TokenAttribution:
    """Attribution by masking one token at a time.

    Masking with the empty string removes the token entirely; a mask token can
    be supplied instead when the encoder has one. Which is used matters for
    interpretation, so it is recorded by the caller in the manifest.
    """
    t0 = time.perf_counter()
    tokens = (tokenize_fn or simple_tokenize)(text)
    truncated = len(tokens) > max_tokens
    tokens = tokens[:max_tokens]

    if not tokens:
        return TokenAttribution([], [], OCCLUSION, 0.0, truncated=False,
                                seconds=time.perf_counter() - t0,
                                error="empty text after tokenisation")

    base = float(np.asarray(predict_fn([text])).ravel()[0])
    variants = []
    for i in range(len(tokens)):
        kept = list(tokens)
        if mask_token:
            kept[i] = mask_token
        else:
            kept.pop(i)
        variants.append(" ".join(kept))

    probs: list[float] = []
    for i in range(0, len(variants), max(1, batch_size)):
        probs.extend(np.asarray(predict_fn(variants[i:i + batch_size])
                                ).ravel().astype(float).tolist())
    if len(probs) != len(tokens):
        raise RuntimeError(f"predict_fn returned {len(probs)} values for "
                           f"{len(tokens)} occlusions")

    attributions = [float(base - p) for p in probs]
    if any(math.isnan(a) or math.isinf(a) for a in attributions):
        raise ValueError("attribution contains NaN or Inf; predict_fn is unstable")

    return TokenAttribution(tokens, attributions, OCCLUSION, base,
                            truncated=truncated, seconds=time.perf_counter() - t0)


def attribution_concentration(attributions: Sequence[float],
                              top_k: Sequence[int] = (1, 3, 5, 10)) -> dict[str, float]:
    """Share of total |attribution| mass held by the top-k tokens.

    Uses absolute values: a strongly negative token is as much "explanation" as
    a positive one. A flat distribution means the explanation identifies no
    particular evidence, which is itself worth reporting.
    """
    a = np.abs(np.asarray(attributions, dtype=np.float64))
    total = float(a.sum())
    if total <= 0 or len(a) == 0:
        return {f"top_{k}": 0.0 for k in top_k}
    order = np.sort(a)[::-1]
    return {f"top_{k}": float(order[:k].sum() / total) for k in top_k}


def deletion_test(text: str, attribution: TokenAttribution,
                  predict_fn: Callable[[Sequence[str]], np.ndarray],
                  fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
                  mask_token: str = "") -> dict[str, Any]:
    """Remove the highest-attribution tokens; a faithful explanation moves P.

    Reported as a curve plus the mean drop. No single pass/fail threshold is
    imposed -- what counts as a large enough drop is corpus-dependent, and
    inventing a cutoff would manufacture a verdict.
    """
    if not attribution.tokens:
        return {"applicable": False, "reason": "no tokens"}
    order = sorted(range(len(attribution.attributions)),
                   key=lambda i: (-attribution.attributions[i], i))
    base = attribution.base_probability
    curve = []
    for frac in fractions:
        n = max(1, int(round(frac * len(attribution.tokens))))
        drop = set(order[:n])
        kept = [(mask_token if mask_token else None) if i in drop else t
                for i, t in enumerate(attribution.tokens)]
        variant = " ".join(t for t in kept if t is not None)
        p = float(np.asarray(predict_fn([variant])).ravel()[0])
        curve.append({"fraction": float(frac), "n_removed": n,
                      "probability": p, "delta": float(p - base)})
    return {"applicable": True, "base_probability": base, "curve": curve,
            "mean_absolute_delta": float(np.mean([abs(c["delta"]) for c in curve])),
            "definition": ("P(fake) after removing the top-attribution tokens, "
                           "minus P(fake) on the full text.")}


def insertion_test(text: str, attribution: TokenAttribution,
                   predict_fn: Callable[[Sequence[str]], np.ndarray],
                   fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.5)) -> dict[str, Any]:
    """Build the input up from nothing using the highest-attribution tokens.

    Tokens are inserted in their ORIGINAL ORDER, not attribution order, so the
    fragment stays readable and the encoder is not handed scrambled text.
    """
    if not attribution.tokens:
        return {"applicable": False, "reason": "no tokens"}
    order = sorted(range(len(attribution.attributions)),
                   key=lambda i: (-attribution.attributions[i], i))
    base = attribution.base_probability
    curve = []
    for frac in fractions:
        n = max(1, int(round(frac * len(attribution.tokens))))
        keep = sorted(order[:n])
        variant = " ".join(attribution.tokens[i] for i in keep)
        p = float(np.asarray(predict_fn([variant])).ravel()[0])
        curve.append({"fraction": float(frac), "n_inserted": n,
                      "probability": p, "recovery": float(p - base)})
    return {"applicable": True, "base_probability": base, "curve": curve,
            "definition": ("P(fake) using only the top-attribution tokens (in "
                           "original order), minus P(fake) on the full text. "
                           "Values near zero mean the fragment already recovers "
                           "the prediction.")}


def stability_check(text: str,
                    explain_fn: Callable[[str], TokenAttribution],
                    n_repeats: int = 2) -> dict[str, Any]:
    """Deterministic settings promise identical output; verify it."""
    runs = [explain_fn(text) for _ in range(max(2, int(n_repeats)))]
    first = runs[0].attributions
    identical = all(len(r.attributions) == len(first)
                    and all(abs(a - b) < 1e-12 for a, b in zip(r.attributions, first))
                    for r in runs[1:])
    maxdiff = 0.0
    for r in runs[1:]:
        if len(r.attributions) == len(first):
            maxdiff = max([maxdiff] + [abs(a - b) for a, b in zip(r.attributions, first)])
    return {"n_repeats": len(runs), "identical": bool(identical),
            "max_absolute_difference": float(maxdiff),
            "definition": "Bit-identical attributions across repeated runs."}
