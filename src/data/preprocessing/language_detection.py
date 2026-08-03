"""Language detection for Module 2.

Project scope only needs to distinguish Tamil vs. English (plus a
catch-all for anything else that slips through). Two-stage approach:

  1. Unicode-block heuristic: if a large-enough fraction of the
     alphabetic characters in the text fall in the Tamil block
     (U+0B80-U+0BFF), call it Tamil outright. This is fast, has zero
     false negatives for real Tamil text, and — critically — doesn't
     get confused by short strings the way statistical detectors do.
  2. Statistical fallback (`langdetect`) for everything else, mapped
     down to {"en", "ta", "other"}. `langdetect` is non-deterministic
     across runs unless seeded, so the seed is fixed at import time.

Kept separate from `cleaning.py` because detection should run on text
that still has enough signal to classify — cleaning already happens
before this is called in the pipeline, which is fine since none of the
cleaning steps strip Tamil or Latin letters.
"""

from __future__ import annotations

import re

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deterministic langdetect output

_TAMIL_RE = re.compile("[஀-௿]")
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)  # any alphabetic char, any script

SUPPORTED_LANGUAGES = ("ta", "en")
FALLBACK_LABEL = "other"
UNKNOWN_LABEL = "unknown"


def tamil_char_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are in the Tamil Unicode block."""
    if not text:
        return 0.0
    alpha_chars = _ALPHA_RE.findall(text)
    if not alpha_chars:
        return 0.0
    tamil_chars = _TAMIL_RE.findall(text)
    return len(tamil_chars) / len(alpha_chars)


def detect_language(text: str, tamil_threshold: float = 0.3) -> str:
    """Return one of "ta", "en", "other", or "unknown" (empty/undetectable input)."""
    if not text or not text.strip():
        return UNKNOWN_LABEL

    if tamil_char_ratio(text) >= tamil_threshold:
        return "ta"

    try:
        detected = detect(text)
    except LangDetectException:
        return UNKNOWN_LABEL

    if detected in SUPPORTED_LANGUAGES:
        return detected
    return FALLBACK_LABEL
