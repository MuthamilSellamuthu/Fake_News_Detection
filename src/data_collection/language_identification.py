"""Lightweight language identification — Module 1.

Scope: this project only needs to distinguish Tamil vs. English (plus a
catch-all "other"/"unknown" for anything that slips through). A
two-stage approach keeps this both fast and reliable:

  1. Unicode-block heuristic (`tamil_char_ratio`): if a large-enough
     fraction of the alphabetic characters in a text fall inside the
     Tamil Unicode block (U+0B80-U+0BFF), classify it "ta" outright.
     This is a single regex pass, has effectively zero false negatives
     for genuine Tamil text, and — unlike a statistical detector —
     doesn't get confused by short strings (headlines, short verdicts).
  2. Statistical fallback (`langdetect`) for anything the heuristic
     doesn't confidently call Tamil. `langdetect` is a port of Google's
     language-detection library and is non-deterministic across runs
     unless seeded, so the seed is fixed at import time.

Performance note (`identify_languages_bulk`): for a large single-source,
single-language dataset (WELFake and ISOT are both documented English-
only datasets, ~72k and ~45k rows respectively), running the statistical
fallback on every row is ~100k+ redundant `langdetect` calls for a
result that is already known from the dataset's own documentation. The
Tamil-ratio heuristic still runs on every single row — it's cheap
(one regex pass) and is the only thing that could catch real
contamination (e.g. a mis-scraped Tamil row ending up in an English
CSV). `langdetect` is then run only on a bounded random sample, purely
to produce a verifiable statistic for the validation report ("N/300
sampled rows independently confirmed as English"), not to make a
per-row decision. Small Tamil-source datasets (hundreds of rows, not
tens of thousands) go through the full per-row two-stage path.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deterministic langdetect output across runs

_TAMIL_RE = re.compile("[஀-௿]")  # U+0B80-U+0BFF, the Tamil Unicode block
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)  # any alphabetic char, any script

SUPPORTED_LANGUAGES = ("ta", "en")
FALLBACK_LABEL = "other"
UNKNOWN_LABEL = "unknown"


def tamil_char_ratio(text: str) -> float:
    """Fraction of alphabetic characters in `text` that fall in the Tamil Unicode block.

    Returns 0.0 for empty/whitespace-only/non-alphabetic input.
    """
    if not text:
        return 0.0
    alpha_chars = _ALPHA_RE.findall(text)
    if not alpha_chars:
        return 0.0
    tamil_chars = _TAMIL_RE.findall(text)
    return len(tamil_chars) / len(alpha_chars)


def detect_language(text: str, tamil_threshold: float = 0.3) -> str:
    """Classify `text` as one of "ta", "en", "other", or "unknown".

    "unknown" means the input was empty or `langdetect` could not form
    any opinion at all (distinct from "other", a confident-but-
    unsupported language).
    """
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


@dataclass
class BulkLanguageResult:
    languages: list[str]
    sample_verified: int
    sample_size: int
    sample_mismatches: list[tuple[int, str, str]]  # (row_index, expected, detected)


def identify_languages_bulk(
    texts: list[str],
    assumed_language: str,
    tamil_threshold: float = 0.3,
    sample_size: int = 300,
    random_seed: int = 42,
) -> BulkLanguageResult:
    """Tag every row via the fast Tamil-ratio heuristic; verify a sample via `langdetect`.

    Every row that clears `tamil_threshold` is tagged "ta" regardless of
    `assumed_language` (contamination should never be silently
    overwritten). Every other row is tagged `assumed_language` directly
    — no per-row `langdetect` call. A random sample of those rows is
    then independently re-checked with `langdetect` purely to produce a
    verifiable statistic; mismatches are returned (not auto-corrected)
    so a human can decide whether they're real contamination or a
    `langdetect` false alarm on a short/noisy string.
    """
    languages: list[str] = []
    non_tamil_indices: list[int] = []

    for i, text in enumerate(texts):
        if tamil_char_ratio(text or "") >= tamil_threshold:
            languages.append("ta")
        else:
            languages.append(assumed_language)
            non_tamil_indices.append(i)

    rng = random.Random(random_seed)
    sample_indices = rng.sample(non_tamil_indices, k=min(sample_size, len(non_tamil_indices)))

    mismatches: list[tuple[int, str, str]] = []
    verified = 0
    for idx in sample_indices:
        text = texts[idx] or ""
        if not text.strip():
            continue
        try:
            detected = detect(text)
        except LangDetectException:
            continue
        verified += 1
        expected = assumed_language
        detected_mapped = detected if detected in SUPPORTED_LANGUAGES else FALLBACK_LABEL
        if detected_mapped != expected:
            mismatches.append((idx, expected, detected_mapped))

    return BulkLanguageResult(
        languages=languages,
        sample_verified=verified,
        sample_size=len(sample_indices),
        sample_mismatches=mismatches,
    )


def identify_languages_per_row(texts: list[str], tamil_threshold: float = 0.3) -> list[str]:
    """Full two-stage `detect_language` on every row. Only for small datasets."""
    return [detect_language(text, tamil_threshold=tamil_threshold) for text in texts]
