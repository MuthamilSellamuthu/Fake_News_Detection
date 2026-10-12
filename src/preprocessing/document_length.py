"""Document-length analysis and short/invalid-document filtering — Module 2.

Two distinct jobs, kept in one module because they share the same
length computation:

  1. `compute_length_stats` — pure measurement (char/word length
     distribution per language), for the preprocessing report. Never
     modifies data.
  2. `flag_invalid_documents` — decides which rows are unusable
     (fewer than `min_word_count` words) and which are extreme-length
     outliers worth a report note (>= `extreme_length_percentile`).
     Outliers are flagged, not dropped — truncation is a Module 3
     tokenizer concern, not a Module 2 cleaning concern; but rows below
     the word-count floor genuinely carry no usable signal and are
     dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def compute_length_stats(df: pd.DataFrame, text_col: str = "text", language_col: str = "language") -> dict[str, Any]:
    char_len = df[text_col].fillna("").astype(str).str.len()
    word_len = df[text_col].fillna("").astype(str).str.count(r"\S+")

    stats: dict[str, Any] = {"overall": {
        "char_len": _describe(char_len),
        "word_len": _describe(word_len),
    }}

    for lang, group in df.groupby(language_col):
        g_char = group[text_col].fillna("").astype(str).str.len()
        g_word = group[text_col].fillna("").astype(str).str.count(r"\S+")
        stats[lang] = {
            "char_len": _describe(g_char),
            "word_len": _describe(g_word),
        }

    return stats


def _describe(series: pd.Series) -> dict[str, float]:
    if len(series) == 0:
        return {}
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


@dataclass
class InvalidDocumentResult:
    too_short_mask: pd.Series
    extreme_length_mask: pd.Series
    n_too_short: int
    n_extreme_length: int
    extreme_length_threshold_chars: float


def flag_invalid_documents(
    df: pd.DataFrame,
    text_col: str = "text",
    min_word_count: int | dict[str, int] = 5,
    extreme_length_percentile: float = 99.5,
    language_col: str = "language",
) -> InvalidDocumentResult:
    """`min_word_count` accepts either a single int (applied uniformly,
    Module 2's original "Version A" behavior) or a `{language: threshold}`
    dict (language-aware "Version B" behavior — see
    `docs/module2_short_document_review.md` for why a uniform threshold
    systematically over-penalizes Tamil: Tamil is agglutinative and Tamil
    news headlines are conventionally terse, so a whitespace-token word
    count under-represents how much information a short Tamil string
    actually carries relative to the same token count in English).
    """
    word_len = df[text_col].fillna("").astype(str).str.count(r"\S+")
    char_len = df[text_col].fillna("").astype(str).str.len()

    if isinstance(min_word_count, dict):
        thresholds = df[language_col].map(min_word_count)
        if thresholds.isna().any():
            missing = sorted(df.loc[thresholds.isna(), language_col].unique())
            raise ValueError(f"min_word_count dict has no entry for language(s): {missing}")
        too_short_mask = word_len < thresholds
    else:
        too_short_mask = word_len < min_word_count

    threshold = float(char_len.quantile(extreme_length_percentile / 100.0))
    extreme_mask = char_len >= threshold

    return InvalidDocumentResult(
        too_short_mask=too_short_mask,
        extreme_length_mask=extreme_mask,
        n_too_short=int(too_short_mask.sum()),
        n_extreme_length=int(extreme_mask.sum()),
        extreme_length_threshold_chars=threshold,
    )
