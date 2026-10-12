"""Exact-duplicate and missing-value removal — Module 1.

Deliberately "exact match" only (no fuzzy/near-duplicate detection —
that kind of similarity-based filtering belongs to Module 2, per the
scope boundary in docs/module1_overview.md). Every function returns
`(filtered_df, n_removed)` so the caller can report precisely how much
was dropped and why, for the validation report.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data_collection.deduplication")


def drop_missing_text(
    df: pd.DataFrame,
    text_col: str = "text",
    min_length: int = 5,
) -> tuple[pd.DataFrame, int]:
    """Drop rows with a null/empty `text_col`, or shorter than `min_length` chars."""
    before = len(df)
    has_text = df[text_col].notna() & (df[text_col].astype(str).str.strip().str.len() >= min_length)
    filtered = df.loc[has_text].copy()
    removed = before - len(filtered)
    if removed:
        logger.info(
            "Dropped %d row(s) with missing/too-short '%s' (< %d chars) (%.1f%% of input).",
            removed,
            text_col,
            min_length,
            100 * removed / before if before else 0.0,
        )
    return filtered, removed


def drop_missing_label(df: pd.DataFrame, label_col: str = "label") -> tuple[pd.DataFrame, int]:
    """Drop rows whose label could not be resolved (None) by label standardization."""
    before = len(df)
    filtered = df.loc[df[label_col].notna()].copy()
    removed = before - len(filtered)
    if removed:
        logger.info(
            "Dropped %d row(s) with unresolved label (%.1f%% of input).",
            removed,
            100 * removed / before if before else 0.0,
        )
    return filtered, removed


def drop_exact_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Drop exact-match duplicate rows on `subset` (default: all columns), keeping the first occurrence."""
    before = len(df)
    filtered = df.drop_duplicates(subset=subset, keep="first").copy()
    removed = before - len(filtered)
    if removed:
        logger.info(
            "Dropped %d exact-duplicate row(s) on %s (%.1f%% of input).",
            removed,
            subset or "all columns",
            100 * removed / before if before else 0.0,
        )
    return filtered, removed
