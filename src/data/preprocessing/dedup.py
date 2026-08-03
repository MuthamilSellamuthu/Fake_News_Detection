"""Missing-value and duplicate removal for Module 2.

Both functions return `(filtered_df, n_removed)` so the pipeline can
report exactly how much was dropped and why — useful for a preprocessing
report and for sanity-checking that a run didn't silently gut the
dataset.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("preprocessing.dedup")


def drop_missing(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label",
    min_length: int = 20,
) -> tuple[pd.DataFrame, int]:
    """Drop rows with a null/empty label, or cleaned text shorter than `min_length`."""
    before = len(df)

    has_label = df[label_col].notna()
    has_text = df[text_col].notna() & (df[text_col].str.len() >= min_length)

    keep_mask = has_label & has_text
    filtered = df.loc[keep_mask].copy()

    removed = before - len(filtered)
    if removed:
        logger.info(
            "Dropped %d row(s) for missing label or text shorter than %d chars (%.1f%% of input).",
            removed,
            min_length,
            100 * removed / before if before else 0.0,
        )
    return filtered, removed


def drop_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Drop exact-match duplicate rows on `subset` (default: all columns), keeping the first occurrence."""
    before = len(df)
    filtered = df.drop_duplicates(subset=subset, keep="first").copy()
    removed = before - len(filtered)
    if removed:
        logger.info(
            "Dropped %d duplicate row(s) on %s (%.1f%% of input).",
            removed,
            subset or "all columns",
            100 * removed / before if before else 0.0,
        )
    return filtered, removed
