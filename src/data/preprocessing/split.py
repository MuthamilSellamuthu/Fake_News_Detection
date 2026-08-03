"""Stratified train/validation/test split for Module 2.

Stratifies on a combined (label, language) key by default, so both the
Fake/Real balance *and* the Tamil/English balance are preserved across
splits — not just the label balance. Falls back to label-only
stratification if the combined key produces a class with fewer members
than splits require (`train_test_split` needs >= 2 members per
stratify class to split at all).

Implementation: two sequential `sklearn.model_selection.train_test_split`
calls (first peel off train, then split the remainder into val/test),
which is the standard way to get a three-way stratified split since
scikit-learn's function is inherently two-way.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger("preprocessing.split")


def _stratify_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return df[columns].astype(str).agg("_".join, axis=1)


def _safe_stratify(df: pd.DataFrame, columns: list[str], min_count: int = 2) -> pd.Series | None:
    """Return a stratify key usable by train_test_split, or None if even
    label-only stratification isn't viable (should not happen in practice).
    """
    key = _stratify_key(df, columns)
    counts = key.value_counts()
    if (counts >= min_count).all():
        return key

    small_classes = counts[counts < min_count].index.tolist()
    logger.warning(
        "Stratify key %s has class(es) with < %d member(s) (%s); "
        "falling back to stratifying on 'label' only.",
        columns,
        min_count,
        small_classes,
    )
    if columns != ["label"]:
        return _safe_stratify(df, ["label"], min_count=min_count)
    return None


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_columns: list[str] | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratios_sum = round(train_ratio + val_ratio + test_ratio, 6)
    if ratios_sum != 1.0:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratios_sum}")

    stratify_columns = stratify_columns or ["label"]
    missing_cols = [c for c in stratify_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Stratify column(s) not found in dataframe: {missing_cols}")

    strat_key = _safe_stratify(df, stratify_columns)

    train_df, remainder_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=strat_key,
    )

    # Re-derive the stratify key on the remainder (index-aligned subset).
    remainder_strat_key = _safe_stratify(remainder_df, stratify_columns) if strat_key is not None else None
    val_share_of_remainder = val_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        remainder_df,
        train_size=val_share_of_remainder,
        random_state=random_seed,
        stratify=remainder_strat_key,
    )

    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        logger.info(
            "%s split: %d rows | labels=%s | languages=%s",
            name,
            len(split_df),
            split_df["label"].value_counts().to_dict() if "label" in split_df else {},
            split_df["language"].value_counts().to_dict() if "language" in split_df else {},
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    total = len(train_df) + len(val_df) + len(test_df)
    return {
        "total_rows": total,
        "train": {
            "rows": len(train_df),
            "fraction": round(len(train_df) / total, 4) if total else 0.0,
            "label_counts": train_df["label"].value_counts().to_dict(),
            "language_counts": train_df["language"].value_counts().to_dict(),
        },
        "val": {
            "rows": len(val_df),
            "fraction": round(len(val_df) / total, 4) if total else 0.0,
            "label_counts": val_df["label"].value_counts().to_dict(),
            "language_counts": val_df["language"].value_counts().to_dict(),
        },
        "test": {
            "rows": len(test_df),
            "fraction": round(len(test_df) / total, 4) if total else 0.0,
            "label_counts": test_df["label"].value_counts().to_dict(),
            "language_counts": test_df["language"].value_counts().to_dict(),
        },
    }
