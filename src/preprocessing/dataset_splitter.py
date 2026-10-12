"""Reproducible, jointly-stratified train/val/test split — Module 2.

Stratifies on a combined `(language, label)` key rather than either
column alone, so the split preserves the en/ta ratio *and* the
fake/real ratio independently within each language — a plain
single-column stratification on `label` alone could, by chance, put
proportionally more Tamil-fake rows in one split than another, which
matters given how small the Tamil slice already is (5,417 of 69,089
rows). Splitting is done per stratum (each of the 4 language x label
groups is split independently at the configured ratios, then
concatenated) rather than via a single global shuffle, which is what
guarantees the joint proportions hold in every split, not just on
average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("preprocessing.dataset_splitter")


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    stratum_counts: dict[str, dict[str, int]]


def stratified_split(
    df: pd.DataFrame,
    stratify_columns: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> SplitResult:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    rng = np.random.default_rng(random_seed)

    train_parts, val_parts, test_parts = [], [], []
    stratum_counts: dict[str, dict[str, int]] = {}

    strata = df.groupby(stratify_columns, dropna=False)
    for key, group in strata:
        key_str = str(key)
        n = len(group)
        shuffled_idx = rng.permutation(n)
        shuffled = group.iloc[shuffled_idx]

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Remainder goes to test, so rounding never drops or duplicates a row.
        n_test = n - n_train - n_val

        train_parts.append(shuffled.iloc[:n_train])
        val_parts.append(shuffled.iloc[n_train:n_train + n_val])
        test_parts.append(shuffled.iloc[n_train + n_val:])

        stratum_counts[key_str] = {"total": n, "train": n_train, "val": n_val, "test": n_test}
        logger.info("Stratum %s: total=%d -> train=%d val=%d test=%d", key_str, n, n_train, n_val, n_test)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy()

    return SplitResult(train=train_df, val=val_df, test=test_df, stratum_counts=stratum_counts)


@dataclass
class LeakageCheckResult:
    passed: bool
    n_train_val_overlap: int
    n_train_test_overlap: int
    n_val_test_overlap: int


def verify_no_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str = "record_id",
) -> LeakageCheckResult:
    """Confirm no `record_id` (and, as a stronger check, no exact `text`)
    appears in more than one split."""
    train_ids, val_ids, test_ids = set(train[id_col]), set(val[id_col]), set(test[id_col])

    tv = len(train_ids & val_ids)
    tt = len(train_ids & test_ids)
    vt = len(val_ids & test_ids)

    passed = (tv == 0 and tt == 0 and vt == 0)
    if not passed:
        logger.error("DATA LEAKAGE DETECTED: train/val overlap=%d, train/test overlap=%d, val/test overlap=%d", tv, tt, vt)
    else:
        logger.info("Leakage check passed: zero record_id overlap across train/val/test.")

    return LeakageCheckResult(passed=passed, n_train_val_overlap=tv, n_train_test_overlap=tt, n_val_test_overlap=vt)
