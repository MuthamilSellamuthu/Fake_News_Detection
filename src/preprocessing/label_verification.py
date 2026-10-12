"""Label verification — Module 2.

Module 1 already standardized every source's label into the project's
fixed convention (fake=1, real=0). This module does not re-derive
labels — it verifies the column is still exactly what it claims to be
after cleaning has run, since a cleaning bug that accidentally
mangles or nulls a label would otherwise fail silently and only
surface much later as an inexplicable training-time error.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class LabelVerificationResult:
    n_null: int
    n_invalid_value: int
    invalid_record_ids: list[str]
    is_valid: bool


def verify_labels(
    df: pd.DataFrame,
    label_col: str = "label",
    record_id_col: str = "record_id",
    valid_labels: tuple[int, ...] = (0, 1),
) -> LabelVerificationResult:
    null_mask = df[label_col].isna()
    n_null = int(null_mask.sum())

    non_null = df.loc[~null_mask]
    invalid_mask = ~non_null[label_col].isin(valid_labels)
    n_invalid = int(invalid_mask.sum())
    invalid_ids = non_null.loc[invalid_mask, record_id_col].tolist()
    if n_null:
        invalid_ids = df.loc[null_mask, record_id_col].tolist() + invalid_ids

    return LabelVerificationResult(
        n_null=n_null,
        n_invalid_value=n_invalid,
        invalid_record_ids=invalid_ids,
        is_valid=(n_null == 0 and n_invalid == 0),
    )
