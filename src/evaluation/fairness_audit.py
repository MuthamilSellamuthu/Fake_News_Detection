"""Fairness / bias audit — Module 6.

Slices classification metrics by a grouping column (language, source)
to check the classifier performs comparably across groups rather than
concentrating its errors in the low-resource groups this project
specifically cares about (Tamil overall, and individual Tamil
sources). This is a "low-resource fairness" check rather than a
demographic-fairness audit — the protected axis here is language/data
source, which is what this project's own research gap analysis is
about, not a generic bias dimension.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.classification_metrics import compute_classification_metrics


def audit_by_group(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob_fake: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    """Return `{group_value: metrics}` for every distinct value in `groups`.

    A group is skipped (with a note, not silently) if it has fewer than
    2 rows or only one true class present — both make most classification
    metrics undefined or meaningless rather than merely small-sample-noisy,
    which is common for individual Tamil sources given current volume
    (e.g. `boom_tamil` currently has only 14 rows total across the whole
    dataset, some fraction of which land in test).
    """
    report: dict[str, Any] = {}
    for group_value in pd.unique(groups):
        mask = groups == group_value
        n_rows = int(mask.sum())

        if n_rows < 2 or len(np.unique(y_true[mask])) < 2:
            report[str(group_value)] = {
                "n_rows": n_rows,
                "note": "too few rows or only one class present in this group; metrics skipped",
            }
            continue

        group_metrics = compute_classification_metrics(y_true[mask], y_pred[mask], y_prob_fake[mask])
        group_metrics["n_rows"] = n_rows
        report[str(group_value)] = group_metrics

    return report
