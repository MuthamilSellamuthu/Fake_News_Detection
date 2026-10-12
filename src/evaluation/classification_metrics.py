"""Standard classification metrics — Module 6.

One function, `compute_classification_metrics`, used identically by
the top-level test-set report, the imbalance-ratio ablation, and the
per-group fairness audit, so all three report the exact same set of
numbers computed the exact same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob_fake: np.ndarray
) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1, Brier loss, ROC-AUC, and
    a confusion matrix.

    `y_prob_fake` (the model's predicted P(fake), a continuous score in
    [0, 1]) is needed separately from `y_pred` (the thresholded 0/1
    decision) because Brier loss and ROC-AUC are both defined over
    probabilities, not hard labels — a model that's barely over the
    0.5 threshold and one that's at 0.99 both count as "correct" for
    accuracy/F1, but Brier loss and ROC-AUC can tell them apart.

    ROC-AUC is `None` (rather than raising) when `y_true` contains only
    one class — this happens legitimately in small per-group fairness
    slices and in extreme imbalance-ablation ratios, where AUC is
    undefined rather than merely inconvenient to compute.
    """
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier_loss": brier_score_loss(y_true, y_prob_fake),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob_fake)
    except ValueError:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
    }
    return metrics
