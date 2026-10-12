"""Module 4 — metrics, per-language breakdown, and baselines.

Metrics are implemented in plain NumPy rather than delegated to
scikit-learn. Two reasons, both practical:

1. They can be unit-tested against hand-computed values in an
   environment without sklearn or torch installed, which is where this
   code was developed.
2. The per-language breakdown is the paper's central claim. An
   aggregate macro-F1 can look strong while Tamil performance is poor,
   because Tamil is 7.8% of the corpus. Owning the computation makes
   the language slicing explicit rather than an afterthought.

`evaluate_predictions()` always returns overall AND per-language
results. There is deliberately no way to compute only the aggregate.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def _prf_per_class(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall, F1 for each class from a confusion matrix.

    Zero-denominator cases yield 0.0 rather than NaN — the convention
    scikit-learn uses with `zero_division=0`, and the one that keeps a
    macro average finite when a class is never predicted.
    """
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Binary ROC-AUC via the rank (Mann-Whitney U) identity.

    Ties receive averaged ranks, which is what makes this agree with
    sklearn on non-strict score distributions. Returns None when only
    one class is present — undefined rather than 0.0, so a
    single-class language slice cannot silently contribute a
    misleadingly bad number to a summary table.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_pos_ranks = ranks[y_true == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Accuracy, macro precision/recall/F1, ROC-AUC, confusion matrix."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    if len(y_true) == 0:
        return {"n": 0}

    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = _prf_per_class(cm)

    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": round(float((y_true == y_pred).mean()), 6),
        "precision_macro": round(float(precision.mean()), 6),
        "recall_macro": round(float(recall.mean()), 6),
        "macro_f1": round(float(f1.mean()), 6),
        "per_class": {
            str(c): {
                "precision": round(float(precision[c]), 6),
                "recall": round(float(recall[c]), 6),
                "f1": round(float(f1[c]), 6),
                "support": int(cm[c].sum()),
            }
            for c in range(cm.shape[0])
        },
        "confusion_matrix": cm.tolist(),
    }
    if y_score is not None:
        out["roc_auc"] = roc_auc(y_true, y_score)
    return out


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    languages: np.ndarray,
    y_score: np.ndarray | None = None,
    languages_to_report: tuple[str, ...] = ("en", "ta"),
) -> dict[str, Any]:
    """Overall + per-language metrics. Both, always."""
    languages = np.asarray(languages).astype(str).reshape(-1)
    result: dict[str, Any] = {"overall": compute_metrics(y_true, y_pred, y_score), "per_language": {}}
    for lang in languages_to_report:
        mask = languages == lang
        if not mask.any():
            result["per_language"][lang] = {"n": 0, "note": "no rows for this language in this split"}
            continue
        result["per_language"][lang] = compute_metrics(
            np.asarray(y_true)[mask],
            np.asarray(y_pred)[mask],
            None if y_score is None else np.asarray(y_score)[mask],
        )
    return result


# --------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------

def majority_class_baseline(y_train: np.ndarray, y_eval: np.ndarray, languages: np.ndarray) -> dict[str, Any]:
    """Predict the training majority class for everything.

    The floor that makes accuracy interpretable. On an imbalanced
    corpus a model can look respectable while beating this by almost
    nothing — reporting it is what stops that from passing unnoticed.
    """
    values, counts = np.unique(np.asarray(y_train).astype(int), return_counts=True)
    majority = int(values[np.argmax(counts)])
    y_pred = np.full(len(y_eval), majority, dtype=int)
    # Constant score => AUC is undefined; pass 0.5-equivalent constant.
    res = evaluate_predictions(y_eval, y_pred, languages, y_score=np.full(len(y_eval), 0.5))
    res["majority_class"] = majority
    res["baseline"] = "majority_class"
    return res


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    seed: int = 42,
    C: float = 1.0,
):
    """Logistic regression on the frozen 768-d embeddings.

    Uses scikit-learn when available (Colab has it) and raises a clear
    error otherwise rather than silently substituting a weaker model —
    a baseline that quietly differs from what the paper claims is worse
    than no baseline.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for the logistic-regression baseline. "
            "Install it with `pip install scikit-learn`."
        ) from exc
    clf = LogisticRegression(max_iter=max_iter, random_state=seed, C=C, n_jobs=-1)
    clf.fit(x_train, np.asarray(y_train).astype(int))
    return clf


def summarise_across_seeds(per_seed: list[dict[str, Any]], keys: tuple[str, ...] = (
    "accuracy", "precision_macro", "recall_macro", "macro_f1", "roc_auc",
)) -> dict[str, Any]:
    """mean ± std across seeds for the final selected configuration.

    Reported so a claimed improvement can be separated from run-to-run
    variance. A gain smaller than the seed standard deviation is not a
    result, and this makes that visible instead of hiding it behind a
    single lucky run.
    """
    out: dict[str, Any] = {"n_seeds": len(per_seed)}
    for key in keys:
        vals = [d.get(key) for d in per_seed if d.get(key) is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[key] = {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=0)), 6),
            "values": [round(float(v), 6) for v in arr],
        }
    return out
