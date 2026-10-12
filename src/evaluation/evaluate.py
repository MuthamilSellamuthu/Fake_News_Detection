"""Module 6 orchestrator logic: run every evaluation component and
assemble one consolidated report dict.

Deliberately classifier-agnostic: it takes a plain `predict_proba_fn`
callable (`np.ndarray -> np.ndarray` of shape `(n, 2)`) rather than a
specific classifier class, so the same evaluation code runs unmodified
against either Module 5 variant (the scikit-learn ensemble or the
torch + Focal Loss MLP) — `scripts/evaluate_model.py` is the only place
that knows which one is actually loaded.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.clustering_metrics import compute_clustering_metrics
from src.evaluation.fairness_audit import audit_by_group
from src.evaluation.imbalance_ablation import subsample_to_ratio
from src.evaluation.retrieval_metrics import compute_retrieval_metrics


def run_full_evaluation(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    test_metadata: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run classification, clustering, retrieval, imbalance-ablation,
    and fairness evaluation, in that order, and return one dict with
    all five sections.

    `test_metadata` must be row-aligned with `X_test`/`y_test` (same
    order, same length) and contain whatever columns
    `config["fairness"]["group_columns"]` names — `scripts/evaluate_model.py`
    is responsible for loading and aligning it correctly before calling this.
    """
    report: dict[str, Any] = {}

    probs = predict_proba_fn(X_test)
    y_prob_fake = probs[:, 1]
    y_pred = (y_prob_fake >= 0.5).astype(int)

    report["classification"] = compute_classification_metrics(y_test, y_pred, y_prob_fake)

    clustering_cfg = config["clustering"]
    report["clustering"] = compute_clustering_metrics(
        X_test, y_test, n_clusters=clustering_cfg["n_clusters"], random_state=clustering_cfg["random_state"]
    )

    retrieval_cfg = config["retrieval"]
    report["retrieval"] = compute_retrieval_metrics(
        X_test, y_test, X_train, y_train, k_values=retrieval_cfg["k_values"]
    )

    ablation_cfg = config["imbalance_ablation"]
    ablation_report: dict[str, Any] = {}
    for ratio in ablation_cfg["fake_ratios"]:
        X_sub, y_sub = subsample_to_ratio(X_test, y_test, fake_ratio=ratio, seed=ablation_cfg["seed"])
        probs_sub = predict_proba_fn(X_sub)
        y_pred_sub = (probs_sub[:, 1] >= 0.5).astype(int)
        ablation_report[f"fake_ratio_{ratio}"] = {
            "n_rows": len(y_sub),
            **compute_classification_metrics(y_sub, y_pred_sub, probs_sub[:, 1]),
        }
    report["imbalance_ablation"] = ablation_report

    fairness_cfg = config["fairness"]
    fairness_report: dict[str, Any] = {}
    for group_col in fairness_cfg["group_columns"]:
        if group_col not in test_metadata.columns:
            continue
        fairness_report[group_col] = audit_by_group(
            y_test, y_pred, y_prob_fake, test_metadata[group_col].to_numpy()
        )
    report["fairness"] = fairness_report

    return report
