#!/usr/bin/env python
"""Module 6 orchestrator: evaluate a trained classifier.

Reports classification metrics (accuracy/precision/recall/F1/Brier
loss/ROC-AUC), embedding clustering quality (topic alignment),
nearest-neighbor retrieval quality (HR@k/MRR@k), robustness across
synthetic class-imbalance ratios, and a language/source fairness
audit. Loads whichever Module 5 checkpoint `configs/evaluation.yaml`
(or --classifier-type) points at; does not retrain or modify anything
upstream.

Usage
-----
  # Default config: torch + Focal Loss classifier on Contrast-XLMR embeddings
  python scripts/evaluate_model.py

  # Evaluate the scikit-learn ensemble variant instead:
  python scripts/evaluate_model.py --classifier-type sklearn_ensemble

  # Ablation: evaluate on Module 3's frozen (non-fine-tuned) embeddings:
  python scripts/evaluate_model.py --embeddings-dir data/embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.data_loader import load_embeddings  # noqa: E402
from src.evaluation.evaluate import run_full_evaluation  # noqa: E402
from src.utils.config_loader import load_yaml, project_root  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.evaluate_model")


def _load_test_metadata(processed_dir: Path, test_split: str, record_ids: np.ndarray) -> pd.DataFrame:
    """Read Module 2's processed test CSV and reindex it to exactly
    match `record_ids`' order — the order embeddings were written in —
    so `test_metadata` stays row-aligned with `X_test`/`y_test` for the
    fairness audit.
    """
    df = pd.read_csv(processed_dir / f"{test_split}.csv")
    return df.set_index("record_id").loc[record_ids].reset_index()


def _build_predict_proba_fn(classifier_type: str, input_cfg: dict):
    """Load the requested Module 5 checkpoint and return a plain
    `np.ndarray -> np.ndarray` function, so `run_full_evaluation` never
    needs to know which classifier implementation is behind it.
    """
    if classifier_type == "sklearn_ensemble":
        from src.classifier.mlp_ensemble import MLPEnsembleClassifier

        clf = MLPEnsembleClassifier.load(project_root() / input_cfg["sklearn_checkpoint"])
        return clf.predict_proba

    if classifier_type == "torch_focal":
        import torch
        import torch.nn.functional as F

        from src.classifier.torch_mlp import load_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_checkpoint(project_root() / input_cfg["torch_checkpoint"], device=device)
        model.eval()

        def predict_proba_fn(X: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                logits = model(torch.tensor(X, dtype=torch.float32, device=device))
                return F.softmax(logits, dim=-1).cpu().numpy()

        return predict_proba_fn

    raise ValueError(f"Unknown classifier_type: {classifier_type!r}. Expected 'torch_focal' or 'sklearn_ensemble'.")


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 6: performance evaluation.")
    parser.add_argument("--classifier-type", choices=["torch_focal", "sklearn_ensemble"], default=None)
    parser.add_argument("--embeddings-dir", default=None)
    args = parser.parse_args()

    config_path = project_root() / "configs" / "evaluation.yaml"
    config = load_yaml(config_path)
    input_cfg = config["input"]

    if args.classifier_type:
        input_cfg["classifier_type"] = args.classifier_type
    if args.embeddings_dir:
        input_cfg["embeddings_dir"] = args.embeddings_dir

    embeddings_dir = project_root() / input_cfg["embeddings_dir"]
    X_train, y_train, _, _ = load_embeddings(embeddings_dir / f"{config['retrieval']['reference_split']}.h5")
    X_test, y_test, test_ids, _ = load_embeddings(embeddings_dir / f"{input_cfg['test_split']}.h5")

    processed_dir = project_root() / input_cfg["processed_dir"]
    test_metadata = _load_test_metadata(processed_dir, input_cfg["test_split"], test_ids)

    classifier_type = input_cfg["classifier_type"]
    predict_proba_fn = _build_predict_proba_fn(classifier_type, input_cfg)

    logger.info("Evaluating classifier_type=%s on %d test rows (train reference pool: %d rows)...", classifier_type, len(y_test), len(y_train))
    report = run_full_evaluation(predict_proba_fn, X_test, y_test, X_train, y_train, test_metadata, config)

    report_path = project_root() / config["output"]["report_dir"] / config["output"]["report_name"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)

    print("\n" + "=" * 72)
    print("MODULE 6 EVALUATION SUMMARY")
    print("=" * 72)
    print(f"classifier: {classifier_type}   test rows: {len(y_test)}")
    print("\nClassification:")
    for key, value in report["classification"].items():
        if key != "confusion_matrix":
            print(f"  {key}: {value}")
    print(f"  confusion_matrix: {report['classification']['confusion_matrix']}")
    print("\nClustering (topic alignment):")
    for key, value in report["clustering"].items():
        print(f"  {key}: {value}")
    print("\nRetrieval:")
    for key, value in report["retrieval"].items():
        print(f"  {key}: {value}")
    print("\nImbalance ablation ratios evaluated:", list(report["imbalance_ablation"].keys()))
    print("Fairness audit groups evaluated:", list(report["fairness"].keys()))
    print(f"\nFull report saved to: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
