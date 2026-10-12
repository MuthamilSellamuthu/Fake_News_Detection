#!/usr/bin/env python
"""Module 5 orchestrator: train the MLP ensemble classifier.

Loads embeddings produced by Module 3 (frozen XLM-R) or Module 4
(fine-tuned Contrast-XLMR) — whichever `configs/classifier.yaml`
(or --embeddings-dir) points at — trains an ensemble of MLPs, each on
its own class-balanced subsample, and saves the ensemble plus its
validation metrics. Nothing upstream (encoder, embeddings) is touched
or retrained here.

Usage
-----
  # Default config (Contrast-XLMR embeddings from Module 4):
  python scripts/train_classifier.py

  # Ablation: classify on Module 3's frozen (non-fine-tuned) embeddings instead:
  python scripts/train_classifier.py --embeddings-dir data/embeddings

  # Different ensemble size / architecture:
  python scripts/train_classifier.py --n-estimators 11 --hidden-sizes 512 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # noqa: E402

from src.classifier.data_loader import load_embeddings  # noqa: E402
from src.classifier.mlp_ensemble import MLPEnsembleClassifier  # noqa: E402
from src.utils.config_loader import load_yaml, project_root  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.train_classifier")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 5: MLP ensemble classifier training.")
    parser.add_argument("--embeddings-dir", default=None, help="Override configs/classifier.yaml input.embeddings_dir.")
    parser.add_argument("--n-estimators", type=int, default=None, help="Override ensemble.n_estimators.")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=None, help="Override ensemble.hidden_layer_sizes.")
    parser.add_argument("--max-iter", type=int, default=None, help="Override ensemble.max_iter.")
    parser.add_argument("--seed", type=int, default=None, help="Override ensemble.random_state.")
    args = parser.parse_args()

    config_path = project_root() / "configs" / "classifier.yaml"
    config = load_yaml(config_path)

    if args.embeddings_dir:
        config["input"]["embeddings_dir"] = args.embeddings_dir
    if args.n_estimators:
        config["ensemble"]["n_estimators"] = args.n_estimators
    if args.hidden_sizes:
        config["ensemble"]["hidden_layer_sizes"] = args.hidden_sizes
    if args.max_iter:
        config["ensemble"]["max_iter"] = args.max_iter
    if args.seed:
        config["ensemble"]["random_state"] = args.seed

    input_cfg = config["input"]
    ensemble_cfg = config["ensemble"]
    output_cfg = config["output"]

    embeddings_dir = project_root() / input_cfg["embeddings_dir"]
    train_path = embeddings_dir / f"{input_cfg['train_split']}.h5"
    val_path = embeddings_dir / f"{input_cfg['val_split']}.h5"
    for path in (train_path, val_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/generate_embeddings.py (Module 3) or "
                "scripts/train_contrastive.py (Module 4) first to produce embeddings."
            )

    X_train, y_train, _, train_meta = load_embeddings(train_path)
    X_val, y_val, _, _ = load_embeddings(val_path)
    logger.info(
        "Loaded embeddings from %s (encoder: %s). Train: %d rows, Val: %d rows.",
        embeddings_dir, train_meta.get("model_name", "unknown"), len(X_train), len(X_val),
    )

    clf = MLPEnsembleClassifier(
        n_estimators=ensemble_cfg["n_estimators"],
        hidden_layer_sizes=ensemble_cfg["hidden_layer_sizes"],
        activation=ensemble_cfg["activation"],
        learning_rate_init=ensemble_cfg["learning_rate_init"],
        max_iter=ensemble_cfg["max_iter"],
        early_stopping=ensemble_cfg["early_stopping"],
        random_state=ensemble_cfg["random_state"],
    )

    logger.info(
        "Training %d-estimator MLP ensemble (hidden_layer_sizes=%s) on %d train rows...",
        ensemble_cfg["n_estimators"], ensemble_cfg["hidden_layer_sizes"], len(X_train),
    )
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    metrics = {
        "val_accuracy": accuracy_score(y_val, val_preds),
        "val_precision": precision_score(y_val, val_preds, zero_division=0),
        "val_recall": recall_score(y_val, val_preds, zero_division=0),
        "val_f1": f1_score(y_val, val_preds, zero_division=0),
    }
    logger.info("Validation metrics: %s", metrics)

    checkpoint_path = project_root() / output_cfg["checkpoint_dir"] / output_cfg["checkpoint_name"]
    clf.save(checkpoint_path)

    print("\n" + "=" * 72)
    print("MODULE 5 CLASSIFIER TRAINING SUMMARY")
    print("=" * 72)
    print(f"  embeddings source: {embeddings_dir} (encoder: {train_meta.get('model_name', 'unknown')})")
    print(f"  train rows: {len(X_train)}   val rows: {len(X_val)}")
    print(f"  n_estimators: {ensemble_cfg['n_estimators']}   hidden_layer_sizes: {ensemble_cfg['hidden_layer_sizes']}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"  checkpoint: {checkpoint_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
