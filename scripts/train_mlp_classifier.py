#!/usr/bin/env python
"""Module 5 orchestrator (PyTorch + Focal Loss variant).

Trains a single MLP classification head on top of Module 3/4's
embeddings, using Focal Loss + AdamW + a linear-warmup schedule, with
an explicit training loop, validation loop, early stopping, and
checkpointing, then runs one final testing loop against the held-out
test split using the best checkpoint. No encoder is touched or
retrained here.

A separate, independent Module 5 implementation also exists —
`scripts/train_classifier.py` (a scikit-learn bagging ensemble of MLPs
on class-balanced subsamples). Use whichever variant your write-up
calls for; both consume the same embedding files.

Usage
-----
  # Default config (configs/mlp_classifier.yaml), Contrast-XLMR embeddings:
  python scripts/train_mlp_classifier.py

  # Ablation: classify on Module 3's frozen (non-fine-tuned) embeddings instead:
  python scripts/train_mlp_classifier.py --embeddings-dir data/embeddings

  # Different architecture / training budget:
  python scripts/train_mlp_classifier.py --hidden-sizes 512 256 --max-epochs 30 --patience 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.classifier_trainer import train  # noqa: E402
from src.utils.config_loader import load_yaml, project_root  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.train_mlp_classifier")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 5 (PyTorch + Focal Loss): MLP classifier training.")
    parser.add_argument("--embeddings-dir", default=None, help="Override configs/mlp_classifier.yaml input.embeddings_dir.")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=None, help="Override model.hidden_layer_sizes.")
    parser.add_argument("--dropout", type=float, default=None, help="Override model.dropout.")
    parser.add_argument("--gamma", type=float, default=None, help="Override loss.gamma (Focal Loss focusing parameter).")
    parser.add_argument("--lr", type=float, default=None, help="Override optimizer.learning_rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override training.max_epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override training.early_stopping_patience.")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed.")
    args = parser.parse_args()

    config_path = project_root() / "configs" / "mlp_classifier.yaml"
    config = load_yaml(config_path)

    if args.embeddings_dir:
        config["input"]["embeddings_dir"] = args.embeddings_dir
    if args.hidden_sizes:
        config["model"]["hidden_layer_sizes"] = args.hidden_sizes
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
    if args.gamma is not None:
        config["loss"]["gamma"] = args.gamma
    if args.lr:
        config["optimizer"]["learning_rate"] = args.lr
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.max_epochs:
        config["training"]["max_epochs"] = args.max_epochs
    if args.patience:
        config["training"]["early_stopping_patience"] = args.patience
    if args.seed:
        config["training"]["seed"] = args.seed

    summary = train(config)

    print("\n" + "=" * 72)
    print("MODULE 5 (MLP + FOCAL LOSS) TRAINING SUMMARY")
    print("=" * 72)
    for epoch_summary in summary["epochs"]:
        t, v = epoch_summary["train"], epoch_summary["val"]
        print(
            f"  epoch {epoch_summary['epoch']}: train_loss={t['loss']:.4f} "
            f"val_loss={v['loss']:.4f} val_f1={v['f1']:.4f} ({epoch_summary['elapsed_seconds']}s)"
        )
    print(f"  best val_loss: {summary['best_val_loss']:.4f}")
    print("  test metrics (best checkpoint):")
    for key, value in summary["test_metrics"].items():
        print(f"    {key}: {value:.4f}")
    print(f"  best checkpoint: {summary['best_checkpoint']}")
    print(f"  last checkpoint: {summary['last_checkpoint']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
