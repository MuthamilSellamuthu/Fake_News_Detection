#!/usr/bin/env python
"""Module 4 orchestrator: fine-tune XLM-RoBERTa with supervised SimCSE.

This produces "Contrast-XLMR": a fine-tuned encoder checkpoint saved
under `models/contrastive/`, trained so that same-label (fake/fake or
real/real) texts end up close together in embedding space and
opposite-label texts end up far apart. No classifier is trained here —
that's Module 5, which loads this checkpoint's embeddings as input.

Usage
-----
  # Default config (configs/contrastive.yaml):
  python scripts/train_contrastive.py

  # Override a few hyperparameters:
  python scripts/train_contrastive.py --epochs 5 --batch-size 32 --lr 1e-5

  # Quick smoke test on a small slice before committing to a full run:
  python scripts/train_contrastive.py --limit 200 --epochs 1

Realistic time expectation: this is full transformer fine-tuning
(forward AND backward pass every step), noticeably slower per-row than
Module 3's frozen-embedding extraction. A GPU is strongly recommended;
on CPU, budget accordingly and use --limit to validate the pipeline
first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.contrastive.trainer import train  # noqa: E402
from src.utils.config_loader import load_yaml, project_root  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.train_contrastive")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 4: Supervised SimCSE contrastive fine-tuning.")
    parser.add_argument("--model-name", default=None, help="Override configs/contrastive.yaml model.name.")
    parser.add_argument("--max-length", type=int, default=None, help="Override model.max_sequence_length.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override training.learning_rate.")
    parser.add_argument("--temperature", type=float, default=None, help="Override training.temperature.")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed.")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N rows of train/val (smoke test).")
    args = parser.parse_args()

    config_path = project_root() / "configs" / "contrastive.yaml"
    config = load_yaml(config_path)

    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.max_length:
        config["model"]["max_sequence_length"] = args.max_length
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.temperature:
        config["training"]["temperature"] = args.temperature
    if args.seed:
        config["training"]["seed"] = args.seed

    logger.info("Starting Module 4 training: model=%s epochs=%d batch_size=%d lr=%g", config["model"]["name"], config["training"]["epochs"], config["training"]["batch_size"], config["training"]["learning_rate"])

    summary = train(config, limit=args.limit)

    print("\n" + "=" * 72)
    print("MODULE 4 CONTRASTIVE TRAINING SUMMARY")
    print("=" * 72)
    for epoch_summary in summary["epochs"]:
        print(f"  epoch {epoch_summary['epoch']}: train_loss={epoch_summary['train_loss']:.4f}  val_loss={epoch_summary['val_loss']:.4f}  ({epoch_summary['elapsed_seconds']}s)")
    print(f"  best val_loss: {summary['best_val_loss']:.4f}")
    print(f"  best checkpoint: {summary['best_checkpoint']}")
    print(f"  last checkpoint: {summary['last_checkpoint']}")
    print("  learned embeddings saved:")
    for split_name, path in summary["embeddings"].items():
        print(f"    {split_name}: {path}")
    print("=" * 72)
    print("No classifier has been trained — this is Module 4 (Contrast-XLMR encoder + embeddings) output only.")


if __name__ == "__main__":
    main()
