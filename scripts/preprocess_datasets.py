#!/usr/bin/env python
"""Module 2 orchestrator: clean, label, merge, and split Module 1's raw data.

Pipeline (see src/data/preprocessing/ for each stage's implementation):
  1. Load every recognized source under data/raw/ into a common schema.
  2. Clean text: strip HTML, URLs, emoji, special symbols; normalize
     whitespace; Tamil Unicode is explicitly preserved throughout.
  3. Encode labels: Fake = 1, Real = 0 (source-specific rules, see
     configs/preprocessing.yaml).
  4. Detect language: Tamil vs. English (Unicode-block heuristic +
     langdetect fallback).
  5. Drop rows with missing text/label, then drop exact-duplicate rows.
  6. Merge every source into one dataset.
  7. Stratified 70/15/15 train/val/test split (stratified on label +
     language so both stay balanced across splits).
  8. Save train/val/test as CSV + JSONL under data/processed/, plus a
     preprocessing_summary.json describing what happened.

This script does not build sentence pairs, extract embeddings, or touch
any model — that begins in Module 3.

Usage
-----
  python scripts/preprocess_datasets.py
  python scripts/preprocess_datasets.py --raw-dir data/raw --output-dir data/processed
  python scripts/preprocess_datasets.py --config configs/preprocessing.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.pipeline import build_dataset  # noqa: E402
from src.data.preprocessing.save import save_all  # noqa: E402
from src.data.preprocessing.split import split_summary, stratified_split  # noqa: E402
from src.utils.config_loader import load_yaml  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.preprocess_datasets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 2 preprocessing pipeline.")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"), help="Root of Module 1 raw output.")
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "data" / "processed"), help="Where to write train/val/test."
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "preprocessing.yaml"), help="Path to preprocessing.yaml."
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

    logger.info("=" * 60)
    logger.info("Module 2: building merged, cleaned, labeled dataset from %s", args.raw_dir)
    df, build_stats = build_dataset(args.raw_dir, config)

    if df.empty:
        logger.error("No data to split — check that Module 1 raw files exist under %s", args.raw_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Splitting: train/val/test = %s", config["split"])
    split_cfg = config["split"]
    train_df, val_df, test_df = stratified_split(
        df,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        stratify_columns=split_cfg["stratify_columns"],
        random_seed=split_cfg["random_seed"],
    )
    summary = split_summary(train_df, val_df, test_df)

    run_stats = {
        "build": build_stats,
        "split": summary,
        "config_used": config,
    }

    logger.info("=" * 60)
    logger.info("Saving processed splits to %s", args.output_dir)
    output_cfg = config["output"]
    saved_paths = save_all(
        train_df,
        val_df,
        test_df,
        output_dir=args.output_dir,
        formats=output_cfg["formats"],
        filenames=output_cfg["filenames"],
        run_stats=run_stats,
    )

    print("\n" + "=" * 72)
    print("MODULE 2 PREPROCESSING SUMMARY")
    print("=" * 72)
    print(f"Rows loaded (raw):        {build_stats.get('rows_loaded', 0)}")
    print(f"Dropped (missing):        {build_stats.get('rows_dropped_missing', 0)}")
    print(f"Dropped (duplicate):      {build_stats.get('rows_dropped_duplicate', 0)}")
    print(f"Final merged rows:        {build_stats.get('rows_final', 0)}")
    print(f"Label counts (1=Fake, 0=Real): {build_stats.get('label_counts', {})}")
    print(f"Language counts:          {build_stats.get('language_counts', {})}")
    print(f"Source counts:            {build_stats.get('source_counts', {})}")
    print("-" * 72)
    for split_name in ("train", "val", "test"):
        s = summary[split_name]
        print(f"{split_name:>5}: {s['rows']:>7} rows ({s['fraction']:.1%}) | labels={s['label_counts']} | langs={s['language_counts']}")
    print("-" * 72)
    print("Output files:")
    for split_name, paths in saved_paths.items():
        print(f"  {split_name}: {paths}")
    print("=" * 72)


if __name__ == "__main__":
    main()
