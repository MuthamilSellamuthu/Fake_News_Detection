#!/usr/bin/env python
"""Module 2 — CLI entry point for the preprocessing pipeline.

Runs `src.preprocessing.preprocessing_pipeline.run_preprocessing_pipeline`
end to end against Module 1's real output, then prints a summary.

Usage
-----
  python scripts/preprocess_dataset.py
  python scripts/preprocess_dataset.py --config configs/module2_preprocessing.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.preprocessing_pipeline import run_preprocessing_pipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.preprocess_dataset")


def _attach_module2_log_file(path: Path) -> None:
    """In addition to the standard `logs/run_<timestamp>.log` every
    module writes, Module 2 also writes a fixed-name `preprocessing.log`
    (as explicitly requested for this module's deliverables) by
    attaching a second FileHandler to the root logger used across
    `src.preprocessing.*`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(formatter)
    logging.getLogger("preprocessing").addHandler(handler)
    logging.getLogger("scripts.preprocess_dataset").addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 2: clean, verify, and split the Module 1 dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to module2_preprocessing.yaml.")
    args = parser.parse_args()

    _attach_module2_log_file(PROJECT_ROOT / "outputs" / "module2" / "preprocessing.log")

    logger.info("Starting Module 2 preprocessing pipeline...")
    report = run_preprocessing_pipeline(config_path=args.config)

    print("\n" + "=" * 72)
    print("MODULE 2 — PREPROCESSING SUMMARY")
    print("=" * 72)
    summary = {
        "input_rows": report["stages"]["input"]["n_rows"],
        "final_rows": report["stages"]["cleaned_dataset_summary"]["n_rows"],
        "by_language": report["stages"]["cleaned_dataset_summary"]["by_language"],
        "by_label": report["stages"]["cleaned_dataset_summary"]["by_label"],
        "split": {
            "train": report["stages"]["split"]["n_train"],
            "val": report["stages"]["split"]["n_val"],
            "test": report["stages"]["split"]["n_test"],
        },
        "leakage_check_passed": report["stages"]["split"]["leakage_check"]["passed"],
        "quality_checks_passed": report["stages"]["final_quality_checks"]["passed"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nOutput files:")
    for key, path in report["output_files"].items():
        print(f"  {key}: {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
