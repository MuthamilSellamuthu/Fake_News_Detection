#!/usr/bin/env python
"""Module 1 — CLI entry point for dataset validation / standardization / stats.

Runs `src.data_collection.dataset_validator.run_validation_pipeline` end to
end against the real files under `datasets/raw/`, then prints a summary.

Usage
-----
  python scripts/validate_and_standardize.py
  python scripts/validate_and_standardize.py --config configs/module1_validation.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.dataset_validator import run_validation_pipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.validate_and_standardize")


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 1: validate, dedup, label-standardize, and report stats.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to module1_validation.yaml (default: configs/module1_validation.yaml).",
    )
    args = parser.parse_args()

    logger.info("Starting Module 1 validation pipeline...")
    result = run_validation_pipeline(config_path=args.config)

    print("\n" + "=" * 72)
    print("MODULE 1 — VALIDATION & STATISTICS SUMMARY")
    print("=" * 72)
    print(json.dumps(result["validation_report"]["stages"], indent=2, ensure_ascii=False, default=str))
    print("\nOutput files:")
    for key, path in result["validation_report"]["output_files"].items():
        print(f"  {key}: {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
