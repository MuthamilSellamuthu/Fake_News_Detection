"""Persist Module 2 output (processed splits + a run summary) to disk.

Writes both CSV and JSONL by default — CSV for quick inspection in a
spreadsheet, JSONL for straightforward streaming/loading in Module 3
(embedding extraction) without a pandas dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("preprocessing.save")


def save_split(df: pd.DataFrame, name: str, output_dir: Path, formats: list[str]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if "csv" in formats:
        csv_path = output_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # BOM helps Excel render Tamil correctly
        paths["csv"] = str(csv_path)
        logger.info("Saved %s (%d rows) -> %s", name, len(df), csv_path)

    if "jsonl" in formats:
        jsonl_path = output_dir / f"{name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        paths["jsonl"] = str(jsonl_path)
        logger.info("Saved %s (%d rows) -> %s", name, len(df), jsonl_path)

    return paths


def save_all(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
    formats: list[str] | None = None,
    filenames: dict[str, str] | None = None,
    run_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    formats = formats or ["csv", "jsonl"]
    filenames = filenames or {"train": "train", "val": "val", "test": "test"}
    output_dir = Path(output_dir)

    result = {
        "train": save_split(train_df, filenames["train"], output_dir, formats),
        "val": save_split(val_df, filenames["val"], output_dir, formats),
        "test": save_split(test_df, filenames["test"], output_dir, formats),
    }

    if run_stats is not None:
        summary_path = output_dir / "preprocessing_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(run_stats, f, ensure_ascii=False, indent=2, default=str)
        result["summary"] = str(summary_path)
        logger.info("Saved run summary -> %s", summary_path)

    return result
