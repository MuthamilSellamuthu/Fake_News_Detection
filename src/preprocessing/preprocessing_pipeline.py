"""Module 2 orchestrator — cleaning, verification, quality checks,
document-length analysis, deduplication, statistics, and the
train/val/test split.

Reads Module 1's `datasets/processed/module1_dataset.csv` (never
modifies it) and writes:

  - `datasets/processed/cleaned_dataset.{csv,jsonl}` — the fully
    cleaned, verified, deduplicated dataset (still one row per
    document; splitting happens separately).
  - `datasets/processed/{train,validation,test}.csv` — the
    stratified split.
  - `outputs/module2/preprocessing_statistics.json` — before/after
    counts, length stats, split stats, leakage check, language/label
    distributions.
  - `outputs/module2/preprocessing_report.md` — human-readable summary.

Deliberately NOT done here (Module 3's job): tokenization, embedding
generation, or anything model-specific. This module only produces a
clean, verified, leak-free, splittable text dataset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.cleaning import clean_text
from src.preprocessing.dataset_splitter import stratified_split, verify_no_leakage
from src.preprocessing.document_length import compute_length_stats, flag_invalid_documents
from src.preprocessing.label_verification import verify_labels
from src.preprocessing.language_verification import verify_languages
from src.preprocessing.quality_checks import (
    run_quality_checks,
    verify_and_remove_duplicates,
    verify_missing_values,
)
from src.preprocessing.report_writer import write_preprocessing_report
from src.utils.config_loader import load_yaml, project_root
from src.utils.logger import get_logger

logger = get_logger("preprocessing.pipeline")


def run_preprocessing_pipeline(config_path: str | Path | None = None) -> dict[str, Any]:
    root = project_root()
    resolved_config_path = Path(config_path) if config_path else (root / "configs" / "module2_preprocessing.yaml")
    cfg = load_yaml(resolved_config_path)

    report: dict[str, Any] = {"stages": {}}

    # ---- Load Module 1 output ----
    input_path = root / cfg["input"]["path"]
    logger.info("Loading Module 1 output from %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    n_input = len(df)
    logger.info("Loaded %d rows.", n_input)

    report["stages"]["input"] = {
        "n_rows": n_input,
        "by_language": df["language"].value_counts().to_dict(),
        "by_label": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
    }

    # ---- Missing-value verification (before cleaning) ----
    missing_before = verify_missing_values(df, critical_fields=("text", "label"))
    report["stages"]["missing_values_before_cleaning"] = missing_before.counts

    # ---- Text cleaning (HTML, URLs, Unicode NFC, zero-width, control
    #      chars, whitespace, English-only lowercasing) ----
    clean_cfg = cfg["cleaning"]
    logger.info("Cleaning text for %d rows...", n_input)
    df["text"] = [clean_text(t, lang, clean_cfg) for t, lang in zip(df["text"], df["language"])]
    df["title"] = [clean_text(t, lang, clean_cfg) for t, lang in zip(df["title"].fillna(""), df["language"])]
    logger.info("Text cleaning complete.")

    # ---- Missing-value verification (after cleaning — cleaning can turn
    #      whitespace-only text into an empty string) ----
    missing_after = verify_missing_values(df, critical_fields=("text", "label"))
    report["stages"]["missing_values_after_cleaning"] = missing_after.counts

    # ---- Label verification ----
    label_result = verify_labels(df)
    report["stages"]["label_verification"] = {
        "n_null": label_result.n_null,
        "n_invalid_value": label_result.n_invalid_value,
        "is_valid": label_result.is_valid,
    }
    if not label_result.is_valid:
        logger.warning("Label verification found issues: %d null, %d invalid value.", label_result.n_null, label_result.n_invalid_value)

    # ---- Language verification (corrected Tamil-ratio formula) ----
    lang_cfg = cfg["language_verification"]
    lang_result = verify_languages(
        record_ids=df["record_id"].tolist(),
        texts=df["text"].tolist(),
        stored_languages=df["language"].tolist(),
        tamil_threshold=lang_cfg["tamil_ratio_threshold"],
    )
    report["stages"]["language_verification"] = {
        "n_mismatches": lang_result.n_mismatches,
        "mismatch_record_ids_sample": lang_result.mismatch_record_ids[:20],
    }

    # ---- Document-length analysis (measurement, pre-filter) ----
    length_stats_before = compute_length_stats(df)
    report["stages"]["length_stats_before_filtering"] = length_stats_before

    # ---- Remove invalid/extremely short documents ----
    doc_len_cfg = cfg["document_length"]
    invalid_result = flag_invalid_documents(
        df,
        min_word_count=doc_len_cfg["min_word_count"],
        extreme_length_percentile=doc_len_cfg["extreme_length_percentile"],
    )
    n_before_short_filter = len(df)
    df = df.loc[~invalid_result.too_short_mask].copy()
    report["stages"]["short_document_removal"] = {
        "n_removed": n_before_short_filter - len(df),
        "min_word_count_threshold": doc_len_cfg["min_word_count"],
        "n_extreme_length_flagged_not_removed": invalid_result.n_extreme_length,
        "extreme_length_threshold_chars": invalid_result.extreme_length_threshold_chars,
    }
    logger.info(
        "Removed %d row(s) below the min-word-count threshold (%s). %d row(s) flagged as "
        "extreme-length outliers (>= %.0f chars) but kept.",
        n_before_short_filter - len(df), doc_len_cfg["min_word_count"],
        invalid_result.n_extreme_length, invalid_result.extreme_length_threshold_chars,
    )

    # ---- Duplicate verification (post-cleaning — catches whitespace-
    #      masked duplicates Module 1's raw-field dedup missed) ----
    dup_cfg = cfg["duplicate_verification"]
    df, dup_report = verify_and_remove_duplicates(df, subset=dup_cfg["subset"])
    report["stages"]["duplicate_verification_post_cleaning"] = {
        "n_before": dup_report.n_before,
        "n_removed": dup_report.n_removed,
        "n_after": dup_report.n_after,
    }

    # ---- Final quality checks ----
    quality_result = run_quality_checks(df, cfg["quality_checks"])
    report["stages"]["final_quality_checks"] = {
        "passed": quality_result.passed,
        "failures": quality_result.failures,
        "checks": quality_result.checks,
    }
    if not quality_result.passed:
        raise RuntimeError(f"Module 2 quality checks failed: {quality_result.failures}")

    # ---- Post-cleaning length stats (for before/after comparison) ----
    length_stats_after = compute_length_stats(df)
    report["stages"]["length_stats_after_cleaning"] = length_stats_after

    n_final = len(df)
    report["stages"]["cleaned_dataset_summary"] = {
        "n_rows": n_final,
        "by_language": df["language"].value_counts().to_dict(),
        "by_label": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "by_language_and_label": (
            df.groupby("language")["label"].value_counts().unstack(fill_value=0).to_dict(orient="index")
        ),
    }

    # ---- Write cleaned dataset ----
    out_cfg = cfg["output"]
    processed_dir = root / out_cfg["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned_csv_path = processed_dir / out_cfg["cleaned_filename_csv"]
    cleaned_jsonl_path = processed_dir / out_cfg["cleaned_filename_jsonl"]
    df.to_csv(cleaned_csv_path, index=False)
    df.to_json(cleaned_jsonl_path, orient="records", lines=True, force_ascii=False)
    logger.info("Wrote cleaned dataset: %d rows -> %s / %s", n_final, cleaned_csv_path, cleaned_jsonl_path)

    # ---- Stratified split ----
    split_cfg = cfg["split"]
    split_result = stratified_split(
        df,
        stratify_columns=split_cfg["stratify_columns"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        random_seed=split_cfg["random_seed"],
    )
    train_path = root / out_cfg["split_dir"] / out_cfg["train_filename"]
    val_path = root / out_cfg["split_dir"] / out_cfg["val_filename"]
    test_path = root / out_cfg["split_dir"] / out_cfg["test_filename"]
    split_result.train.to_csv(train_path, index=False)
    split_result.val.to_csv(val_path, index=False)
    split_result.test.to_csv(test_path, index=False)
    logger.info(
        "Wrote split: train=%d val=%d test=%d -> %s / %s / %s",
        len(split_result.train), len(split_result.val), len(split_result.test),
        train_path, val_path, test_path,
    )

    # ---- Leakage verification ----
    leakage_result = verify_no_leakage(split_result.train, split_result.val, split_result.test)
    if not leakage_result.passed:
        raise RuntimeError(
            f"Data leakage detected across splits: train/val={leakage_result.n_train_val_overlap}, "
            f"train/test={leakage_result.n_train_test_overlap}, val/test={leakage_result.n_val_test_overlap}"
        )

    report["stages"]["split"] = {
        "train_ratio": split_cfg["train_ratio"],
        "val_ratio": split_cfg["val_ratio"],
        "test_ratio": split_cfg["test_ratio"],
        "random_seed": split_cfg["random_seed"],
        "stratify_columns": split_cfg["stratify_columns"],
        "n_train": len(split_result.train),
        "n_val": len(split_result.val),
        "n_test": len(split_result.test),
        "stratum_counts": split_result.stratum_counts,
        "leakage_check": {
            "passed": leakage_result.passed,
            "train_val_overlap": leakage_result.n_train_val_overlap,
            "train_test_overlap": leakage_result.n_train_test_overlap,
            "val_test_overlap": leakage_result.n_val_test_overlap,
        },
    }

    # ---- Write statistics + report ----
    reports_dir = root / out_cfg["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    stats_path = reports_dir / out_cfg["statistics_filename"]
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    report_md_path = reports_dir / out_cfg["report_filename"]
    write_preprocessing_report(report, report_md_path)

    # Snapshot the exact config used for this run alongside the other
    # reports — `preprocessing_config.yaml`, requested as its own
    # deliverable file so a given cleaned_dataset/report pair is always
    # traceable to the exact settings that produced it, independent of
    # whatever configs/module2_preprocessing.yaml looks like later.
    config_snapshot_path = reports_dir / "preprocessing_config.yaml"
    shutil.copyfile(resolved_config_path, config_snapshot_path)

    report["output_files"] = {
        "cleaned_csv": str(cleaned_csv_path),
        "cleaned_jsonl": str(cleaned_jsonl_path),
        "train_csv": str(train_path),
        "val_csv": str(val_path),
        "test_csv": str(test_path),
        "statistics_json": str(stats_path),
        "report_md": str(report_md_path),
        "config_snapshot": str(config_snapshot_path),
    }
    # Re-write the JSON once more now that output_files is populated, so
    # the on-disk statistics file includes it too (matches Module 1's pattern).
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Module 2 preprocessing pipeline complete. Final dataset: %d rows (from %d input rows).", n_final, n_input)
    return report
