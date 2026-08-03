"""Module 2 pipeline: load -> clean -> label -> detect language -> dedup -> merge.

`build_dataset()` is the single entrypoint the CLI (and, later, notebooks
or tests) should call. It returns the merged, cleaned, labeled DataFrame
plus a stats dict describing what happened at each stage — no file I/O
happens here (see `save.py` for that), so this function is easy to unit
test or call from a notebook without touching disk.

No model code, tokenization, sentence-pair construction, or filtering
(consistency/threshold filtering from the base paper) happens here —
that begins in Module 3.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.preprocessing.cleaning import clean_text
from src.data.preprocessing.dedup import drop_duplicates, drop_missing
from src.data.preprocessing.label_encoder import apply_label_rules
from src.data.preprocessing.language_detection import detect_language
from src.data.preprocessing.loaders import load_all_raw
from src.utils.logger import get_logger

logger = get_logger("preprocessing.pipeline")


def build_dataset(raw_dir: str, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {}

    # 1) Load + merge all recognized raw sources into one schema.
    df = load_all_raw(raw_dir)
    stats["rows_loaded"] = len(df)
    if df.empty:
        logger.error("No data loaded from %s; nothing to process.", raw_dir)
        return df, stats

    # 2) Clean title + body text (HTML/URL/emoji/symbol removal, whitespace normalization).
    logger.info("Cleaning text...")
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_body"] = df["body_text"].apply(clean_text)
    df["clean_text"] = (df["clean_title"] + " " + df["clean_body"]).str.strip()

    # 3) Label encoding (Fake=1, Real=0) — source-specific rules from config.
    logger.info("Assigning labels...")
    label_rules = config["label_rules"]
    df["label"] = df.apply(lambda row: apply_label_rules(row.to_dict(), label_rules), axis=1)
    unresolved = df["label"].isna().sum()
    if unresolved:
        logger.warning("%d row(s) had no resolvable label (will be dropped as missing).", unresolved)
    stats["unresolved_labels"] = int(unresolved)

    # 4) Language detection (Tamil vs. English, mirroring the two dataset families).
    logger.info("Detecting language...")
    lang_cfg = config["language_detection"]
    threshold = lang_cfg["tamil_char_ratio_threshold"]
    df["language_detected"] = df["clean_text"].apply(lambda t: detect_language(t, tamil_threshold=threshold))
    # Prefer the detector's output; fall back to the source's declared language
    # when detection is inconclusive (very short/ambiguous cleaned text).
    df["language"] = df["language_detected"].where(
        df["language_detected"].isin(["ta", "en"]), df["language_declared"]
    )

    # 5) Missing-value removal (empty/too-short text, unresolved label).
    logger.info("Removing rows with missing text/label...")
    df, n_missing = drop_missing(
        df,
        text_col="clean_text",
        label_col="label",
        min_length=config["cleaning"]["min_text_length"],
    )
    stats["rows_dropped_missing"] = n_missing

    # 6) Duplicate removal (exact match on cleaned text).
    logger.info("Removing duplicate rows...")
    df, n_duplicates = drop_duplicates(df, subset=config["cleaning"]["dedup_subset"])
    stats["rows_dropped_duplicate"] = n_duplicates

    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)
    df["record_id"] = range(len(df))

    stats["rows_final"] = len(df)
    stats["label_counts"] = df["label"].value_counts().to_dict()
    stats["language_counts"] = df["language"].value_counts().to_dict()
    stats["source_counts"] = df["source"].value_counts().to_dict()

    logger.info(
        "Final dataset: %d rows | labels=%s | languages=%s | sources=%s",
        stats["rows_final"],
        stats["label_counts"],
        stats["language_counts"],
        stats["source_counts"],
    )

    output_columns = [
        "record_id",
        "source",
        "source_type",
        "language",
        "language_declared",
        "language_detected",
        "title",
        "clean_title",
        "body_text",
        "clean_body",
        "clean_text",
        "label",
        "url",
    ]
    return df[output_columns], stats
