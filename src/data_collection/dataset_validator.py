"""Module 1 orchestrator — validation, deduplication, language ID, label
standardization, and statistics reporting.

This ties together `deduplication.py`, `language_identification.py`,
`label_standardization.py`, and `dataset_stats.py` into one pipeline
that reads the real files under `datasets/raw/` and writes:

  - `datasets/processed/module1_dataset.{csv,jsonl}` — one unified,
    standardized-but-uncleaned dataset across every source.
  - `outputs/module1/validation_report.json` — what was dropped and why.
  - `outputs/module1/dataset_statistics.{json,md}` — final counts.

Deliberately NOT done here (Module 2's job — see
docs/module1_overview.md): text normalization/cleaning, contrastive
pair construction, near-duplicate filtering, tokenization, or
train/val/test splitting. This module only makes the raw collected
data *trustworthy and countable*, nothing more.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_collection.deduplication import (
    drop_exact_duplicates,
    drop_missing_label,
    drop_missing_text,
)
from src.data_collection.dataset_stats import compute_dataset_statistics, write_statistics_reports
from src.data_collection.label_standardization import standardize_label
from src.data_collection.language_identification import (
    identify_languages_bulk,
    identify_languages_per_row,
)
from src.utils.config_loader import load_yaml, project_root
from src.utils.logger import get_logger

logger = get_logger("data_collection.dataset_validator")

UNIFIED_COLUMNS = [
    "record_id",
    "source",
    "source_type",
    "language",
    "title",
    "body_text",
    "text",
    "label",
    "url",
    "date",
]


# --------------------------------------------------------------------------
# Loading + normalizing each raw source into the unified schema
# --------------------------------------------------------------------------

def _load_welfake(cfg: dict[str, Any], root: Path) -> pd.DataFrame:
    """Load WELFake and standardize its label inline, vectorized.

    Label standardization for WELFake is a fixed value flip (0<->1), not
    a per-row decision — running it as a `pandas.DataFrame.apply(axis=1)`
    dispatch (as `label_standardization.standardize_label` does for the
    Tamil sources) would mean ~72k Python-level function calls for a
    rule that is really just `1 - original_label`. Vectorizing it here
    keeps the *rule* defined once in
    `label_standardization.encode_welfake_label` (used to validate this
    logic and reused as-is for any single-row/ad-hoc lookup) while
    making the bulk path fast.
    """
    from src.data_collection.label_standardization import encode_welfake_label

    raw_cfg = cfg["raw_inputs"]["english"]["welfake"]
    label_rules = cfg["label_rules"]["welfake"]
    path = root / raw_cfg["path"]
    if not path.exists():
        logger.warning("WELFake file not found at %s — skipping.", path)
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    df = pd.read_csv(path)
    n_raw = len(df)

    fake_value = label_rules["original_fake_value"]
    real_value = label_rules["original_real_value"]
    original_labels = df[raw_cfg["label_col"]]
    valid_mask = original_labels.isin([fake_value, real_value])
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        logger.warning("WELFake: %d row(s) have an unrecognized label value and will be dropped.", n_invalid)
        # Confirm the rule is correct on at least one real example (cheap sanity check).
    assert encode_welfake_label(fake_value, fake_value, real_value) == 1
    assert encode_welfake_label(real_value, fake_value, real_value) == 0

    df = df.loc[valid_mask].copy()
    standardized_label = (df[raw_cfg["label_col"]] == fake_value).astype(int)  # fake_value -> 1 (FAKE), else 0 (REAL)

    out = pd.DataFrame(
        {
            "record_id": [f"welfake_{i}" for i in df.index],
            "source": "welfake",
            "source_type": "mixed_news",
            "language": None,
            "title": df[raw_cfg["title_col"]].fillna(""),
            "body_text": df[raw_cfg["text_col"]].fillna(""),
            "url": None,
            "date": None,
            "label": standardized_label,
        }
    )
    out["text"] = (out["title"].astype(str) + " " + out["body_text"].astype(str)).str.strip()
    logger.info("Loaded WELFake: %d raw rows (%d after dropping unrecognized labels) from %s", n_raw, len(out), path)
    return out


def _load_isot(cfg: dict[str, Any], root: Path) -> pd.DataFrame:
    """Load ISOT (Fake.csv / True.csv) and standardize labels inline.

    The label here is the file of origin, a per-file constant — no
    per-row lookup is needed at all, so this assigns the standardized
    label directly rather than routing through the generic
    `standardize_label` dispatch (kept for the Tamil, genuinely
    per-row, case). `encode_isot_label` is still exercised via the
    assertions below as a correctness check on the constant.
    """
    from src.data_collection.label_standardization import encode_isot_label

    raw_cfg = cfg["raw_inputs"]["english"]["isot"]
    isot_rules = cfg["label_rules"]["isot"]
    fake_path = root / raw_cfg["fake_path"]
    real_path = root / raw_cfg["real_path"]

    assert encode_isot_label(isot_rules["fake_source_file"], isot_rules["fake_source_file"], isot_rules["real_source_file"]) == 1
    assert encode_isot_label(isot_rules["real_source_file"], isot_rules["fake_source_file"], isot_rules["real_source_file"]) == 0

    frames = []
    for path, source_file, label in (
        (fake_path, isot_rules["fake_source_file"], 1),
        (real_path, isot_rules["real_source_file"], 0),
    ):
        if not path.exists():
            logger.warning("ISOT file not found at %s — skipping.", path)
            continue
        df = pd.read_csv(path)
        frame = pd.DataFrame(
            {
                "record_id": [f"isot_{source_file}_{i}" for i in df.index],
                "source": "isot",
                "source_type": "mixed_news",
                "language": None,
                "title": df[raw_cfg["title_col"]].fillna(""),
                "body_text": df[raw_cfg["text_col"]].fillna(""),
                "url": None,
                "date": df["date"] if "date" in df.columns else None,
                "label": label,
            }
        )
        frame["text"] = (frame["title"].astype(str) + " " + frame["body_text"].astype(str)).str.strip()
        logger.info("Loaded ISOT/%s: %d raw rows from %s", source_file, len(df), path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _load_aanirudh_headlines(cfg: dict[str, Any], root: Path) -> pd.DataFrame:
    """Load the AAnirudh07 Tamil headlines corpus; label is already explicit.

    Unlike the scraped Tamil sources (`_load_tamil_sources`), this
    dataset ships a real fake/real column ("Authenticity") — no
    keyword-scan/source-type-default approximation is used for these
    rows. The only genuine unknown is which polarity Authenticity uses
    (0=real/1=fake, or the reverse) — no README text this project could
    reach states it explicitly. Resolved here by comparing the observed
    0/1 split against the dataset's own documented class counts (2,949
    fake / 2,324 real) and picking whichever orientation is closer; the
    chosen orientation and the actual observed counts are always logged
    so this is checkable, not a silent guess.
    """
    tamil_direct_cfg = cfg["raw_inputs"]["tamil"].get("direct_downloads", {})
    raw_cfg = tamil_direct_cfg.get("aanirudh_headlines")
    if raw_cfg is None:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    path = root / raw_cfg["path"]
    if not path.exists():
        logger.warning(
            "AAnirudh07 Tamil headlines file not found at %s — skipping "
            "(see docs/module1_overview.md for the download link; this source "
            "requires network access this project's dev sandbox does not have).",
            path,
        )
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    df = pd.read_csv(path)
    n_raw = len(df)

    authenticity_col = raw_cfg["authenticity_col"]
    news_col = raw_cfg["news_col"]
    valid_mask = df[authenticity_col].isin([0, 1])
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        logger.warning(
            "AAnirudh07 headlines: %d row(s) have a non-binary Authenticity value and will be dropped.",
            n_invalid,
        )
    df = df.loc[valid_mask].copy()

    count_1 = int((df[authenticity_col] == 1).sum())
    count_0 = int((df[authenticity_col] == 0).sum())
    expected_fake = raw_cfg["expected_fake_count"]
    expected_real = raw_cfg["expected_real_count"]

    direct_error = abs(count_1 - expected_fake) + abs(count_0 - expected_real)
    flipped_error = abs(count_0 - expected_fake) + abs(count_1 - expected_real)

    if direct_error <= flipped_error:
        label = df[authenticity_col].astype(int)
        orientation = "direct (Authenticity 1=fake, 0=real)"
    else:
        label = 1 - df[authenticity_col].astype(int)
        orientation = "flipped (Authenticity 1=real, 0=fake)"

    logger.info(
        "AAnirudh07 headlines: observed Authenticity split 1=%d, 0=%d (dataset's documented "
        "split: fake=%d, real=%d) -> using %s orientation. If this looks wrong, spot-check a "
        "few known-fake headlines against the source file manually.",
        count_1, count_0, expected_fake, expected_real, orientation,
    )

    out = pd.DataFrame(
        {
            "record_id": [f"aanirudh_headlines_{i}" for i in df.index],
            "source": "aanirudh_headlines",
            "source_type": "mixed_news",
            "language": None,
            "title": df[news_col].fillna(""),
            "body_text": "",
            "url": None,
            "date": df["Date"] if "Date" in df.columns else None,
            "label": label,
        }
    )
    out["text"] = out["title"].astype(str).str.strip()
    logger.info(
        "Loaded AAnirudh07 Tamil headlines: %d raw rows (%d after dropping invalid labels) from %s",
        n_raw, len(out), path,
    )
    return out


def _load_tamil_sources(cfg: dict[str, Any], root: Path) -> pd.DataFrame:
    source_type_map: dict[str, str] = cfg["raw_inputs"]["tamil"]["sources"]
    tamil_root = root / "datasets" / "raw" / "tamil"

    frames = []
    if not tamil_root.exists():
        logger.warning("Tamil raw directory not found at %s — skipping.", tamil_root)
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    for jsonl_path in sorted(tamil_root.glob("*/*.jsonl")):
        records: list[dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON line %d in %s", line_no, jsonl_path)

        if not records:
            continue

        df = pd.DataFrame(records)
        source_name = df["source"].iloc[0] if "source" in df.columns and len(df) else jsonl_path.parent.name
        source_type = source_type_map.get(source_name, df.get("source_type", pd.Series(["unknown"])).iloc[0])

        frame = pd.DataFrame(
            {
                "record_id": [f"{source_name}_{jsonl_path.stem}_{i}" for i in df.index],
                "source": source_name,
                "source_type": source_type,
                "language": df["language"] if "language" in df.columns else "ta",
                "title": df["title"].fillna("") if "title" in df.columns else "",
                "body_text": df["body_text"].fillna("") if "body_text" in df.columns else "",
                "url": df["url"] if "url" in df.columns else None,
                "date": df["date_hint"] if "date_hint" in df.columns else None,
            }
        )
        frame["text"] = (frame["title"].astype(str) + " " + frame["body_text"].astype(str)).str.strip()
        logger.info("Loaded Tamil source '%s': %d raw rows from %s", source_name, len(df), jsonl_path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------

def _apply_label_standardization(df: pd.DataFrame, rules: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    if df.empty:
        df["label"] = pd.Series(dtype="object")
        return df, 0

    labels = df.apply(lambda row: standardize_label(row.to_dict(), rules), axis=1)
    df = df.copy()
    df["label"] = labels
    return drop_missing_label(df, label_col="label")


def _apply_language_identification(
    english_df: pd.DataFrame,
    tamil_df: pd.DataFrame,
    lang_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    lang_report: dict[str, Any] = {}

    if not english_df.empty:
        result = identify_languages_bulk(
            texts=english_df["text"].tolist(),
            assumed_language="en",
            tamil_threshold=lang_cfg["tamil_char_ratio_threshold"],
            sample_size=lang_cfg["bulk_langdetect_sample_size"],
            random_seed=lang_cfg["bulk_langdetect_random_seed"],
        )
        english_df = english_df.copy()
        english_df["language"] = result.languages
        lang_report["english_sources"] = {
            "sample_size": result.sample_size,
            "sample_verified": result.sample_verified,
            "sample_mismatches": len(result.sample_mismatches),
            "tamil_contamination_rows": int((english_df["language"] == "ta").sum()),
        }

    if not tamil_df.empty:
        detected = identify_languages_per_row(
            tamil_df["text"].tolist(),
            tamil_threshold=lang_cfg["tamil_char_ratio_threshold"],
        )
        tamil_df = tamil_df.copy()
        tamil_df["language"] = detected
        lang_report["tamil_sources"] = {
            "rows_checked": len(tamil_df),
            "by_detected_language": pd.Series(detected).value_counts().to_dict(),
        }

    return english_df, tamil_df, lang_report


def run_validation_pipeline(config_path: str | Path | None = None) -> dict[str, Any]:
    """Run the full Module 1 validation pipeline end-to-end and return a report dict."""
    root = project_root()
    cfg = load_yaml(config_path or (root / "configs" / "module1_validation.yaml"))

    validation_report: dict[str, Any] = {"stages": {}}

    # ---- Load ----
    welfake_df = _load_welfake(cfg, root)
    isot_df = _load_isot(cfg, root)
    tamil_scraped_df = _load_tamil_sources(cfg, root)
    tamil_direct_df = _load_aanirudh_headlines(cfg, root)

    english_df = pd.concat([welfake_df, isot_df], ignore_index=True) if len(welfake_df) or len(isot_df) else pd.DataFrame(columns=UNIFIED_COLUMNS)
    validation_report["stages"]["rows_loaded"] = {
        "welfake": int(len(welfake_df)),
        "isot": int(len(isot_df)),
        "tamil_scraped_total": int(len(tamil_scraped_df)),
        "by_tamil_scraped_source": tamil_scraped_df["source"].value_counts().to_dict() if not tamil_scraped_df.empty else {},
        "aanirudh_headlines": int(len(tamil_direct_df)),
    }

    # ---- Label standardization ----
    # English sources (WELFake, ISOT) and the AAnirudh07 Tamil headlines
    # corpus already have their `label` column assigned vectorized at
    # load time (see the respective `_load_*` docstrings for why) —
    # only the missing-label safety check applies to them here. The
    # scraped Tamil sources genuinely need a per-row decision (keyword
    # scan + source-type fallback), so only they go through the generic
    # `standardize_label` dispatch.
    english_df, english_label_dropped = drop_missing_label(english_df, label_col="label") if not english_df.empty else (english_df, 0)
    tamil_direct_df, tamil_direct_label_dropped = drop_missing_label(tamil_direct_df, label_col="label") if not tamil_direct_df.empty else (tamil_direct_df, 0)

    tamil_rules = cfg["label_rules"]
    tamil_scraped_df, tamil_scraped_label_dropped = _apply_label_standardization(tamil_scraped_df, tamil_rules)

    tamil_df = pd.concat([tamil_scraped_df, tamil_direct_df], ignore_index=True, sort=False)

    validation_report["stages"]["dropped_unresolved_label"] = {
        "english": int(english_label_dropped),
        "tamil_scraped": int(tamil_scraped_label_dropped),
        "aanirudh_headlines": int(tamil_direct_label_dropped),
    }

    # ---- Drop missing/too-short text ----
    val_cfg = cfg["validation"]
    english_df, english_text_dropped = drop_missing_text(english_df, text_col="text", min_length=val_cfg["min_text_length"])
    tamil_df, tamil_text_dropped = drop_missing_text(tamil_df, text_col="text", min_length=val_cfg["min_text_length"])
    validation_report["stages"]["dropped_missing_text"] = {
        "english": int(english_text_dropped),
        "tamil": int(tamil_text_dropped),
    }

    # ---- Exact-duplicate removal ----
    english_df, english_dupes = drop_exact_duplicates(english_df, subset=val_cfg["english_dedup_subset"])

    tamil_dedup_key = val_cfg["tamil_dedup_key"]
    if not tamil_df.empty and tamil_dedup_key in tamil_df.columns and tamil_df[tamil_dedup_key].notna().any():
        # Rows with a real URL dedup on URL; rows without one (listing-only
        # records with a null url) fall back to title-based dedup so they
        # aren't silently exempted from deduplication entirely.
        has_key = tamil_df[tamil_dedup_key].notna()
        with_key, dupes_with_key = drop_exact_duplicates(tamil_df.loc[has_key], subset=[tamil_dedup_key])
        without_key, dupes_without_key = drop_exact_duplicates(tamil_df.loc[~has_key], subset=["title"])
        tamil_df = pd.concat([with_key, without_key], ignore_index=True)
        tamil_dupes = dupes_with_key + dupes_without_key
    else:
        tamil_df, tamil_dupes = drop_exact_duplicates(tamil_df, subset=["title"])

    validation_report["stages"]["dropped_exact_duplicates"] = {
        "english": int(english_dupes),
        "tamil": int(tamil_dupes),
    }

    # ---- Language identification ----
    english_df, tamil_df, lang_report = _apply_language_identification(
        english_df, tamil_df, cfg["language_identification"]
    )
    validation_report["stages"]["language_identification"] = lang_report

    # ---- Combine, finalize schema ----
    combined = pd.concat([english_df, tamil_df], ignore_index=True, sort=False)
    combined["label"] = combined["label"].astype(int)
    final_columns = [c for c in UNIFIED_COLUMNS if c in combined.columns]
    combined = combined[final_columns]

    validation_report["stages"]["rows_final"] = {
        "total": int(len(combined)),
        "by_source": combined["source"].value_counts().to_dict(),
    }

    # ---- Write processed dataset ----
    out_cfg = cfg["output"]
    processed_dir = root / out_cfg["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = processed_dir / out_cfg["dataset_filename_csv"]
    jsonl_path = processed_dir / out_cfg["dataset_filename_jsonl"]
    combined.to_csv(csv_path, index=False)
    # `DataFrame.to_json(..., lines=True)` is pandas' vectorized/C-accelerated
    # JSONL writer — materially faster than building a Python list of dicts
    # via `to_dict(orient="records")` and calling `json.dumps` per row, which
    # becomes the pipeline's slowest step once the combined frame exceeds
    # ~60k rows (confirmed during development: the manual-loop version was
    # slow enough to blow past this project's own tooling timeouts).
    combined.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    logger.info("Wrote standardized dataset: %d rows -> %s / %s", len(combined), csv_path, jsonl_path)

    # ---- Statistics ----
    stats = compute_dataset_statistics(combined)
    reports_dir = root / out_cfg["reports_dir"]
    write_statistics_reports(
        stats,
        json_path=reports_dir / out_cfg["statistics_report_json_filename"],
        md_path=reports_dir / out_cfg["statistics_report_md_filename"],
    )

    # ---- Validation report ----
    validation_report["output_files"] = {
        "processed_csv": str(csv_path),
        "processed_jsonl": str(jsonl_path),
        "statistics_json": str(reports_dir / out_cfg["statistics_report_json_filename"]),
        "statistics_md": str(reports_dir / out_cfg["statistics_report_md_filename"]),
    }
    validation_report_path = reports_dir / out_cfg["validation_report_filename"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    with validation_report_path.open("w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False, default=str)
    validation_report["output_files"]["validation_report"] = str(validation_report_path)

    logger.info("Module 1 validation pipeline complete. Final dataset: %d rows.", len(combined))
    return {"validation_report": validation_report, "statistics": stats}
