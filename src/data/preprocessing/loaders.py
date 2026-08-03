"""Load each Module 1 raw source into a common, unlabeled schema.

Every loader returns a DataFrame with (at least) these columns, so
downstream pipeline code never needs to know which source a row came
from:

    source           - short source name, e.g. "welfake", "boom_tamil"
    source_type      - "mainstream_news" | "fact_check"
    language_declared- "en" | "ta" (known a priori from the source)
    title            - str
    body_text        - str
    url              - str | None
    _source_file     - (ISOT only) which file the row came from
    _original_label  - (WELFake only) the dataset's own 0/1 label

Loaders are intentionally forgiving: a missing raw file logs a warning
and contributes zero rows rather than raising, so the pipeline can run
on a partial Module 1 output (e.g. only the datasets that were actually
downloaded/scraped so far).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("preprocessing.loaders")

STANDARD_COLUMNS = [
    "source",
    "source_type",
    "language_declared",
    "title",
    "body_text",
    "url",
]

TAMIL_SOURCES = [
    ("boom_tamil", "fact_check"),
    ("youturn", "fact_check"),
    ("pib_factcheck", "fact_check"),
    ("dailythanthi", "mainstream_news"),
    ("dinamalar", "mainstream_news"),
    ("hindutamil", "mainstream_news"),
]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def load_welfake(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "english" / "welfake" / "WELFake_Dataset.csv"
    if not path.exists():
        logger.warning("WELFake raw file not found at %s; skipping (0 rows).", path)
        return _empty_frame()

    df = pd.read_csv(path)
    # Published WELFake columns: an unnamed index, "title", "text", "label".
    df = df.rename(columns={"text": "body_text"})
    for col in ("title", "body_text"):
        if col not in df.columns:
            df[col] = ""

    df["source"] = "welfake"
    df["source_type"] = "mainstream_news"  # WELFake is a merged corpus of news-style articles, not a fact-check desk
    df["language_declared"] = "en"
    df["url"] = None
    df["_original_label"] = df["label"]

    logger.info("Loaded %d rows from WELFake.", len(df))
    return df[STANDARD_COLUMNS + ["_original_label"]]


def load_isot(raw_dir: Path) -> pd.DataFrame:
    extracted_dir = raw_dir / "english" / "isot" / "extracted"
    fake_path = _find_file(extracted_dir, "Fake.csv")
    real_path = _find_file(extracted_dir, "True.csv")

    frames = []
    if fake_path is not None:
        fake_df = pd.read_csv(fake_path)
        fake_df["_source_file"] = "Fake.csv"
        frames.append(fake_df)
    else:
        logger.warning("ISOT Fake.csv not found under %s; skipping.", extracted_dir)

    if real_path is not None:
        real_df = pd.read_csv(real_path)
        real_df["_source_file"] = "True.csv"
        frames.append(real_df)
    else:
        logger.warning("ISOT True.csv not found under %s; skipping.", extracted_dir)

    if not frames:
        return _empty_frame()

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"text": "body_text"})
    for col in ("title", "body_text"):
        if col not in df.columns:
            df[col] = ""

    df["source"] = "isot"
    df["source_type"] = "mainstream_news"
    df["language_declared"] = "en"
    df["url"] = None

    logger.info("Loaded %d rows from ISOT.", len(df))
    return df[STANDARD_COLUMNS + ["_source_file"]]


def _find_file(directory: Path, filename: str) -> Path | None:
    """ISOT's zip sometimes nests files a level deep; search shallowly for the filename."""
    if not directory.exists():
        return None
    direct = directory / filename
    if direct.exists():
        return direct
    matches = list(directory.rglob(filename))
    return matches[0] if matches else None


def load_tamil_jsonl(raw_dir: Path, source: str, source_type: str) -> pd.DataFrame:
    path = raw_dir / "tamil" / source / f"{source}.jsonl"
    if not path.exists():
        logger.warning("%s raw file not found at %s; skipping (0 rows).", source, path)
        return _empty_frame()

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line %d in %s", line_num, path)

    if not records:
        return _empty_frame()

    df = pd.DataFrame(records)
    for col in ("title", "body_text", "url"):
        if col not in df.columns:
            df[col] = None

    df["source"] = source
    df["source_type"] = source_type
    df["language_declared"] = "ta"

    logger.info("Loaded %d rows from %s.", len(df), source)
    return df[STANDARD_COLUMNS]


def load_all_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate every recognized Module 1 source found under `raw_dir`.

    Sources whose raw file is missing simply contribute 0 rows (with a
    logged warning) rather than aborting the whole run.
    """
    raw_dir = Path(raw_dir)
    frames = [load_welfake(raw_dir), load_isot(raw_dir)]
    frames.extend(load_tamil_jsonl(raw_dir, source, source_type) for source, source_type in TAMIL_SOURCES)

    non_empty = [f for f in frames if len(f) > 0]
    if not non_empty:
        logger.error("No raw data found under %s for any recognized source.", raw_dir)
        return _empty_frame()

    combined = pd.concat(non_empty, ignore_index=True)
    combined["record_id"] = range(len(combined))
    logger.info("Combined raw dataset: %d rows across %d source(s).", len(combined), combined["source"].nunique())
    return combined
