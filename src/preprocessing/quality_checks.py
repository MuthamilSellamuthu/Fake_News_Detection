"""Missing-value verification, duplicate verification, and final
dataset quality checks — Module 2.

These are checks, not transformations — they report what's wrong (and,
for duplicates, remove exact repeats), but the heavier decisions
(what counts as "invalid") live in `document_length.py` and the
pipeline orchestrator, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("preprocessing.quality_checks")


@dataclass
class MissingValueReport:
    counts: dict[str, int]
    critical_fields_clean: bool  # True if text/label have zero nulls


def verify_missing_values(df: pd.DataFrame, critical_fields: tuple[str, ...] = ("text", "label")) -> MissingValueReport:
    counts = {col: int(df[col].isna().sum()) for col in df.columns}
    critical_clean = all(counts.get(f, 0) == 0 for f in critical_fields)
    if not critical_clean:
        logger.warning("Critical field(s) have null values: %s", {f: counts.get(f, 0) for f in critical_fields})
    return MissingValueReport(counts=counts, critical_fields_clean=critical_clean)


@dataclass
class DuplicateReport:
    n_before: int
    n_removed: int
    n_after: int
    removed_record_ids: list[str]


def verify_and_remove_duplicates(
    df: pd.DataFrame,
    subset: list[str],
    record_id_col: str = "record_id",
) -> tuple[pd.DataFrame, DuplicateReport]:
    """Exact-match dedup on `subset`, run *after* text cleaning (this is
    the point of running this again post-Module-1: cleaning/normalization
    can surface duplicates that differed only in whitespace/HTML/etc.
    before cleaning — confirmed on the real data, see
    docs/module2_preprocessing_analysis section of the completion report).
    """
    n_before = len(df)
    dup_mask = df.duplicated(subset=subset, keep="first")
    removed_ids = df.loc[dup_mask, record_id_col].tolist()
    filtered = df.loc[~dup_mask].copy()
    n_after = len(filtered)

    if n_before - n_after:
        logger.info("Duplicate verification (post-cleaning, subset=%s): removed %d row(s).", subset, n_before - n_after)

    return filtered, DuplicateReport(
        n_before=n_before,
        n_removed=n_before - n_after,
        n_after=n_after,
        removed_record_ids=removed_ids,
    )


@dataclass
class QualityCheckResult:
    passed: bool
    failures: list[str]
    checks: dict[str, Any]


def run_quality_checks(
    df: pd.DataFrame,
    config: dict,
    text_col: str = "text",
    label_col: str = "label",
    language_col: str = "language",
) -> QualityCheckResult:
    """Final automated assertions on the fully-cleaned dataset. Returns
    a result object rather than raising, so the caller can decide
    whether to halt or just log — but every individual check result is
    always reported, pass or fail, for the preprocessing report.
    """
    checks: dict[str, Any] = {}
    failures: list[str] = []

    n_null_text = int(df[text_col].isna().sum())
    checks["no_null_text"] = {"passed": n_null_text == 0, "n_null": n_null_text}
    if config.get("require_no_null_text", True) and n_null_text:
        failures.append(f"{n_null_text} row(s) have null text")

    n_empty_text = int((df[text_col].fillna("").astype(str).str.strip() == "").sum())
    checks["no_empty_text"] = {"passed": n_empty_text == 0, "n_empty": n_empty_text}
    if config.get("require_no_empty_text", True) and n_empty_text:
        failures.append(f"{n_empty_text} row(s) have empty text")

    n_null_label = int(df[label_col].isna().sum())
    checks["no_null_label"] = {"passed": n_null_label == 0, "n_null": n_null_label}
    if config.get("require_no_null_label", True) and n_null_label:
        failures.append(f"{n_null_label} row(s) have null label")

    valid_labels = {0, 1}
    n_invalid_label = int((~df[label_col].isin(valid_labels)).sum())
    checks["valid_labels"] = {"passed": n_invalid_label == 0, "n_invalid": n_invalid_label}
    if config.get("require_valid_labels", True) and n_invalid_label:
        failures.append(f"{n_invalid_label} row(s) have a label outside {{0,1}}")

    valid_languages = {"en", "ta"}
    n_invalid_lang = int((~df[language_col].isin(valid_languages)).sum())
    checks["valid_languages"] = {"passed": n_invalid_lang == 0, "n_invalid": n_invalid_lang}
    if config.get("require_valid_languages", True) and n_invalid_lang:
        failures.append(f"{n_invalid_lang} row(s) have a language outside {{en,ta}}")

    n_exact_dupes = int(df.duplicated(subset=[text_col]).sum())
    checks["no_exact_duplicates"] = {"passed": n_exact_dupes == 0, "n_duplicates": n_exact_dupes}
    if config.get("require_no_exact_duplicates", True) and n_exact_dupes:
        failures.append(f"{n_exact_dupes} exact-duplicate text row(s) remain")

    passed = len(failures) == 0
    if not passed:
        logger.warning("Quality checks FAILED: %s", failures)
    else:
        logger.info("All quality checks passed.")

    return QualityCheckResult(passed=passed, failures=failures, checks=checks)
