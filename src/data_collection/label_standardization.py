"""Label standardization — Module 1.

Every raw source expresses ground truth differently. This module maps
all of them into one fixed project-wide convention:

    Fake = 1
    Real = 0

  - ISOT ships as two separate files (Fake.csv / True.csv) — the file
    of origin *is* the label.
  - WELFake ships a `label` column using the *opposite* convention
    (0=fake, 1=real, verified against the real downloaded file: label
    0 -> 35,028 rows, label 1 -> 37,106 rows), so it is flipped:
    target = 1 - original.
  - Tamil sources collected in Module 1 have no structured verdict
    field (title/body text only), so labels are inferred by (1) a
    keyword scan of the title+body for an explicit verdict, falling
    back to (2) a per-source-type default. This is a documented
    approximation, not a substitute for manual verification — see
    docs/module1_overview.md.

This is standardization only — it does not classify anything. No model
is involved; every rule here is a fixed lookup or keyword match.
"""

from __future__ import annotations

from typing import Any

FAKE = 1
REAL = 0


def encode_isot_label(source_file: str, fake_filename: str = "Fake.csv", real_filename: str = "True.csv") -> int:
    if source_file == fake_filename:
        return FAKE
    if source_file == real_filename:
        return REAL
    raise ValueError(f"Unrecognized ISOT source file: {source_file!r}")


def encode_welfake_label(original_label: int, original_fake_value: int = 0, original_real_value: int = 1) -> int:
    if int(original_label) == original_fake_value:
        return FAKE
    if int(original_label) == original_real_value:
        return REAL
    raise ValueError(f"Unrecognized WELFake label value: {original_label!r}")


def infer_tamil_label(
    title: str,
    body_text: str | None,
    source: str,
    fact_check_sources: list[str],
    mainstream_sources: list[str],
    fake_keywords: list[str],
    real_keywords: list[str],
    default_fact_check_label: int = FAKE,
    default_mainstream_label: int = REAL,
) -> int | None:
    """Best-effort label for a Tamil record collected in Module 1.

    Priority: explicit verdict keyword found in title+body > source-type
    default. Returns None if the source isn't recognized as either
    fact-check or mainstream — the caller should treat that as a
    missing label and drop the row (never silently guess).
    """
    haystack = f"{title or ''} {body_text or ''}".lower()

    if any(kw.lower() in haystack for kw in fake_keywords):
        return FAKE
    if any(kw.lower() in haystack for kw in real_keywords):
        return REAL

    if source in fact_check_sources:
        return default_fact_check_label
    if source in mainstream_sources:
        return default_mainstream_label

    return None


def standardize_label(record: dict[str, Any], rules: dict[str, Any]) -> int | None:
    """Dispatch to the right encoder based on `record["source"]`.

    `record` must already carry whatever raw label info its source
    provides: `_source_file` for ISOT, `_original_label` for WELFake,
    or `title`/`body_text`/`source` for a Tamil source. Returns None
    (never raises) for a Tamil record whose source isn't in either
    source-type list, so the caller can drop it as unresolved rather
    than crash a whole-dataset run over one bad record.
    """
    source = record.get("source")

    if source == "isot":
        isot_rules = rules["isot"]
        return encode_isot_label(
            record["_source_file"],
            fake_filename=isot_rules["fake_source_file"],
            real_filename=isot_rules["real_source_file"],
        )

    if source == "welfake":
        welfake_rules = rules["welfake"]
        return encode_welfake_label(
            record["_original_label"],
            original_fake_value=welfake_rules["original_fake_value"],
            original_real_value=welfake_rules["original_real_value"],
        )

    # Everything else is a Tamil source collected in Module 1.
    return infer_tamil_label(
        title=record.get("title", ""),
        body_text=record.get("body_text"),
        source=source,
        fact_check_sources=rules["fact_check_sources"],
        mainstream_sources=rules["mainstream_news_sources"],
        fake_keywords=rules["fake_verdict_keywords"],
        real_keywords=rules["real_verdict_keywords"],
        default_fact_check_label=rules["default_fact_check_label"],
        default_mainstream_label=rules["default_mainstream_label"],
    )
