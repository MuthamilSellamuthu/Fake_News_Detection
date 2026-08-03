"""Label encoding for Module 2.

Target convention for the whole project (fixed by spec):
    Fake = 1
    Real = 0

Each raw source expresses truthfulness differently, so encoding logic
is source-specific:

  - ISOT ships as two separate files (Fake.csv / True.csv) — the file
    of origin *is* the label, tracked via a `_source_file` column set
    at load time.
  - WELFake ships a `label` column using the *opposite* convention
    (0=fake, 1=real per the dataset's own documentation), so it must be
    flipped: target = 1 - original.
  - The Tamil sources collected in Module 1 have no structured verdict
    field (only title/body text), so labels are inferred by: (1) a
    keyword scan for an explicit verdict in the title+body, falling
    back to (2) a per-source-type default. This is an approximation —
    see configs/preprocessing.yaml for the caveat and keyword lists.
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

    Priority: explicit verdict keyword > source-type default. Returns
    None if the source isn't recognized as either fact-check or
    mainstream (caller should treat that as a missing label and drop
    the row).
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


def apply_label_rules(record: dict[str, Any], rules: dict[str, Any]) -> int | None:
    """Dispatch to the right encoder based on `record["source"]`.

    `record` is expected to already carry whatever raw label info that
    source provides (`_source_file` for ISOT, `_original_label` for
    WELFake, `title`/`body_text`/`source` for Tamil sources).
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

    # Everything else is a Tamil source from Module 1.
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
