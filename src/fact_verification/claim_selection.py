"""Module 5 — deterministic claim selection and normalisation.

No learned claim-extraction model. Check-worthiness labels do not exist
in this project, so an extractor could be neither trained nor validated;
a documented deterministic rule is reproducible and can be reported
honestly. Claim *detection* is declared out of scope rather than
approximated by an unvalidated model.

The rule is driven by a measured property of the corpus (audited
2026-08-09 over all 69,008 rows):

    Tamil   : median text 79 chars, 96.5% have NO body_text
              -> the headline already IS the claim; selection is a no-op
    English : median text 2,518 chars, 99.9% have body_text
              -> the title is the claim, the body becomes candidate
                 evidence for retrieval

Normalisation is Tamil-safe and mirrors Module 2: NFC (never NFKC/NFKD,
which can collapse linguistically distinct Tamil characters), and
lowercasing applied to Latin script only, since Tamil has no case.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
# Tamil block U+0B80-U+0BFF: letters, vowel signs (matras), virama.
_TAMIL_START, _TAMIL_END = "஀", "௿"


@dataclass
class Claim:
    """One selected claim, carrying its provenance.

    `dataset` and `source` are retained end-to-end so results can be
    reported per corpus. Merging datasets would hide domain mismatch,
    which is a stated reporting requirement.
    """

    claim_id: str
    text: str                 # normalised claim text
    raw_text: str             # exactly as selected, before normalisation
    language: str
    dataset: str              # "project" | "x_fact" | "fever"
    source: str | None = None
    label: Any = None         # corpus-native label; NOT remapped here
    record_id: str | None = None
    url: str | None = None
    evidence: list[str] = field(default_factory=list)
    selection_rule: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id, "text": self.text, "raw_text": self.raw_text,
            "language": self.language, "dataset": self.dataset, "source": self.source,
            "label": self.label, "record_id": self.record_id, "url": self.url,
            "evidence": self.evidence, "selection_rule": self.selection_rule,
            "metadata": self.metadata,
        }


def tamil_ratio(text: str) -> float:
    """Fraction of non-whitespace characters in the Tamil block.

    Uses the same bounded-in-[0,1] formula adopted in Module 2 after the
    Module 1 formula was found capable of exceeding 1.0.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _TAMIL_START <= c <= _TAMIL_END) / len(chars)


def normalize_claim(text: str, config: dict[str, Any] | None = None) -> str:
    """Normalise claim text for comparison and hashing.

    Deliberately conservative: it must not alter Tamil script. NFC only,
    Latin-only lowercasing, URL stripping, whitespace collapsing.
    """
    cfg = config or {}
    if text is None:
        return ""
    s = str(text)

    if cfg.get("strip_urls", True):
        s = _URL_RE.sub(" ", s)

    s = unicodedata.normalize(cfg.get("unicode_form", "NFC"), s)

    if cfg.get("lowercase_latin_only", True):
        # str.lower() is a no-op on Tamil codepoints, but restricting it
        # to ASCII makes the intent explicit and removes any dependence
        # on locale-sensitive casing behaviour.
        s = "".join(c.lower() if "A" <= c <= "Z" else c for c in s)

    if cfg.get("collapse_whitespace", True):
        s = _WS_RE.sub(" ", s).strip()
    return s


def hash_key(text: str, strip_punctuation: bool = True) -> str:
    """Aggressive key for EXACT-duplicate grouping.

    Punctuation is dropped so that claims differing only by a trailing
    '!' or quoting style collapse together. This is used only for
    grouping, never for the text fed to a model.
    """
    s = normalize_claim(text)
    if strip_punctuation:
        s = "".join(c for c in s if c.isalnum() or c.isspace() or _TAMIL_START <= c <= _TAMIL_END)
        s = _WS_RE.sub(" ", s).strip()
    return s


def select_claim_from_record(
    record: dict[str, Any],
    config: dict[str, Any],
    dataset: str = "project",
) -> Claim | None:
    """Apply the deterministic selection rule to one project record.

    Returns None when no usable claim can be formed (title missing and
    fallback too short). Dropped records are counted and reported rather
    than silently discarded.
    """
    cs = config["claim_selection"]
    norm_cfg = config["normalization"]

    title = record.get("title")
    title_str = "" if title is None or (isinstance(title, float)) else str(title).strip()

    rule = "title_as_claim"
    raw = title_str
    if len(hash_key(title_str)) < cs["min_claim_chars"]:
        # Fall back to the leading portion of `text`. For Tamil this is
        # usually identical to the title anyway; for English it is the
        # article opening.
        fallback = record.get(cs.get("fallback_field", "text")) or ""
        raw = str(fallback).strip()
        rule = "fallback_text_prefix"

    normalised = normalize_claim(raw, norm_cfg)
    if cs.get("truncate_long_titles", True) and len(normalised) > cs["max_claim_chars"]:
        normalised = normalised[: cs["max_claim_chars"]].rstrip()
        rule += "+truncated"

    if cs.get("drop_claims_below_min", True) and len(normalised) < cs["min_claim_chars"]:
        return None

    language = str(record.get("language", "")) or ("ta" if tamil_ratio(normalised) > 0.3 else "en")
    rid = record.get("record_id")

    return Claim(
        claim_id=f"{dataset}:{rid}" if rid else f"{dataset}:{hash_key(normalised)[:40]}",
        text=normalised,
        raw_text=raw,
        language=language,
        dataset=dataset,
        source=record.get("source"),
        label=record.get("label"),
        record_id=str(rid) if rid is not None else None,
        url=record.get("url") if isinstance(record.get("url"), str) else None,
        selection_rule=rule,
        metadata={
            "tamil_ratio": round(tamil_ratio(normalised), 4),
            "has_body_text": bool(record.get("body_text")) and str(record.get("body_text")) != "nan",
            "raw_char_len": len(raw),
        },
    )


def select_claims(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    dataset: str = "project",
) -> tuple[list[Claim], dict[str, Any]]:
    """Select claims across records and return a reporting summary.

    The summary records how many claims were dropped and why -- a silent
    drop would change the effective dataset size without appearing in
    any report.
    """
    claims: list[Claim] = []
    dropped = 0
    rules: dict[str, int] = {}
    for rec in records:
        c = select_claim_from_record(rec, config, dataset)
        if c is None:
            dropped += 1
            continue
        claims.append(c)
        rules[c.selection_rule] = rules.get(c.selection_rule, 0) + 1

    by_lang: dict[str, int] = {}
    for c in claims:
        by_lang[c.language] = by_lang.get(c.language, 0) + 1

    lengths = sorted(len(c.text) for c in claims)
    summary = {
        "dataset": dataset,
        "n_input_records": len(records),
        "n_claims_selected": len(claims),
        "n_dropped_below_min_chars": dropped,
        "by_language": by_lang,
        "selection_rules": rules,
        "claim_char_length": {
            "min": lengths[0] if lengths else 0,
            "median": lengths[len(lengths) // 2] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
        },
    }
    return claims, summary
