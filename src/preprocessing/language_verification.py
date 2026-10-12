"""Language verification — Module 2.

Module 1 already tagged every row `en`/`ta` and this project's own
analysis (see the Module 2 kickoff conversation / `docs/module2_overview.md`)
confirmed those tags are correct on the real data: all 5,416 Tamil-tagged
rows score well above threshold on a Tamil-script ratio, and zero
English-tagged rows show Tamil contamination. So this module does not
re-classify from scratch — it *verifies* the existing tags against a
corrected ratio formula and flags (not silently fixes) any disagreement.

Bug fixed relative to Module 1's `src/data_collection/language_identification.py`
(that file is left untouched, per this project's "do not modify Module 1
further" instruction — this is new, separate code):
Module 1's ratio was (Tamil-block codepoints) / (regex "\\w"-matched
codepoints). Python's Unicode `\\w` under-counts Tamil combining vowel
signs relative to the Tamil Unicode block character class, which *does*
count them — so that ratio could exceed 1.0 (confirmed on the real
Module 1 output: hundreds of Tamil rows scored ratios up to 2.0). It
never produced a wrong `en`/`ta` classification (anything scoring
"too high" only made the >=0.3 threshold check for Tamil more certain,
never less), but it's a real bug worth not carrying forward.

Fix: ratio = (Tamil-block codepoints) / (non-whitespace codepoints).
The numerator is a subset of the denominator by construction, so the
ratio is mathematically guaranteed to fall in [0, 1].
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TAMIL_RE = re.compile("[஀-௿]")  # U+0B80-U+0BFF, the Tamil Unicode block
_NON_WHITESPACE_RE = re.compile(r"\S")


def tamil_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that fall in the Tamil
    Unicode block. Always in [0.0, 1.0]."""
    if not text:
        return 0.0
    non_ws = _NON_WHITESPACE_RE.findall(text)
    if not non_ws:
        return 0.0
    tamil_chars = _TAMIL_RE.findall(text)
    return len(tamil_chars) / len(non_ws)


@dataclass
class LanguageVerificationResult:
    verified_language: list[str]
    n_mismatches: int
    mismatch_record_ids: list[str]


def verify_languages(
    record_ids: list[str],
    texts: list[str],
    stored_languages: list[str],
    tamil_threshold: float = 0.3,
) -> LanguageVerificationResult:
    """Recompute the corrected Tamil ratio for every row and compare
    against the stored `language` tag. Rows where stored=="ta" but the
    corrected ratio falls below threshold (or vice versa) are flagged
    as mismatches; the *stored* tag is kept as `verified_language`
    either way (flagging, not silently overwriting — a mismatch is a
    signal for a human to look at, not license to auto-relabel)."""
    verified: list[str] = []
    mismatch_ids: list[str] = []

    for record_id, text, stored in zip(record_ids, texts, stored_languages):
        ratio = tamil_ratio(text)
        corrected = "ta" if ratio >= tamil_threshold else "en"
        if corrected != stored:
            mismatch_ids.append(record_id)
        verified.append(stored)

    return LanguageVerificationResult(
        verified_language=verified,
        n_mismatches=len(mismatch_ids),
        mismatch_record_ids=mismatch_ids,
    )
