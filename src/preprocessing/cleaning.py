"""Text-cleaning primitives — Module 2.

Each function does exactly one thing so the pipeline's stage order is
explicit and independently testable. `clean_text()` composes them in
the order that matters: HTML must be stripped before URL removal (a
`<a href="...">` tag's href is not "a URL in the text" until the tag
is unwrapped), Unicode normalization must run before whitespace
collapsing (composing combining marks can itself introduce/remove
characters), and whitespace collapsing must run last (every earlier
step tends to leave gaps behind where it deleted something).

Tamil-preservation note: nothing here transliterates, case-folds, or
otherwise touches Tamil script beyond canonical Unicode composition
(NFC) and removing genuinely non-linguistic noise (HTML tags, URLs,
zero-width characters, control characters). `lowercase_english()` is
applied only to rows tagged `en` by the caller — it is never run on
Tamil text.
"""

from __future__ import annotations

import html
import re
import unicodedata

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------

def remove_html(text: str) -> str:
    """Strip HTML tags and unescape entities (e.g. &amp; -> &).

    Uses BeautifulSoup's html.parser (not regex) because arbitrary
    scraped HTML is not a regular language — nested/malformed tags are
    common in the WELFake source data and a regex tag-stripper would
    mishandle them silently.
    """
    if not text:
        return ""
    stripped = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return html.unescape(stripped)


# ---------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)


def remove_urls(text: str) -> str:
    """Replace raw URLs with a space. URLs carry no classification
    signal as raw strings and only inflate downstream token counts."""
    if not text:
        return ""
    return _URL_RE.sub(" ", text)


# ---------------------------------------------------------------------
# Unicode normalization (Tamil-safe)
# ---------------------------------------------------------------------

# Zero-width characters seen in the real Module 1 data (ZWSP, ZWNJ,
# ZWJ, BOM/ZWNBSP) — copy-paste noise from web scraping, not
# meaningful content. Tamil script does not rely on ZWJ/ZWNJ for
# conjunct formation the way some other Indic scripts do, so stripping
# these does not affect Tamil rendering or meaning.
_ZERO_WIDTH_RE = re.compile("[​‌‍﻿]")

# C0 control characters excluding the whitespace ones normalize_whitespace
# already handles (\t \n \r), plus C1 controls.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Canonical Unicode normalization. NFC (the default) composes a
    base character + combining marks into their canonical precomposed
    sequence without altering meaning — critical for Tamil, where a
    consonant + dependent vowel sign must remain distinguishable from
    a different consonant + vowel sign. NFKC/NFKD ("compatibility")
    forms are deliberately not used here: they can collapse
    visually/linguistically distinct Tamil characters, which this
    project never wants.
    """
    if not text:
        return ""
    return unicodedata.normalize(form, text)


def remove_zero_width_chars(text: str) -> str:
    if not text:
        return ""
    return _ZERO_WIDTH_RE.sub("", text)


def strip_control_chars(text: str) -> str:
    if not text:
        return ""
    return _CONTROL_CHAR_RE.sub("", text)


# ---------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse any run of whitespace (spaces, tabs, newlines) to a
    single space and strip leading/trailing whitespace."""
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------
# Text normalization (language-aware)
# ---------------------------------------------------------------------

def lowercase_english(text: str) -> str:
    """Lowercase English text. Never call this on Tamil text — Tamil
    has no case distinction, so this would be a no-op at best and an
    unnecessary Unicode-library risk at worst."""
    if not text:
        return ""
    return text.lower()


# ---------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------

def clean_text(text: str | None, language: str, config: dict) -> str:
    """Run the full Module 2 cleaning pipeline on a single string.

    Order: HTML -> URLs -> Unicode normalize -> zero-width strip ->
    control-char strip -> whitespace collapse -> (English-only)
    lowercase. Returns "" for None/NaN/empty input rather than raising.
    `config` is `configs/module2_preprocessing.yaml`'s `cleaning:` block.
    """
    if text is None:
        return ""
    text = str(text)

    if config.get("remove_html", True):
        text = remove_html(text)
    if config.get("remove_urls", True):
        text = remove_urls(text)

    unicode_form = config.get("unicode_form", "NFC")
    if unicode_form:
        text = normalize_unicode(text, form=unicode_form)
    if config.get("remove_zero_width_chars", True):
        text = remove_zero_width_chars(text)
    if config.get("strip_control_chars", True):
        text = strip_control_chars(text)
    if config.get("collapse_whitespace", True):
        text = normalize_whitespace(text)

    if config.get("lowercase_english", True) and language == "en":
        text = lowercase_english(text)
        # Lowercasing can affect nothing whitespace-wise, but re-collapse
        # defensively in case a future rule changes that assumption.
        text = normalize_whitespace(text)

    return text
