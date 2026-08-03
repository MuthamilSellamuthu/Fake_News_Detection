"""Text-cleaning primitives for Module 2.

Each function does exactly one thing so the pipeline stage order is
explicit and testable in isolation. `clean_text()` composes them in the
order that matters: HTML must be stripped before URL/symbol removal (an
<a href="..."> tag's href is not a "URL in text" until unwrapped), and
whitespace normalization must run last (every earlier step tends to
leave gaps behind).

Tamil-preservation note: the "remove special symbols" step keeps
Python's Unicode-aware `\\w` (which already includes Tamil letters,
since Python's `re` treats `\\w` as "any Unicode word character", not
just ASCII) *and* explicitly whitelists the Tamil Unicode block
(U+0B80-U+0BFF) so the property holds even if that default behaviour
ever changes. Nothing in this module transliterates, normalizes script,
or otherwise touches Tamil text beyond removing genuinely non-linguistic
symbols/whitespace.
"""

from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------
def remove_html(text: str) -> str:
    """Strip HTML tags and unescape entities (e.g. &amp; -> &)."""
    if not text:
        return ""
    stripped = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return html.unescape(stripped)


# ---------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)


def remove_urls(text: str) -> str:
    if not text:
        return ""
    return _URL_RE.sub(" ", text)


# ---------------------------------------------------------------------
# Emoji
# ---------------------------------------------------------------------
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F000-\U0001F0FF"  # mahjong / playing cards / dominoes
    "\U00002B00-\U00002BFF"  # misc symbols and arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "]+",
    flags=re.UNICODE,
)


def remove_emoji(text: str) -> str:
    if not text:
        return ""
    return _EMOJI_RE.sub(" ", text)


# ---------------------------------------------------------------------
# Special symbols (Tamil-preserving)
# ---------------------------------------------------------------------
# Whitelist: Unicode word chars (\w, includes Tamil letters), whitespace,
# the Tamil block explicitly (belt-and-braces), and a small set of
# sentence-level punctuation that carries meaning in news text.
_ALLOWED_CHARS = r"\w\s஀-௿.,!?'\"()\-:;%/@#&+"
_DISALLOWED_RE = re.compile(f"[^{_ALLOWED_CHARS}]", re.UNICODE)


def remove_special_symbols(text: str) -> str:
    if not text:
        return ""
    return _DISALLOWED_RE.sub(" ", text)


# ---------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------
def clean_text(text: str | None) -> str:
    """Run the full Module 2 cleaning pipeline on a single string.

    Order: HTML -> URLs -> emoji -> special symbols -> whitespace.
    Returns "" for None/NaN/empty input rather than raising.
    """
    if text is None:
        return ""
    text = str(text)
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_emoji(text)
    text = remove_special_symbols(text)
    text = normalize_whitespace(text)
    return text
