"""HTML parsing helpers shared by every Tamil collector — Module 1.

Kept separate from any single collector because every Tamil source
needs the same three operations: pull article links matching a URL
shape out of a listing page, guess a publish-date hint from nearby
text, and extract an article's main body text. Centralizing these
means a parsing-quality fix benefits every collector at once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup


@dataclass
class ExtractedLink:
    url: str
    title: str | None
    context_text: str | None


def extract_links_by_pattern(html: str, base_url: str, link_pattern: str) -> list[ExtractedLink]:
    """Find every `<a href>` on the page whose *path* (not full URL)
    matches `link_pattern`, and return each as an `ExtractedLink` with
    its visible text as a title guess and the surrounding text as
    context (useful for date-hint extraction).

    Matching on path shape rather than a CSS class is deliberate: CSS
    classes/IDs change on every redesign, but a news site's URL
    structure (e.g. `/news/tamilnadu/<slug>`) tends to stay stable far
    longer, since changing it would break every existing inbound link
    and search-engine index entry.
    """
    soup = BeautifulSoup(html, "lxml")
    pattern = re.compile(link_pattern)
    seen: set[str] = set()
    results: list[ExtractedLink] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        # Normalize to an absolute-path comparison; treat both relative
        # (`/news/...`) and absolute (`https://site.com/news/...`) hrefs.
        path = href
        if href.startswith(base_url):
            path = href[len(base_url):]
        elif href.startswith("http"):
            continue  # absolute link to a different host — not this site's article

        if not pattern.match(path):
            continue

        full_url = href if href.startswith("http") else f"{base_url}{path}"
        if full_url in seen:
            continue
        seen.add(full_url)

        title = anchor.get_text(strip=True) or None
        parent_text = anchor.parent.get_text(" ", strip=True) if anchor.parent else None
        results.append(ExtractedLink(url=full_url, title=title, context_text=parent_text))

    return results


_DATE_HINT_RE = re.compile(r"(\d{1,2}[\s./-]\w+[\s./-]\d{2,4})|(\d{4}-\d{2}-\d{2})")


def extract_date_hint(context_text: str | None) -> str | None:
    """Best-effort date string pulled from a listing page's surrounding
    text (article pages usually carry a proper structured date, but
    listing pages often only show it as plain text near the link).
    """
    if not context_text:
        return None
    match = _DATE_HINT_RE.search(context_text)
    return match.group(0) if match else None


def extract_article_body(html: str) -> str | None:
    """Best-effort main-content extraction: prefer an `<article>` tag,
    fall back to the largest `<div>`/`<section>` by visible text length
    (a simple, dependency-free heuristic — main content is usually the
    single largest contiguous text block on a news article page, ads
    and navigation are smaller, and this fails safely to `None` rather
    than to a wall of the wrong content when the guess is off).
    """
    soup = BeautifulSoup(html, "lxml")

    article_tag = soup.find("article")
    if article_tag is not None:
        text = article_tag.get_text(" ", strip=True)
        if text:
            return text

    candidates = soup.find_all(["div", "section"])
    if not candidates:
        return None
    best = max(candidates, key=lambda tag: len(tag.get_text(strip=True)))
    text = best.get_text(" ", strip=True)
    return text or None
