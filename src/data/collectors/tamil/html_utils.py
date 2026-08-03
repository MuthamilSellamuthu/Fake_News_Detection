"""Shared HTML-parsing helpers for the Tamil scrapers.

Rather than hard-coding CSS classes (which drift as sites redesign),
these helpers extract article links by *URL pattern* — every article on
these sites has a distinctive, stable-looking slug/id in its href even
when the surrounding markup's class names change. This makes the
scrapers more resilient, at the cost of occasionally picking up a
non-article link that happens to match the pattern (acceptable for a
raw-collection pass; the preprocessing stage can filter further).
"""

from __future__ import annotations

import re
from typing import NamedTuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup

# Matches "18 March 2025 3:15 PM IST", "8 hours ago", "01 Aug 2026", etc.
_DATE_HINT_RE = re.compile(
    r"(\d{1,2}\s+\w+\s+\d{4}[^|\n]*)|(\d+\s+(?:hour|minute|day)s?\s+ago)",
    re.IGNORECASE,
)


class ExtractedLink(NamedTuple):
    url: str
    title: str
    context_text: str  # raw text found near the link (may contain author/date)


def fetch_soup(html: str, parser: str = "lxml") -> BeautifulSoup:
    try:
        return BeautifulSoup(html, parser)
    except Exception:  # lxml not installed / parser issue -> fall back
        return BeautifulSoup(html, "html.parser")


def extract_links_by_pattern(
    html: str,
    base_url: str,
    link_pattern: str,
    min_title_length: int = 8,
) -> list[ExtractedLink]:
    """Find all <a href> matching `link_pattern`, returning de-duplicated
    (url, title, nearby-text) tuples.

    `link_pattern` is matched against the *path* of the href (i.e. after
    stripping the domain), so it works whether the site emits absolute
    or relative links.
    """
    soup = fetch_soup(html)
    pattern = re.compile(link_pattern)

    seen: set[str] = set()
    results: list[ExtractedLink] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        path = href
        # Strip scheme+domain if present, so the pattern only needs to
        # describe the path shape.
        if href.startswith("http"):
            try:
                path = "/" + href.split("/", 3)[3]
            except IndexError:
                path = "/"

        if not pattern.match(path):
            continue

        title = anchor.get_text(strip=True)
        if len(title) < min_title_length:
            # Anchor might just wrap an <img>; look for a heading sibling/parent instead.
            heading = anchor.find_parent(["h1", "h2", "h3", "h4"]) or anchor.find(["h1", "h2", "h3", "h4"])
            if heading:
                title = heading.get_text(strip=True)

        if len(title) < min_title_length:
            continue

        full_url = urljoin(base_url, href)
        if full_url in seen:
            continue
        seen.add(full_url)

        context_text = _nearby_text(anchor)
        results.append(ExtractedLink(url=full_url, title=title, context_text=context_text))

    return results


def _nearby_text(anchor) -> str:
    """Best-effort grab of the text in the container around a link —
    typically where a byline/date sits on these sites.
    """
    container = anchor.find_parent(["article", "div", "li"])
    if container is None:
        return ""
    text = container.get_text(" ", strip=True)
    return text[:400]  # cap length; this is just a hint, not the article body


def extract_date_hint(text: str) -> str | None:
    match = _DATE_HINT_RE.search(text)
    return match.group(0).strip() if match else None


def extract_article_body(html: str, min_paragraph_length: int = 40) -> str:
    """Best-effort main-content extraction for a single article page.

    Strategy: prefer an <article> tag if present, else the <main> tag,
    else the whole document; join <p> tags long enough to plausibly be
    body text (filters out nav/footer one-liners). This is intentionally
    simple — it is a *collection* aid, not a cleaning step.
    """
    soup = fetch_soup(html)
    scope = soup.find("article") or soup.find("main") or soup

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in scope.find_all("p")
        if len(p.get_text(strip=True)) >= min_paragraph_length
    ]
    return "\n".join(paragraphs)
