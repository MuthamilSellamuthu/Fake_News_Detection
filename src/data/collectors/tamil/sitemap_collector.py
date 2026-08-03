"""Sitemap-driven collector for mainstream Tamil news portals.

Why this exists: the section-listing scraper (`generic_news_scraper.py`)
only ever sees each section's current front page — confirmed against a
real run where `dailythanthi.com`'s `?page=2` query param returned the
exact same articles as page 1 (Quintype's site doesn't page that way).
That caps yield at roughly (sections x ~15-20 articles), nowhere near
enough for a 25,000-row target.

News CMSs almost always publish a proper XML sitemap for SEO, and it is
the correct way to discover a large, historical set of article URLs
efficiently. `dailythanthi.com/sitemap.xml` was confirmed during
development to be a *sitemap index* pointing at one XML file per day
(`sitemap-daily-YYYY-MM-DD.xml`), each containing hundreds of article
URLs — walking even a few weeks of these comfortably clears 25,000
URLs from this single source.

Cost, to be upfront about it: getting the *title* for each article still
requires one page fetch per article (sitemaps only list URLs, not
titles), so this remains rate-limited to ~1 request per
`rate_limit_seconds` — at the default 1.5s that's ~2,400 articles/hour,
so a 25,000-article run is realistically an overnight job, not a five-
minute one. `max_articles` lets you cap a single run and resumable
dedup (inherited from `BaseCollector`) means repeated runs accumulate
rather than restart.
"""

from __future__ import annotations

import re
from typing import Any

from src.data.collectors.base_collector import BaseCollector
from src.data.collectors.tamil.html_utils import extract_article_body

_LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)
_LASTMOD_RE = re.compile(r"<lastmod>\s*([^<\s]+)\s*</lastmod>", re.IGNORECASE)


class SitemapNewsCollector(BaseCollector):
    """Not meant to be used directly — subclass and set `source_name`,
    `BASE_URL`, `SITEMAP_INDEX_URL`, and `ARTICLE_LINK_PATTERN`.
    """

    source_name = "tamil/_generic_sitemap"
    BASE_URL: str = ""
    SITEMAP_INDEX_URL: str = ""
    # Matches the *path* of a genuine article URL; used to filter out
    # sitemap entries for static pages, tag/topic indexes, etc.
    ARTICLE_LINK_PATTERN: str = r"^/[a-z0-9\-]+/[a-z0-9\-]+/[a-z0-9\-]+/?$"

    def _extract_locs(self, xml_text: str) -> list[tuple[str, str | None]]:
        """Return [(loc, lastmod_or_None), ...] preserving document order."""
        locs = _LOC_RE.findall(xml_text)
        lastmods = _LASTMOD_RE.findall(xml_text)
        # <lastmod> may be absent or not 1:1 with <loc> (e.g. sitemap index
        # entries sometimes omit it) — pad defensively rather than zip-cut.
        if len(lastmods) != len(locs):
            lastmods = [None] * len(locs)
        return list(zip(locs, lastmods))

    def collect(
        self,
        max_daily_sitemaps: int = 30,
        max_articles: int = 2000,
        fetch_full_text: bool = True,
        filename: str | None = None,
        skip_existing: bool = True,
    ) -> dict[str, Any]:
        filename = filename or f"{self.source_name.split('/')[-1]}_sitemap.jsonl"
        existing_urls = self.load_existing_urls(filename) if skip_existing else set()

        self.logger.info("Fetching sitemap index: %s", self.SITEMAP_INDEX_URL)
        index_response = self.fetch(self.SITEMAP_INDEX_URL)
        if index_response is None:
            return {"source": self.source_name, "status": "failed", "reason": "sitemap_index_unreachable"}

        entries = self._extract_locs(index_response.text)
        # A sitemap *index* points at other .xml files; a flat sitemap
        # points directly at articles. Detect which we got.
        sub_sitemaps = [(loc, lastmod) for loc, lastmod in entries if loc.endswith(".xml")]
        direct_urls = [(loc, lastmod) for loc, lastmod in entries if not loc.endswith(".xml")]

        pattern = re.compile(self.ARTICLE_LINK_PATTERN)

        def is_article(url: str) -> bool:
            path = url[len(self.BASE_URL):] if url.startswith(self.BASE_URL) else url
            return bool(pattern.match(path))

        total_records = 0
        articles_processed = 0
        sitemaps_processed = 0

        if sub_sitemaps:
            # Newest-first isn't guaranteed by document order alone, but
            # dailythanthi's index does list daily sitemaps newest-first;
            # we just take the first `max_daily_sitemaps` entries as-is.
            for sitemap_url, _ in sub_sitemaps[:max_daily_sitemaps]:
                if articles_processed >= max_articles:
                    break
                self.logger.info("Fetching daily sitemap %d/%d: %s", sitemaps_processed + 1, max_daily_sitemaps, sitemap_url)
                sub_response = self.fetch(sitemap_url)
                sitemaps_processed += 1
                if sub_response is None:
                    self.logger.warning("Could not fetch %s; skipping.", sitemap_url)
                    continue

                article_urls = [loc for loc, _ in self._extract_locs(sub_response.text) if is_article(loc)]
                new_urls = [u for u in article_urls if u not in existing_urls]
                self.logger.info(
                    "%s: %d article URL(s), %d new.", sitemap_url, len(article_urls), len(new_urls)
                )

                records = []
                for url in new_urls:
                    if articles_processed >= max_articles:
                        break
                    article_response = self.fetch(url)
                    articles_processed += 1
                    if article_response is None:
                        continue
                    title = _extract_title(article_response.text)
                    if not title:
                        continue
                    record: dict[str, Any] = {
                        "source": self.source_name.split("/")[-1],
                        "source_type": "mainstream_news",
                        "language": "ta",
                        "url": url,
                        "title": title,
                        "body_text": extract_article_body(article_response.text) if fetch_full_text else None,
                        "sitemap": sitemap_url,
                    }
                    records.append(record)
                    existing_urls.add(url)

                total_records += self.save_jsonl(records, filename)
        else:
            # Flat sitemap: URLs are articles directly.
            new_urls = [loc for loc, _ in direct_urls if is_article(loc) and loc not in existing_urls]
            for url in new_urls:
                if articles_processed >= max_articles:
                    break
                article_response = self.fetch(url)
                articles_processed += 1
                if article_response is None:
                    continue
                title = _extract_title(article_response.text)
                if not title:
                    continue
                record = {
                    "source": self.source_name.split("/")[-1],
                    "source_type": "mainstream_news",
                    "language": "ta",
                    "url": url,
                    "title": title,
                    "body_text": extract_article_body(article_response.text) if fetch_full_text else None,
                    "sitemap": self.SITEMAP_INDEX_URL,
                }
                total_records += self.save_jsonl([record], filename)
                existing_urls.add(url)

        return {
            "source": self.source_name,
            "status": "ok" if total_records else "no_records",
            "sitemaps_processed": sitemaps_processed,
            "articles_fetched": articles_processed,
            "records_written": total_records,
            "output_file": str(self.output_dir / filename),
        }


_TITLE_TAG_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_H1_TAG_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def _extract_title(html: str) -> str | None:
    """Best-effort <h1> (preferred, usually the actual headline) falling
    back to <title> (often has a site-name suffix like ' | Daily Thanthi').
    """
    match = _H1_TAG_RE.search(html)
    if not match:
        match = _TITLE_TAG_RE.search(html)
    if not match:
        return None
    title = _TAG_STRIP_RE.sub("", match.group(1)).strip()
    return title or None
