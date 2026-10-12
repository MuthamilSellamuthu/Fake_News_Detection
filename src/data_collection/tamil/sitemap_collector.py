"""Generic sitemap-driven collector — Module 1.

The section-listing scraper (`generic_news_scraper.py`) only ever sees
each section's current front page (confirmed: dailythanthi.com's
`?page=N` doesn't paginate), capping yield at a few hundred articles.
News CMSs almost always publish a proper XML sitemap for SEO purposes;
walking it is the correct way to reach real historical volume — a
sitemap *index* pointing at one file per day, each listing hundreds of
article URLs, comfortably clears tens of thousands of URLs from a
single domain over a few weeks of history.

Cost, stated upfront: sitemaps list URLs only, not titles/body text, so
one page fetch per article is still required — this remains rate-
limited to about one request per `rate_limit_seconds`, making a
large-scale run realistically an overnight job. `max_articles` caps a
single run; resumable dedup (`BaseCollector.load_existing_urls`) means
repeated runs accumulate rather than restart.
"""

from __future__ import annotations

import re
from typing import Any

from src.data_collection.base_collector import BaseCollector
from src.data_collection.tamil.html_utils import extract_article_body

_LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)
_H1_TAG_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_TITLE_TAG_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def _extract_title(html: str) -> str | None:
    match = _H1_TAG_RE.search(html) or _TITLE_TAG_RE.search(html)
    if not match:
        return None
    title = _TAG_STRIP_RE.sub("", match.group(1)).strip()
    return title or None


class SitemapNewsCollector(BaseCollector):
    """Not meant to be used directly — subclass and set `source_name`,
    `BASE_URL`, `SITEMAP_INDEX_URL`, and `ARTICLE_LINK_PATTERN`.
    """

    source_name = "_generic_sitemap"
    BASE_URL: str = ""
    SITEMAP_INDEX_URL: str = ""
    ARTICLE_LINK_PATTERN: str = r"^/[a-z0-9\-]+/[a-z0-9\-]+/[a-z0-9\-]+/?$"

    def _extract_locs(self, xml_text: str) -> list[str]:
        return _LOC_RE.findall(xml_text)

    def collect(
        self,
        max_daily_sitemaps: int = 30,
        max_articles: int = 2000,
        fetch_full_text: bool = True,
        filename: str | None = None,
        skip_existing: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        filename = filename or f"{self.source_name}_sitemap.jsonl"
        existing_urls = self.load_existing_urls(filename) if skip_existing else set()

        self.logger.info("Fetching sitemap index: %s", self.SITEMAP_INDEX_URL)
        index_response = self.fetch(self.SITEMAP_INDEX_URL)
        if index_response is None:
            return {"source": self.source_name, "status": "failed", "reason": "sitemap_index_unreachable_or_disallowed"}

        entries = self._extract_locs(index_response.text)
        sub_sitemaps = [loc for loc in entries if loc.endswith(".xml")]
        direct_urls = [loc for loc in entries if not loc.endswith(".xml")]

        pattern = re.compile(self.ARTICLE_LINK_PATTERN)

        def is_article(url: str) -> bool:
            path = url[len(self.BASE_URL):] if url.startswith(self.BASE_URL) else url
            return bool(pattern.match(path))

        total_records = 0
        articles_processed = 0
        sitemaps_processed = 0

        def process_article_urls(urls: list[str], sitemap_source: str) -> None:
            nonlocal total_records, articles_processed
            new_urls = [u for u in urls if is_article(u) and u not in existing_urls]
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
                records.append({
                    "source": self.source_name,
                    "source_type": "mainstream_news",
                    "language": "ta",
                    "url": url,
                    "title": title,
                    "body_text": extract_article_body(article_response.text) if fetch_full_text else None,
                    "sitemap": sitemap_source,
                })
                existing_urls.add(url)
            total_records += self.save_jsonl(records, filename)

        if sub_sitemaps:
            for sitemap_url in sub_sitemaps[:max_daily_sitemaps]:
                if articles_processed >= max_articles:
                    break
                self.logger.info("Fetching daily sitemap %d/%d: %s", sitemaps_processed + 1, max_daily_sitemaps, sitemap_url)
                sub_response = self.fetch(sitemap_url)
                sitemaps_processed += 1
                if sub_response is None:
                    self.logger.warning("Could not fetch %s; skipping.", sitemap_url)
                    continue
                article_urls = self._extract_locs(sub_response.text)
                process_article_urls(article_urls, sitemap_url)
        else:
            process_article_urls(direct_urls, self.SITEMAP_INDEX_URL)

        return {
            "source": self.source_name,
            "status": "ok" if total_records else "no_records",
            "sitemaps_processed": sitemaps_processed,
            "articles_fetched": articles_processed,
            "records_written": total_records,
            "output_file": str(self.output_dir / filename),
        }
