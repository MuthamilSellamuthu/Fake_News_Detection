"""Reusable section-listing scraper for mainstream Tamil news portals — Module 1.

Daily Thanthi, Dinamalar, and The Hindu Tamil are collected here as the
"real / legitimate news" side of the dataset — mirroring how the base
paper pairs a fact-checking outlet (majority-fake claims) with
reputable newsrooms (majority-real articles). Site-specific differences
(domain, section paths, link shape) are config (`SECTIONS`, `BASE_URL`
on each subclass); the scraping logic lives once here.

Known limitation, confirmed during development: this strategy only
ever sees each section's *current front page* — dailythanthi.com's
`?page=N` convention does not actually change results (confirmed: page
2 returned identical articles to page 1). For real historical volume
on Daily Thanthi, use `DailyThanthiSitemapCollector` instead. This
class remains useful for a fast, shallow "what's on the front page
right now" pass, and is the only option currently available for
Dinamalar and The Hindu Tamil (neither has a confirmed sitemap format).
"""

from __future__ import annotations

from typing import Any

from src.data_collection.base_collector import BaseCollector
from src.data_collection.tamil.html_utils import extract_article_body, extract_links_by_pattern


class GenericNewsCollector(BaseCollector):
    """Not meant to be used directly with the default `source_name` —
    subclass and override `source_name`, `BASE_URL`, and `SECTIONS`.
    """

    source_name = "_generic_news"
    BASE_URL: str = ""
    SECTIONS: list[str] = []
    LINK_DEPTH_MIN: int = 2

    def _link_pattern_for_section(self, section: str) -> str:
        depth_extra = max(self.LINK_DEPTH_MIN - section.count("/") - 1, 1)
        suffix = r"/[a-z0-9\-]+" * depth_extra
        return rf"^/{section}{suffix}/?$"

    def collect(
        self,
        sections: list[str] | None = None,
        max_pages_per_section: int = 3,
        fetch_full_text: bool = True,
        filename: str | None = None,
        skip_existing: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        sections = sections or self.SECTIONS
        filename = filename or f"{self.source_name}.jsonl"

        total_records = 0
        total_links_seen = 0
        pages_fetched = 0
        existing_urls = self.load_existing_urls(filename) if skip_existing else set()

        for section in sections:
            pattern = self._link_pattern_for_section(section)
            section_url = f"{self.BASE_URL}/{section}"

            for page_num in range(1, max_pages_per_section + 1):
                url = section_url if page_num == 1 else f"{section_url}?page={page_num}"
                self.logger.info("Fetching %s (page %d): %s", section, page_num, url)
                response = self.fetch(url)
                pages_fetched += 1
                if response is None:
                    self.logger.warning("Request failed/blocked for %s; skipping rest of this section.", url)
                    break

                links = extract_links_by_pattern(response.text, self.BASE_URL, pattern)
                if not links:
                    self.logger.info("No article links matched pattern %r on %s.", pattern, url)
                    break

                total_links_seen += len(links)
                new_links = [link for link in links if link.url not in existing_urls]
                if not new_links:
                    self.logger.info("Page %d of %s had no new links (already collected); moving to next section.", page_num, section)
                    break

                records = []
                for link in new_links:
                    record: dict[str, Any] = {
                        "source": self.source_name,
                        "source_type": "mainstream_news",
                        "language": "ta",
                        "section": section,
                        "url": link.url,
                        "title": link.title,
                        "listing_context": link.context_text,
                        "page_number": page_num,
                    }
                    if fetch_full_text:
                        article_response = self.fetch(link.url)
                        record["body_text"] = extract_article_body(article_response.text) if article_response is not None else None
                    records.append(record)
                    existing_urls.add(link.url)

                total_records += self.save_jsonl(records, filename)

        return {
            "source": self.source_name,
            "status": "ok" if total_records else "no_records",
            "sections": sections,
            "pages_fetched": pages_fetched,
            "links_seen": total_links_seen,
            "records_written": total_records,
            "output_file": str(self.output_dir / filename),
        }
