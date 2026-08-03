"""Collector for BOOM Tamil (tamil.boomlive.in) — Tamil fact-check desk.

BOOM Tamil investigates and debunks/verifies viral Tamil-language claims.
Listing pages are server-rendered (confirmed during development) and
paginate at `/fact-check/page/{n}`.

This collects article *metadata* (title, url, byline/date hint) from the
listing pages, and optionally the full article body text. It does NOT
infer a true/false label — BOOM's verdict (false/misleading/true/etc.)
is stated in the article prose, not in a structured field visible on
the listing, so label extraction belongs in the Module 2 preprocessing
stage where the raw text is actually analyzed.

Output: data/raw/tamil/boom_tamil/boom_tamil.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.collectors.base_collector import BaseCollector
from src.data.collectors.tamil.html_utils import (
    extract_article_body,
    extract_date_hint,
    extract_links_by_pattern,
)


class BoomTamilCollector(BaseCollector):
    source_name = "tamil/boom_tamil"

    BASE_URL = "https://tamil.boomlive.in"
    LISTING_PATH = "/fact-check"
    LINK_PATTERN = r"^/fact-check/[a-z0-9\-]+-\d+$"

    def collect(
        self,
        max_pages: int = 5,
        fetch_full_text: bool = True,
        filename: str = "boom_tamil.jsonl",
        skip_existing: bool = True,
    ) -> dict[str, Any]:
        total_records = 0
        total_links_seen = 0
        existing_urls = self.load_existing_urls(filename) if skip_existing else set()

        for page_num in range(1, max_pages + 1):
            url = (
                f"{self.BASE_URL}{self.LISTING_PATH}"
                if page_num == 1
                else f"{self.BASE_URL}{self.LISTING_PATH}/page/{page_num}"
            )
            self.logger.info("Fetching listing page %d: %s", page_num, url)
            response = self.fetch(url)
            if response is None:
                self.logger.warning("Stopping pagination at page %d (request failed).", page_num)
                break

            links = extract_links_by_pattern(response.text, self.BASE_URL, self.LINK_PATTERN)
            if not links:
                self.logger.info("No article links found on page %d; assuming end of listing.", page_num)
                break

            total_links_seen += len(links)
            new_links = [link for link in links if link.url not in existing_urls]
            if not new_links:
                self.logger.info(
                    "Page %d had no new links (all %d already collected in a previous run); stopping — "
                    "older pages will also be entirely already-seen since listing is newest-first.",
                    page_num,
                    len(links),
                )
                break

            records = []
            for link in new_links:
                record: dict[str, Any] = {
                    "source": "boom_tamil",
                    "source_type": "fact_check",
                    "language": "ta",
                    "url": link.url,
                    "title": link.title,
                    "date_hint": extract_date_hint(link.context_text),
                    "listing_context": link.context_text,
                    "page_number": page_num,
                }

                if fetch_full_text:
                    article_response = self.fetch(link.url)
                    if article_response is not None:
                        record["body_text"] = extract_article_body(article_response.text)
                    else:
                        record["body_text"] = None

                records.append(record)
                existing_urls.add(link.url)

            total_records += self.save_jsonl(records, filename)

        return {
            "source": self.source_name,
            "status": "ok" if total_records else "no_records",
            "pages_fetched": page_num,
            "links_seen": total_links_seen,
            "records_written": total_records,
            "output_file": str(self.output_dir / filename),
        }


if __name__ == "__main__":
    collector = BoomTamilCollector(output_root=Path(__file__).resolve().parents[4] / "data" / "raw")
    print(collector.collect(max_pages=2))
