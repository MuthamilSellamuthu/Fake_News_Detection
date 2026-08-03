"""Collector for YouTurn (youturn.in) — Tamil fact-checking outlet.

DEVELOPMENT NOTE — read before use:
youturn.in is a client-side rendered React single-page application. A
plain HTTP GET (what `requests` does) returns an empty HTML shell with
no article markup — confirmed during development by fetching both the
homepage and a tag-listing page. A static requests+BeautifulSoup
scraper genuinely cannot see YouTurn's article listing.

Two ways to make this collector work for real:
  1. Render with a headless browser (Selenium/Playwright) and scrape the
     rendered DOM. Not implemented here to avoid adding a heavy browser
     dependency to Module 1 by default; see `render_fn` below.
  2. Discover YouTurn's backend JSON API (most React SPAs fetch data
     from a REST/GraphQL endpoint under the hood) via browser devtools,
     then hit that endpoint directly with `requests` — far cheaper than
     driving a browser. Not confirmed during development.

This collector is intentionally honest about the gap: `collect()` tries
the plain HTTP path, detects that no article links were found, and
returns a clear "unsupported_js_rendered" status rather than silently
writing zero records and looking like it worked. If you provide a
`render_fn(url) -> html` callable (e.g. wrapping Selenium), it will be
used instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.data.collectors.base_collector import BaseCollector
from src.data.collectors.tamil.html_utils import extract_article_body, extract_links_by_pattern


class YouturnCollector(BaseCollector):
    source_name = "tamil/youturn"

    BASE_URL = "https://youturn.in"
    LISTING_PATH = "/tag/tamil-fact-checker"
    # Best-effort guess at article URL shape; unconfirmed (see module docstring).
    LINK_PATTERN = r"^/[a-z0-9\-]+/?$"

    def collect(
        self,
        max_pages: int = 5,
        fetch_full_text: bool = True,
        filename: str = "youturn.jsonl",
        render_fn: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        listing_url = f"{self.BASE_URL}{self.LISTING_PATH}"

        if render_fn is not None:
            self.logger.info("Using provided render_fn to fetch JS-rendered HTML.")
            html = render_fn(listing_url)
        else:
            response = self.fetch(listing_url)
            if response is None:
                return {"source": self.source_name, "status": "failed", "reason": "request_failed"}
            html = response.text

        links = extract_links_by_pattern(html, self.BASE_URL, self.LINK_PATTERN)

        if not links:
            self.logger.error(
                "No article links found for YouTurn. This site is a client-side "
                "rendered React app — a plain HTTP fetch cannot see its content. "
                "Pass a `render_fn` backed by Selenium/Playwright, or discover and "
                "call YouTurn's backend API directly. See module docstring."
            )
            return {
                "source": self.source_name,
                "status": "unsupported_js_rendered",
                "records_written": 0,
                "note": "Static scraping unsupported for this site; see docstring for options.",
            }

        records = []
        for link in links:
            record: dict[str, Any] = {
                "source": "youturn",
                "source_type": "fact_check",
                "language": "ta",
                "url": link.url,
                "title": link.title,
                "listing_context": link.context_text,
            }
            if fetch_full_text:
                article_response = self.fetch(link.url)
                record["body_text"] = extract_article_body(article_response.text) if article_response else None
            records.append(record)

        written = self.save_jsonl(records, filename)
        return {
            "source": self.source_name,
            "status": "ok",
            "records_written": written,
            "output_file": str(self.output_dir / filename),
        }


if __name__ == "__main__":
    collector = YouturnCollector(output_root=Path(__file__).resolve().parents[4] / "data" / "raw")
    print(collector.collect())
