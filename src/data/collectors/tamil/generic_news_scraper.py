"""Reusable scraping engine for mainstream Tamil news portals.

Dina Thanthi, Dinamalar, and The Hindu Tamil are collected here as the
"real / legitimate news" side of the eventual dataset — mirroring how
the base paper (Ni'mah et al., 2025) pairs a fact-checking outlet
(majority-fake claims) with reputable newsrooms (majority-real
articles). Rather than writing three near-identical scrapers, the
site-specific differences (domain, section paths, link shape) are
config, and the scraping logic lives once in `GenericNewsCollector`.

Link extraction: article links on these portals typically look like
`https://<domain>/<section>/<subsection>/<slug>` — i.e. at least
`link_depth_min` path segments deeper than the section root, with no
numeric ID required (unlike BOOM Tamil). We match anything under a
configured section path with enough depth to look like an article
rather than a subsection index.

IMPORTANT: only `dailythanthi` had its live HTML structure confirmed
during development. `dinamalar` and `hindutamil` use best-effort
defaults — check `configs/sources.yaml` (`verified: false`) and inspect
the live site before relying on a full run from those two.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.collectors.base_collector import BaseCollector
from src.data.collectors.tamil.html_utils import extract_article_body, extract_links_by_pattern


class GenericNewsCollector(BaseCollector):
    """Not meant to be used directly with the default `source_name` —
    subclass and override `source_name`, `BASE_URL`, and `SECTIONS`.
    """

    source_name = "tamil/_generic_news"
    BASE_URL: str = ""
    SECTIONS: list[str] = []
    LINK_DEPTH_MIN: int = 2

    def _link_pattern_for_section(self, section: str) -> str:
        # e.g. section "news/tamilnadu" -> path must be
        # "/news/tamilnadu/<slug>" (one extra segment, no trailing slash).
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
    ) -> dict[str, Any]:
        sections = sections or self.SECTIONS
        filename = filename or f"{self.source_name.split('/')[-1]}.jsonl"

        total_records = 0
        total_links_seen = 0
        pages_fetched = 0
        existing_urls = self.load_existing_urls(filename) if skip_existing else set()

        for section in sections:
            pattern = self._link_pattern_for_section(section)
            section_url = f"{self.BASE_URL}/{section}"

            for page_num in range(1, max_pages_per_section + 1):
                # Common pagination convention on these CMSs; harmless if
                # the param is ignored (we detect "no new links" and stop).
                url = section_url if page_num == 1 else f"{section_url}?page={page_num}"
                self.logger.info("Fetching %s (page %d): %s", section, page_num, url)
                response = self.fetch(url)
                pages_fetched += 1
                if response is None:
                    self.logger.warning("Request failed for %s; skipping rest of this section.", url)
                    break

                links = extract_links_by_pattern(response.text, self.BASE_URL, pattern)
                if not links:
                    self.logger.info(
                        "No article links matched pattern %r on %s. If this section is "
                        "known-good, verify the link pattern in configs/sources.yaml.",
                        pattern,
                        url,
                    )
                    break

                total_links_seen += len(links)
                new_links = [link for link in links if link.url not in existing_urls]
                if not new_links:
                    self.logger.info(
                        "Page %d of %s had no new links (all already collected); moving to next section.",
                        page_num,
                        section,
                    )
                    break

                records = []
                for link in new_links:
                    record: dict[str, Any] = {
                        "source": self.source_name.split("/")[-1],
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
                        record["body_text"] = (
                            extract_article_body(article_response.text) if article_response is not None else None
                        )
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


if __name__ == "__main__":
    # Smoke-test with Dina Thanthi (the one verified source) when run directly.
    from src.data.collectors.tamil.dailythanthi_collector import DailyThanthiCollector

    collector = DailyThanthiCollector(output_root=Path(__file__).resolve().parents[4] / "data" / "raw")
    print(collector.collect(max_pages_per_section=1))
