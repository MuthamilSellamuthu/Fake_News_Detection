"""Collector for PIB Fact Check (Government of India) — Module 1.

The Press Information Bureau's dedicated fact-check portal
(factcheck.pib.gov.in) is a claim-*submission* form, not a public
article archive — there is nothing there to list or scrape. Published
verdicts are instead distributed primarily via the @PIBFactCheck
account on X/Twitter and, secondarily, as press releases on the main
pib.gov.in site.

This collector queries PIB's press-release search for the Tamil
regional-language feed, filtered by a Tamil "false information"
keyword. Coverage will necessarily be partial — most PIB fact-check
activity happens on a platform (X/Twitter) this collector does not
touch — and that limitation is reported in the result, not hidden.

Output: datasets/raw/tamil/pib_factcheck/pib_factcheck.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from src.data_collection.base_collector import BaseCollector
from src.data_collection.tamil.html_utils import extract_article_body, extract_links_by_pattern


class PibFactCheckCollector(BaseCollector):
    source_name = "pib_factcheck"

    BASE_URL = "https://www.pib.gov.in"
    SEARCH_PATH = "/allRel.aspx"
    LANG_PARAM = 4
    KEYWORD = "தவறான தகவல்"
    LINK_PATTERN = r"^/PressRelease(Iframe)?Page\.aspx\?PRID=\d+$"

    def collect(
        self,
        fetch_full_text: bool = True,
        filename: str = "pib_factcheck.jsonl",
        skip_existing: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        existing_urls = self.load_existing_urls(filename) if skip_existing else set()

        query = urlencode({"lang": self.LANG_PARAM, "keyword": self.KEYWORD})
        url = f"{self.BASE_URL}{self.SEARCH_PATH}?{query}"
        self.logger.info("Fetching PIB press-release search: %s", url)

        response = self.fetch(url)
        if response is None:
            return {
                "source": self.source_name,
                "status": "failed",
                "reason": "request_failed_or_disallowed",
                "note": "PIB's fact-check activity is mostly on X/Twitter, outside this collector's scope regardless.",
            }

        links = extract_links_by_pattern(response.text, self.BASE_URL, self.LINK_PATTERN)
        new_links = [link for link in links if link.url not in existing_urls]

        records = []
        for link in new_links:
            record: dict[str, Any] = {
                "source": self.source_name,
                "source_type": "fact_check",
                "language": "ta",
                "url": link.url,
                "title": link.title,
                "listing_context": link.context_text,
            }
            if fetch_full_text:
                article_response = self.fetch(link.url)
                record["body_text"] = extract_article_body(article_response.text) if article_response is not None else None
            records.append(record)
            existing_urls.add(link.url)

        total_records = self.save_jsonl(records, filename)

        return {
            "source": self.source_name,
            "status": "ok" if total_records else "no_records",
            "links_seen": len(links),
            "records_written": total_records,
            "output_file": str(self.output_dir / filename),
            "note": "Partial coverage by design — most PIB fact-check verdicts are distributed via X/Twitter, not this press-release search.",
        }


if __name__ == "__main__":
    collector = PibFactCheckCollector(
        output_root=Path(__file__).resolve().parents[3] / "datasets" / "raw" / "tamil",
        user_agent="Mozilla/5.0 (compatible; FakeNewsResearchBot/1.0; +mailto:research@example.edu)",
    )
    print(collector.collect())
