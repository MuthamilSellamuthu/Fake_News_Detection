"""Collector for the PIB Fact Check Unit (Government of India).

DEVELOPMENT NOTE — read before use:
`factcheck.pib.gov.in` (the dedicated Fact Check domain) is a claim
*submission* portal, not a public article archive — there is no
listing page there to scrape. PIB's actual fact-check verdicts are
distributed primarily via the @PIBFactCheck account on X/Twitter, and
secondarily as press releases on the main pib.gov.in site (an ASP.NET
WebForms application whose full release archive is filtered through
postback forms rather than plain query strings).

This collector takes the pragmatic subset that *is* reachable with
plain GET requests: `pib.gov.in`'s regional-language release listing,
filtered client-side for fact-check-flavoured keywords. This will only
surface a partial subset of actual PIB fact-checks and should be
treated as supplementary, not authoritative coverage. For complete
coverage, supplement with a manual Twitter/X export of @PIBFactCheck.

Output: data/raw/tamil/pib_factcheck/pib_factcheck.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.collectors.base_collector import BaseCollector
from src.data.collectors.tamil.html_utils import extract_article_body, extract_links_by_pattern

# PIB press-release detail pages look like:
#   /PressReleasePage.aspx?PRID=1234567
#   /PressReleseDetail.aspx?PRID=1234567&reg=...&lang=...
_RELEASE_LINK_PATTERN = r"^/(PressReleasePage|PressReleseDetail)\.aspx.*$"

_DEFAULT_KEYWORDS = ("Fact Check", "FactCheck", "PIB Fact Check", "தவறான தகவல்", "வதந்தி")


class PibFactCheckCollector(BaseCollector):
    source_name = "tamil/pib_factcheck"

    BASE_URL = "https://www.pib.gov.in"
    SEARCH_PATH = "/allRel.aspx"

    def collect(
        self,
        lang_param: int = 4,
        keywords: tuple[str, ...] = _DEFAULT_KEYWORDS,
        fetch_full_text: bool = True,
        filename: str = "pib_factcheck.jsonl",
    ) -> dict[str, Any]:
        url = f"{self.BASE_URL}{self.SEARCH_PATH}?lang={lang_param}"
        self.logger.info("Fetching PIB regional release listing: %s", url)

        response = self.fetch(url)
        if response is None:
            return {"source": self.source_name, "status": "failed", "reason": "request_failed"}

        links = extract_links_by_pattern(response.text, self.BASE_URL, _RELEASE_LINK_PATTERN)
        keyword_matches = [
            link for link in links if any(kw.lower() in (link.title + link.context_text).lower() for kw in keywords)
        ]

        if not keyword_matches:
            self.logger.warning(
                "No fact-check-flavoured press releases found via keyword filter on %s. "
                "This is a known coverage gap (see module docstring) — PIB's fact-check "
                "output is mostly distributed via X/Twitter, not this listing. Consider "
                "supplementing with a manual Twitter/X export of @PIBFactCheck.",
                url,
            )
            return {
                "source": self.source_name,
                "status": "no_results",
                "links_seen": len(links),
                "note": "Partial-coverage source; see module docstring.",
            }

        records = []
        for link in keyword_matches:
            record: dict[str, Any] = {
                "source": "pib_factcheck",
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
            "links_seen": len(links),
            "keyword_matches": len(keyword_matches),
            "records_written": written,
            "output_file": str(self.output_dir / filename),
        }


if __name__ == "__main__":
    collector = PibFactCheckCollector(output_root=Path(__file__).resolve().parents[4] / "data" / "raw")
    print(collector.collect())
