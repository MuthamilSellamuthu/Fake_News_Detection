"""Collector for YouTurn (youturn.in) — Module 1.

YouTurn is a Tamil-language fact-checking outlet. It is documented here
as **currently uncollectable** with this project's static-HTTP scraping
approach: the site is a client-side-rendered React single-page app — a
plain HTTP GET returns an empty page shell with no article markup in
the raw HTML, so `requests` + `BeautifulSoup` (which never execute
JavaScript) cannot see any listing content.

Collecting this source for real would require either a JS-capable
fetch (Playwright/Selenium) or a discovered backend API — neither is
implemented in Module 1. This class exists so that gap is explicit and
loud (it raises rather than silently returning an empty result that
could be mistaken for "the site just had nothing new"), matching this
project's standing rule against silently fabricating or hand-waving
missing data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data_collection.base_collector import BaseCollector


class YouturnCollector(BaseCollector):
    source_name = "youturn"

    BASE_URL = "https://youturn.in"
    LISTING_PATH = "/tag/tamil-fact-checker"

    def collect(self, **kwargs: Any) -> dict[str, Any]:
        self.logger.error(
            "YouTurn (youturn.in) is a client-side-rendered React SPA; a static "
            "HTTP GET cannot see article listings. This collector is not "
            "implemented — see this file's module docstring for what would be "
            "needed (Playwright/Selenium or a discovered backend API)."
        )
        return {
            "source": self.source_name,
            "status": "not_implemented",
            "reason": "js_rendered_spa_requires_browser_automation",
        }


if __name__ == "__main__":
    collector = YouturnCollector(
        output_root=Path(__file__).resolve().parents[3] / "datasets" / "raw" / "tamil",
        user_agent="Mozilla/5.0 (compatible; FakeNewsResearchBot/1.0; +mailto:research@example.edu)",
    )
    print(collector.collect())
