"""Sitemap-based collector for dailythanthi.com — the real-volume path — Module 1.

`DailyThanthiCollector` only ever sees each section's current front
page (~15-20 articles). This class instead walks the site's XML
sitemap index at `https://www.dailythanthi.com/sitemap.xml` (confirmed
real: a sitemap index pointing at one `sitemap-daily-YYYY-MM-DD.xml`
file per day, each listing several hundred article URLs) — enough
daily volume that ~10-15 days of history clears 25,000+ Tamil article
URLs from this single domain.

Output: datasets/raw/tamil/dailythanthi/dailythanthi_sitemap.jsonl
"""

from __future__ import annotations

from pathlib import Path

from src.data_collection.tamil.sitemap_collector import SitemapNewsCollector


class DailyThanthiSitemapCollector(SitemapNewsCollector):
    source_name = "dailythanthi"
    BASE_URL = "https://www.dailythanthi.com"
    SITEMAP_INDEX_URL = "https://www.dailythanthi.com/sitemap.xml"
    ARTICLE_LINK_PATTERN = r"^/[a-z0-9\-]+/[a-z0-9\-]+/[a-z0-9\-]+/?$"


if __name__ == "__main__":
    collector = DailyThanthiSitemapCollector(
        output_root=Path(__file__).resolve().parents[3] / "datasets" / "raw" / "tamil",
        user_agent="Mozilla/5.0 (compatible; FakeNewsResearchBot/1.0; +mailto:research@example.edu)",
    )
    print(collector.collect(max_daily_sitemaps=3, max_articles=50))
