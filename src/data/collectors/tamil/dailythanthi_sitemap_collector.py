"""Sitemap-based collector for dailythanthi.com — the real-volume path.

`DailyThanthiCollector` (generic_news_scraper.py) only ever sees each
section's current front page (~15-20 articles), because dailythanthi's
`?page=N` param doesn't paginate. This class instead walks the site's
XML sitemap index, confirmed live during development at
`https://www.dailythanthi.com/sitemap.xml`:

  <sitemapindex>
    <sitemap><loc>.../sitemap-daily-2026-08-01.xml</loc></sitemap>
    <sitemap><loc>.../sitemap-daily-2026-07-31.xml</loc></sitemap>
    ...
    <sitemap><loc>.../sitemap-static-pages.xml</loc></sitemap>
  </sitemapindex>

Each `sitemap-daily-YYYY-MM-DD.xml` lists several hundred article URLs
for that day, shaped like:
  https://www.dailythanthi.com/news/tamilnadu/some-article-slug-123456

That's enough daily volume that ~10-15 days of history clears 25,000
Tamil article URLs from this single domain — dwarfing what the section
scraper can reach.

Output: data/raw/tamil/dailythanthi/dailythanthi_sitemap.jsonl
(kept separate from dailythanthi.jsonl so the two collection strategies
never fight over dedup state; Module 2 merges every .jsonl under each
source's raw directory anyway).
"""

from __future__ import annotations

from pathlib import Path

from src.data.collectors.tamil.sitemap_collector import SitemapNewsCollector


class DailyThanthiSitemapCollector(SitemapNewsCollector):
    source_name = "tamil/dailythanthi"
    BASE_URL = "https://www.dailythanthi.com"
    SITEMAP_INDEX_URL = "https://www.dailythanthi.com/sitemap.xml"
    # /<section>/<subsection>/<slug> — matches URLs like
    # /news/tamilnadu/some-slug-123 or /cinema/cinemanews/some-slug-123.
    ARTICLE_LINK_PATTERN = r"^/[a-z0-9\-]+/[a-z0-9\-]+/[a-z0-9\-]+/?$"


if __name__ == "__main__":
    collector = DailyThanthiSitemapCollector(output_root=Path(__file__).resolve().parents[4] / "data" / "raw")
    print(collector.collect(max_daily_sitemaps=3, max_articles=50))
