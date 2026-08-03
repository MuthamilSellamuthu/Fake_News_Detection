"""Collector for Dina Thanthi / Daily Thanthi (dailythanthi.com).

Mainstream Tamil newspaper — used as a "real/legitimate news" source.
Built on the Quintype CMS; section listing pages were confirmed
reachable and server-rendered during development (see
configs/sources.yaml: dailythanthi -> verified: true).

Output: data/raw/tamil/dailythanthi/dailythanthi.jsonl
"""

from __future__ import annotations

from pathlib import Path

from src.data.collectors.tamil.generic_news_scraper import GenericNewsCollector


class DailyThanthiCollector(GenericNewsCollector):
    source_name = "tamil/dailythanthi"
    BASE_URL = "https://www.dailythanthi.com"
    # Kept in sync with configs/sources.yaml -> tamil.dailythanthi.sections.
    # NOTE: this only surfaces each section's current front page (the site's
    # `?page=N` query param does not appear to change results — confirmed by
    # a real run where page 2 returned the same articles as page 1). For
    # real volume, use SitemapDailyThanthiCollector instead, which walks the
    # site's daily XML sitemaps and reaches thousands of historical articles.
    SECTIONS = [
        "news/tamilnadu",
        "news/india",
        "news/world",
        "news/politics",
        "news/sirappukatturaigal",
        "cinema/cinemanews",
        "cinema/ott",
        "cinema/movie-review",
        "sports/cricket",
        "sports/football",
        "sports/tennis",
        "sports/hockey",
        "sports/othersports",
        "business",
        "devotional",
    ]
    LINK_DEPTH_MIN = 2


if __name__ == "__main__":
    collector = DailyThanthiCollector(output_root=Path(__file__).resolve().parents[4] / "data" / "raw")
    print(collector.collect(max_pages_per_section=1))
