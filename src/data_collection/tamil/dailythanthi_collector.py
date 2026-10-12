"""Collector for Dina Thanthi / Daily Thanthi (dailythanthi.com) — Module 1.

Mainstream Tamil newspaper — the "real/legitimate news" source. See
`generic_news_scraper.py` for the shared scraping mechanics and its
documented pagination limitation (front page only, per section).

Output: datasets/raw/tamil/dailythanthi/dailythanthi.jsonl
"""

from __future__ import annotations

from pathlib import Path

from src.data_collection.tamil.generic_news_scraper import GenericNewsCollector


class DailyThanthiCollector(GenericNewsCollector):
    source_name = "dailythanthi"
    BASE_URL = "https://www.dailythanthi.com"
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
    collector = DailyThanthiCollector(
        output_root=Path(__file__).resolve().parents[3] / "datasets" / "raw" / "tamil",
        user_agent="Mozilla/5.0 (compatible; FakeNewsResearchBot/1.0; +mailto:research@example.edu)",
    )
    print(collector.collect(max_pages_per_section=1))
