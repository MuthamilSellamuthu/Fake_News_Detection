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
    SECTIONS = ["news/tamilnadu", "news/india", "news/world"]
    LINK_DEPTH_MIN = 2


if __name__ == "__main__":
    collector = DailyThanthiCollector(output_root=Path(__file__).resolve().parents[4] / "data" / "raw")
    print(collector.collect(max_pages_per_section=1))
