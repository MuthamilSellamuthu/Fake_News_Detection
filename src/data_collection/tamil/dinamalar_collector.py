"""Collector for Dinamalar (dinamalar.com) — Module 1.

Mainstream Tamil newspaper. Homepage was reachable but not fully
inspected during development — `SECTIONS` below is a best-effort
placeholder; confirm the real section slug(s) against the live site
before relying on this for a full run (see configs/module1_sources.yaml,
verified: false).

Output: datasets/raw/tamil/dinamalar/dinamalar.jsonl
"""

from __future__ import annotations

from pathlib import Path

from src.data_collection.tamil.generic_news_scraper import GenericNewsCollector


class DinamalarCollector(GenericNewsCollector):
    source_name = "dinamalar"
    BASE_URL = "https://www.dinamalar.com"
    SECTIONS = ["news"]  # unverified — confirm against the live site
    LINK_DEPTH_MIN = 2


if __name__ == "__main__":
    collector = DinamalarCollector(
        output_root=Path(__file__).resolve().parents[3] / "datasets" / "raw" / "tamil",
        user_agent="Mozilla/5.0 (compatible; FakeNewsResearchBot/1.0; +mailto:research@example.edu)",
    )
    print(collector.collect(max_pages_per_section=1))
