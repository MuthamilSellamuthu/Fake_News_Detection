"""Collector for The Hindu Tamil (hindutamil.in) — Module 1.

Mainstream Tamil newspaper. This domain was unreachable from the
development sandbox, so `SECTIONS` below is an unverified placeholder
— confirm the real section slug(s) from an environment with real
network access before relying on this for a full run (see
configs/module1_sources.yaml, verified: false).

Output: datasets/raw/tamil/hindutamil/hindutamil.jsonl
"""

from __future__ import annotations

from pathlib import Path

from src.data_collection.tamil.generic_news_scraper import GenericNewsCollector


class HinduTamilCollector(GenericNewsCollector):
    source_name = "hindutamil"
    BASE_URL = "https://www.hindutamil.in"
    SECTIONS = ["news"]  # unverified — confirm against the live site
    LINK_DEPTH_MIN = 2


if __name__ == "__main__":
    collector = HinduTamilCollector(
        output_root=Path(__file__).resolve().parents[3] / "datasets" / "raw" / "tamil",
        user_agent="Mozilla/5.0 (compatible; FakeNewsResearchBot/1.0; +mailto:research@example.edu)",
    )
    print(collector.collect(max_pages_per_section=1))
