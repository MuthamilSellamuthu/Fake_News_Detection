"""Collector for Dinamalar (dinamalar.com).

Mainstream Tamil newspaper — used as a "real/legitimate news" source.

NOT VERIFIED: the homepage was reachable during development but its
section structure was not fully inspected (response exceeded the
inspection size limit). `SECTIONS` below is a best-effort placeholder —
confirm actual section slugs (e.g. by viewing dinamalar.com's nav bar)
and update `configs/sources.yaml` / this file before a full run.

Output: data/raw/tamil/dinamalar/dinamalar.jsonl
"""

from __future__ import annotations

from pathlib import Path

from src.data.collectors.tamil.generic_news_scraper import GenericNewsCollector


class DinamalarCollector(GenericNewsCollector):
    source_name = "tamil/dinamalar"
    BASE_URL = "https://www.dinamalar.com"
    # TODO: verify real section slugs on dinamalar.com before relying on this list.
    SECTIONS = ["news"]
    LINK_DEPTH_MIN = 2


if __name__ == "__main__":
    collector = DinamalarCollector(output_root=Path(__file__).resolve().parents[4] / "data" / "raw")
    print(collector.collect(max_pages_per_section=1))
