"""Collector for The Hindu Tamil (hindutamil.in).

Mainstream Tamil newspaper — used as a "real/legitimate news" source.

NOT VERIFIED: this domain could not be reached from the development
sandbox (blocked by the environment's fetch policy), so `SECTIONS`
below is an unverified placeholder. Confirm actual section slugs and
link shape by inspecting hindutamil.in directly before relying on this
collector for a full run.

Output: data/raw/tamil/hindutamil/hindutamil.jsonl
"""

from __future__ import annotations

from pathlib import Path

from src.data.collectors.tamil.generic_news_scraper import GenericNewsCollector


class HinduTamilCollector(GenericNewsCollector):
    source_name = "tamil/hindutamil"
    BASE_URL = "https://www.hindutamil.in"
    # TODO: verify real section slugs on hindutamil.in before relying on this list.
    SECTIONS = ["news"]
    LINK_DEPTH_MIN = 2


if __name__ == "__main__":
    collector = HinduTamilCollector(output_root=Path(__file__).resolve().parents[4] / "data" / "raw")
    print(collector.collect(max_pages_per_section=1))
