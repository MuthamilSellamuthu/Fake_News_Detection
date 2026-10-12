#!/usr/bin/env python
"""Module 1 — Tamil dataset collection CLI.

Runs one, several, or all Tamil collectors. Each fails independently —
one source erroring out (or being blocked, or unimplemented) does not
stop the others. A summary table is printed at the end.

Usage
-----
  python scripts/collect_tamil.py --all
  python scripts/collect_tamil.py --sources dailythanthi_sitemap boom_tamil
  python scripts/collect_tamil.py --sources dailythanthi_sitemap --max-daily-sitemaps 15 --max-articles 5000
  python scripts/collect_tamil.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.tamil.aanirudh_headlines_collector import AanirudhHeadlinesCollector  # noqa: E402
from src.data_collection.tamil.boom_tamil_collector import BoomTamilCollector  # noqa: E402
from src.data_collection.tamil.dailythanthi_collector import DailyThanthiCollector  # noqa: E402
from src.data_collection.tamil.dailythanthi_sitemap_collector import DailyThanthiSitemapCollector  # noqa: E402
from src.data_collection.tamil.dinamalar_collector import DinamalarCollector  # noqa: E402
from src.data_collection.tamil.hindutamil_collector import HinduTamilCollector  # noqa: E402
from src.data_collection.tamil.pib_factcheck_collector import PibFactCheckCollector  # noqa: E402
from src.data_collection.tamil.youturn_collector import YouturnCollector  # noqa: E402
from src.utils.config_loader import load_yaml, project_root  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.collect_tamil")

_REGISTRY: dict[str, type] = {
    "aanirudh_headlines": AanirudhHeadlinesCollector,
    "boom_tamil": BoomTamilCollector,
    "dailythanthi": DailyThanthiCollector,
    "dailythanthi_sitemap": DailyThanthiSitemapCollector,
    "dinamalar": DinamalarCollector,
    "hindutamil": HinduTamilCollector,
    "youturn": YouturnCollector,
    "pib_factcheck": PibFactCheckCollector,
}


def _build_kwargs(source: str, args: argparse.Namespace) -> dict[str, Any]:
    if source == "boom_tamil":
        return {"max_pages": args.max_pages, "fetch_full_text": not args.no_full_text}
    if source in ("dailythanthi", "dinamalar", "hindutamil"):
        return {"max_pages_per_section": args.max_pages, "fetch_full_text": not args.no_full_text}
    if source == "dailythanthi_sitemap":
        return {
            "max_daily_sitemaps": args.max_daily_sitemaps,
            "max_articles": args.max_articles,
            "fetch_full_text": not args.no_full_text,
        }
    if source == "pib_factcheck":
        return {"fetch_full_text": not args.no_full_text}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 1: collect Tamil datasets.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Run every registered Tamil source.")
    group.add_argument("--sources", nargs="+", choices=sorted(_REGISTRY.keys()))
    parser.add_argument("--list", action="store_true", help="List available source names and exit.")
    parser.add_argument("--max-pages", type=int, default=50, help="Listing pages (boom_tamil) or pages-per-section (dailythanthi/dinamalar/hindutamil). Default: 50.")
    parser.add_argument("--max-daily-sitemaps", type=int, default=30, help="dailythanthi_sitemap only. Default: 30.")
    parser.add_argument("--max-articles", type=int, default=2000, help="dailythanthi_sitemap only. Default: 2000.")
    parser.add_argument("--no-full-text", action="store_true", help="Collect listing metadata only; skip full article body text.")
    parser.add_argument("--rate-limit", type=float, default=None, help="Override configs/module1_sources.yaml http.rate_limit_seconds.")
    args = parser.parse_args()

    if args.list:
        print("Tamil sources:", ", ".join(sorted(_REGISTRY.keys())))
        return

    if args.all:
        sources = list(_REGISTRY.keys())
    elif args.sources:
        sources = args.sources
    else:
        parser.error("Specify --all, --sources <name...>, or --list.")
        return

    config = load_yaml(project_root() / "configs" / "module1_sources.yaml")
    http_cfg = config["http"]
    output_root = project_root() / "datasets" / "raw" / "tamil"

    results: list[dict[str, Any]] = []
    for source in sources:
        logger.info("=" * 60)
        logger.info("Starting collector: %s", source)
        collector = _REGISTRY[source](
            output_root=output_root,
            user_agent=http_cfg["user_agent"],
            rate_limit_seconds=args.rate_limit or http_cfg["rate_limit_seconds"],
            timeout_seconds=http_cfg["timeout_seconds"],
            max_retries=http_cfg["max_retries"],
            backoff_factor=http_cfg["backoff_factor"],
            respect_robots_txt=http_cfg["respect_robots_txt"],
        )
        try:
            result = collector.collect(**_build_kwargs(source, args))
        except Exception as exc:  # noqa: BLE001 - keep going, report the failure
            logger.exception("Collector %r raised an unhandled exception", source)
            result = {"source": source, "status": "error", "error": str(exc)}
        results.append(result)

    print("\n" + "=" * 72)
    print("MODULE 1 — TAMIL COLLECTION SUMMARY")
    print("=" * 72)
    for result in results:
        print(f"  [{result.get('status', 'unknown').upper():^18}] {result.get('source', '?')}")
        for key, value in result.items():
            if key not in ("source", "status"):
                print(f"      {key}: {value}")
    print("=" * 72)


if __name__ == "__main__":
    main()
