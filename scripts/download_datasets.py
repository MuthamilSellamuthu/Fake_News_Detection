#!/usr/bin/env python
"""Module 1 orchestrator: run one, several, or all dataset collectors.

This script only *collects* data (downloads / scrapes) and writes it,
unmodified, under `data/raw/`. It does not clean, label, deduplicate,
tokenize, or otherwise preprocess anything — that is Module 2.

Usage
-----
  # Everything (English direct-downloads + all Tamil scrapers):
  python scripts/download_datasets.py --all

  # Just the English datasets:
  python scripts/download_datasets.py --sources welfake isot

  # Just BOOM Tamil, 3 pages, metadata only (skip full article text):
  python scripts/download_datasets.py --sources boom_tamil --max-pages 3 --no-full-text

  # List available source names:
  python scripts/download_datasets.py --list

Each collector fails independently — one source erroring out does not
stop the others. A summary table is printed at the end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Make `src` importable when run as `python scripts/download_datasets.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collectors.english.isot_collector import IsotCollector  # noqa: E402
from src.data.collectors.english.welfake_collector import WelfakeCollector  # noqa: E402
from src.data.collectors.tamil.boom_tamil_collector import BoomTamilCollector  # noqa: E402
from src.data.collectors.tamil.dailythanthi_collector import DailyThanthiCollector  # noqa: E402
from src.data.collectors.tamil.dinamalar_collector import DinamalarCollector  # noqa: E402
from src.data.collectors.tamil.hindutamil_collector import HinduTamilCollector  # noqa: E402
from src.data.collectors.tamil.pib_factcheck_collector import PibFactCheckCollector  # noqa: E402
from src.data.collectors.tamil.youturn_collector import YouturnCollector  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.download_datasets")

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Registry: source name -> (collector class, kwargs builder)
# Every collector class is a BaseCollector subclass whose .collect(**kwargs)
# writes into data/raw/<source_name>/ and returns a summary dict.
_REGISTRY: dict[str, type] = {
    "welfake": WelfakeCollector,
    "isot": IsotCollector,
    "boom_tamil": BoomTamilCollector,
    "youturn": YouturnCollector,
    "pib_factcheck": PibFactCheckCollector,
    "dailythanthi": DailyThanthiCollector,
    "dinamalar": DinamalarCollector,
    "hindutamil": HinduTamilCollector,
}

_ENGLISH_SOURCES = ["welfake", "isot"]
_TAMIL_SOURCES = ["boom_tamil", "youturn", "pib_factcheck", "dailythanthi", "dinamalar", "hindutamil"]


def _build_kwargs(source: str, args: argparse.Namespace) -> dict[str, Any]:
    """Translate shared CLI flags into each collector's collect() kwargs."""
    if source in ("welfake", "isot"):
        return {"force": args.force}

    if source == "boom_tamil":
        return {"max_pages": args.max_pages, "fetch_full_text": not args.no_full_text}

    if source == "youturn":
        return {"max_pages": args.max_pages, "fetch_full_text": not args.no_full_text}

    if source == "pib_factcheck":
        return {"fetch_full_text": not args.no_full_text}

    if source in ("dailythanthi", "dinamalar", "hindutamil"):
        return {"max_pages_per_section": args.max_pages, "fetch_full_text": not args.no_full_text}

    return {}


def run_source(source: str, args: argparse.Namespace) -> dict[str, Any]:
    collector_cls = _REGISTRY[source]
    collector = collector_cls(
        output_root=DATA_RAW_DIR,
        rate_limit_seconds=args.rate_limit,
    )
    kwargs = _build_kwargs(source, args)
    try:
        return collector.collect(**kwargs)
    except Exception as exc:  # noqa: BLE001 - we want to keep going and report the failure
        logger.exception("Collector %r raised an unhandled exception", source)
        return {"source": source, "status": "error", "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 1 dataset collectors.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--all", action="store_true", help="Run every registered source.")
    group.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(_REGISTRY.keys()),
        help="Run only the named source(s).",
    )
    parser.add_argument("--list", action="store_true", help="List available source names and exit.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Max listing pages (or sections-pages) to fetch per Tamil source. Default: 5.",
    )
    parser.add_argument(
        "--no-full-text",
        action="store_true",
        help="Collect listing metadata only; skip fetching each article's full body text.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download English datasets even if already present in data/raw/.",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.5,
        help="Minimum seconds between HTTP requests to the same host. Default: 1.5.",
    )
    args = parser.parse_args()

    if args.list:
        print("English sources: ", ", ".join(_ENGLISH_SOURCES))
        print("Tamil sources:   ", ", ".join(_TAMIL_SOURCES))
        return

    if args.all:
        sources = list(_REGISTRY.keys())
    elif args.sources:
        sources = args.sources
    else:
        parser.error("Specify --all, --sources <name...>, or --list.")
        return

    logger.info("Running %d collector(s): %s", len(sources), ", ".join(sources))
    results = []
    for source in sources:
        logger.info("=" * 60)
        logger.info("Starting collector: %s", source)
        result = run_source(source, args)
        results.append(result)

    print("\n" + "=" * 72)
    print("MODULE 1 COLLECTION SUMMARY")
    print("=" * 72)
    for result in results:
        status = result.get("status", "unknown")
        source = result.get("source", "?")
        extra = {k: v for k, v in result.items() if k not in ("source", "status")}
        print(f"  [{status.upper():^20}] {source}")
        for key, value in extra.items():
            if key in ("output_file", "records_written", "size_mb", "note", "reason", "path", "zip_path"):
                print(f"      {key}: {value}")
    print("=" * 72)
    print("Raw data root:", DATA_RAW_DIR)
    print("Nothing has been preprocessed — this is Module 1 output only.")


if __name__ == "__main__":
    main()