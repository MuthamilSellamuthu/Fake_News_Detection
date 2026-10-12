#!/usr/bin/env python
"""Module 1 — English dataset collection CLI.

Downloads WELFake and ISOT into `datasets/raw/english/`. Both are
direct, no-login downloads from their official hosts (Zenodo /
University of Victoria) — no scraping, no robots.txt ambiguity.

Usage
-----
  python scripts/collect_english.py                 # both sources
  python scripts/collect_english.py --sources welfake
  python scripts/collect_english.py --force          # re-download even if already present
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.english.isot_collector import IsotCollector  # noqa: E402
from src.data_collection.english.welfake_collector import WelfakeCollector  # noqa: E402
from src.utils.config_loader import load_yaml, project_root  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.collect_english")

_REGISTRY: dict[str, type] = {
    "welfake": WelfakeCollector,
    "isot": IsotCollector,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 1: download English datasets (WELFake, ISOT).")
    parser.add_argument("--sources", nargs="+", choices=sorted(_REGISTRY.keys()), default=None)
    parser.add_argument("--force", action="store_true", help="Re-download even if already present.")
    args = parser.parse_args()

    config = load_yaml(project_root() / "configs" / "module1_sources.yaml")
    http_cfg = config["http"]
    output_root = project_root() / "datasets" / "raw" / "english"

    sources = args.sources or list(_REGISTRY.keys())
    results: list[dict[str, Any]] = []

    for source in sources:
        logger.info("=" * 60)
        logger.info("Collecting: %s", source)
        collector = _REGISTRY[source](
            output_root=output_root,
            user_agent=http_cfg["user_agent"],
            rate_limit_seconds=http_cfg["rate_limit_seconds"],
            timeout_seconds=http_cfg["timeout_seconds"],
            max_retries=http_cfg["max_retries"],
            backoff_factor=http_cfg["backoff_factor"],
            respect_robots_txt=http_cfg["respect_robots_txt"],
        )
        try:
            result = collector.collect(force=args.force)
        except Exception as exc:  # noqa: BLE001 - keep going, report the failure
            logger.exception("Collector %r raised an unhandled exception", source)
            result = {"source": source, "status": "error", "error": str(exc)}
        results.append(result)

    print("\n" + "=" * 72)
    print("MODULE 1 — ENGLISH COLLECTION SUMMARY")
    print("=" * 72)
    for result in results:
        print(f"  [{result.get('status', 'unknown').upper():^18}] {result.get('source', '?')}")
        for key, value in result.items():
            if key not in ("source", "status"):
                print(f"      {key}: {value}")
    print("=" * 72)


if __name__ == "__main__":
    main()
