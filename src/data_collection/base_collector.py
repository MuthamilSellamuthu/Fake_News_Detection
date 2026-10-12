"""Shared collector base class — Module 1.

Every source-specific collector (`WelfakeCollector`, `BoomTamilCollector`,
etc.) inherits from `BaseCollector` so the things that must behave
identically across every source — polite rate-limiting, retry/backoff
on transient failures, `robots.txt` compliance, resumable
deduplication, and consistent logging — are implemented exactly once,
rather than risking subtle drift between a dozen near-duplicate
implementations.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data_collection.robots_checker import RobotsChecker
from src.utils.logger import get_logger


class BaseCollector(ABC):
    """Abstract base for all Module 1 dataset collectors.

    Subclasses must set `source_name` (used for the output directory
    and log messages) and implement `collect()`.
    """

    source_name: str = "base"

    def __init__(
        self,
        output_root: Path,
        user_agent: str,
        rate_limit_seconds: float = 1.5,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        respect_robots_txt: bool = True,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_dir = self.output_root / self.source_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.user_agent = user_agent
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout_seconds = timeout_seconds
        self.respect_robots_txt = respect_robots_txt

        self.logger = get_logger(f"data_collection.{self.source_name}")
        self._robots_checker = RobotsChecker(user_agent=user_agent, timeout_seconds=timeout_seconds)
        self._session = self._build_session(max_retries=max_retries, backoff_factor=backoff_factor)
        self._last_request_time: float = 0.0

    def _build_session(self, max_retries: int, backoff_factor: float) -> requests.Session:
        """A `requests.Session` with automatic retry/backoff on
        transient failures (connection errors, 429/500/502/503/504).
        Deliberately does *not* retry on 403 (Forbidden) — a block is a
        deliberate response, not a transient failure, and retrying it
        would look like an attempt to force through an explicit denial,
        which is exactly the opposite of what "respect website
        policies" means.
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": self.user_agent})
        return session

    def _throttle(self) -> None:
        """Enforce `rate_limit_seconds` between requests, regardless of
        how fast the previous request actually returned.
        """
        elapsed = time.time() - self._last_request_time
        remaining = self.rate_limit_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_time = time.time()

    def fetch(self, url: str, **kwargs: Any) -> requests.Response | None:
        """GET `url`, honoring rate-limiting and `robots.txt` first.

        Returns `None` (rather than raising) on a `robots.txt`
        disallow, or on a request that fails even after the session's
        built-in retries — callers are expected to treat `None` as
        "skip this URL, log already happened here" rather than crash
        the whole collection run over one bad URL.
        """
        if self.respect_robots_txt and not self._robots_checker.can_fetch(url):
            return None

        self._throttle()
        try:
            response = self._session.get(url, timeout=self.timeout_seconds, **kwargs)
            if response.status_code == 403:
                self.logger.warning("HTTP 403 (Forbidden) for %s — site is declining this request; not retrying.", url)
                return None
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            self.logger.warning("Request failed for %s: %s", url, exc)
            return None

    def load_existing_urls(self, filename: str, url_key: str = "url") -> set[str]:
        """Read whatever's already in `output_dir/filename` (a JSONL
        file) and return the set of URLs already collected, so a
        collector can be re-run later without re-fetching or
        duplicating rows it already has — important for sources like
        the Daily Thanthi sitemap crawl, which is designed to be run in
        multiple chunked sessions rather than one continuous run.
        """
        path = self.output_dir / filename
        if not path.exists():
            return set()

        urls: set[str] = set()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                url = record.get(url_key)
                if url:
                    urls.add(url)

        if urls:
            self.logger.info("Found %d already-collected URL(s) in %s; will skip re-fetching them.", len(urls), path)
        return urls

    def save_jsonl(self, records: list[dict[str, Any]], filename: str) -> int:
        """Append `records` to `output_dir/filename` as JSON Lines,
        one record per line. Returns the number of records written.
        """
        if not records:
            return 0
        path = self.output_dir / filename
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(records)

    def save_binary(self, content: bytes, filename: str) -> Path:
        """Write raw bytes (e.g. a downloaded CSV or zip) to `output_dir/filename`."""
        path = self.output_dir / filename
        path.write_bytes(content)
        return path

    def already_collected(self, filename: str) -> bool:
        """True if `output_dir/filename` already exists and is non-empty
        — used by direct-download collectors (WELFake, ISOT) to skip
        re-downloading a large file unless `force=True` is passed.
        """
        path = self.output_dir / filename
        return path.exists() and path.stat().st_size > 0

    @abstractmethod
    def collect(self, **kwargs: Any) -> dict[str, Any]:
        """Run this source's collection and return a summary dict. Must
        be implemented by every subclass.
        """
        raise NotImplementedError
