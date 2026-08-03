"""Abstract base class shared by every dataset collector.

Design goals:
  - Every collector (English direct-download or Tamil scraper) gets the
    same HTTP session behaviour (retries, timeout, polite rate limiting,
    a descriptive User-Agent) without repeating that setup.
  - Output is always appended as JSON Lines under `data/raw/<source>/`,
    or written as a raw binary file (zip/csv) for direct-download
    sources — collectors never transform/clean the data they collect.
  - `collect()` is the only method subclasses must implement; it should
    return a summary dict so the orchestrator script can report what
    happened without parsing logs.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logger import get_logger


class BaseCollector(ABC):
    """Common HTTP + storage plumbing for a single data source."""

    #: subclasses should override with a short, filesystem-safe name
    #: (used for the output subfolder and log messages).
    source_name: str = "base"

    def __init__(
        self,
        output_root: str | Path,
        rate_limit_seconds: float = 1.5,
        timeout_seconds: int = 15,
        max_retries: int = 3,
        user_agent: str | None = None,
    ) -> None:
        self.output_dir = Path(output_root) / self.source_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limit_seconds = rate_limit_seconds
        self.timeout_seconds = timeout_seconds
        self._last_request_ts: float = 0.0

        self.logger = get_logger(f"collector.{self.source_name}")
        self.session = self._build_session(max_retries, user_agent)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    def _build_session(self, max_retries: int, user_agent: str | None) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "User-Agent": user_agent
                or "Mozilla/5.0 (compatible; FakeNewsResearchBot/0.1; +mailto:research@example.edu)",
                "Accept-Language": "en-US,en;q=0.9,ta;q=0.8",
            }
        )
        return session

    def _throttle(self) -> None:
        """Sleep just enough to respect `rate_limit_seconds` between requests."""
        elapsed = time.monotonic() - self._last_request_ts
        remaining = self.rate_limit_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_ts = time.monotonic()

    def fetch(self, url: str, **kwargs: Any) -> requests.Response | None:
        """GET a URL with retry/backoff/rate-limiting already applied.

        Returns None (and logs a warning) on failure instead of raising,
        so a single bad page does not abort an entire collection run.
        """
        self._throttle()
        try:
            response = self.session.get(url, timeout=self.timeout_seconds, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            self.logger.warning("Request failed for %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Storage (raw only — no cleaning/normalization happens here)
    # ------------------------------------------------------------------
    def save_jsonl(self, records: Iterable[dict[str, Any]], filename: str) -> int:
        """Append records as JSON Lines to `output_dir/filename`.

        Returns the number of records written. Each record is stamped
        with `collected_at` (UTC ISO-8601) if not already present.
        """
        path = self.output_dir / filename
        count = 0
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                record.setdefault("collected_at", datetime.now(timezone.utc).isoformat())
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        if count:
            self.logger.info("Wrote %d record(s) to %s", count, path)
        return count

    def save_binary(self, content: bytes, filename: str) -> Path:
        """Write raw bytes (e.g. a downloaded .csv or .zip) to disk unmodified."""
        path = self.output_dir / filename
        path.write_bytes(content)
        self.logger.info("Saved %s (%.2f MB)", path, len(content) / (1024 * 1024))
        return path

    def already_collected(self, filename: str) -> bool:
        return (self.output_dir / filename).exists()

    def load_existing_urls(self, filename: str, url_key: str = "url") -> set[str]:
        """Read whatever's already in `output_dir/filename` and return the
        set of URLs already collected.

        Lets a scraper be re-run (same day, next day, next week) without
        re-fetching or duplicating rows it already has — important for
        sources like Boom Tamil or PIB Fact Check where reaching a large
        target count means running the collector repeatedly as sites
        publish new content, not just once.
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

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    @abstractmethod
    def collect(self, **kwargs: Any) -> dict[str, Any]:
        """Run the collection and return a summary dict, e.g.
        {"source": self.source_name, "records": 123, "status": "ok"}.
        """
        raise NotImplementedError
