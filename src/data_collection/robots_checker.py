"""robots.txt compliance checker — Module 1.

A tiny, dedicated wrapper around Python's standard-library
`urllib.robotparser` so every collector checks site policy the same
way, and so a `robots.txt` fetch/parse only happens once per host per
process (cached), not once per URL.

This is treated as a hard gate, not advisory: `RobotsChecker.can_fetch()`
returning `False` means the collector must not request that URL, full
stop — see `BaseCollector.fetch()`, which calls this before every
request.
"""

from __future__ import annotations

from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from src.utils.logger import get_logger

logger = get_logger("data_collection.robots_checker")


class RobotsChecker:
    """Fetches and caches `robots.txt` per host, then answers
    "can I fetch this URL with this user-agent?" for any URL on that host.
    """

    def __init__(self, user_agent: str, timeout_seconds: float = 10.0) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self._parsers: dict[str, RobotFileParser] = {}
        self._unreachable_hosts: set[str] = set()

    def _get_parser(self, url: str) -> RobotFileParser | None:
        """Return a `RobotFileParser` already loaded with `<host>/robots.txt`,
        fetching and caching it on first use for this host. Returns
        `None` if `robots.txt` itself couldn't be fetched — in that
        case, per the standard interpretation of a missing/unreachable
        robots.txt, access is not blocked (there is nothing prohibiting
        it), but this is logged so the ambiguity is visible rather than
        silently assumed.
        """
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"

        if host in self._unreachable_hosts:
            return None
        if host in self._parsers:
            return self._parsers[host]

        robots_url = urljoin(host, "/robots.txt")
        parser = RobotFileParser()
        parser.set_url(robots_url)

        try:
            response = requests.get(
                robots_url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout_seconds,
            )
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
                logger.info("Loaded robots.txt for %s", host)
            else:
                # A non-200 (404 in particular) is the standard signal
                # that no robots.txt exists, i.e. no restrictions apply.
                parser.parse([])
                logger.info(
                    "No robots.txt found at %s (HTTP %d) — treating as unrestricted.",
                    robots_url, response.status_code,
                )
        except requests.RequestException as exc:
            logger.warning(
                "Could not fetch %s (%s) — cannot confirm robots.txt policy for %s; "
                "treating as unrestricted per default interpretation, but this should "
                "be verified manually before a large-scale run.",
                robots_url, exc, host,
            )
            self._unreachable_hosts.add(host)
            return None

        self._parsers[host] = parser
        return parser

    def can_fetch(self, url: str) -> bool:
        """Return whether `self.user_agent` is permitted to fetch `url`
        per that host's `robots.txt`. Fails open (returns True) only
        when robots.txt itself is unreachable — never when it explicitly
        disallows the path.
        """
        parser = self._get_parser(url)
        if parser is None:
            return True
        allowed = parser.can_fetch(self.user_agent, url)
        if not allowed:
            logger.warning("robots.txt disallows fetching %s — skipping.", url)
        return allowed
