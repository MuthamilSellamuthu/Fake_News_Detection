"""Shared logging utility.

Gives every collector/module a consistently formatted logger without
each module re-configuring `logging` (and without duplicate handlers
when a logger is requested more than once).
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED_LOGGERS: set[str] = set()

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a single stdout handler.

    Safe to call multiple times with the same `name` (e.g. once per
    collector instantiation) — handlers are only attached once.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if name not in _CONFIGURED_LOGGERS:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED_LOGGERS.add(name)

    return logger
