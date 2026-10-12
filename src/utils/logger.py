"""Shared logging utility.

Gives every collector/module a consistently formatted logger without
each module re-configuring `logging` (and without duplicate handlers
when a logger is requested more than once).

Module 1 addition: every run now also writes a timestamped file under
`logs/`, in addition to the console — so a long-running collection job
(Tamil sitemap crawling can run for hours) leaves a permanent,
reviewable record even after the terminal that started it is closed.
Console-only output disappears the moment the window is closed or the
buffer scrolls past it.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_CONFIGURED_LOGGERS: set[str] = set()

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_RUN_LOG_FILE: Path | None = None


def _project_root() -> Path:
    """`src/utils/logger.py` -> project root is two parents up."""
    return Path(__file__).resolve().parents[2]


def _run_log_path() -> Path:
    """One log file per process run (not per logger name), timestamped
    at first use, so every module invoked within a single run writes to
    the same file instead of scattering one file per logger.
    """
    global _RUN_LOG_FILE
    if _RUN_LOG_FILE is None:
        logs_dir = _project_root() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _RUN_LOG_FILE = logs_dir / f"run_{timestamp}.log"
    return _RUN_LOG_FILE


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with console + run-file handlers.

    Safe to call multiple times with the same `name` (e.g. once per
    collector instantiation) — handlers are only attached once.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if name not in _CONFIGURED_LOGGERS:
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(_run_log_path(), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.propagate = False
        _CONFIGURED_LOGGERS.add(name)

    return logger
