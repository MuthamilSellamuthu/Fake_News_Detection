"""Small helper for loading YAML configuration files.

Kept deliberately dumb (no schema validation, no defaults merging) —
collectors are responsible for validating the keys they need. This
just centralizes "read YAML off disk into a dict" so every module
doesn't repeat the same three lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Raises FileNotFoundError if the path does not exist and
    ValueError if the file does not parse to a mapping at the top level.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in {path}, got {type(data)}")
    return data


def project_root() -> Path:
    """Return the project root (parent of `src/`)."""
    return Path(__file__).resolve().parents[2]


def default_sources_config_path() -> Path:
    return project_root() / "configs" / "sources.yaml"
