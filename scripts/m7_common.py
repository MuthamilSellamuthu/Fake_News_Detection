"""Module 7 — shared plumbing. Reuses Module 6's project-root resolution.

Deliberately thin: `scripts/m6_common.py` already solved root resolution,
manifests and atomic JSON, and duplicating that logic is exactly how the two
would drift apart. Module 7 adds only what is specific to it.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from scripts.m6_common import (  # noqa: F401  - reused, not reimplemented
    PROJECT_ROOT, detect_device, load_jsonl, module5_fingerprint,
    read_manifest, resolve_project_root, write_json, write_manifest)
from src.utils.config_loader import load_yaml

M7_CONFIG = "configs/module7_integration.yaml"


def load_m7_config(path: str | Path | None = None, root: Path | None = None) -> dict[str, Any]:
    base = root or PROJECT_ROOT
    return load_yaml(Path(path) if path else base / M7_CONFIG)


def m7_dir(cfg: dict[str, Any], sub: str, root: Path | None = None) -> Path:
    d = (root or PROJECT_ROOT) / cfg["output"]["root"] / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reproducibility_block(cfg: dict[str, Any], extra: dict[str, Any] | None = None
                          ) -> dict[str, Any]:
    """Metadata every Module 7 manifest carries, so a result can be traced."""
    info: dict[str, Any] = {
        "seed": cfg["reproducibility"]["seed"],
        "deterministic": cfg["reproducibility"]["deterministic"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "device": detect_device(),
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        info["torch"] = "not installed"
    if extra:
        info.update(extra)
    return info


def set_all_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass


def banner(stage: str) -> None:
    print("=" * 70)
    print(f"MODULE 7 — {stage}")
    print("=" * 70)
    print(f"  project root: {PROJECT_ROOT}")
    print(f"  device      : {detect_device()}")
