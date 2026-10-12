"""Module 10 — shared plumbing. Reuses Module 6/7's helpers unchanged.

Deliberately thin, for the same reason `m7_common.py` is thin: root
resolution, atomic JSON and manifests were solved once in `m6_common.py`, and
a second copy is how two implementations drift apart.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from scripts.m6_common import (  # noqa: F401  - reused, not reimplemented
    PROJECT_ROOT, detect_device, read_manifest, resolve_project_root,
    write_json, write_manifest)
from scripts.m7_common import load_m7_config, sha256_file  # noqa: F401
from src.utils.config_loader import load_yaml

M10_CONFIG = "configs/module10_inference.yaml"


def load_m10_config(path: str | Path | None = None,
                    root: Path | None = None) -> dict[str, Any]:
    base = root or PROJECT_ROOT
    return load_yaml(Path(path) if path else base / M10_CONFIG)


def m10_dir(cfg: dict[str, Any], sub: str, root: Path | None = None) -> Path:
    d = (root or PROJECT_ROOT) / cfg["output"]["root"] / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def banner(stage: str) -> None:
    print(f"\n{'=' * 70}\nMODULE 10 — {stage}\n{'=' * 70}")


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def software_versions() -> dict[str, Any]:
    """Record what is actually importable, not what is assumed.

    faiss and sentence-transformers are reported only if present; Module 10's
    selected implementation (BM25 + xlm-roberta-base) requires neither, and
    listing an absent package as a dependency would misdescribe the run.
    """
    out: dict[str, Any] = {"python": sys.version.split()[0],
                           "platform": platform.platform()}
    for name in ("torch", "transformers", "numpy", "faiss", "sentence_transformers"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:                                   # noqa: BLE001
            out[name] = None
    return out


def provenance_block(cfg: dict[str, Any], extra: dict[str, Any] | None = None
                     ) -> dict[str, Any]:
    block = {"seed": int(cfg["reproducibility"]["seed"]),
             "deterministic": bool(cfg["reproducibility"]["deterministic"]),
             "config_hash": config_hash(cfg),
             "device": detect_device(),
             "software": software_versions(),
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        block.update(extra)
    return block


def set_all_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:                                       # noqa: BLE001
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:                                       # noqa: BLE001
        pass
