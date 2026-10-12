"""Module 9 — shared plumbing."""
from __future__ import annotations

import json, sys, time
from pathlib import Path
from typing import Any

from scripts.m6_common import (  # noqa: F401
    PROJECT_ROOT, detect_device, write_json, write_manifest)
from src.utils.config_loader import load_yaml

UNAVAILABLE = "unavailable"


def load_m9_config(path=None, root=None) -> dict[str, Any]:
    base = root or PROJECT_ROOT
    return load_yaml(Path(path) if path else base / "configs/module9_evaluation.yaml")


def m9_dir(cfg, sub="", root=None) -> Path:
    base = (root or PROJECT_ROOT) / cfg["output"]["root"]
    d = base / sub if sub else base
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(path: Path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def macro_f1_binary(y, p) -> float:
    import numpy as np
    y, p = np.asarray(y), np.asarray(p)
    f1s = []
    for c in (0, 1):
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return float(sum(f1s) / 2)


def accuracy(y, p) -> float:
    import numpy as np
    y, p = np.asarray(y), np.asarray(p)
    return float((y == p).mean()) if len(y) else 0.0


def manifest_block(cfg, stage, split, inputs, outputs, status, extra=None):
    b = {"module": 9, "stage": stage, "dataset": "project", "split": split,
         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "input_artefacts": inputs, "output_artefacts": outputs,
         "configuration": {"statistics": cfg.get("statistics"),
                           "metrics": cfg.get("metrics")},
         "random_seed": cfg.get("reproducibility", {}).get("seed"),
         "status": status,
         "runtime": {"python": sys.version.split()[0], "device": detect_device()}}
    if extra:
        b.update(extra)
    return b


def banner(stage: str) -> None:
    print("=" * 70); print(f"MODULE 9 — {stage}"); print("=" * 70)
