"""Module 8 — shared plumbing. Reuses Module 6/7 utilities."""
from __future__ import annotations

import hashlib, json, sys, time
from pathlib import Path
from typing import Any

from scripts.m6_common import (  # noqa: F401
    PROJECT_ROOT, detect_device, read_manifest, write_json, write_manifest)
from src.utils.config_loader import load_yaml


def load_m8_config(path=None, root=None) -> dict[str, Any]:
    base = root or PROJECT_ROOT
    return load_yaml(Path(path) if path else base / "configs/module8_explainability.yaml")


def m8_dir(cfg, sub, root=None) -> Path:
    d = (root or PROJECT_ROOT) / cfg["output"]["root"] / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def manifest_block(cfg, module: int, stage: str, dataset: str, split: str,
                   inputs: list, outputs: list, status: str,
                   extra: dict | None = None) -> dict[str, Any]:
    """The manifest shape required across Modules 7-9."""
    b = {"module": module, "stage": stage, "dataset": dataset, "split": split,
         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "input_artefacts": inputs, "output_artefacts": outputs,
         "configuration": {k: cfg.get(k) for k in ("explanation", "faithfulness",
                                                   "leakage", "reproducibility")
                           if k in cfg},
         "random_seed": cfg.get("reproducibility", {}).get("seed"),
         "status": status,
         "runtime": {"python": sys.version.split()[0], "device": detect_device()}}
    try:
        import numpy; b["runtime"]["numpy"] = numpy.__version__
    except Exception: pass
    try:
        import torch; b["runtime"]["torch"] = torch.__version__
    except Exception: b["runtime"]["torch"] = "unavailable"
    try:
        import transformers; b["runtime"]["transformers"] = transformers.__version__
    except Exception: b["runtime"]["transformers"] = "unavailable"
    if extra: b.update(extra)
    return b


def write_jsonl(path: Path, rows) -> Path:
    """Atomic JSONL write."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def banner(stage: str) -> None:
    print("=" * 70); print(f"MODULE 8 — {stage}"); print("=" * 70)
    print(f"  project root: {PROJECT_ROOT}"); print(f"  device      : {detect_device()}")
