"""Module 9 — reproducibility metadata.

Anything that cannot be determined is recorded as the literal string
"unavailable". It is never guessed, never omitted, and never filled with a
plausible-looking default -- a fabricated environment record would make a
failed replication impossible to diagnose.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

UNAVAILABLE = "unavailable"


def _try(fn, default=UNAVAILABLE):
    try:
        v = fn()
        return v if v is not None else default
    except Exception:  # noqa: BLE001
        return default


def software_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0],
           "platform": _try(lambda: platform.platform()),
           "processor": _try(lambda: platform.processor() or UNAVAILABLE)}
    for name in ("numpy", "torch", "transformers", "sentence_transformers",
                 "sklearn", "scipy", "faiss", "shap"):
        def _v(n=name):
            mod = __import__(n)
            return getattr(mod, "__version__", UNAVAILABLE)
        out[name] = _try(_v)
    return out


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cpu_count": _try(lambda: __import__("os").cpu_count())}
    def _mem():
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return f"{float(line.split()[1]) / 1024 / 1024:.1f} GB"
        return UNAVAILABLE
    info["memory_total"] = _try(_mem)
    def _gpu():
        import torch
        if not torch.cuda.is_available():
            return "no CUDA device"
        return {"name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count()}
    info["gpu"] = _try(_gpu)
    return info


def git_commit(root: Path) -> str:
    def _c():
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else UNAVAILABLE
    return _try(_c)


def sha256_file(path: Path) -> str:
    def _h():
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    return _try(_h)


def hash_artefacts(root: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rel in relative_paths:
        p = Path(root) / rel
        if p.is_file():
            out[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size,
                        "exists": True}
        elif p.is_dir():
            files = sorted(f for f in p.rglob("*")
                           if f.is_file() and "__pycache__" not in str(f))
            out[rel] = {"exists": True, "n_files": len(files),
                        "total_bytes": sum(f.stat().st_size for f in files)}
        else:
            out[rel] = {"exists": False, "sha256": UNAVAILABLE}
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def build_reproducibility_report(root: Path, configs: dict[str, dict[str, Any]],
                                 artefacts: Sequence[str], seeds: dict[str, Any]
                                 ) -> dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(root),
        "software": software_versions(),
        "hardware": hardware_info(),
        "seeds": seeds,
        "config_hashes": {name: config_hash(c) for name, c in configs.items()},
        "artefact_hashes": hash_artefacts(root, artefacts),
        "policy": ("Fields that could not be determined are recorded as "
                   f"'{UNAVAILABLE}'. No value in this report is inferred or "
                   "filled with a default."),
    }
