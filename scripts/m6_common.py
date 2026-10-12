"""Module 6 — shared stage plumbing and project-root resolution.

Every m6_* script loads the same corpus, writes the same manifest shape, and
verifies Module 5 is untouched. Centralising that here keeps the nine stage
scripts short and stops them drifting apart.

PROJECT ROOT RESOLUTION
-----------------------
This module must import cleanly in four different execution contexts:

  1. `python scripts/m6_01_encode_evidence.py`   (cwd anywhere)
  2. `from scripts.m6_common import ...`         (notebook, after sys.path setup)
  3. `importlib.util.spec_from_file_location(...)` (direct file load)
  4. pasted verbatim into a notebook cell        (no `__file__` at all)

The previous version used `PROJECT_ROOT = Path(__file__).resolve().parents[1]`
as its only strategy. That breaks in context 4 with `NameError: name '__file__'
is not defined`, and resolves to the wrong directory in context 3 when the file
is loaded from a copy. Because `from src.utils.config_loader import load_yaml`
runs at import time and depends on that root already being on `sys.path`, a
wrong root surfaces later as the very confusing `ModuleNotFoundError: No module
named 'src'` -- on a project where `src/` obviously exists.

`resolve_project_root()` therefore tries, in order:

  1. $FAKE_NEWS_PROJECT_ROOT            explicit, wins over everything
  2. the directory containing this file (only when `__file__` exists)
  3. cwd and each of its parents        searching for the project markers
  4. /content/drive/MyDrive/Fake_News_Detection   (Colab default)

A candidate only counts if it contains BOTH `src/utils/config_loader.py` and
`configs/module5_fact_verification.yaml`. Checking two markers rather than one
means a stray `src/` directory elsewhere on the path cannot masquerade as the
project root.

MODULE 5 IS READ-ONLY. `load_corpus()` opens files in read mode only, and
`module5_fingerprint()` lets any stage prove nothing under Module 5's
directories changed while it ran.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Two markers, not one: a lone `src/` somewhere else must not win.
_ROOT_MARKERS = ("src/utils/config_loader.py", "configs/module5_fact_verification.yaml")
_COLAB_DEFAULT = Path("/content/drive/MyDrive/Fake_News_Detection")
_ENV_VAR = "FAKE_NEWS_PROJECT_ROOT"


def _looks_like_project_root(path: Path) -> bool:
    try:
        return all((path / marker).is_file() for marker in _ROOT_MARKERS)
    except OSError:
        return False


def resolve_project_root(explicit: str | Path | None = None) -> Path:
    """Locate the project root without depending on `__file__`.

    Raises RuntimeError with the candidates tried, rather than returning a
    wrong root that fails later as an unrelated-looking import error.
    """
    tried: list[str] = []

    def consider(cand: Path | None, why: str) -> Path | None:
        if cand is None:
            return None
        try:
            cand = Path(cand).expanduser().resolve()
        except OSError:
            return None
        tried.append(f"{why}: {cand}")
        return cand if _looks_like_project_root(cand) else None

    # 1. explicit argument, then environment variable
    for cand, why in ((explicit, "explicit argument"), (os.environ.get(_ENV_VAR), _ENV_VAR)):
        found = consider(cand, why) if cand else None
        if found:
            return found

    # 2. this file's parent directory -- guarded, since `__file__` may not exist
    try:
        here = Path(__file__).resolve()          # noqa: F821 - may be undefined
    except NameError:
        here = None
    if here is not None:
        for parent in list(here.parents)[:4]:
            found = consider(parent, "relative to m6_common.py")
            if found:
                return found

    # 3. cwd and its parents
    cwd = Path.cwd().resolve()
    for cand in [cwd, *cwd.parents]:
        found = consider(cand, "cwd or parent")
        if found:
            return found

    # 4. Colab default
    found = consider(_COLAB_DEFAULT, "Colab default")
    if found:
        return found

    raise RuntimeError(
        "Could not locate the Fake_News_Detection project root.\n"
        f"  A valid root must contain both: {', '.join(_ROOT_MARKERS)}\n"
        "  Candidates tried:\n    " + "\n    ".join(tried or ["(none)"]) + "\n"
        f"  Fix: set {_ENV_VAR}=/content/drive/MyDrive/Fake_News_Detection")


def ensure_project_on_path(root: Path | None = None) -> Path:
    """Put the project root on `sys.path` so `src` and `scripts` import.

    Idempotent and safe to call from any stage script or notebook cell.
    """
    root = Path(root) if root else resolve_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# Resolved once at import. Every stage script imports PROJECT_ROOT from here,
# so this is the single place the root is determined.
PROJECT_ROOT = ensure_project_on_path()

from src.utils.config_loader import load_yaml  # noqa: E402

M5_READONLY = [
    "datasets/fact_verification",
    "src/fact_verification",
    "configs/module5_fact_verification.yaml",
]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else PROJECT_ROOT / "configs" / "module6_retrieval_verification.yaml"
    return load_yaml(p)


def stage_dir(cfg: dict[str, Any], stage: str, root: Path | None = None) -> Path:
    d = (root or PROJECT_ROOT) / cfg["output"]["root"] / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"corrupt JSONL {path}:{ln}: {e}") from e
    return rows


def load_corpus(cfg: dict[str, Any], dataset: str, root: Path | None = None
                ) -> dict[str, Any]:
    """Read one Module 5 corpus. Read-only, always."""
    base = (root or PROJECT_ROOT) / cfg["inputs"]["corpus_dir"] / dataset
    claims_p = base / cfg["inputs"]["claims_filename"]
    ev_p = base / cfg["inputs"]["evidence_filename"]
    sp_p = base / cfg["inputs"]["splits_filename"]
    for p in (claims_p, ev_p, sp_p):
        if not p.is_file():
            raise FileNotFoundError(f"Module 5 output missing: {p}. Run Module 5 first.")

    claims = load_jsonl(claims_p)
    evidence = load_jsonl(ev_p)
    splits_doc = json.loads(sp_p.read_text(encoding="utf-8"))
    splits = splits_doc.get("splits", {})

    split_of: dict[int, str] = {}
    for s, idxs in splits.items():
        for i in idxs:
            split_of[i] = s

    return {
        "dataset": dataset,
        "claims": claims,
        "evidence": evidence,
        "evidence_texts": {e["evidence_id"]: e.get("text", "") for e in evidence},
        "splits": splits,
        "split_of": split_of,
        "claim_id_of": {i: c.get("claim_id", str(i)) for i, c in enumerate(claims)},
        "index_of_claim_id": {c.get("claim_id", str(i)): i for i, c in enumerate(claims)},
        "paths": {"claims": str(claims_p), "evidence": str(ev_p), "splits": str(sp_p)},
    }


def gold_evidence_map(corpus: dict[str, Any]) -> dict[str, list[str]]:
    return {c.get("claim_id", str(i)): list(c.get("evidence") or [])
            for i, c in enumerate(corpus["claims"])}


def module5_fingerprint(root: Path | None = None) -> dict[str, Any]:
    """Size+count signature of every Module 5 path, for before/after comparison."""
    base = root or PROJECT_ROOT
    out: dict[str, Any] = {}
    for rel in M5_READONLY:
        p = base / rel
        if not p.exists():
            out[rel] = None
        elif p.is_file():
            out[rel] = {"bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
        else:
            files = sorted(f for f in p.rglob("*") if f.is_file() and "__pycache__" not in str(f))
            out[rel] = {"n_files": len(files),
                        "total_bytes": sum(f.stat().st_size for f in files)}
    return out


def assert_module5_untouched(before: dict[str, Any], root: Path | None = None) -> None:
    after = module5_fingerprint(root)
    changed = {k: {"before": before.get(k), "after": after.get(k)}
               for k in after if before.get(k) != after.get(k)}
    if changed:
        raise AssertionError(f"MODULE 5 WAS MODIFIED: {json.dumps(changed, indent=2)}")


def write_manifest(stage_path: Path, payload: dict[str, Any]) -> Path:
    # Create the directory, as write_json() does. Omitting this worked only
    # while every caller happened to target a directory some earlier stage had
    # already made: stage_dir() creates outputs/module6/<stage>, but NOT the
    # <dataset>/ or ablations/<dataset>/<run_name>/ subtree beneath it. The
    # first caller to pass a genuinely new nested path -- an isolated ablation
    # directory -- hit FileNotFoundError on manifest.tmp.json.
    stage_path = Path(stage_path)
    stage_path.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("written_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    tmp = stage_path / "manifest.tmp.json"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(stage_path / "manifest.json")
    return stage_path / "manifest.json"


def read_manifest(stage_path: Path) -> dict[str, Any]:
    p = Path(stage_path) / "manifest.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)
    return path


def config_hash(cfg: dict[str, Any], keys: list[str] | None = None) -> str:
    """Hash of the config slice a stage depends on, so a changed setting is
    visible in the manifest rather than silently reusing stale shards."""
    sub = {k: cfg.get(k) for k in (keys or sorted(cfg))}
    return hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:16]


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def banner(stage: str, cfg: dict[str, Any]) -> None:
    print("=" * 70)
    print(f"MODULE 6 — {stage}")
    print("=" * 70)
    print(f"  project root: {PROJECT_ROOT}")
    print(f"  device      : {detect_device()}")
