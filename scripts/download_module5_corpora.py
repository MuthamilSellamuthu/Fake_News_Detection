#!/usr/bin/env python
"""Module 5 Stage 2 — licence-gated corpus acquisition. COLAB ONLY.

Downloads X-FACT and the FEVER gold-evidence resource into a NEW
location, `datasets/module5_external/`, keeping each corpus in its own
directory. Nothing under `datasets/processed/`, `embeddings/`,
`outputs/module3/`, `outputs/module4/` or `models/module4/` is read for
writing or modified.

Refuses to run unless approval is explicit. Records dataset name,
source URL, revision, download timestamp, licence, SHA-256 and byte
size for every file, so a later audit can prove exactly which artefact
produced which number.

Licences (verified 2026-08-09, recorded in configs/module5_fact_verification.yaml):
  X-FACT  MIT, (c) 2021 Utah NLP  https://github.com/utahnlp/x-fact
  FEVER   Wikipedia terms, fallback CC BY-SA 3.0  https://fever.ai

FEVER DESIGN CHOICE — recorded here because it constrains every later claim:
  This script fetches FEVER claims WITH THEIR GOLD EVIDENCE SENTENCES
  already attached, not the ~40 GB Wikipedia `wiki-pages` dump. That
  makes the corpus tractable, but it changes what "retrieval" means:
  the candidate pool becomes the union of gold evidence sentences, not
  all of Wikipedia. Any retrieval result obtained this way is
  CLOSED-POOL and must not be described as full-Wikipedia open-world
  retrieval. Original page/sentence identifiers are preserved wherever
  the source provides them so the link back to Wikipedia survives.

Usage (in Colab, from the project root):
    python scripts/download_module5_corpora.py --approve-download
    python scripts/download_module5_corpora.py --approve-download --only x_fact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_yaml  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("module5.download")

# Paths that must never be written by this module.
FROZEN = ["datasets/processed", "embeddings", "outputs/module3", "outputs/module4", "models/module4"]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def assert_frozen_untouched(root: Path, before: dict[str, Any]) -> dict[str, Any]:
    """Compare a cheap signature of the frozen trees before/after.

    Uses (file count, total bytes, newest mtime) per tree. Cheap enough
    to run either side of a download and sufficient to detect any write.
    """
    now = frozen_signature(root)
    changed = {k: {"before": before.get(k), "after": now.get(k)} for k in now if before.get(k) != now.get(k)}
    return {"unchanged": not changed, "changed": changed, "signature": now}


def frozen_signature(root: Path) -> dict[str, Any]:
    sig: dict[str, Any] = {}
    for rel in FROZEN:
        p = root / rel
        if not p.exists():
            sig[rel] = None
            continue
        files = [f for f in p.rglob("*") if f.is_file()]
        sig[rel] = {
            "n_files": len(files),
            "total_bytes": sum(f.stat().st_size for f in files),
            "newest_mtime": max((f.stat().st_mtime for f in files), default=0.0),
        }
    return sig


# --------------------------------------------------------------------
# Downloaders
# --------------------------------------------------------------------

def download_x_fact(dest: Path, hf_config: str = "ta") -> dict[str, Any]:
    """X-FACT via the HuggingFace datasets library.

    The HF dataset REQUIRES an explicit config name. Calling
    `load_dataset("utahnlp/x-fact")` with no config raises
    "ValueError: Config name is missing" — observed 2026-08-09. The
    config is therefore always passed explicitly and always recorded in
    the manifest, so a later audit can tell which language subset a
    given number came from.

    `hf_config` defaults to "ta": the immediate research question is the
    Tamil subset, and "all_languages" would download roughly 25x the
    data for no current purpose. This is a deliberate scope decision,
    not a limitation of the loader — pass a different config to widen it.

    The dataset revision is captured where the installed `datasets`
    version exposes it, so the exact snapshot is reproducible.
    """
    dest.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "dataset": "x-fact",
        "hf_config": hf_config,
        "hf_config_rationale": (
            "Tamil-only subset selected deliberately; 'all_languages' not downloaded "
            "because the current research question concerns Tamil. Statistics from this "
            "download describe the Tamil config ONLY and must not be reported as X-FACT overall."
        ),
        "licence": "MIT",
        "licence_url": "https://github.com/utahnlp/x-fact/blob/main/LICENSE",
        "attribution": "X-FACT (Gupta & Srikumar, ACL 2021), Copyright (c) 2021 Utah NLP",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }
    try:
        from datasets import get_dataset_config_names, load_dataset

        record["source_url"] = "https://huggingface.co/datasets/utahnlp/x-fact"
        record["method"] = "huggingface_datasets"

        try:
            available = get_dataset_config_names("utahnlp/x-fact")
            record["configs_available"] = available
            if hf_config not in available:
                raise ValueError(
                    f"Requested X-FACT config {hf_config!r} is not available. "
                    f"Available: {available}")
        except ValueError:
            raise
        except Exception:  # noqa: BLE001 - config listing is best-effort
            record["configs_available"] = None

        ds = load_dataset("utahnlp/x-fact", hf_config)
        record["splits_reported_by_source"] = {k: len(v) for k, v in ds.items()}

        # Capture the resolved revision where the installed datasets
        # version exposes it; absence is recorded rather than guessed.
        revision = None
        try:
            first = next(iter(ds.values()))
            info = getattr(first, "info", None)
            revision = getattr(info, "version", None)
            if revision is not None:
                revision = str(revision)
            dl_checksums = getattr(info, "download_checksums", None)
            if dl_checksums:
                record["source_download_checksums"] = {str(k): v for k, v in list(dl_checksums.items())[:10]}
        except Exception:  # noqa: BLE001
            revision = None
        record["dataset_revision"] = revision or "not_exposed_by_datasets_library"

        # Persist verbatim as JSONL: no filtering, no column renaming.
        for split, data in ds.items():
            out = dest / f"x_fact_{split}.jsonl"
            data.to_json(str(out), orient="records", lines=True, force_ascii=False)
            record["files"].append({
                "path": str(out.relative_to(PROJECT_ROOT)), "split": split,
                "hf_config": hf_config,
                "n_rows": len(data), "columns": list(data.column_names),
                "bytes": out.stat().st_size, "sha256": sha256_of(out),
            })
        logger.info("X-FACT (config=%s) downloaded: %s", hf_config, record["splits_reported_by_source"])
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["note"] = ("HuggingFace route failed. Fallback: clone "
                          "https://github.com/utahnlp/x-fact and place the TSVs under "
                          f"{dest.relative_to(PROJECT_ROOT)}, then re-run the audit. "
                          "Note the HF dataset requires an explicit config name.")
        logger.error("X-FACT download failed: %s", record["error"])
    return record


def download_fever_gold_evidence(dest: Path) -> dict[str, Any]:
    """FEVER claims WITH gold evidence sentences attached.

    Deliberately NOT the Wikipedia dump — see the module docstring for
    why, and for the limitation this imposes on retrieval claims.
    """
    dest.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "dataset": "fever_gold_evidence",
        "licence": "Wikipedia terms; fallback CC BY-SA 3.0",
        "licence_url": "https://fever.ai/dataset/fever.html",
        "attribution": ("FEVER (Thorne et al., NAACL 2018). Annotations under the licence "
                        "terms of the underlying Wikipedia pages, fallback CC BY-SA 3.0."),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "GOLD_EVIDENCE_ONLY",
        "evidence_scope_note": (
            "Candidate pool is the union of gold evidence sentences, NOT all of Wikipedia. "
            "Retrieval results from this corpus are closed-pool and must not be reported as "
            "full-Wikipedia open-world retrieval."
        ),
        "files": [],
    }
    try:
        from datasets import load_dataset

        record["source_url"] = "https://huggingface.co/datasets/copenlu/fever_gold_evidence"
        record["method"] = "huggingface_datasets"
        ds = load_dataset("copenlu/fever_gold_evidence")
        record["splits_reported_by_source"] = {k: len(v) for k, v in ds.items()}
        for split, data in ds.items():
            out = dest / f"fever_gold_{split}.jsonl"
            data.to_json(str(out), orient="records", lines=True, force_ascii=False)
            record["files"].append({
                "path": str(out.relative_to(PROJECT_ROOT)), "split": split,
                "n_rows": len(data), "columns": list(data.column_names),
                "bytes": out.stat().st_size, "sha256": sha256_of(out),
            })
        logger.info("FEVER gold-evidence downloaded: %s", record["splits_reported_by_source"])
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("FEVER download failed: %s", record["error"])
    return record


def main() -> None:
    p = argparse.ArgumentParser(description="Module 5 Stage 2: download external corpora (Colab only).")
    p.add_argument("--approve-download", action="store_true",
                   help="Required. Without it nothing is fetched.")
    p.add_argument("--only", choices=["x_fact", "fever"], default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--dest", default="datasets/module5_external")
    p.add_argument("--x-fact-config", default=None,
                   help="X-FACT HF config name. Defaults to corpora.x_fact.hf_config "
                        "from the YAML ('ta'). The HF dataset requires an explicit config.")
    args = p.parse_args()

    root = PROJECT_ROOT
    cfg = load_yaml(Path(args.config) if args.config else root / "configs" / "module5_fact_verification.yaml")

    if not args.approve_download:
        print("\nDOWNLOAD NOT APPROVED — nothing fetched.")
        print("Re-run with --approve-download once licence terms are accepted:")
        print(f"  X-FACT : {cfg['corpora']['x_fact']['licence']}  ({cfg['corpora']['x_fact']['licence_url']})")
        print(f"  FEVER  : {cfg['corpora']['fever']['licence']}  ({cfg['corpora']['fever']['licence_url']})")
        sys.exit(2)

    if shutil.disk_usage(root).free < 3 * (1 << 30):
        logger.warning("Less than 3 GB free — downloads may fail partway.")

    before = frozen_signature(root)
    dest_root = root / args.dest
    dest_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "module": "Module 5 Stage 2 — corpus acquisition",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "destination": str(dest_root.relative_to(root)),
        "approval_flag": True,
        "corpora": {},
    }

    x_cfg = args.x_fact_config or cfg["corpora"]["x_fact"].get("hf_config", "ta")
    manifest["x_fact_config_selected"] = x_cfg

    if args.only in (None, "x_fact"):
        logger.info("Downloading X-FACT with config=%r", x_cfg)
        manifest["corpora"]["x_fact"] = download_x_fact(dest_root / "x_fact", hf_config=x_cfg)
    if args.only in (None, "fever"):
        manifest["corpora"]["fever_gold_evidence"] = download_fever_gold_evidence(dest_root / "fever_gold_evidence")

    manifest["frozen_modules_1_4"] = assert_frozen_untouched(root, before)

    out_dir = root / cfg["output"]["reports_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    mp = out_dir / "download_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print("MODULE 5 STAGE 2 — DOWNLOAD MANIFEST")
    print("=" * 72)
    for name, rec in manifest["corpora"].items():
        print(f"\n{name}:")
        print(f"  licence : {rec.get('licence')}")
        print(f"  source  : {rec.get('source_url')}")
        if rec.get("hf_config"):
            print(f"  config  : {rec['hf_config']}   (revision: {rec.get('dataset_revision')})")
        if rec.get("evidence_scope"):
            print(f"  scope   : {rec['evidence_scope']}")
        if rec.get("error"):
            print(f"  ERROR   : {rec['error']}")
            continue
        print(f"  splits  : {rec.get('splits_reported_by_source')}")
        for f in rec.get("files", []):
            print(f"    {f['path']}  rows={f['n_rows']}  {f['bytes']:,} B")
            print(f"      sha256 {f['sha256']}")
            print(f"      columns: {f['columns']}")
    fr = manifest["frozen_modules_1_4"]
    print(f"\nModules 1-4 unchanged: {fr['unchanged']}")
    if not fr["unchanged"]:
        print(f"  CHANGED: {fr['changed']}")
    print(f"\nWrote: {mp}")
    print("=" * 72)

    if any(r.get("error") for r in manifest["corpora"].values()) or not fr["unchanged"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
