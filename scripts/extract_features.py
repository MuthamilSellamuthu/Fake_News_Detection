#!/usr/bin/env python
"""Module 3 — CLI entry point for multilingual feature extraction.

Runs `src.embeddings.feature_extraction.run_feature_extraction` end to
end against Module 2's finalized (Version B) train/validation/test
splits, then prints a summary.

Usage
-----
  python scripts/extract_features.py
  python scripts/extract_features.py --config configs/feature_config.yaml
  python scripts/extract_features.py --splits test        # just one split

Requires `torch`, `transformers`, and `sentencepiece` to be installed,
and (unless `model.local_path` is set in the config) network access to
the Hugging Face Hub to download the XLM-RoBERTa checkpoint. If either
is unavailable, run this in the provided Colab notebook instead:
notebooks/module3_colab.ipynb.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.feature_extraction import run_feature_extraction  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.extract_features")


def _attach_module3_log_file(path: Path) -> None:
    """Fixed-name `feature_extraction.log` deliverable, in addition to
    the standard timestamped `logs/run_<timestamp>.log` — same pattern
    as Module 2's `scripts/preprocess_dataset.py`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(formatter)
    logging.getLogger("features").addHandler(handler)
    logging.getLogger("scripts.extract_features").addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 3: extract XLM-RoBERTa embeddings for the Module 2 splits.")
    parser.add_argument("--config", type=str, default=None, help="Path to feature_config.yaml.")
    parser.add_argument("--splits", nargs="+", default=None, choices=["train", "validation", "test"], help="Subset of splits to run (default: all three).")
    args = parser.parse_args()

    _attach_module3_log_file(PROJECT_ROOT / "outputs" / "module3" / "feature_extraction.log")

    logger.info("Starting Module 3 feature extraction pipeline...")
    report = run_feature_extraction(config_path=args.config, splits=args.splits)

    print("\n" + "=" * 72)
    print("MODULE 3 — FEATURE EXTRACTION SUMMARY")
    print("=" * 72)

    # Read defensively via .get(): this summary runs *after* all the GPU
    # work is finished, so a KeyError here would throw away a completed
    # extraction over a cosmetic print. The embeddings and
    # feature_statistics.json are already safely on disk by this point.
    env = report.get("execution_environment", {})
    totals = report.get("totals", {})
    peak = report.get("peak_gpu_memory", {})

    splits_summary = {}
    for name, s in report.get("splits", {}).items():
        if s.get("skipped_already_complete"):
            splits_summary[name] = {"skipped_already_complete": True, "n_rows": s.get("n_rows")}
            continue
        seq = s.get("sequence_length", {})
        rt = s.get("runtime", {})
        splits_summary[name] = {
            "n_source_rows": s.get("n_source_rows"),
            "n_embeddings_generated": s.get("n_embeddings_generated"),
            "n_truncated": seq.get("n_truncated"),
            "pct_truncated": seq.get("pct_truncated"),
            "avg_token_length": seq.get("avg_token_length"),
            "max_token_length": seq.get("max_token_length"),
            "runtime": rt.get("total_human"),
            "docs_per_second": rt.get("throughput_docs_per_second"),
            "storage": s.get("storage", {}).get("human"),
            "n_errors": s.get("n_errors"),
        }

    summary = {
        "status": report.get("status"),
        "model": report.get("model", {}).get("name"),
        "embedding_dim": report.get("model", {}).get("embedding_dim"),
        "device": env.get("device_used"),
        "gpu": env.get("gpu_name") or "none (CPU run)",
        "mixed_precision_active": report.get("extraction_config", {}).get("mixed_precision_active"),
        "peak_gpu_memory": peak.get("peak_allocated_human"),
        "total_documents_embedded": totals.get("total_documents_embedded"),
        "total_embedding_vectors": totals.get("total_embedding_vectors"),
        "total_runtime": totals.get("total_runtime_human"),
        "overall_throughput_docs_per_second": totals.get("overall_throughput_docs_per_second"),
        "total_storage": totals.get("embeddings_dir_storage", {}).get("human"),
        "splits": splits_summary,
    }
    print(json.dumps(summary, indent=2))
    print("=" * 72)


if __name__ == "__main__":
    main()
