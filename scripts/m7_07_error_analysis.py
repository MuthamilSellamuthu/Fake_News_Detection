#!/usr/bin/env python
"""Module 7 Stage 7 — error categorisation and examples. CPU.

    python scripts/m7_07_error_analysis.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m7_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m7_config, m7_dir, reproducibility_block,
    set_all_seeds, write_json, write_manifest)
from src.integration.error_analysis import categorise_all, sample_examples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Module 7 error analysis.")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m7_config(args.config, root=root)
    banner(f"Stage 7 error analysis [{args.split}]")
    set_all_seeds(int(cfg["reproducibility"]["seed"]))

    fname = "integrated_predictions.json" if args.split == "test" else "fusion_predictions.json"
    p = m7_dir(cfg, args.split, root) / fname
    if not p.is_file():
        raise SystemExit(f"No predictions at {p}. Run m7_05 first.")
    rows = json.loads(p.read_text(encoding="utf-8"))["records"]

    claims = {}
    cpath = m7_dir(cfg, "claims", root) / f"{args.split}_claims.json"
    if cpath.is_file():
        claims = {str(c["record_id"]): c["text"]
                  for c in json.loads(cpath.read_text(encoding="utf-8"))["claims"]}

    for r in rows:
        r["claim_text"] = claims.get(str(r["record_id"]), "")
        vp = r.get("verifier_probs")
        r["verifier_prediction"] = (None if not r.get("has_verifier")
                                    else int(r["p_fake_verifier"] >= 0.5))

    ea = cfg["error_analysis"]
    cat = categorise_all(rows)
    examples = sample_examples(cat, int(ea["max_examples_per_category"]),
                               ea["include_fields"], int(ea["truncate_text_chars"]))

    out = m7_dir(cfg, "error_analysis", root)
    write_json(out / "error_analysis.json",
               {"split": args.split, "counts": cat["counts"],
                "per_language": cat["per_language"],
                "counts_balance": cat["counts_balance"],
                "net_fusion_effect": cat["net_fusion_effect"],
                "interpretation": cat["interpretation"],
                "examples": examples})
    write_manifest(out, {"stage": "m7_07_error_analysis", "split": args.split,
                         "status": "complete", "counts": cat["counts"],
                         "net_fusion_effect": cat["net_fusion_effect"],
                         "reproducibility": reproducibility_block(cfg)})
    print(f"  counts: {cat['counts']}")
    print(f"  net fusion effect (fixed - broke): {cat['net_fusion_effect']}")
    print(f"  balance: {cat['counts_balance']}  ->  {out}")


if __name__ == "__main__":
    main()
