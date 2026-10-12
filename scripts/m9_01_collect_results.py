#!/usr/bin/env python
"""Module 9 Stage 1 — collect measured results from Modules 4/6/7/8. CPU.

Reads only. Anything absent is recorded as unavailable with the path that was
looked for, so a gap in the paper's tables is traceable to a stage that was
never run rather than to a silent failure here.

    python scripts/m9_01_collect_results.py --split test
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m9_common import (  # noqa: E402
    PROJECT_ROOT, UNAVAILABLE, banner, load_m9_config, m9_dir, manifest_block,
    read_json, write_json, write_manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Module 4/6/7/8 results.")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m9_config(args.config, root=root)
    banner(f"Stage 1 collect results [{args.split}]")
    inp = cfg["inputs"]
    found, missing = {}, []

    def grab(label, rel):
        p = root / rel
        d = read_json(p)
        if d is None:
            missing.append({"label": label, "path": rel})
            print(f"  MISSING  {label:34} {rel}")
        else:
            found[label] = d
            print(f"  found    {label:34} {rel}")
        return d

    grab("module4_metrics", f"{inp['module4_reports']}/evaluation_metrics.json")
    for ds in ("x_fact", "fever"):
        for split in ("validation", "test"):
            for r in ("dense", "bm25"):
                grab(f"retrieval_{ds}_{split}_{r}",
                     f"{inp['module6_root']}/retrieve/{ds}_{split}/{r}_metrics.json")
        grab(f"verification_{ds}",
             f"{inp['module6_root']}/evaluation/{ds}_test/verification_metrics.json")
    grab("module7_metrics", f"{inp['module7_root']}/{args.split}/integrated_metrics.json"
         if args.split == "test" else f"{inp['module7_root']}/{args.split}/integrated_metrics.json")
    grab("module7_predictions",
         f"{inp['module7_root']}/{args.split}/"
         + ("integrated_predictions.json" if args.split == "test" else "fusion_predictions.json"))
    grab("module7_fusion_parameters", f"{inp['module7_root']}/validation/fusion_parameters.json")
    grab("module7_ablations", f"{inp['module7_root']}/ablations/ablation_results.json")
    grab("module7_error_analysis", f"{inp['module7_root']}/error_analysis/error_analysis.json")
    grab("module8_summary", f"{inp['module8_root']}/metrics/explanation_summary.json")
    grab("module8_metrics", f"{inp['module8_root']}/metrics/explanation_metrics.json")

    out = m9_dir(cfg, "", root)
    write_json(out / "collected_results.json",
               {"split": args.split, "n_found": len(found), "n_missing": len(missing),
                "found": found, "missing": missing,
                "policy": "Missing artefacts are listed, never imputed."})
    write_manifest(out, manifest_block(
        cfg, "m9_01_collect_results", args.split,
        [m["path"] for m in missing] + list(found), ["collected_results.json"],
        "complete" if not missing else "partial",
        {"n_found": len(found), "n_missing": len(missing),
         "missing_labels": [m["label"] for m in missing]}))
    print(f"\n  found {len(found)} / missing {len(missing)}")
    print(f"  wrote -> {out/'collected_results.json'}")
    if missing:
        print("\n  Stages that must run before the final report is complete:")
        for m in missing:
            print(f"    - {m['label']}")


if __name__ == "__main__":
    main()
