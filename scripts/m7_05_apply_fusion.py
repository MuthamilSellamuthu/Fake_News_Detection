#!/usr/bin/env python
"""Module 7 Stage 5 — apply FROZEN fusion parameters. CPU.

Contains no fitting. Reads the parameters written by m7_04 and applies them
unchanged, which is what makes the test evaluation honest.

    python scripts/m7_05_apply_fusion.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m7_04_fit_fusion import assemble  # noqa: E402
from scripts.m7_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m7_config, m7_dir, reproducibility_block,
    set_all_seeds, write_json, write_manifest)
from src.integration.fusion import FusionParameters, apply_fusion  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply frozen fusion parameters.")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m7_config(args.config, root=root)
    banner(f"Stage 5 apply fusion [{args.split}]")
    set_all_seeds(int(cfg["reproducibility"]["seed"]))

    pfile = m7_dir(cfg, "validation", root) / cfg["output"]["fusion_params_filename"]
    if not pfile.is_file():
        raise SystemExit(f"No fusion parameters at {pfile}. Run m7_04 first.")
    params = FusionParameters.from_dict(
        json.loads(pfile.read_text(encoding="utf-8"))["parameters"])
    if params.fit_split != "validation":
        raise SystemExit(f"parameters were fitted on {params.fit_split!r}, not validation")
    print(f"  frozen params: alpha={params.alpha} threshold={params.threshold} "
          f"T={params.temperature:.4f} (fitted on {params.fit_split})")

    rows = assemble(cfg, root, args.split)
    res = apply_fusion([r["p_fake"] for r in rows], [r["p_fake_verifier"] for r in rows],
                       [r["has_verifier"] for r in rows], params)
    for r, pf, pred in zip(rows, res["p_fake"], res["prediction"]):
        r["module4_prediction"] = int(r["prediction"])
        r["module4_p_fake"] = float(r["p_fake"])
        r["fused_p_fake"] = float(pf)
        r["fused_p_real"] = float(1.0 - pf)
        r["final_prediction"] = int(pred)

    out = m7_dir(cfg, args.split, root)
    write_json(out / ("integrated_predictions.json" if args.split == "test"
                      else "fusion_predictions.json"),
               {"split": args.split, "parameters": params.to_dict(), "records": rows})
    write_manifest(out, {"stage": "m7_05_apply_fusion", "split": args.split,
                         "status": "complete", "n_records": len(rows),
                         "n_with_verifier": sum(1 for r in rows if r["has_verifier"]),
                         "parameters_used": params.to_dict(),
                         "fitting_performed": False,
                         "reproducibility": reproducibility_block(cfg)})
    print(f"  records: {len(rows):,}  ->  {out}")


if __name__ == "__main__":
    main()
