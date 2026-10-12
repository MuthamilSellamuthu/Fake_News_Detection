#!/usr/bin/env python
"""Module 7 Stage 4 — fit fusion parameters on VALIDATION ONLY. CPU.

Every tunable in Module 7 is chosen here: the fusion weight alpha, the decision
threshold, and the calibration temperature. All from the validation split. The
underlying routines raise TestSplitLeakage if handed test, so this cannot be
misused by passing a different flag.

    python scripts/m7_04_fit_fusion.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.m7_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m7_config, m7_dir, reproducibility_block,
    set_all_seeds, write_json, write_manifest)
from src.integration.fusion import (  # noqa: E402
    apply_fusion, fit_binary_temperature, fit_fusion, verifier_to_fake_prob)


def assemble(cfg, root, split):
    """Join Module 4 predictions with verifier output on record_id."""
    m4 = json.loads((m7_dir(cfg, "module4_predictions", root)
                     / f"{split}_predictions.json").read_text(encoding="utf-8"))
    vpath = m7_dir(cfg, "verification", root) / f"{split}_verification.json"
    ver = (json.loads(vpath.read_text(encoding="utf-8"))["records"]
           if vpath.is_file() else [])
    roles = {"fever": cfg["fusion"]["fever_label_roles"],
             "x_fact": cfg["fusion"]["x_fact_label_roles"]}
    by_rec = {}
    for v in ver:
        r = roles[v["dataset"]]
        p = v["probs"]
        by_rec[str(v["record_id"])] = {
            "p_fake_verifier": verifier_to_fake_prob(
                p[r["fake"]], p[r["real"]], p.get(r["neutral"])),
            "verdict": v["verdict"], "probs": p,
            "evidence_ids": v.get("evidence_ids", []),
            "has_evidence": bool(v.get("has_evidence"))}

    rows = []
    for m in m4["records"]:
        v = by_rec.get(str(m["record_id"]))
        rows.append({**m,
                     "p_fake_verifier": v["p_fake_verifier"] if v else 0.5,
                     "has_verifier": v is not None,
                     "has_evidence": bool(v["has_evidence"]) if v else False,
                     "verifier_verdict": v["verdict"] if v else None,
                     "verifier_probs": v["probs"] if v else None,
                     "evidence_ids": v["evidence_ids"] if v else []})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit fusion parameters on validation.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m7_config(args.config, root=root)
    banner("Stage 4 fit fusion (VALIDATION ONLY)")
    set_all_seeds(int(cfg["reproducibility"]["seed"]))

    fus = cfg["fusion"]
    if fus["fit_split"] != "validation":
        raise SystemExit(f"fusion.fit_split must be 'validation', got {fus['fit_split']!r}")

    rows = assemble(cfg, root, "validation")
    n_ver = sum(1 for r in rows if r["has_verifier"])
    print(f"  validation records : {len(rows):,}")
    print(f"  with verifier out  : {n_ver:,} "
          f"({100 * n_ver / max(len(rows), 1):.1f}%)")
    if n_ver == 0:
        print("  NOTE: no verifier output present. Fitting will select alpha=1.0,")
        print("        i.e. the fusion correctly reduces to classifier-only.")

    thr = np.arange(float(fus["threshold_grid_start"]),
                    float(fus["threshold_grid_stop"]) + 1e-9,
                    float(fus["threshold_grid_step"]))
    params, trace = fit_fusion(
        [r["gold_label"] for r in rows], [r["p_fake"] for r in rows],
        [r["p_fake_verifier"] for r in rows], [r["has_verifier"] for r in rows],
        alpha_grid=fus["alpha_grid"], threshold_grid=thr.tolist(),
        split_name="validation", select_on=fus["select_on"])

    if cfg["calibration"]["temperature_scaling"]:
        fused = apply_fusion([r["p_fake"] for r in rows],
                             [r["p_fake_verifier"] for r in rows],
                             [r["has_verifier"] for r in rows], params)
        temp = fit_binary_temperature([r["gold_label"] for r in rows],
                                      fused["p_fake"], split_name="validation")
        params.temperature = temp["temperature"]
        print(f"  temperature        : {temp['temperature']:.4f} "
              f"(NLL {temp['nll_before']:.4f} -> {temp['nll_after']:.4f})")

    out = m7_dir(cfg, "validation", root)
    write_json(out / cfg["output"]["fusion_params_filename"],
               {"parameters": params.to_dict(),
                "grid_trace_size": len(trace),
                "top_10_grid_points": sorted(trace, key=lambda d: -d["score"])[:10],
                "leakage_control": (
                    "Fitted on the Module 2 validation split only. The test split "
                    "was not read by this stage.")})
    write_manifest(out, {"stage": "m7_04_fit_fusion", "status": "complete",
                         "fit_split": "validation",
                         "n_records": len(rows), "n_with_verifier": n_ver,
                         "parameters": params.to_dict(),
                         "reproducibility": reproducibility_block(cfg)})

    print(f"\n  alpha              : {params.alpha}")
    print(f"  threshold          : {params.threshold}")
    print(f"  {fus['select_on']} (val)   : {params.selected_score:.4f}")
    print(f"  note               : {params.notes}")
    print(f"\n  wrote -> {out}")


if __name__ == "__main__":
    main()
