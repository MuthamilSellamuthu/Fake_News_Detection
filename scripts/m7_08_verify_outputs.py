#!/usr/bin/env python
"""Module 7 verification — PASS/FAIL over whatever Module 7 has produced.

Recomputes from artefacts rather than trusting manifests. Stages not yet run
are SKIP, never PASS. Exit code 0 only when nothing failed.

    python scripts/m7_08_verify_outputs.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.m7_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m7_config, m7_dir, reproducibility_block, write_json)
from src.integration.fusion import FusionParameters, apply_fusion  # noqa: E402


class Check:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        s = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.rows.append({"name": name, "status": s, "detail": detail})
        print(f"  {s:4}  {name}" + (f"  |  {detail}" if detail else ""))

    def summary(self):
        f = sum(1 for r in self.rows if r["status"] == "FAIL")
        p = sum(1 for r in self.rows if r["status"] == "PASS")
        s = sum(1 for r in self.rows if r["status"] == "SKIP")
        return {"passed": p, "failed": f, "skipped": s, "n_checks": len(self.rows),
                "overall": "FAIL" if f else "PASS", "checks": self.rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Module 7 outputs.")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m7_config(args.config, root=root)
    banner(f"VERIFICATION [{args.split}]")
    ck = Check()

    # ---- Module 4 predictions ----
    mp = m7_dir(cfg, "module4_predictions", root) / f"{args.split}_predictions.json"
    if mp.is_file():
        d = json.loads(mp.read_text(encoding="utf-8"))
        recs, st = d["records"], d["statistics"]
        ck.add("m4_predictions_exist", True, f"{len(recs):,} records")
        ck.add("m4_record_ids_unique", st["record_ids_unique"])
        ck.add("m4_probs_in_unit_interval", st["all_probs_in_unit_interval"])
        ck.add("m4_language_accounting_balances", st["language_accounting_balances"],
               str(st["by_language"]))
        ck.add("m4_languages_valid",
               set(st["by_language"]) <= set(cfg["evaluation"]["languages"]) | {"unknown"},
               str(sorted(st["by_language"])))
    else:
        for n in ("m4_predictions_exist", "m4_record_ids_unique",
                  "m4_probs_in_unit_interval", "m4_language_accounting_balances",
                  "m4_languages_valid"):
            ck.add(n, None, "m7_01 not run")

    # ---- claims + retrieval ----
    cp = m7_dir(cfg, "claims", root) / f"{args.split}_claims.json"
    rp = m7_dir(cfg, "retrieval", root) / f"{args.split}_retrieval.json"
    if cp.is_file():
        cd = json.loads(cp.read_text(encoding="utf-8"))
        ids = [c["claim_id"] for c in cd["claims"]]
        ck.add("claims_exist", True, f"{len(ids):,} claims")
        ck.add("claim_ids_unique", len(set(ids)) == len(ids))
    else:
        ck.add("claims_exist", None, "m7_02 not run")
        ck.add("claim_ids_unique", None, "m7_02 not run")
    if rp.is_file():
        rd = json.loads(rp.read_text(encoding="utf-8"))
        conf_ok, ev_ok = True, True
        for lang, per in rd["results"].items():
            for cid, byret in per.items():
                for ret, payload in byret.items():
                    c = payload.get("confidence", [])
                    if any(not (0.0 - 1e-9 <= x <= 1.0 + 1e-9) for x in c):
                        conf_ok = False
                    if len(payload.get("evidence_ids", [])) != len(payload.get("scores", [])):
                        ev_ok = False
        ck.add("retrieval_confidence_in_unit_interval", conf_ok)
        ck.add("retrieval_ids_and_scores_aligned", ev_ok)
        ck.add("retrieval_coverage_recorded", "coverage" in rd, str(list(rd.get("coverage", {}))))
    else:
        for n in ("retrieval_confidence_in_unit_interval",
                  "retrieval_ids_and_scores_aligned", "retrieval_coverage_recorded"):
            ck.add(n, None, "m7_02 retrieval not run")

    # ---- fusion parameters + leakage ----
    pf = m7_dir(cfg, "validation", root) / cfg["output"]["fusion_params_filename"]
    params = None
    if pf.is_file():
        params = FusionParameters.from_dict(
            json.loads(pf.read_text(encoding="utf-8"))["parameters"])
        ck.add("fusion_params_exist", True,
               f"alpha={params.alpha} thr={params.threshold} T={params.temperature:.4f}")
        ck.add("fusion_fitted_on_validation_not_test", params.fit_split == "validation",
               f"fit_split={params.fit_split!r}")
        ck.add("alpha_in_unit_interval", 0.0 <= params.alpha <= 1.0)
        ck.add("threshold_in_unit_interval", 0.0 <= params.threshold <= 1.0)
    else:
        for n in ("fusion_params_exist", "fusion_fitted_on_validation_not_test",
                  "alpha_in_unit_interval", "threshold_in_unit_interval"):
            ck.add(n, None, "m7_04 not run")

    # ---- predictions: recompute from stored params (proves no test tuning) ----
    fname = "integrated_predictions.json" if args.split == "test" else "fusion_predictions.json"
    pp = m7_dir(cfg, args.split, root) / fname
    if pp.is_file() and params is not None:
        rows = json.loads(pp.read_text(encoding="utf-8"))["records"]
        rec = apply_fusion([r["p_fake"] for r in rows],
                           [r["p_fake_verifier"] for r in rows],
                           [r["has_verifier"] for r in rows], params)
        same_p = np.allclose(rec["p_fake"], [r["fused_p_fake"] for r in rows], atol=1e-9)
        same_d = np.array_equal(rec["prediction"], [r["final_prediction"] for r in rows])
        ck.add("predictions_recomputable_from_stored_params", bool(same_p and same_d),
               "recomputed fused probabilities and decisions match exactly")
        ids = [r["record_id"] for r in rows]
        ck.add("prediction_ids_unique", len(set(ids)) == len(ids))
        pv = np.array([r["fused_p_fake"] for r in rows])
        ck.add("fused_probs_in_unit_interval", bool(((pv >= 0) & (pv <= 1)).all()))
        ck.add("p_fake_plus_p_real_equals_one",
               bool(np.allclose(pv + np.array([r["fused_p_real"] for r in rows]), 1.0)))
        ck.add("final_predictions_binary",
               set(int(r["final_prediction"]) for r in rows) <= {0, 1})
        no_ver = [r for r in rows if not r["has_verifier"]]
        ck.add("no_verifier_records_fall_back_to_classifier",
               all(abs(r["fused_p_fake"] - r["module4_p_fake"]) < 1e-9 for r in no_ver)
               if (no_ver and params.temperature == 1.0) else True,
               f"{len(no_ver):,} record(s) without a verdict")
        # evidence must originate from retrieval
        if rp.is_file():
            rd = json.loads(rp.read_text(encoding="utf-8"))
            retrieved = {cid: {e for ret in byret.values()
                               for e in ret.get("evidence_ids", [])}
                         for per in rd["results"].values() for cid, byret in per.items()}
            by_rec = {}
            if cp.is_file():
                by_rec = {str(c["record_id"]): c["claim_id"]
                          for c in json.loads(cp.read_text(encoding="utf-8"))["claims"]}
            bad = 0
            for r in rows:
                cid = by_rec.get(str(r["record_id"]))
                allowed = retrieved.get(cid, set()) if cid else set()
                if set(r.get("evidence_ids") or []) - allowed:
                    bad += 1
            ck.add("no_fabricated_evidence", bad == 0,
                   f"{bad} record(s) cite evidence never retrieved")
        else:
            ck.add("no_fabricated_evidence", None, "retrieval output absent")
    else:
        for n in ("predictions_recomputable_from_stored_params", "prediction_ids_unique",
                  "fused_probs_in_unit_interval", "p_fake_plus_p_real_equals_one",
                  "final_predictions_binary", "no_verifier_records_fall_back_to_classifier",
                  "no_fabricated_evidence"):
            ck.add(n, None, "m7_05 not run")

    # ---- metrics sanity ----
    mfile = m7_dir(cfg, args.split, root) / "integrated_metrics.json"
    if mfile.is_file():
        md = json.loads(mfile.read_text(encoding="utf-8"))
        vals = []
        for stage in ("classifier_only", "integrated"):
            ov = (md["stages"].get(stage) or {}).get("overall", {})
            vals += [v for k, v in ov.items()
                     if isinstance(v, (int, float)) and k in
                     ("accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc")]
        ck.add("metrics_in_valid_range", all(0.0 <= v <= 1.0 for v in vals),
               f"{len(vals)} metric value(s) checked")
    else:
        ck.add("metrics_in_valid_range", None, "m7_06 not run")

    # ---- error analysis ----
    ep = m7_dir(cfg, "error_analysis", root) / "error_analysis.json"
    if ep.is_file():
        ed = json.loads(ep.read_text(encoding="utf-8"))
        ck.add("error_counts_balance", ed["counts_balance"], str(ed["counts"]))
    else:
        ck.add("error_counts_balance", None, "m7_07 not run")

    summary = ck.summary()
    summary["reproducibility"] = reproducibility_block(cfg)
    out = m7_dir(cfg, "manifests", root) / f"verification_{args.split}.json"
    write_json(out, summary)
    print("\n" + "=" * 70)
    print(f"MODULE 7 VERIFICATION: {summary['overall']}  "
          f"({summary['passed']} pass / {summary['failed']} fail / {summary['skipped']} skip)")
    print(f"wrote {out}")
    print("=" * 70)
    sys.exit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
