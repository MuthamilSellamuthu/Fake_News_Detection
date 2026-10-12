#!/usr/bin/env python
"""Module 9 verification — final PASS/FAIL. CPU.

    python scripts/m9_06_verify_outputs.py --split test
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m9_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m9_config, m9_dir, read_json, write_json)


class Check:
    def __init__(self): self.rows = []; self.warnings = []
    def add(self, name, ok, detail=""):
        s = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.rows.append({"name": name, "status": s, "detail": detail})
        print(f"  {s:4}  {name}" + (f"  |  {detail}" if detail else ""))
    def warn(self, msg):
        self.warnings.append(msg); print(f"  WARN  {msg}")
    def summary(self):
        f = sum(1 for r in self.rows if r["status"] == "FAIL")
        return {"passed": sum(1 for r in self.rows if r["status"] == "PASS"),
                "failed": f, "skipped": sum(1 for r in self.rows if r["status"] == "SKIP"),
                "warnings": self.warnings, "n_checks": len(self.rows),
                "overall": "FAIL" if f else "PASS", "checks": self.rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Module 9 outputs.")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m9_config(args.config, root=root)
    banner(f"VERIFICATION [{args.split}]")
    ck = Check()
    base = m9_dir(cfg, "", root)

    coll = read_json(base / "collected_results.json")
    ck.add("collected_results_exist", coll is not None,
           f"{coll['n_found']} found / {coll['n_missing']} missing" if coll else "run m9_01")
    if coll and coll["n_missing"]:
        ck.warn(f"{coll['n_missing']} upstream artefact(s) missing: "
                f"{[m['label'] for m in coll['missing']][:6]}")

    found = (coll or {}).get("found", {})
    preds = found.get("module7_predictions")
    if preds:
        recs = preds["records"]
        ids = [r["record_id"] for r in recs]
        ck.add("no_duplicate_record_ids", len(set(ids)) == len(ids),
               f"{len(ids) - len(set(ids))} duplicate(s)")
        ck.add("no_missing_predictions",
               all(r.get("final_prediction") is not None for r in recs),
               f"{len(recs)} record(s)")
        probs = [float(r["fused_p_fake"]) for r in recs]
        ck.add("probabilities_valid",
               all(0.0 <= p <= 1.0 and math.isfinite(p) for p in probs))
        langs = {}
        for r in recs:
            langs[r.get("language")] = langs.get(r.get("language"), 0) + 1
        ck.add("language_counts_balance", sum(langs.values()) == len(recs), str(langs))
    else:
        for n in ("no_duplicate_record_ids", "no_missing_predictions",
                  "probabilities_valid", "language_counts_balance"):
            ck.add(n, None, "Module 7 predictions unavailable")

    stat = read_json(base / "statistical_analysis.json")
    if stat:
        bad = []
        for name, t in stat["tests"].items():
            p = t.get("p_value")
            if p is not None and not (0.0 <= p <= 1.0 and math.isfinite(p)):
                bad.append(name)
        ck.add("p_values_valid", not bad, f"{len(stat['tests'])} test(s); bad: {bad}")
        inapplicable = [n for n, t in stat["tests"].items() if not t.get("applicable")]
        ck.add("inapplicable_tests_explained",
               all(stat["tests"][n].get("reason") for n in inapplicable),
               f"{len(inapplicable)} inapplicable test(s), each with a reason")
        boot = stat.get("bootstrap", {})
        bad_ci = [k for k, v in boot.items()
                  if isinstance(v, dict) and v.get("applicable")
                  and isinstance(v.get("ci_lower"), float)
                  and v["ci_lower"] > v["ci_upper"]]
        ck.add("confidence_intervals_valid", not bad_ci, f"bad: {bad_ci}")
    else:
        for n in ("p_values_valid", "inapplicable_tests_explained",
                  "confidence_intervals_valid"):
            ck.add(n, None, "m9_02 not run")

    tdir = m9_dir(cfg, "tables", root)
    present = sorted(p.name for p in tdir.glob("*.csv")) if tdir.exists() else []
    ck.add("tables_generated", bool(present), f"{len(present)}: {present}")

    fdir = m9_dir(cfg, "figures", root)
    fman = read_json(fdir / "figures_manifest.json")
    if fman:
        missing = [f for f in fman["generated"] if not (root / f).is_file()]
        ck.add("figures_correspond_to_files", not missing,
               f"{len(fman['generated'])} generated, {len(fman['skipped'])} skipped")
        if fman["skipped"]:
            ck.warn(f"{len(fman['skipped'])} figure(s) skipped for missing data")
    else:
        ck.add("figures_correspond_to_files", None, "m9_04 not run")

    rep = read_json(base / cfg["output"]["report"])
    if rep:
        audit = rep["leakage_audit"]
        for k in cfg["leakage_audit"]["checks"]:
            if k in audit:
                ck.add(f"leakage::{k}", bool(audit[k]))
        kinds = {f["kind"] for items in rep["findings"].values() for f in items}
        ck.add("findings_are_kind_tagged",
               kinds <= set(cfg["reporting"]["result_kinds"]), str(sorted(kinds)))
        repro = rep.get("reproducibility", {})
        ck.add("reproducibility_recorded",
               bool(repro.get("software")) and "seeds" in repro,
               f"git={repro.get('git_commit')} python={repro.get('software', {}).get('python')}")
        unavailable = [k for k, v in (repro.get("software") or {}).items()
                       if v == "unavailable"]
        if unavailable:
            ck.warn(f"software versions unavailable: {unavailable}")
    else:
        for n in ("findings_are_kind_tagged", "reproducibility_recorded"):
            ck.add(n, None, "m9_05 not run")

    m8 = found.get("module8_summary")
    if m8 and preds:
        ck.add("explanations_correspond_to_predictions",
               (m8.get("overall") or {}).get("n_records", 0) <= len(preds["records"]),
               f"{(m8.get('overall') or {}).get('n_records')} explanations "
               f"<= {len(preds['records'])} predictions")
    else:
        ck.add("explanations_correspond_to_predictions", None,
               "Module 8 or Module 7 outputs unavailable")

    s = ck.summary()
    out = base / "verification_report.json"
    write_json(out, s)
    print("\n" + "=" * 70)
    print(f"MODULE 9 VERIFICATION: {s['overall']}")
    print(f"  passed  : {s['passed']}")
    print(f"  failed  : {s['failed']}")
    print(f"  skipped : {s['skipped']}")
    print(f"  warnings: {len(s['warnings'])}")
    print(f"wrote {out}"); print("=" * 70)
    sys.exit(1 if s["failed"] else 0)


if __name__ == "__main__":
    main()
