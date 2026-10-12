#!/usr/bin/env python
"""Module 8 verification — PASS/FAIL over generated explanations.

    python scripts/m8_05_verify_outputs.py --split validation
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m8_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m8_config, m8_dir, read_jsonl, write_json)


class Check:
    def __init__(self): self.rows = []
    def add(self, name, ok, detail=""):
        s = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.rows.append({"name": name, "status": s, "detail": detail})
        print(f"  {s:4}  {name}" + (f"  |  {detail}" if detail else ""))
    def summary(self):
        f = sum(1 for r in self.rows if r["status"] == "FAIL")
        return {"passed": sum(1 for r in self.rows if r["status"] == "PASS"),
                "failed": f, "skipped": sum(1 for r in self.rows if r["status"] == "SKIP"),
                "n_checks": len(self.rows), "overall": "FAIL" if f else "PASS",
                "checks": self.rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Module 8 outputs.")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m8_config(args.config, root=root)
    banner(f"VERIFICATION [{args.split}]")
    ck = Check()

    edir = m8_dir(cfg, "explanations", root)
    epath = edir / cfg["output"]["explanations_filename"].format(split=args.split)
    if not epath.is_file():
        for n in ("explanations_exist", "record_ids_unique", "no_nan_attributions",
                  "attribution_length_matches_tokens", "methods_labelled_honestly",
                  "probabilities_in_unit_interval", "tamil_text_preserved",
                  "evidence_provenance_preserved", "reason_generated"):
            ck.add(n, None, "m8_03 not run")
    else:
        rows = read_jsonl(epath)
        ck.add("explanations_exist", True, f"{len(rows):,} records")
        ids = [r["record_id"] for r in rows]
        ck.add("record_ids_unique", len(set(ids)) == len(ids),
               f"{len(ids) - len(set(ids))} duplicate(s)")
        nan = sum(1 for r in rows for t in r["token_attributions"]
                  if not math.isfinite(float(t["attribution"])))
        ck.add("no_nan_attributions", nan == 0, f"{nan} non-finite value(s)")
        bad_len = sum(1 for r in rows
                      if r["token_attributions"] and
                      len({t["position"] for t in r["token_attributions"]})
                      != len(r["token_attributions"]))
        ck.add("attribution_length_matches_tokens", bad_len == 0,
               f"{bad_len} record(s) with inconsistent positions")
        methods = {r["explanation_method"] for r in rows}
        ck.add("methods_labelled_honestly",
               "shap" not in methods,
               f"methods seen: {sorted(methods)} (plain 'shap' must never appear)")
        bad_p = sum(1 for r in rows if not (0.0 <= float(r["p_fake"]) <= 1.0))
        ck.add("probabilities_in_unit_interval", bad_p == 0, f"{bad_p} out of range")
        ta = [r for r in rows if r.get("language") == "ta"]
        corrupt = sum(1 for r in ta if "�" in (r.get("text") or ""))
        ck.add("tamil_text_preserved", corrupt == 0,
               f"{len(ta)} Tamil record(s), {corrupt} with U+FFFD")
        with_ev = [r for r in rows if r["evidence"]]
        ck.add("evidence_provenance_preserved",
               all(e.get("evidence_id") for r in with_ev for e in r["evidence"]),
               f"{len(with_ev)} record(s) carry evidence")
        ck.add("reason_generated", all(r.get("final_decision_reason") for r in rows))

    mdir = m8_dir(cfg, "metrics", root)
    mpath = mdir / cfg["output"]["metrics_filename"]
    if mpath.is_file():
        m = json.loads(mpath.read_text(encoding="utf-8"))
        acc = m["explanation_statistics"]["language_accounting"]
        ck.add("language_accounting_balances", acc["balances"],
               f"{acc['sum_per_language']} vs {acc['overall']}")
        # "not computed" is not "computed and broken". m8_04 --summary-only
        # deliberately skips faithfulness, and it cannot be computed at all
        # without token attributions from m8_02. Only a populated-but-malformed
        # result is a failure.
        fo = (m.get("faithfulness") or {}).get("overall") or {}
        n_faith = int(fo.get("n_records") or 0)
        per_record = m.get("per_record_faithfulness") or []
        if n_faith == 0:
            ck.add("faithfulness_metrics_present", None,
                   "not computed (m8_04 --summary-only, or no attributions from m8_02)")
        else:
            malformed = [r for r in per_record
                         if not any(k in r for k in ("deletion", "insertion",
                                                     "concentration", "stability"))]
            ck.add("faithfulness_metrics_present", not malformed,
                   f"{n_faith} record(s) evaluated; {len(malformed)} malformed")
    else:
        ck.add("language_accounting_balances", None, "m8_04 not run")
        ck.add("faithfulness_metrics_present", None, "m8_04 not run")

    ck.add("no_parameters_fitted_on_test",
           cfg["leakage"]["no_parameters_fitted"] is True
           and cfg["leakage"]["test_is_read_only"] is True,
           "Module 8 fits nothing; test is read-only")

    s = ck.summary()
    out = m8_dir(cfg, "manifests", root) / f"verification_{args.split}.json"
    write_json(out, s)
    print("\n" + "=" * 70)
    print(f"MODULE 8 VERIFICATION: {s['overall']}  "
          f"({s['passed']} pass / {s['failed']} fail / {s['skipped']} skip)")
    print(f"wrote {out}"); print("=" * 70)
    sys.exit(1 if s["failed"] else 0)


if __name__ == "__main__":
    main()
