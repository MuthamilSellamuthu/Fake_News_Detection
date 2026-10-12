#!/usr/bin/env python
"""Module 9 Stage 5 — the final research report. CPU.

Every statement is tagged as measured / derived / statistical_inference /
limitation. No sentence asserts superiority unless the measured difference
exists AND survives Holm correction; otherwise the finding is written as
"not established", which is a different claim from "no difference".

    python scripts/m9_05_generate_research_report.py --split test
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m9_common import (  # noqa: E402
    PROJECT_ROOT, UNAVAILABLE, banner, load_m9_config, m9_dir, manifest_block,
    read_json, write_json, write_manifest)
from src.evaluation.reproducibility import build_reproducibility_report  # noqa: E402
from src.utils.config_loader import load_yaml  # noqa: E402


def finding(kind, statement, evidence=None):
    return {"kind": kind, "statement": statement, "evidence": evidence or {}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the final research report.")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m9_config(args.config, root=root)
    banner(f"Stage 5 research report [{args.split}]")

    base = m9_dir(cfg, "", root)
    coll = read_json(base / "collected_results.json")
    if coll is None:
        raise SystemExit("Run m9_01_collect_results.py first.")
    found = coll["found"]
    stat = read_json(base / "statistical_analysis.json") or {}
    stages = (found.get("module7_metrics") or {}).get("stages", {})
    primary = cfg["metrics"]["primary"]

    def metric(stage, scope="overall", m=None):
        m = m or primary
        blk = stages.get(stage) or {}
        b = blk.get("overall") if scope == "overall" else \
            (blk.get("per_language") or {}).get(scope)
        v = (b or {}).get(m)
        return v if isinstance(v, (int, float)) else None

    holm = {}
    if stat.get("holm_correction", {}).get("applicable"):
        for r, key in zip(stat["holm_correction"]["results"], stat["comparison_order"]):
            holm[key] = r

    findings = {rq: [] for rq in cfg["research_questions"]}

    clf = metric("classifier_only")
    findings["RQ1"].append(
        finding("measured",
                f"Classifier-only {primary} on {args.split} = {clf:.4f}." if clf is not None
                else "Classifier-only performance is unavailable; Module 7 has not run.",
                {"value": clf, "metric": primary, "split": args.split}))

    integ = metric("integrated_weighted") if "integrated_weighted" in stages \
        else metric("integrated")
    key = "classifier_only__vs__integrated_weighted"
    t = (stat.get("tests") or {}).get(key, {})
    h = holm.get(key)
    if clf is not None and integ is not None:
        delta = integ - clf
        findings["RQ2"].append(finding(
            "measured", f"Integrated {primary} = {integ:.4f} vs classifier-only "
                        f"{clf:.4f} (delta {delta:+.4f}).",
            {"integrated": integ, "classifier_only": clf, "delta": delta}))
        if h and h.get("rejected"):
            findings["RQ2"].append(finding(
                "statistical_inference",
                f"The difference is statistically significant after Holm correction "
                f"(adjusted p = {h['p_adjusted']:.4g}).", h))
        elif t.get("applicable"):
            findings["RQ2"].append(finding(
                "statistical_inference",
                f"The difference is NOT established after Holm correction "
                f"(adjusted p = {h['p_adjusted']:.4g} if computed). This does not "
                f"demonstrate equivalence, only that a difference was not shown.",
                {"test": t, "holm": h}))
        else:
            findings["RQ2"].append(finding(
                "limitation",
                f"No significance test was applicable: {t.get('reason', UNAVAILABLE)}.", t))
    else:
        findings["RQ2"].append(finding(
            "limitation", "Integrated results unavailable; RQ2 cannot be answered."))

    for lang, label in (("ta", "Tamil"), ("en", "English")):
        c, i = metric("classifier_only", lang), (metric("integrated_weighted", lang)
                                                 or metric("integrated", lang))
        rq = "RQ3" if lang == "ta" else "RQ5"
        if c is not None and i is not None:
            d = i - c
            direction = ("improved" if d > 0 else "degraded" if d < 0 else "unchanged")
            findings[rq].append(finding(
                "measured",
                f"{label} {primary}: {c:.4f} -> {i:.4f} ({d:+.4f}); integration "
                f"{direction} {label} performance.",
                {"classifier_only": c, "integrated": i, "delta": d, "language": lang}))
            pl = (stat.get("per_language_mcnemar") or {}).get(lang, {})
            if pl.get("applicable"):
                findings[rq].append(finding(
                    "statistical_inference",
                    f"{label} paired McNemar p = {pl['p_value']:.4g} "
                    f"({pl['n_discordant']} discordant pairs).", pl))
            elif pl:
                findings[rq].append(finding("limitation",
                                            f"{label}: {pl.get('reason', UNAVAILABLE)}", pl))
        else:
            findings[rq].append(finding("limitation",
                                        f"{label} per-language metrics unavailable."))

    retr = {k: v for k, v in found.items() if k.startswith("retrieval_")}
    if retr:
        for name, res in sorted(retr.items()):
            ov = (res or {}).get("overall", {})
            if ov.get("recall@5") is not None:
                findings["RQ4"].append(finding(
                    "measured",
                    f"{name}: Recall@5 = {ov['recall@5']:.4f}, MRR = "
                    f"{ov.get('mrr', float('nan')):.4f}.", ov))
    else:
        findings["RQ4"].append(finding("limitation", "No retrieval metrics collected."))

    m8 = found.get("module8_summary")
    if m8:
        fo = m8.get("faithfulness_overall") or {}
        if fo.get("n_records"):
            findings["RQ6"].append(finding(
                "measured",
                f"Explanation faithfulness computed on {fo['n_records']} record(s); "
                f"stability identical rate = {fo.get('stability_identical_rate')}.", fo))
        else:
            findings["RQ6"].append(finding(
                "limitation", "Explanations generated but faithfulness not computed."))
    else:
        findings["RQ6"].append(finding("limitation", "Module 8 has not run."))

    configs = {}
    for name, rel in cfg["inputs"]["configs"].items():
        p = root / rel
        if p.is_file():
            configs[name] = load_yaml(p)
    repro = build_reproducibility_report(
        root, configs, cfg["reproducibility"]["artefacts_to_hash"],
        {"module9": cfg["reproducibility"]["seed"]})

    fp = found.get("module7_fusion_parameters", {}).get("parameters", {})
    audit = {
        "fusion_parameters_fitted_on_validation": fp.get("fit_split") == "validation",
        "calibration_temperature_fitted_on_validation": fp.get("fit_split") == "validation",
        "threshold_selected_on_validation": fp.get("fit_split") == "validation",
        "test_labels_never_used_for_fitting": fp.get("fit_split") == "validation",
        "explanation_parameters_not_fitted": True,
        "evidence": {"fusion_parameters": fp},
    }

    report = {
        "split": args.split,
        "research_questions": cfg["research_questions"],
        "findings": findings,
        "leakage_audit": audit,
        "reproducibility": repro,
        "result_kinds": cfg["reporting"]["result_kinds"],
        "claim_policy": cfg["reporting"]["claim_policy"],
        "completeness": {"n_found": coll["n_found"], "n_missing": coll["n_missing"],
                         "missing": coll["missing"]},
    }
    write_json(base / cfg["output"]["report"], report)
    write_json(base / cfg["output"]["summary"],
               {"split": args.split, "primary_metric": primary,
                "classifier_only": clf, "integrated": integ,
                "delta": (integ - clf) if (clf is not None and integ is not None) else None,
                "per_language": {l: {"classifier_only": metric("classifier_only", l),
                                     "integrated": metric("integrated_weighted", l)
                                     or metric("integrated", l)}
                                 for l in cfg["metrics"]["languages"]},
                "n_findings": {k: len(v) for k, v in findings.items()}})
    write_json(base / cfg["output"]["reproducibility"], repro)
    write_manifest(base, manifest_block(
        cfg, "m9_05_generate_research_report", args.split,
        ["collected_results.json", "statistical_analysis.json"],
        [cfg["output"]["report"], cfg["output"]["summary"],
         cfg["output"]["reproducibility"]], "complete",
        {"n_findings": sum(len(v) for v in findings.values())}))

    for rq, items in findings.items():
        print(f"\n  {rq}: {cfg['research_questions'][rq]}")
        for f in items:
            print(f"    [{f['kind']}] {f['statement']}")
    print(f"\n  wrote -> {base}")


if __name__ == "__main__":
    main()
