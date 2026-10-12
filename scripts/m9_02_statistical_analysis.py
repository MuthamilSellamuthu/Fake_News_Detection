#!/usr/bin/env python
"""Module 9 Stage 2 — paired significance tests + Holm correction. CPU.

Paired McNemar over identical test records, bootstrap CIs for metric
differences, and Holm correction across the whole comparison family. A test
whose requirements are unmet reports why instead of a p-value.

    python scripts/m9_02_statistical_analysis.py --split test
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.m9_common import (  # noqa: E402
    PROJECT_ROOT, accuracy, banner, load_m9_config, m9_dir, macro_f1_binary,
    manifest_block, read_json, write_json, write_manifest)
from src.evaluation.confidence_intervals import bootstrap_metric  # noqa: E402
from src.evaluation.statistical_tests import (  # noqa: E402
    holm_correction, mcnemar_test, paired_bootstrap_difference)


def system_predictions(records, name):
    """Predictions for one system, or None when that experiment does not exist."""
    if name == "classifier_only":
        return np.array([int(r["module4_prediction"]) for r in records])
    if name == "integrated_weighted":
        return np.array([int(r["final_prediction"]) for r in records])
    if name == "verifier_only":
        if not any(r.get("has_verifier") for r in records):
            return None
        return np.array([int(float(r.get("p_fake_verifier", 0.5)) >= 0.5)
                         for r in records])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Statistical comparisons.")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m9_config(args.config, root=root)
    banner(f"Stage 2 statistical analysis [{args.split}]")

    coll = read_json(m9_dir(cfg, "", root) / "collected_results.json")
    if coll is None:
        raise SystemExit("Run m9_01_collect_results.py first.")
    preds_doc = coll["found"].get("module7_predictions")
    if preds_doc is None:
        raise SystemExit(
            "Module 7 predictions are unavailable, so no statistical comparison "
            "can be computed. Run m7_05_apply_fusion.py first.")
    records = preds_doc["records"]
    y = np.array([int(r["gold_label"]) for r in records])
    langs = np.array([str(r.get("language")) for r in records])
    print(f"  records: {len(y):,}")

    st = cfg["statistics"]
    tests, order = {}, []
    for a, b in st["comparisons"]:
        pa, pb = system_predictions(records, a), system_predictions(records, b)
        key = f"{a}__vs__{b}"
        order.append(key)
        if pa is None or pb is None:
            missing = a if pa is None else b
            tests[key] = {"test": "mcnemar", "applicable": False,
                          "reason": f"system {missing!r} has no predictions "
                                    f"(experiment not run)"}
            print(f"  SKIP  {key}: {missing} unavailable")
            continue
        tests[key] = mcnemar_test(y, pa, pb, int(st["min_discordant_pairs"]), a, b)
        r = tests[key]
        print(f"  {'OK  ' if r['applicable'] else 'SKIP'}  {key}: "
              + (f"p={r['p_value']:.4g} discordant={r['n_discordant']}"
                 if r["applicable"] else r["reason"]))

    corrected = holm_correction([tests[k].get("p_value") for k in order],
                                float(st["alpha"]))

    boot = {}
    if st["bootstrap"]["enabled"]:
        b = st["bootstrap"]
        for name in ("classifier_only", "integrated_weighted", "verifier_only"):
            p = system_predictions(records, name)
            if p is None:
                boot[name] = {"applicable": False, "reason": "experiment not run"}
                continue
            boot[name] = {
                "macro_f1": bootstrap_metric(y, p, macro_f1_binary,
                                             int(b["n_resamples"]), int(b["seed"]),
                                             float(b["confidence_level"])),
                "accuracy": bootstrap_metric(y, p, accuracy, int(b["n_resamples"]),
                                             int(b["seed"]), float(b["confidence_level"]))}
        pa = system_predictions(records, "classifier_only")
        pb = system_predictions(records, "integrated_weighted")
        if pa is not None and pb is not None:
            boot["difference_integrated_minus_classifier"] = paired_bootstrap_difference(
                y, pb, pa, macro_f1_binary, int(b["n_resamples"]), int(b["seed"]),
                float(b["confidence_level"]))

    per_lang = {}
    for lang in cfg["metrics"]["languages"]:
        m = langs == lang
        if m.sum() < 20:
            per_lang[lang] = {"applicable": False, "n": int(m.sum()),
                              "reason": "fewer than 20 records"}
            continue
        pa, pb = system_predictions(records, "classifier_only"), \
                 system_predictions(records, "integrated_weighted")
        per_lang[lang] = mcnemar_test(y[m], pa[m], pb[m],
                                      int(st["min_discordant_pairs"]),
                                      "classifier_only", "integrated_weighted")

    out = m9_dir(cfg, "", root)
    write_json(out / "statistical_analysis.json",
               {"split": args.split, "comparison_order": order, "tests": tests,
                "holm_correction": corrected, "bootstrap": boot,
                "per_language_mcnemar": per_lang,
                "policy": ("A difference is reported as significant only if the "
                           "Holm-adjusted p-value is below alpha.")})
    write_manifest(out, manifest_block(
        cfg, "m9_02_statistical_analysis", args.split, ["collected_results.json"],
        ["statistical_analysis.json"], "complete",
        {"n_comparisons": len(tests),
         "n_applicable": sum(1 for t in tests.values() if t.get("applicable"))}))
    print(f"\n  wrote -> {out/'statistical_analysis.json'}")


if __name__ == "__main__":
    main()
