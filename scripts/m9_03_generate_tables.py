#!/usr/bin/env python
"""Module 9 Stage 3 — CSV + JSON tables for the paper. CPU.

    python scripts/m9_03_generate_tables.py --split test
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.m9_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m9_config, m9_dir, macro_f1_binary, manifest_block,
    read_json, write_json, write_manifest)
from src.evaluation.comparison_tables import (  # noqa: E402
    ablation_table, classification_table, language_comparison_table,
    retrieval_table, statistical_table, write_csv)
from src.evaluation.robustness import (  # noqa: E402
    confidence_robustness, error_breakdown, evidence_availability_robustness, slice_by)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Module 9 tables.")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m9_config(args.config, root=root)
    banner(f"Stage 3 tables [{args.split}]")

    coll = read_json(m9_dir(cfg, "", root) / "collected_results.json")
    if coll is None:
        raise SystemExit("Run m9_01_collect_results.py first.")
    found = coll["found"]
    tdir = m9_dir(cfg, "tables", root)
    written = []

    m7 = found.get("module7_metrics")
    stages = (m7 or {}).get("stages", {})
    if stages:
        rows = classification_table(stages, cfg["metrics"]["classification"], "overall")
        written.append(str(write_csv(tdir / "final_classification_metrics.csv", rows)))
        write_json(tdir / "final_classification_metrics.json", rows)
        lang_rows = []
        for lang in cfg["metrics"]["languages"]:
            lang_rows += classification_table(stages, cfg["metrics"]["classification"], lang)
        lang_rows += language_comparison_table(stages, cfg["metrics"]["primary"],
                                               cfg["metrics"]["languages"])
        written.append(str(write_csv(tdir / "language_comparison.csv", lang_rows)))
        write_json(tdir / "language_comparison.json", lang_rows)
        ab = ablation_table(stages, cfg["metrics"]["primary"])
        written.append(str(write_csv(tdir / "ablation_results.csv", ab)))
        write_json(tdir / "ablation_results.json", ab)
        print(f"  classification/language/ablation tables: {len(rows)} systems")
    else:
        print("  SKIP classification tables: Module 7 metrics unavailable")

    retr = {k.replace("retrieval_", ""): v for k, v in found.items()
            if k.startswith("retrieval_")}
    if retr:
        rows = retrieval_table(retr)
        written.append(str(write_csv(tdir / "retrieval_comparison.csv", rows)))
        write_json(tdir / "retrieval_comparison.json", rows)
        print(f"  retrieval table: {len(rows)} row(s)")
    else:
        print("  SKIP retrieval table: no Module 6 retrieval metrics found")

    ver = {k: v for k, v in found.items() if k.startswith("verification_")}
    if ver:
        rows = [{"dataset": k.replace("verification_", ""),
                 **{m: (v.get("metrics", {}) or {}).get(m, "unavailable")
                    for m in cfg["metrics"]["verification"]}} for k, v in ver.items()]
        written.append(str(write_csv(tdir / "verification_metrics.csv", rows)))
        write_json(tdir / "verification_metrics.json", rows)
    else:
        print("  SKIP verification table: Module 6 Stage 7 not run")

    stat = read_json(m9_dir(cfg, "", root) / "statistical_analysis.json")
    if stat:
        rows = statistical_table(stat["tests"], stat.get("holm_correction"))
        written.append(str(write_csv(tdir / "statistical_tests.csv", rows)))
        write_json(tdir / "statistical_tests.json", rows)
        print(f"  statistical table: {len(rows)} comparison(s)")
    else:
        print("  SKIP statistical table: run m9_02 first")

    preds = found.get("module7_predictions")
    if preds:
        recs = preds["records"]
        rob = {
            "language": slice_by(recs, lambda r: r.get("language"), macro_f1_binary,
                                 min_slice=int(cfg["robustness"]["min_slice_size"])),
            "gold_class": slice_by(recs, lambda r: f"gold_{r['gold_label']}",
                                   macro_f1_binary,
                                   min_slice=int(cfg["robustness"]["min_slice_size"])),
            "evidence_availability": evidence_availability_robustness(
                recs, macro_f1_binary, int(cfg["robustness"]["min_slice_size"])),
            "confidence": confidence_robustness(
                recs, macro_f1_binary,
                float(cfg["robustness"]["confidence"]["high_confidence_threshold"]),
                int(cfg["robustness"]["min_slice_size"])),
            "error_breakdown": error_breakdown(recs)}
        rows = [{"slice_type": k, "slice": s,
                 "applicable": v.get("applicable") if isinstance(v, dict) else "",
                 "n": v.get("n") if isinstance(v, dict) else "",
                 "macro_f1": v.get("value") if isinstance(v, dict) else "",
                 "reason": v.get("reason", "") if isinstance(v, dict) else ""}
                for k, block in rob.items() if isinstance(block, dict)
                for s, v in block.items() if isinstance(v, dict) and "applicable" in v]
        written.append(str(write_csv(tdir / "robustness_results.csv", rows)))
        write_json(tdir / "robustness_results.json", rob)
        print(f"  robustness table: {len(rows)} slice(s)")
    else:
        print("  SKIP robustness: Module 7 predictions unavailable")

    m8 = found.get("module8_summary")
    if m8:
        rows = [{"scope": scope,
                 "n_records": (m8.get(key) or {}).get("n_records", "unavailable"),
                 "failure_rate": (m8.get(key) or {}).get("failure_rate", "unavailable"),
                 "attribution_coverage": (m8.get(key) or {}).get("attribution_coverage",
                                                                 "unavailable")}
                for scope, key in (("overall", "overall"), ("en", "english"),
                                   ("ta", "tamil"))]
        written.append(str(write_csv(tdir / "explainability_results.csv", rows)))
        write_json(tdir / "explainability_results.json", rows)
        print(f"  explainability table: {len(rows)} row(s)")
    else:
        print("  SKIP explainability table: Module 8 not run")

    write_manifest(m9_dir(cfg, "", root), manifest_block(
        cfg, "m9_03_generate_tables", args.split, ["collected_results.json"],
        written, "complete", {"n_tables": len(written)}))
    print(f"\n  wrote {len(written)} table file(s) -> {tdir}")


if __name__ == "__main__":
    main()
