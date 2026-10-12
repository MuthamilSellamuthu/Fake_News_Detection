#!/usr/bin/env python
"""Module 8 Stage 4 — faithfulness metrics. GPU (re-encodes perturbed text).

Deletion, insertion, concentration and stability, each defined and reported
separately. No composite score.

    python scripts/m8_04_evaluate_explanations.py --split validation --limit 100
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m8_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m8_config, m8_dir, manifest_block, read_jsonl,
    write_json, write_manifest)
from src.explainability.explanation_metrics import (  # noqa: E402
    summarise_explanations, summarise_faithfulness)
from src.explainability.token_attribution import (  # noqa: E402
    TokenAttribution, attribution_concentration, deletion_test, insertion_test,
    occlusion_attribution, stability_check)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate explanation faithfulness.")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--summary-only", action="store_true",
                    help="Aggregate existing explanations without re-encoding.")
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m8_config(args.config, root=root)
    banner(f"Stage 4 explanation metrics [{args.split}]")

    edir = m8_dir(cfg, "explanations", root)
    epath = edir / cfg["output"]["explanations_filename"].format(split=args.split)
    if not epath.is_file():
        raise SystemExit(f"Run m8_03 first: {epath} not found")
    explanations = read_jsonl(epath)
    summary = summarise_explanations(explanations, cfg["evaluation"]["languages"])
    print(f"  explanations : {summary['overall']['n_records']:,}")
    print(f"  per language : "
          f"{ {k: v['n_records'] for k, v in summary['per_language'].items()} }")

    faith = []
    if not args.summary_only:
        from scripts.m8_02_generate_classifier_explanations import build_text_predict_fn
        predict, _ = build_text_predict_fn(cfg, root)
        f = cfg["faithfulness"]
        rows = [e for e in explanations if e.get("token_attributions")]
        if args.limit:
            rows = rows[:int(args.limit)]
        print(f"  faithfulness on {len(rows):,} record(s) with attributions")
        t0 = time.perf_counter()
        for i, e in enumerate(rows, 1):
            toks = [t["token"] for t in e["token_attributions"]]
            vals = [t["attribution"] for t in e["token_attributions"]]
            att = TokenAttribution(toks, vals, e["explanation_method"],
                                   float(e.get("p_fake", 0.0)))
            item = {"record_id": e["record_id"], "language": e.get("language")}
            if f["deletion"]["enabled"]:
                item["deletion"] = deletion_test(e["text"], att, predict,
                                                 f["deletion"]["fractions"])
            if f["insertion"]["enabled"]:
                item["insertion"] = insertion_test(e["text"], att, predict,
                                                   f["insertion"]["fractions"])
            if f["attribution_concentration"]["enabled"]:
                item["concentration"] = attribution_concentration(
                    vals, f["attribution_concentration"]["top_k"])
            if f["stability"]["enabled"] and i <= 20:
                item["stability"] = stability_check(
                    e["text"],
                    lambda t: occlusion_attribution(
                        t, predict, mask_token=cfg["explanation"]["mask_token"] or "",
                        max_tokens=int(cfg["explanation"]["max_tokens_per_record"]),
                        batch_size=int(cfg["explanation"]["batch_size"])),
                    int(f["stability"]["n_repeats"]))
            faith.append(item)
            if i % 20 == 0:
                print(f"    {i}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)")

    fsum = summarise_faithfulness(faith, cfg["evaluation"]["languages"]) if faith else {
        "overall": {"n_records": 0},
        "note": "faithfulness not computed (summary-only run or no attributions)"}

    mdir = m8_dir(cfg, "metrics", root)
    write_json(mdir / cfg["output"]["metrics_filename"],
               {"split": args.split, "explanation_statistics": summary,
                "faithfulness": fsum, "per_record_faithfulness": faith})
    write_json(mdir / cfg["output"]["summary_filename"],
               {"split": args.split,
                "overall": summary["overall"],
                "english": summary["per_language"].get("en"),
                "tamil": summary["per_language"].get("ta"),
                "language_accounting": summary["language_accounting"],
                "faithfulness_overall": fsum.get("overall"),
                "faithfulness_per_language": fsum.get("per_language"),
                "definitions": fsum.get("definitions")})
    write_manifest(mdir, manifest_block(
        cfg, 8, "m8_04_evaluate_explanations", "project", args.split,
        [str(epath.relative_to(root))],
        [str((mdir / cfg["output"]["metrics_filename"]).relative_to(root))],
        "complete", {"n_explanations": len(explanations), "n_faithfulness": len(faith)}))
    print(f"\n  wrote -> {mdir}")


if __name__ == "__main__":
    main()
