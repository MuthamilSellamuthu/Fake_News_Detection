#!/usr/bin/env python
"""Module 8 Stage 1 — select the records to explain. CPU.

Occlusion costs one forward pass per token, so explaining a whole split is not
feasible on free Colab. A deterministic stratified sample is drawn instead and
its size is recorded, so the reported statistics are never mistaken for
full-split coverage.

    python scripts/m8_01_prepare_explanations.py --split validation
"""
from __future__ import annotations

import argparse, csv, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m8_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m8_config, m8_dir, manifest_block, write_json,
    write_manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select records for explanation.")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m8_config(args.config, root=root)
    banner(f"Stage 1 prepare explanations [{args.split}]")

    m7 = root / cfg["inputs"]["module7_root"] / args.split
    fname = "integrated_predictions.json" if args.split == "test" else "fusion_predictions.json"
    pred_path = m7 / fname
    if not pred_path.is_file():
        raise SystemExit(
            f"Module 7 predictions not found: {pred_path}\n"
            f"  Run scripts/m7_05_apply_fusion.py --split {args.split} first.")
    rows = json.loads(pred_path.read_text(encoding="utf-8"))["records"]

    csv.field_size_limit(10 ** 9)
    tp = root / cfg["inputs"]["processed_dir"] / cfg["inputs"]["splits"][args.split]
    texts = {}
    if tp.is_file():
        with tp.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                texts[str(r["record_id"])] = r.get("title") or r.get("text") or ""
    for r in rows:
        r["text"] = texts.get(str(r["record_id"]), "")

    n_target = int(cfg["explanation"]["sample"][args.split])
    seed = int(cfg["explanation"]["sample"]["seed"])
    strata: dict[tuple, list] = {}
    for r in rows:
        strata.setdefault((r.get("language"), int(r["gold_label"])), []).append(r)
    rng = random.Random(seed)
    selected = []
    total = len(rows)
    for key in sorted(strata, key=lambda k: (str(k[0]), k[1])):
        group = sorted(strata[key], key=lambda x: str(x["record_id"]))
        take = max(1, round(n_target * len(group) / total)) if total else 0
        rng.shuffle(group)
        selected.extend(sorted(group[:take], key=lambda x: str(x["record_id"])))
    selected = sorted(selected, key=lambda x: str(x["record_id"]))[:n_target]

    by_lang, by_gold = {}, {}
    for r in selected:
        by_lang[r.get("language")] = by_lang.get(r.get("language"), 0) + 1
        by_gold[int(r["gold_label"])] = by_gold.get(int(r["gold_label"]), 0) + 1
    missing_text = sum(1 for r in selected if not r["text"])

    out = m8_dir(cfg, "explanations", root)
    write_json(out / f"selected_{args.split}.json",
               {"split": args.split, "n_selected": len(selected),
                "n_available": total, "coverage": len(selected) / total if total else 0,
                "by_language": by_lang, "by_gold_label": by_gold,
                "n_missing_text": missing_text, "seed": seed,
                "sampling": "deterministic stratified by (language, gold_label)",
                "records": selected})
    write_manifest(out, manifest_block(
        cfg, 8, "m8_01_prepare_explanations", "project", args.split,
        [str(pred_path.relative_to(root)), str(tp.relative_to(root)) if tp.is_file() else "MISSING"],
        [str((out / f'selected_{args.split}.json').relative_to(root))],
        "complete",
        {"n_selected": len(selected), "n_available": total,
         "by_language": by_lang, "n_missing_text": missing_text}))

    print(f"  available        : {total:,}")
    print(f"  selected         : {len(selected):,} "
          f"({100 * len(selected) / max(total, 1):.1f}% of the split)")
    print(f"  by language      : {by_lang}")
    print(f"  by gold label    : {by_gold}")
    print(f"  missing text     : {missing_text}")
    print(f"\n  wrote -> {out}")


if __name__ == "__main__":
    main()
