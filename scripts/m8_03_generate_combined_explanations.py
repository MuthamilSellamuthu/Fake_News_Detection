#!/usr/bin/env python
"""Module 8 Stage 3 — assemble combined explanations. CPU.

Joins classifier attributions, retrieved evidence and verifier probabilities
into one record per item, and generates the decision sentence from the actual
values. Runs without a verifier: records simply carry has_verification=false.

    python scripts/m8_03_generate_combined_explanations.py --split validation
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m8_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m8_config, m8_dir, manifest_block, read_jsonl,
    write_json, write_jsonl, write_manifest)
from src.explainability.combined_explanation import (  # noqa: E402
    build_explanation_record, render_text_report)
from src.explainability.token_attribution import TokenAttribution  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble combined explanations.")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m8_config(args.config, root=root)
    banner(f"Stage 3 combined explanations [{args.split}]")

    edir = m8_dir(cfg, "explanations", root)
    sel = json.loads((edir / f"selected_{args.split}.json").read_text(encoding="utf-8"))
    records = {r["record_id"]: r for r in sel["records"]}

    att_path = edir / f"attributions_{args.split}.jsonl"
    atts = {a["record_id"]: a for a in read_jsonl(att_path)} if att_path.is_file() else {}
    print(f"  records      : {len(records):,}")
    print(f"  attributions : {len(atts):,}"
          + ("  (m8_02 not run; explanations will have no token attributions)"
             if not atts else ""))

    m7root = root / cfg["inputs"]["module7_root"]
    vpath = m7root / "verification" / f"{args.split}_verification.json"
    ver = ({v["record_id"]: v for v in
            json.loads(vpath.read_text(encoding="utf-8"))["records"]}
           if vpath.is_file() else {})
    rpath = m7root / "retrieval" / f"{args.split}_retrieval.json"
    retr = json.loads(rpath.read_text(encoding="utf-8")) if rpath.is_file() else {"results": {}}
    ev_text: dict[str, str] = {}
    try:
        from scripts.m6_common import load_config as load_m6, load_corpus
        m6cfg = load_m6(root / cfg["inputs"]["module6_config"])
        for ds in ("fever", "x_fact"):
            try:
                ev_text.update(load_corpus(m6cfg, ds, root=root)["evidence_texts"])
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    print(f"  verification : {len(ver):,}   evidence texts: {len(ev_text):,}")

    params_path = m7root / "validation" / "fusion_parameters.json"
    alpha = 1.0
    if params_path.is_file():
        alpha = float(json.loads(params_path.read_text(encoding="utf-8"))
                      ["parameters"]["alpha"])
    print(f"  fusion alpha : {alpha}")

    out_rows, reports = [], []
    for rid in sorted(records):
        rec = records[rid]
        a = atts.get(rid)
        att_obj = None
        if a and a.get("tokens"):
            att_obj = TokenAttribution(a["tokens"], a["attributions"],
                                       a.get("method", "occlusion"),
                                       float(a.get("base_probability") or 0.0))
            att_obj.truncated = bool(a.get("truncated"))
            att_obj.seconds = float(a.get("seconds") or 0.0)
            att_obj.error = a.get("error")
        v = ver.get(rid)
        evidence = [{"evidence_id": e, "text": ev_text.get(e, "")}
                    for e in (v.get("evidence_ids") if v else [])]
        conf = None
        for lang_res in retr.get("results", {}).values():
            hit = lang_res.get(rec.get("claim_id", ""), {})
            for payload in hit.values():
                conf = payload.get("confidence")
                break
        row = build_explanation_record(
            rec, att_obj, (v or {}).get("probs"), evidence, conf, alpha,
            method_label=(a.get("method") if a else "none"),
            top_k=int(cfg["explanation"]["top_k_tokens"]),
            max_evidence=int(cfg["verification_explanation"]["max_evidence_shown"]),
            truncate_chars=int(cfg["verification_explanation"]["truncate_evidence_chars"]))
        out_rows.append(row)
        if len(reports) < 20:
            reports.append(render_text_report(row))

    path = edir / cfg["output"]["explanations_filename"].format(split=args.split)
    write_jsonl(path, out_rows)
    (edir / f"sample_reports_{args.split}.txt").write_text(
        "\n\n" + ("\n\n" + "-" * 70 + "\n\n").join(reports), encoding="utf-8")
    write_manifest(edir, manifest_block(
        cfg, 8, "m8_03_generate_combined_explanations", "project", args.split,
        [str(att_path.relative_to(root)) if att_path.is_file() else "attributions:MISSING",
         str(vpath.relative_to(root)) if vpath.is_file() else "verification:MISSING"],
        [str(path.relative_to(root))], "complete",
        {"n_explanations": len(out_rows),
         "n_with_attributions": sum(1 for r in out_rows if r["token_attributions"]),
         "n_with_verification": sum(1 for r in out_rows if r["verification_probabilities"]),
         "fusion_alpha": alpha}))
    print(f"\n  explanations : {len(out_rows):,}  ->  {path}")


if __name__ == "__main__":
    main()
