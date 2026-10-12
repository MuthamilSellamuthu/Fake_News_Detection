#!/usr/bin/env python
"""Module 6 Stage 8 — run the six configured ablations. Mixed CPU/GPU.

Retrieval ablations (bm25_vs_dense, top_k sweep) read the already-computed
Stage 4 outputs and are pure CPU. Verification ablations need a trained
checkpoint per variant and are therefore reported as pending until those runs
exist -- claiming a number without running the variant would be fabrication.

    python scripts/m6_08_run_ablations.py --dataset fever --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m6_common import (  # noqa: E402
    PROJECT_ROOT, assert_module5_untouched, banner, gold_evidence_map, load_config,
    load_corpus, module5_fingerprint, stage_dir, write_json, write_manifest)
from src.retrieval.retrieval_metrics import evaluate_retrieval  # noqa: E402
from src.verification.ablations import build_ablation_specs, run_all_ablations  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Module 6 ablations.")
    ap.add_argument("--dataset", required=True, choices=["fever", "x_fact"])
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_config(args.config)
    banner(f"Stage 8 ablations [{args.dataset}/{args.split}]", cfg)
    before = module5_fingerprint(root)

    corpus = load_corpus(cfg, args.dataset, root=root)
    gold = gold_evidence_map(corpus)
    ret_dir = stage_dir(cfg, "retrieve", root) / f"{args.dataset}_{args.split}"
    rr_dir = stage_dir(cfg, "rerank", root) / f"{args.dataset}_{args.split}"

    def _load(p: Path):
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    dense = _load(ret_dir / "dense_candidates.json")
    bm25 = _load(ret_dir / "bm25_candidates.json")
    reranked = _load(rr_dir / "reranked_top5.json")

    def score(cands, k):
        if not cands:
            return {"error": "candidates not available; run the earlier stage"}
        m = evaluate_retrieval(cands, gold, (k,), ndcg_k=min(k, 10))
        return {f"recall@{k}": m["overall"][f"recall@{k}"], "mrr": m["overall"]["mrr"],
                "n_scored": m["accounting"]["n_scored"],
                "n_excluded_no_gold": m["accounting"]["n_excluded_no_gold_evidence"]}

    runners = {
        "bm25_vs_dense": lambda v: (
            {**score(bm25 if v["retriever"] == "bm25" else dense, 5),
             "macro_f1": None, "recall@5": score(
                 bm25 if v["retriever"] == "bm25" else dense, 5).get("recall@5")}),
        "top_k_sweep": lambda v: {"recall@k": score(dense, int(v["top_k"])).get(
            f"recall@{int(v['top_k'])}"), **score(dense, int(v["top_k"]))},
        "retrieval_only_vs_retrieval_rerank": lambda v: score(
            reranked if v["rerank"] else dense, 5),
    }

    out = run_all_ablations(cfg, runners, seed=int(cfg["reproducibility"]["seed"]))
    for spec in build_ablation_specs(cfg):
        if spec.name not in runners:
            out.setdefault("pending", []).append({
                "name": spec.name, "question": spec.question,
                "reason": "requires a trained verifier checkpoint per variant; "
                          "run m6_06/m6_07 with the corresponding flags"})

    d = stage_dir(cfg, "ablations", root) / f"{args.dataset}_{args.split}"
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "ablation_results.json", out)
    write_manifest(d, {"stage": "ablations", "dataset": args.dataset, "split": args.split,
                       "status": "complete", "n_run": len(out["ablations"]),
                       "n_pending": len(out.get("pending", [])),
                       "n_skipped": len(out.get("skipped", []))})

    assert_module5_untouched(before, root)
    for name, res in out["ablations"].items():
        print(f"  {name}: best={res['best_variant']} value={res['best_value']} "
              f"delta={res['delta']}")
    for p in out.get("pending", []):
        print(f"  PENDING {p['name']}: {p['reason']}")


if __name__ == "__main__":
    main()
