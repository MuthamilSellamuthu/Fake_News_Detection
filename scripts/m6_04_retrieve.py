#!/usr/bin/env python
"""Module 6 Stage 4 — dense and/or BM25 retrieval + real gold metrics. CPU.

Runs the top-50 search, then scores it against Module 5's gold evidence.
Claims with no gold evidence -- FEVER's NOT ENOUGH INFO among them -- are
excluded from recall denominators and counted explicitly, never dropped
silently.

    python scripts/m6_04_retrieve.py --dataset fever --split test --retriever dense
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m6_common import (  # noqa: E402
    PROJECT_ROOT, assert_module5_untouched, banner, gold_evidence_map, load_config,
    load_corpus, module5_fingerprint, read_manifest, stage_dir, write_json, write_manifest)
from src.retrieval.bm25_baseline import BM25Retriever  # noqa: E402
from src.retrieval.encoders import ShardedEncoder  # noqa: E402
from src.retrieval.faiss_index import EvidenceIndex  # noqa: E402
from src.retrieval.index_scope import (  # noqa: E402
    assert_index_within_scope, build_evidence_scope)
from src.retrieval.retrieval_metrics import evaluate_retrieval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieve candidates and score them.")
    ap.add_argument("--dataset", required=True, choices=["fever", "x_fact"])
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--retriever", default="dense", choices=["dense", "bm25", "both"])
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_config(args.config)
    top_k = int(args.top_k or cfg["retrieval"]["top_k_candidates"])
    phase = args.split
    banner(f"Stage 4 retrieve [{args.dataset}/{args.split}] top-{top_k}", cfg)
    before = module5_fingerprint(root)

    corpus = load_corpus(cfg, args.dataset, root=root)
    scope = build_evidence_scope(corpus["claims"], corpus["splits"], phase,
                                 cfg["leakage"]["index_scope"])
    idxs = sorted(corpus["splits"].get(args.split, []))
    claim_ids = [corpus["claim_id_of"][i] for i in idxs]
    claim_texts = [corpus["claims"][i].get("text", "") for i in idxs]
    gold = gold_evidence_map(corpus)
    langs = {corpus["claim_id_of"][i]: str(corpus["claims"][i].get("language", "unknown"))
             for i in idxs}

    out = stage_dir(cfg, "retrieve", root) / f"{args.dataset}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    if args.retriever in ("dense", "both"):
        idx_dir = stage_dir(cfg, "build_index", root) / f"{args.dataset}_{phase}"
        if not (idx_dir / "index_meta.json").is_file():
            raise SystemExit(f"no index at {idx_dir}; run m6_02 first")
        index = EvidenceIndex.load(idx_dir)
        assert_index_within_scope(index.evidence_ids, scope)

        q_dir = stage_dir(cfg, "encode_claims", root) / f"{args.dataset}_{args.split}"
        if read_manifest(q_dir).get("status") != "complete":
            raise SystemExit(f"claim embeddings incomplete at {q_dir}; run m6_03 first")
        q_enc = ShardedEncoder(q_dir)
        queries = q_enc.load_all()
        q_ids = q_enc.load_ids()
        if q_ids != claim_ids:
            raise SystemExit("claim embedding ids do not match the split; re-run m6_03")

        t0 = time.perf_counter()
        ranked, scores = index.search(queries, top_k)
        dense = {cid: list(r) for cid, r in zip(claim_ids, ranked)}
        write_json(out / "dense_candidates.json", dense)
        m = evaluate_retrieval(dense, gold, tuple(k for k in (1, 5, 10, top_k)),
                               ndcg_k=int(cfg["evaluation"]["ndcg_k"]), languages=langs)
        m.pop("per_claim", None)
        write_json(out / "dense_metrics.json", m)
        results["dense"] = {"seconds": round(time.perf_counter() - t0, 2), "metrics": m}
        print(f"  dense  : {m['overall']}")
        print(f"           scored {m['accounting']['n_scored']:,}, "
              f"excluded (no gold) {m['accounting']['n_excluded_no_gold_evidence']:,}")

    if args.retriever in ("bm25", "both"):
        ev_ids = sorted(scope.evidence_ids)
        t0 = time.perf_counter()
        bm = BM25Retriever(ev_ids, [corpus["evidence_texts"].get(e, "") for e in ev_ids],
                           k1=float(cfg["bm25"]["k1"]), b=float(cfg["bm25"]["b"]))
        lex = {cid: bm.retrieve(txt, top_k)[0] for cid, txt in zip(claim_ids, claim_texts)}
        write_json(out / "bm25_candidates.json", lex)
        m = evaluate_retrieval(lex, gold, tuple(k for k in (1, 5, 10, top_k)),
                               ndcg_k=int(cfg["evaluation"]["ndcg_k"]), languages=langs)
        m.pop("per_claim", None)
        write_json(out / "bm25_metrics.json", m)
        results["bm25"] = {"seconds": round(time.perf_counter() - t0, 2), "metrics": m}
        print(f"  bm25   : {m['overall']}")

    write_manifest(out, {"stage": "retrieve", "dataset": args.dataset, "split": args.split,
                         "phase": phase, "top_k": top_k, "status": "complete",
                         "retrievers": sorted(results), "scope": scope.stats,
                         "summary": {k: v["metrics"]["overall"] for k, v in results.items()}})
    assert_module5_untouched(before, root)
    print(f"\n  wrote -> {out}")


if __name__ == "__main__":
    main()
