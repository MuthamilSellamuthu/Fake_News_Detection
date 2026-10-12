#!/usr/bin/env python
"""Module 6 Stage 5 — cross-encoder reranking. GPU. Resumable.

Reranks the top-50 candidates down to top-5. The candidate set is never
widened: `rerank_all()` asserts the output is a subset of what retrieval
proposed, because inventing evidence would turn retrieval misses into
apparent hits and inflate every downstream number.

Sequence length is 256, not the originally configured 512, and only the
evaluation splits plus a documented train subset are reranked. Both are
free-Colab compute reductions and are recorded in the manifest.

    python scripts/m6_05_rerank.py --dataset fever --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m6_common import (  # noqa: E402
    PROJECT_ROOT, assert_module5_untouched, banner, detect_device, gold_evidence_map,
    load_config, load_corpus, module5_fingerprint, stage_dir, write_json, write_manifest)
from src.retrieval.reranker import (  # noqa: E402
    CrossEncoderReranker, build_cross_encoder_score_fn, load_reranked)
from src.retrieval.retrieval_metrics import evaluate_retrieval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Rerank retrieved candidates.")
    ap.add_argument("--dataset", required=True, choices=["fever", "x_fact"])
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--candidates", default="dense", choices=["dense", "bm25"])
    ap.add_argument("--checkpoint", default=None,
                    help="Fine-tuned cross-encoder. Without one the scores are "
                         "from an untrained head and are NOT meaningful.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap claims reranked (train subset). Recorded in the manifest.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_config(args.config)
    rr = cfg["reranking"]
    banner(f"Stage 5 rerank [{args.dataset}/{args.split}]", cfg)
    before = module5_fingerprint(root)

    cand_path = (stage_dir(cfg, "retrieve", root) / f"{args.dataset}_{args.split}"
                 / f"{args.candidates}_candidates.json")
    if not cand_path.is_file():
        raise SystemExit(f"no candidates at {cand_path}; run m6_04 first")
    candidates = json.loads(cand_path.read_text(encoding="utf-8"))

    corpus = load_corpus(cfg, args.dataset, root=root)
    claims = {corpus["claim_id_of"][i]: corpus["claims"][i].get("text", "")
              for i in corpus["splits"].get(args.split, [])}

    limit = args.limit
    if limit is None and args.split == "train":
        limit = int(rr.get("train_subset_size") or 0) or None
    if limit:
        keep = sorted(candidates)[:limit]          # deterministic subset
        candidates = {k: candidates[k] for k in keep}
    print(f"  claims to rerank: {len(candidates):,}"
          + (f"  (limited from the full split)" if limit else ""))

    out = stage_dir(cfg, "rerank", root) / f"{args.dataset}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        write_manifest(out, {"stage": "rerank", "dataset": args.dataset, "split": args.split,
                             "status": "dry_run", "n_claims": len(candidates)})
        print("  DRY RUN: no model loaded.")
        assert_module5_untouched(before, root)
        return

    if not args.checkpoint:
        print("  WARNING: no --checkpoint given. An untrained cross-encoder head "
              "produces arbitrary scores; these results are not reportable.")

    reranker = CrossEncoderReranker(
        score_fn=build_cross_encoder_score_fn(
            args.checkpoint or rr["cross_encoder"], max_length=int(rr["max_length"]),
            device=detect_device()),
        top_n=int(rr["top_n_after_rerank"]), batch_size=int(rr["batch_size"]))
    man = reranker.rerank_all(claims, candidates, corpus["evidence_texts"], out)

    merged = load_reranked(out)
    top5 = {cid: v["evidence_ids"] for cid, v in merged.items()}
    write_json(out / "reranked_top5.json", top5)

    gold = gold_evidence_map(corpus)
    langs = {corpus["claim_id_of"][i]: str(corpus["claims"][i].get("language", "unknown"))
             for i in corpus["splits"].get(args.split, [])}
    m = evaluate_retrieval(top5, gold, (1, 5), ndcg_k=5, languages=langs)
    m.pop("per_claim", None)
    write_json(out / "rerank_metrics.json", m)

    man.update({"stage": "rerank", "dataset": args.dataset, "split": args.split,
                "candidates_from": args.candidates, "max_length": rr["max_length"],
                "checkpoint": args.checkpoint, "limited_to": limit,
                "compute_note": "max_length 256 (config original 512); "
                                "train reranking limited to a documented subset",
                "metrics": m["overall"], "status": "complete"})
    write_manifest(out, man)
    assert_module5_untouched(before, root)
    print(f"  rerank metrics: {m['overall']}")
    print(f"  wrote -> {out}")


if __name__ == "__main__":
    main()
