#!/usr/bin/env python
"""Module 7 Stage 2 — claims from project articles, then retrieval. GPU/CPU.

Module 6 retrieved only for FEVER and X-FACT claims; it never touched the
project's own news articles. So Module 7 runs its own retrieval, reusing
Module 5's claim selector and Module 6's encoder/index/BM25 unchanged.

Unlike m6_04, this stage PERSISTS retrieval scores, because fusion may use a
retrieval confidence. Raw scores are not comparable across retrievers -- dense
is cosine in [-1, 1], BM25 is unbounded -- so a per-query rank-normalised
confidence in [0, 1] is stored alongside them.

Evidence pool per language: English claims are retrieved against FEVER, Tamil
against X-FACT, as configured. This is a genuine domain mismatch (news
articles vs fact-checking corpora) and is recorded, not hidden.

    python scripts/m7_02_prepare_claims_and_retrieve.py --split validation
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.m6_common import load_config as load_m6_config  # noqa: E402
from scripts.m6_common import load_corpus, stage_dir  # noqa: E402
from scripts.m7_common import (  # noqa: E402
    PROJECT_ROOT, banner, detect_device, load_m7_config, m7_dir,
    reproducibility_block, set_all_seeds, write_json, write_manifest)
from src.fact_verification.claim_selection import select_claims  # noqa: E402
from src.retrieval.bm25_baseline import BM25Retriever  # noqa: E402
from src.retrieval.faiss_index import EvidenceIndex  # noqa: E402
from src.retrieval.index_scope import build_evidence_scope  # noqa: E402
from src.integration.fusion import rank_normalise  # noqa: E402


def read_articles(root: Path, cfg: dict, split: str) -> list[dict]:
    """Module 2 rows for one split. Read-only."""
    import csv
    csv.field_size_limit(10 ** 9)
    p = root / cfg["inputs"]["processed_dir"] / cfg["inputs"]["splits"][split]
    if not p.is_file():
        raise SystemExit(f"Module 2 split not found: {p}")
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Select project claims and retrieve evidence.")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--retriever", default="both", choices=["dense", "bm25", "both"])
    ap.add_argument("--limit", type=int, default=None, help="Debug only; recorded.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--claims-only", action="store_true",
                    help="Stop after claim selection (no model, no retrieval).")
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m7_config(args.config, root=root)
    m6cfg = load_m6_config(root / cfg["inputs"]["module6_config"])
    banner(f"Stage 2 claims + retrieval [{args.split}]")
    set_all_seeds(int(cfg["reproducibility"]["seed"]))

    # ---- claim selection: Module 5's implementation, reused unchanged -------
    m5_cfg = {"claim_selection": load_m6_config(
                  root / m6cfg["inputs"]["module5_config"])["claim_selection"],
              "normalization": load_m6_config(
                  root / m6cfg["inputs"]["module5_config"])["normalization"]}
    articles = read_articles(root, cfg, args.split)
    if args.limit:
        articles = articles[:int(args.limit)]
    claims, sel_stats = select_claims(articles, m5_cfg,
                                      dataset=cfg["claim_selection"]["dataset_tag"])
    print(f"  articles         : {sel_stats['n_input_records']:,}")
    print(f"  claims selected  : {sel_stats['n_claims_selected']:,}")
    print(f"  dropped (short)  : {sel_stats.get('n_dropped_below_min_chars', 0):,}")
    print(f"  by language      : {sel_stats.get('by_language')}")

    out = m7_dir(cfg, "claims", root)
    write_json(out / f"{args.split}_claims.json", {
        "split": args.split, "statistics": sel_stats,
        "claims": [{"claim_id": c.claim_id, "record_id": c.record_id,
                    "text": c.text, "language": c.language,
                    "label": c.label, "selection_rule": c.selection_rule}
                   for c in claims]})
    write_manifest(out, {"stage": "m7_02_claims", "split": args.split,
                         "status": "complete", "statistics": sel_stats,
                         "limit": args.limit,
                         "claim_selection": "src/fact_verification/claim_selection.py (reused)",
                         "reproducibility": reproducibility_block(cfg)})
    if args.claims_only:
        print(f"\n  claims-only run. wrote -> {out}")
        return

    # ---- retrieval, per language against its configured evidence pool ------
    ret = cfg["retrieval"]
    pools = cfg["inputs"]["evidence_datasets"]
    results: dict[str, dict] = {}
    coverage: dict[str, dict] = {}

    for lang, dataset in pools.items():
        subset = [c for c in claims if c.language == lang]
        if not subset:
            coverage[lang] = {"n_claims": 0, "note": "no claims in this language"}
            continue
        corpus = load_corpus(m6cfg, dataset, root=root)
        scope = build_evidence_scope(corpus["claims"], corpus["splits"],
                                     ret["phase"], m6cfg["leakage"]["index_scope"])
        ev_ids = sorted(scope.evidence_ids)
        print(f"\n  [{lang}] {len(subset):,} claims vs {dataset} pool "
              f"({len(ev_ids):,} evidence items)")

        per_claim: dict[str, dict] = {}
        t0 = time.perf_counter()

        if args.retriever in ("bm25", "both"):
            bm = BM25Retriever(ev_ids, [corpus["evidence_texts"][e] for e in ev_ids],
                               k1=float(m6cfg["bm25"]["k1"]), b=float(m6cfg["bm25"]["b"]))
            for c in subset:
                ids, scores = bm.retrieve(c.text, int(ret["top_k"]))
                per_claim.setdefault(c.claim_id, {})["bm25"] = {
                    "evidence_ids": ids, "scores": [float(s) for s in scores],
                    "confidence": rank_normalise(scores)}

        if args.retriever in ("dense", "both"):
            idx_dir = stage_dir(m6cfg, "build_index", root) / f"{dataset}_{ret['phase']}"
            if (idx_dir / "index_meta.json").is_file():
                from src.retrieval.encoders import build_labse_encode_fn
                index = EvidenceIndex.load(idx_dir)
                enc = build_labse_encode_fn(
                    m6cfg["retrieval"]["bi_encoder"],
                    batch_size=int(ret["encode_batch_size"]),
                    max_seq_length=int(ret["max_seq_length"]),
                    device=detect_device(), num_threads=int(ret["num_threads"]))
                vecs = enc([c.text for c in subset])
                ids_batch, sc_batch = index.search(np.asarray(vecs), int(ret["top_k"]))
                for c, ids, scores in zip(subset, ids_batch, sc_batch):
                    per_claim.setdefault(c.claim_id, {})["dense"] = {
                        "evidence_ids": list(ids), "scores": [float(s) for s in scores],
                        "confidence": rank_normalise(list(scores))}
            else:
                print(f"      dense skipped: no index at {idx_dir}")

        results[lang] = per_claim
        found = sum(1 for v in per_claim.values()
                    if any(r["evidence_ids"] for r in v.values()))
        coverage[lang] = {
            "n_claims": len(subset), "n_with_evidence": found,
            "coverage": found / len(subset) if subset else 0.0,
            "evidence_pool": dataset, "pool_size": len(ev_ids),
            "retrievers": sorted({k for v in per_claim.values() for k in v}),
            "seconds": round(time.perf_counter() - t0, 2),
            "domain_mismatch_note": (
                f"{lang} project news claims retrieved against the {dataset} "
                f"fact-checking corpus. Low coverage here is a corpus-mismatch "
                f"finding, not a retrieval bug."),
        }
        print(f"      evidence found for {found:,}/{len(subset):,} claims "
              f"({100 * found / max(len(subset), 1):.1f}%)")

    rdir = m7_dir(cfg, "retrieval", root)
    write_json(rdir / f"{args.split}_retrieval.json",
               {"split": args.split, "retriever": args.retriever,
                "top_k": int(ret["top_k"]), "coverage": coverage,
                "score_normalisation": ret["score_normalisation"],
                "results": results})
    write_manifest(rdir, {"stage": "m7_02_retrieval", "split": args.split,
                          "status": "complete", "coverage": coverage,
                          "primary_retriever": ret["primary"],
                          "reproducibility": reproducibility_block(cfg)})
    print(f"\n  wrote -> {rdir}")


if __name__ == "__main__":
    main()
