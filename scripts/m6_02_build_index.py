#!/usr/bin/env python
"""Module 6 Stage 2 — build the FAISS flat-IP index. CPU. Fast.

Loads the shards from Stage 1, builds an exact inner-product index over
L2-normalised vectors (== cosine), and then AUDITS the built index against
the phase scope. The audit is deliberately after construction: it checks what
was actually indexed, not what was intended.

    python scripts/m6_02_build_index.py --dataset fever --phase test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m6_common import (  # noqa: E402
    PROJECT_ROOT, assert_module5_untouched, banner, load_config, load_corpus,
    module5_fingerprint, read_manifest, stage_dir, write_manifest)
from src.retrieval.encoders import ShardedEncoder  # noqa: E402
from src.retrieval.faiss_index import EvidenceIndex, faiss_available  # noqa: E402
from src.retrieval.index_scope import (  # noqa: E402
    assert_index_within_scope, build_evidence_scope)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the evidence index.")
    ap.add_argument("--dataset", required=True, choices=["fever", "x_fact"])
    ap.add_argument("--phase", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_config(args.config)
    banner(f"Stage 2 build index [{args.dataset}/{args.phase}]", cfg)
    print(f"  faiss available: {faiss_available()} (numpy fallback is exact either way)")
    before = module5_fingerprint(root)

    corpus = load_corpus(cfg, args.dataset, root=root)
    scope = build_evidence_scope(corpus["claims"], corpus["splits"], args.phase,
                                 cfg["leakage"]["index_scope"])

    # Each phase needs its OWN index, because each phase may see a different
    # slice of the evidence (validation: train+validation; test: everything).
    # But a narrower phase's evidence is a strict subset of a wider one's, and
    # the vectors are identical -- same model, same text. So if this phase has
    # not been encoded, reuse a wider phase's embeddings and SELECT the
    # permitted subset rather than spending another hour re-encoding.
    # Safety is unchanged: assert_index_within_scope() still audits the result.
    FALLBACK_ORDER = {"train": ["train", "validation", "test"],
                      "validation": ["validation", "test"],
                      "test": ["test"]}
    src = None
    derived_from = None
    for cand_phase in FALLBACK_ORDER.get(args.phase, [args.phase]):
        cand = stage_dir(cfg, "encode_evidence", root) / f"{args.dataset}_{cand_phase}"
        if read_manifest(cand).get("status") == "complete":
            src, derived_from = cand, cand_phase
            break
    if src is None:
        tried = ", ".join(f"{args.dataset}_{p}" for p in FALLBACK_ORDER.get(args.phase, []))
        raise SystemExit(
            f"No completed evidence embeddings usable for {args.dataset}/{args.phase}.\n"
            f"  Looked for (in order): {tried}\n"
            f"  Run: python scripts/m6_01_encode_evidence.py --dataset {args.dataset} "
            f"--phase {args.phase}")

    man = read_manifest(src)
    enc = ShardedEncoder(src, shard_size=int(man.get("shard_size", 10000)))
    vectors = enc.load_all()
    ev_ids = enc.load_ids()
    if len(ev_ids) != vectors.shape[0]:
        raise SystemExit(f"{len(ev_ids)} ids but {vectors.shape[0]} vectors")

    if derived_from != args.phase:
        available = set(ev_ids)
        missing = scope.evidence_ids - available
        if missing:
            raise SystemExit(
                f"Cannot derive the {args.phase!r} index from {derived_from!r} embeddings: "
                f"{len(missing)} permitted evidence item(s) were never encoded. "
                f"Run m6_01 with --phase {args.phase}.")
        keep = [i for i, e in enumerate(ev_ids) if e in scope.evidence_ids]
        vectors = vectors[keep]
        ev_ids = [ev_ids[i] for i in keep]
        print(f"  derived from '{derived_from}' embeddings: selected "
              f"{len(ev_ids):,} of {len(available):,} vectors for this phase's scope")

    index = EvidenceIndex(ev_ids, vectors, normalize=bool(cfg["retrieval"]["normalize"]))
    assert_index_within_scope(index.evidence_ids, scope)   # raises on leakage

    out = stage_dir(cfg, "build_index", root) / f"{args.dataset}_{args.phase}"
    paths = index.save(out)
    write_manifest(out, {"stage": "build_index", "dataset": args.dataset,
                         "phase": args.phase, "status": "complete",
                         "n_evidence": len(index), "dim": index.dim,
                         "backend": "faiss" if index.use_faiss else "numpy",
                         "embeddings_from_phase": derived_from,
                         "derived_by_scope_selection": derived_from != args.phase,
                         "scope_audit": "PASS", "scope": scope.stats, "paths": paths})

    assert_module5_untouched(before, root)
    print(f"  indexed {len(index):,} evidence vectors (dim {index.dim})")
    print(f"  scope audit: PASS  ->  {out}")


if __name__ == "__main__":
    main()
