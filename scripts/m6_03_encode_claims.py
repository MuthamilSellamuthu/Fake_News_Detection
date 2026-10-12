#!/usr/bin/env python
"""Module 6 Stage 3 — encode claims (queries) with LaBSE. GPU. Resumable.

Only the claims that will actually be queried are encoded: the configured
evaluation splits, plus optionally train when building verifier inputs.

    python scripts/m6_03_encode_claims.py --dataset fever --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m6_common import (  # noqa: E402
    PROJECT_ROOT, assert_module5_untouched, banner, config_hash, detect_device,
    load_config, load_corpus, module5_fingerprint, stage_dir, write_manifest)
from src.retrieval.encoders import ShardedEncoder, build_labse_encode_fn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode claims with LaBSE.")
    ap.add_argument("--dataset", required=True, choices=["fever", "x_fact"])
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=None,
                    help="Override the bi-encoder (smaller model = less RAM).")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--shard-size", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_config(args.config)
    banner(f"Stage 3 encode claims [{args.dataset}/{args.split}]", cfg)
    before = module5_fingerprint(root)

    corpus = load_corpus(cfg, args.dataset, root=root)
    idxs = sorted(corpus["splits"].get(args.split, []))
    if not idxs:
        raise SystemExit(f"split {args.split!r} is empty for {args.dataset}")
    claim_ids = [corpus["claim_id_of"][i] for i in idxs]
    texts = [corpus["claims"][i].get("text", "") for i in idxs]
    print(f"  claims to encode: {len(texts):,}")

    out = stage_dir(cfg, "encode_claims", root) / f"{args.dataset}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        write_manifest(out, {"stage": "encode_claims", "dataset": args.dataset,
                             "split": args.split, "status": "dry_run", "n_claims": len(texts)})
        print("  DRY RUN: no model loaded.")
        assert_module5_untouched(before, root)
        return

    r = cfg["retrieval"]
    model_name = args.model or r["bi_encoder"]
    batch = int(args.batch_size or r["encode_batch_size"])
    seq = int(args.max_seq_length or r["max_seq_length"])
    shard = int(args.shard_size or r["shard_size"])
    print(f"  model={model_name} batch={batch} seq={seq} shard={shard}")
    enc = ShardedEncoder(out, shard_size=shard,
                         encode_fn=build_labse_encode_fn(
                             model_name, batch_size=batch, max_seq_length=seq,
                             device=detect_device()),
                         model_name=model_name)
    man = enc.encode(texts, ids=claim_ids)
    man.update({"stage": "encode_claims", "dataset": args.dataset, "split": args.split,
                "config_hash": config_hash(cfg, ["retrieval"])})
    write_manifest(out, man)
    assert_module5_untouched(before, root)
    print(f"  wrote {man['n_shards']} shard(s) -> {out}")


if __name__ == "__main__":
    main()
