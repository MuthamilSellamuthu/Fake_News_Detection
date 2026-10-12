#!/usr/bin/env python
"""Module 6 Stage 1 — encode the evidence pool with LaBSE. GPU. Resumable.

Encodes only the evidence permitted for the requested phase, derived from
Module 5's build_index_scope(). Encoding the whole pool and filtering later
would work, but scoping here means an out-of-scope item never even gets a
vector -- the leakage guarantee holds by construction, not by discipline.

Shards of 10,000, atomic temp-then-rename, manifest-driven resume.

    python scripts/m6_01_encode_evidence.py --dataset fever --phase test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m6_common import (  # noqa: E402
    PROJECT_ROOT, assert_module5_untouched, banner, config_hash, detect_device,
    load_config, load_corpus, module5_fingerprint, stage_dir, write_manifest)
from src.retrieval.encoders import (  # noqa: E402
    ENCODER_CAPABILITIES, ENCODERS_VERSION, LowMemoryStop, ShardedEncoder,
    available_mb, build_labse_encode_fn, encoder_runtime_info)

# The memory-safety work lives in src/retrieval/encoders.py -- micro-batching,
# torch.inference_mode(), gc.collect(), torch.set_num_threads(). This script only
# drives it. A partially synced project (new CLI here, stale encoders.py there)
# would accept every flag, start normally, and then OOM-kill the runtime with no
# traceback. Assert the capabilities up front so that failure is loud.
REQUIRED_ENCODER_CAPABILITIES = {
    "micro_batching", "inference_mode", "gc_collect", "thread_limit",
    "ram_floor", "settings_guard", "lazy_text_fn",
}


def assert_encoder_is_memory_safe() -> None:
    missing = REQUIRED_ENCODER_CAPABILITIES - set(ENCODER_CAPABILITIES)
    if missing:
        raise SystemExit(
            "STALE src/retrieval/encoders.py\n"
            f"  found version : {ENCODERS_VERSION}\n"
            f"  missing       : {sorted(missing)}\n"
            "  The CPU memory-safety implementation is in that file, not in this\n"
            "  script. Sync src/retrieval/encoders.py to Drive and re-run.\n"
            "  Running as-is would OOM-kill the runtime with no traceback.")
from src.retrieval.index_scope import build_evidence_scope  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode scoped evidence with LaBSE.")
    ap.add_argument("--dataset", required=True, choices=["fever", "x_fact"])
    ap.add_argument("--phase", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be encoded; download no model.")
    ap.add_argument("--model", default=None,
                    help="Override the bi-encoder. A smaller model (e.g. "
                         "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) "
                         "cuts RAM ~4x if LaBSE is being OOM-killed.")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--shard-size", type=int, default=None,
                    help="Smaller shards checkpoint more often and cap peak RAM.")
    ap.add_argument("--min-available-mb", type=float, default=1200.0,
                    help="Stop cleanly if system RAM drops below this, instead of "
                         "being OOM-killed with no traceback. 0 disables.")
    ap.add_argument("--num-threads", type=int, default=2,
                    help="torch CPU threads. Each carries its own workspace, so an "
                         "unbounded count multiplies peak RAM on a small container.")
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_config(args.config)
    banner(f"Stage 1 encode evidence [{args.dataset}/{args.phase}]", cfg)
    assert_encoder_is_memory_safe()
    before = module5_fingerprint(root)

    corpus = load_corpus(cfg, args.dataset, root=root)
    scope = build_evidence_scope(corpus["claims"], corpus["splits"], args.phase,
                                 cfg["leakage"]["index_scope"])
    ev_ids = sorted(scope.evidence_ids)                 # sorted => deterministic shards
    ev_texts = corpus["evidence_texts"]
    # Views, not a second full copy of the corpus text (requirement: never hold
    # the whole pool twice). text_fn materialises one shard at a time.
    texts = [ev_texts.get(e, "") for e in ev_ids]

    def text_fn(lo: int, hi: int) -> list[str]:
        return [ev_texts.get(e, "") for e in ev_ids[lo:hi]]

    print(f"  claims in scope   : {scope.n_claims_in_scope:,}")
    print(f"  evidence in scope : {len(ev_ids):,}")
    print(f"  evidence excluded : {scope.n_evidence_excluded:,} (out-of-scope only)")

    out = stage_dir(cfg, "encode_evidence", root) / f"{args.dataset}_{args.phase}"
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        write_manifest(out, {"stage": "encode_evidence", "dataset": args.dataset,
                             "phase": args.phase, "status": "dry_run",
                             "n_evidence": len(ev_ids), "scope": scope.stats})
        print("  DRY RUN: no model loaded, no vectors written.")
        assert_module5_untouched(before, root)
        return

    r = cfg["retrieval"]
    device = detect_device()
    # CPU-safe defaults. The config values (batch 64 / seq 256 / shard 10000) are
    # sized for a GPU; on CPU that shard size means the whole 6,968-item pool goes
    # through one encode() call, and the peak is what gets the process OOM-killed.
    cpu = device == "cpu"
    model_name = args.model or r["bi_encoder"]
    batch = int(args.batch_size or (8 if cpu else r["encode_batch_size"]))
    seq = int(args.max_seq_length or (128 if cpu else r["max_seq_length"]))
    shard = int(args.shard_size or (500 if cpu else r["shard_size"]))
    settings = {"model": model_name, "batch_size": batch, "max_seq_length": seq}

    print(f"  model      : {model_name}")
    print(f"  batch={batch}  seq={seq}  shard={shard}  threads={args.num_threads}"
          + ("   [CPU-safe defaults]" if cpu and not (args.batch_size or args.max_seq_length
                                                     or args.shard_size) else ""))
    print(f"  RAM floor  : {args.min_available_mb:.0f} MB "
          f"(available now {available_mb():.0f} MB)")
    n_shards = (len(ev_ids) + shard - 1) // shard
    print(f"  shards     : {n_shards}  ({len(ev_ids):,} items)")

    enc = ShardedEncoder(out, shard_size=shard,
                         encode_fn=build_labse_encode_fn(
                             model_name, batch_size=batch, max_seq_length=seq,
                             device=device, num_threads=int(args.num_threads),
                             log=print),
                         model_name=model_name)

    rt = encoder_runtime_info()
    print(f"  encoders   : {rt['encoders_version']}  "
          f"torch={rt.get('torch_version', 'n/a')}  "
          f"threads={rt.get('torch_num_threads', 'n/a')}/"
          f"{rt.get('torch_num_interop_threads', 'n/a')} (intra/inter)  "
          f"inference_mode={rt.get('inference_mode_available', 'n/a')}")

    t0 = time.perf_counter()
    ram_lo = available_mb()
    stopped = None
    try:
        man = enc.encode(texts, ids=ev_ids, min_available_mb=float(args.min_available_mb),
                         settings=settings, text_fn=text_fn)
    except LowMemoryStop as exc:
        stopped = str(exc)
        man = enc.load_manifest()

    man.update({"stage": "encode_evidence", "dataset": args.dataset, "phase": args.phase,
                "scope": scope.stats, "settings": settings, "device": device,
                "encoder_runtime": rt,
                "config_hash": config_hash(cfg, ["retrieval", "leakage"])})
    write_manifest(out, man)
    assert_module5_untouched(before, root)

    done = sorted(man.get("completed_shards", []))
    remaining = [i for i in range(n_shards) if i not in set(done)]
    written = sum(1 for i in range(n_shards) if (out / f"shard_{i:05d}.npy").is_file())
    ram_now = available_mb()
    print("\n" + "-" * 62)
    print("STAGE 1a SUMMARY")
    print("-" * 62)
    print(f"  completed shards   : {len(done)}/{n_shards}")
    print(f"  remaining shards   : {len(remaining)}"
          + (f"  (next: {remaining[0]})" if remaining else ""))
    print(f"  embeddings written : {written} shard file(s)")
    print(f"  elapsed            : {time.perf_counter() - t0:.1f}s")
    print(f"  RAM available      : start {ram_lo:.0f} MB -> now {ram_now:.0f} MB"
          f"  (lowest seen {man.get('min_available_mb_observed') or ram_now:.0f} MB)")
    print(f"  status             : {man.get('status')}")
    print(f"  resume-safe        : YES - re-run the identical command; "
          f"completed shards are skipped")
    print(f"  output             : {out}")
    print("  Module 5 untouched : True")
    if stopped:
        print("\n  STOPPED EARLY (clean, not a crash):")
        print(f"  {stopped}")
        sys.exit(2)


if __name__ == "__main__":
    main()
