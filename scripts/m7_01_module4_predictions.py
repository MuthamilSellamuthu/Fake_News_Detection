#!/usr/bin/env python
"""Module 7 Stage 1 — regenerate Module 4 per-record predictions. CPU.

Module 4 persisted only aggregate metrics; the per-record P(fake) was computed
and discarded. Fusion needs it, so this stage reconstructs it by FORWARD
INFERENCE ONLY from the frozen checkpoint and the frozen Module 3 embeddings.
Nothing in Module 3 or Module 4 is written to.

    python scripts/m7_01_module4_predictions.py --split validation
    python scripts/m7_01_module4_predictions.py --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m7_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m7_config, m7_dir, reproducibility_block,
    set_all_seeds, sha256_file, write_json, write_manifest)
from src.integration.module4_inference import (  # noqa: E402
    build_prediction_records, load_split_arrays, predict_p_fake, summarise)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate Module 4 per-record predictions.")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Only for the reported hard prediction; fusion refits its own.")
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m7_config(args.config, root=root)
    banner(f"Stage 1 Module 4 predictions [{args.split}]")
    set_all_seeds(int(cfg["reproducibility"]["seed"]))

    inp = cfg["inputs"]
    ckpt = root / inp["module4_models_dir"] / inp["module4_checkpoint"]
    if not ckpt.is_file():
        raise SystemExit(
            f"Module 4 checkpoint not found: {ckpt}\n"
            f"  Module 7 cannot regenerate predictions without it. Confirm "
            f"inputs.module4_checkpoint in {cfg.get('name')} config.")

    arrays = load_split_arrays(root / inp["embeddings_dir"], args.split, inp["pooling"])
    print(f"  embeddings : {arrays['embeddings'].shape}  pooling={inp['pooling']}")
    print(f"  checkpoint : {ckpt.name}  sha256={sha256_file(ckpt)[:16]}...")

    p_fake = predict_p_fake(
        arrays["embeddings"], ckpt,
        input_dim=int(inp["module4_input_dim"]),
        hidden_dims=list(inp["module4_hidden_dims"]),
        n_classes=int(inp["module4_n_classes"]),
        dropout=float(inp["module4_dropout"]))
    records = build_prediction_records(arrays, p_fake, args.split, float(args.threshold))
    stats = summarise(records)

    out = m7_dir(cfg, "module4_predictions", root)
    write_json(out / f"{args.split}_predictions.json",
               {"split": args.split, "threshold": float(args.threshold),
                "statistics": stats, "records": records})
    write_manifest(out, {
        "stage": "m7_01_module4_predictions", "split": args.split,
        "status": "complete", "statistics": stats,
        "checkpoint": str(ckpt.relative_to(root)),
        "checkpoint_sha256": sha256_file(ckpt),
        "pooling": inp["pooling"],
        "source": "forward inference over frozen Module 3 embeddings; no training",
        "reproducibility": reproducibility_block(cfg)})

    print(f"\n  records          : {stats['n_records']:,}")
    print(f"  by language      : {stats['by_language']}")
    print(f"  by gold label    : {stats['by_gold_label']}")
    print(f"  P(fake) range    : [{stats['p_fake_min']:.6f}, {stats['p_fake_max']:.6f}] "
          f"mean {stats['p_fake_mean']:.6f}")
    print(f"  ids unique       : {stats['record_ids_unique']}")
    print(f"  probs in [0,1]   : {stats['all_probs_in_unit_interval']}")
    print(f"  language balance : {stats['language_accounting_balances']}")
    print(f"\n  wrote -> {out}")


if __name__ == "__main__":
    main()
