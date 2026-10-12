#!/usr/bin/env python
"""Module 8 Stage 2 — token attributions for the classifier. GPU recommended.

Occlusion requires re-encoding the text once per token, so this is the only
expensive Module 8 stage. It is resumable: completed records are skipped.

The method label written to every record is the method that ACTUALLY ran. A
fallback is never labelled "shap".

    python scripts/m8_02_generate_classifier_explanations.py --split validation
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.m8_common import (  # noqa: E402
    PROJECT_ROOT, banner, detect_device, load_m8_config, m8_dir, manifest_block,
    read_jsonl, read_manifest, sha256_file, write_jsonl, write_manifest)
from src.explainability.shap_explainer import resolve_method  # noqa: E402
from src.explainability.token_attribution import occlusion_attribution  # noqa: E402


def build_text_predict_fn(cfg, root):
    """text -> P(fake), through XLM-R/LaBSE then the frozen CE-MLP."""
    import torch
    from sentence_transformers import SentenceTransformer
    from src.contrastive.projection_head import build_ce_mlp_baseline

    inp = cfg["inputs"]
    ckpt = root / inp["module4_models_dir"] / inp["module4_checkpoint"]
    if not ckpt.is_file():
        raise SystemExit(f"Module 4 checkpoint missing: {ckpt}")
    enc = SentenceTransformer(inp["encoder"], device=detect_device())
    enc.max_seq_length = int(cfg["explanation"]["max_seq_length"])
    enc.eval()
    clf = build_ce_mlp_baseline(input_dim=768, hidden_dims=[512, 128],
                                n_classes=2, dropout=0.1)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    clf.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
    clf.eval()

    def predict(texts):
        with torch.inference_mode():
            v = enc.encode(list(texts), batch_size=int(cfg["explanation"]["batch_size"]),
                           convert_to_numpy=True, show_progress_bar=False)
            logits = clf(torch.from_numpy(np.asarray(v, dtype=np.float32)))
            return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return predict, sha256_file(ckpt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate classifier explanations.")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the plan without loading any model.")
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m8_config(args.config, root=root)
    banner(f"Stage 2 classifier explanations [{args.split}]")

    sel_path = m8_dir(cfg, "explanations", root) / f"selected_{args.split}.json"
    if not sel_path.is_file():
        raise SystemExit(f"Run m8_01 first: {sel_path} not found")
    records = json.loads(sel_path.read_text(encoding="utf-8"))["records"]
    if args.limit:
        records = records[:int(args.limit)]

    method, reason = resolve_method(cfg["explanation"]["primary_method"],
                                    bool(cfg["explanation"]["allow_shap_embedding_space"]))
    print(f"  method    : {method}   ({reason})")
    print(f"  records   : {len(records):,}")
    print(f"  max tokens: {cfg['explanation']['max_tokens_per_record']}")

    out_dir = m8_dir(cfg, "explanations", root)
    part = out_dir / f"attributions_{args.split}.jsonl"
    done = {r["record_id"] for r in read_jsonl(part)} if part.is_file() else set()
    todo = [r for r in records if r["record_id"] not in done]
    print(f"  already done: {len(done):,}   to do: {len(todo):,}")

    if args.dry_run:
        write_manifest(out_dir, manifest_block(
            cfg, 8, "m8_02_generate_classifier_explanations", "project", args.split,
            [str(sel_path.relative_to(root))], [], "dry_run",
            {"method": method, "n_records": len(records), "n_pending": len(todo)}))
        print("  DRY RUN: no model loaded.")
        return

    predict, ckpt_hash = build_text_predict_fn(cfg, root)
    exp = cfg["explanation"]
    rows, failures, t0 = [], 0, time.perf_counter()
    for i, r in enumerate(todo, 1):
        try:
            att = occlusion_attribution(
                r["text"], predict, mask_token=exp["mask_token"] or "",
                max_tokens=int(exp["max_tokens_per_record"]),
                batch_size=int(exp["batch_size"]))
            rows.append({"record_id": r["record_id"], "language": r.get("language"),
                         "method": att.method, **att.to_dict()})
        except Exception as e:  # noqa: BLE001
            failures += 1
            rows.append({"record_id": r["record_id"], "language": r.get("language"),
                         "method": method, "tokens": [], "attributions": [],
                         "base_probability": None, "n_tokens": 0,
                         "error": f"{type(e).__name__}: {e}"})
        if i % 25 == 0 or i == len(todo):
            write_jsonl(part, read_jsonl(part) + rows if part.is_file() else rows)
            rows = []
            print(f"    {i}/{len(todo)}  ({time.perf_counter() - t0:.0f}s)")
    if rows:
        write_jsonl(part, (read_jsonl(part) + rows) if part.is_file() else rows)

    allrows = read_jsonl(part)
    write_manifest(out_dir, manifest_block(
        cfg, 8, "m8_02_generate_classifier_explanations", "project", args.split,
        [str(sel_path.relative_to(root))], [str(part.relative_to(root))], "complete",
        {"method": method, "method_reason": reason, "n_records": len(allrows),
         "n_failures": failures,
         "module4_checkpoint_sha256": ckpt_hash,
         "seconds": round(time.perf_counter() - t0, 1)}))
    print(f"\n  attributions: {len(allrows):,}   failures: {failures}")
    print(f"  wrote -> {part}")


if __name__ == "__main__":
    main()
