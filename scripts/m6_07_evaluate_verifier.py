#!/usr/bin/env python
"""Module 6 Stage 7 — evaluate + calibrate the verifier. GPU for inference.

Temperature is FITTED ON VALIDATION and then applied unchanged to test.
Fitting on test would be tuning on test, which the project's protocol forbids
("test evaluated once"); `fit_temperature()` refuses a split named "test".

FEVER and X-FACT are evaluated and reported SEPARATELY, against their own
label spaces. There is no combined score, because there is no shared task.

    python scripts/m6_07_evaluate_verifier.py --dataset fever --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m6_common import (  # noqa: E402
    PROJECT_ROOT, assert_module5_untouched, banner, detect_device, load_config,
    load_corpus, module5_fingerprint, stage_dir, write_json, write_manifest)
from scripts.m6_06_train_verifier import load_retrieved  # noqa: E402
from src.verification.calibration import (  # noqa: E402
    calibrate_and_evaluate, expected_calibration_error, fit_temperature, softmax)
from src.verification.dataset import build_examples, get_label_space  # noqa: E402
from src.verification.trainer import CheckpointManager, classification_report  # noqa: E402
from src.verification.verifier_model import VerifierConfig, build_verifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate and calibrate the verifier.")
    ap.add_argument("--dataset", required=True, choices=["fever", "x_fact"])
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--claim-only", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--evidence-languages", default=None,
                    help="Must MATCH the flag used at training time.")
    ap.add_argument("--run-name", default=None,
                    help="Evaluate the ablation trained under this name. Reads "
                         "models/module6/ablations/{dataset}/{run_name}/ and writes "
                         "outputs/module6/evaluation/ablations/{dataset}_{split}/{run_name}/.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_config(args.config)
    v = cfg["verifier"]
    banner(f"Stage 7 evaluate [{args.dataset}/{args.split}]", cfg)
    before = module5_fingerprint(root)

    space = get_label_space(args.dataset)
    corpus = load_corpus(cfg, args.dataset, root=root)
    ev_lang = {e["evidence_id"]: str(e.get("language", "unknown"))
               for e in corpus["evidence"]}
    keep_langs = ([s.strip() for s in args.evidence_languages.split(",") if s.strip()]
                  if args.evidence_languages else None)

    if args.run_name:
        out = (stage_dir(cfg, "evaluation", root) / "ablations"
               / f"{args.dataset}_{args.split}" / args.run_name)
    else:
        out = stage_dir(cfg, "evaluation", root) / f"{args.dataset}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        write_manifest(out, {"stage": "evaluation", "dataset": args.dataset,
                             "split": args.split, "status": "dry_run",
                             "label_space": space.labels})
        print("  DRY RUN: no model loaded.")
        assert_module5_untouched(before, root)
        return

    import torch
    from transformers import AutoTokenizer

    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint)
    elif args.run_name:
        ckpt_dir = (root / cfg["output"]["models_dir"] / "ablations"
                    / args.dataset / args.run_name)
    else:
        ckpt_dir = root / cfg["output"]["models_dir"] / args.dataset
    vcfg = VerifierConfig(encoder_name=v["encoder"], num_labels=len(space),
                          hidden_dim=int(v["hidden_dim"]), dropout=float(v["dropout"]),
                          max_length=int(v["max_length"]), label_space=space.name)
    model = build_verifier(vcfg)
    cm = CheckpointManager(ckpt_dir)
    if cm.load(model, path=cm.best if cm.best.is_file() else None) is None:
        raise SystemExit(f"no checkpoint in {ckpt_dir}; run m6_06 first")
    dev = detect_device()
    model.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(v["encoder"])

    ev_used: dict[str, dict] = {}

    @torch.no_grad()
    def logits_for(split: str):
        retr, src = load_retrieved(cfg, root, args.dataset, split)
        built, st = build_examples(
            corpus["claims"], corpus["evidence_texts"], retr, args.dataset,
            corpus["split_of"], max_evidence=int(v["evidence_per_claim"]),
            use_gold_evidence=bool(args.oracle), claim_only=bool(args.claim_only),
            evidence_lang=ev_lang, keep_languages=keep_langs)
        ex = [e for e in built if e.split == split]
        # Record WHICH evidence file fed this split. Its absence is why the
        # audit could not determine whether the first x_fact test evaluation
        # consumed the untrained reranker.
        ev_used[split] = {
            "candidate_file": src,
            "attached_source": st["evidence_source"],
            "n_examples": len(ex),
            "n_with_evidence": sum(1 for e in ex if e.evidence_texts),
            "mean_evidence_per_example": (
                sum(len(e.evidence_texts) for e in ex) / len(ex) if ex else 0.0),
            "evidence_languages_kept": keep_langs}
        print(f"  [{split}] candidates from {src!r}; "
              f"{ev_used[split]['n_with_evidence']:,}/{len(ex):,} examples have evidence")
        all_logits, labels, langs = [], [], []
        bs = int(v["batch_size"])
        for i in range(0, len(ex), bs):
            chunk = ex[i:i + bs]
            pairs = [e.to_text_pair(max_evidence=int(v["evidence_per_claim"])) for e in chunk]
            enc = tok([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                      truncation=True, max_length=int(v["max_length"]),
                      return_tensors="pt").to(dev)
            all_logits.append(model(enc["input_ids"], enc["attention_mask"]).float().cpu().numpy())
            labels.extend(e.label_id for e in chunk)
            langs.extend(e.language for e in chunk)
        return (np.concatenate(all_logits) if all_logits else np.zeros((0, len(space))),
                labels, langs)

    val_logits, val_labels, _ = logits_for("validation")
    temp = fit_temperature(val_logits, val_labels, split_name="validation")
    print(f"  temperature fitted on VALIDATION: {temp['temperature']:.4f} "
          f"(nll {temp['nll_before']:.4f} -> {temp['nll_after']:.4f})")

    logits, labels, langs = logits_for(args.split)
    preds = softmax(logits, temp["temperature"]).argmax(1).tolist()
    report = classification_report(labels, preds, space.labels)
    calib = calibrate_and_evaluate(logits, labels, temp["temperature"],
                                   n_bins=int(cfg["calibration"]["ece_bins"]))

    per_lang = {}
    for lang in sorted(set(langs)):
        sel = [i for i, x in enumerate(langs) if x == lang]
        if sel:
            per_lang[lang] = classification_report([labels[i] for i in sel],
                                                   [preds[i] for i in sel], space.labels)

    # Majority-class baseline on the SAME test items, for honest comparison.
    counts = [sum(1 for y in labels if y == c) for c in range(len(space))]
    maj = max(range(len(space)), key=lambda c: counts[c]) if labels else 0
    baseline = classification_report(labels, [maj] * len(labels), space.labels)

    payload = {"dataset": args.dataset, "split": args.split, "task": space.task,
               "label_space": space.labels, "n": len(labels),
               "run_name": args.run_name, "variant": args.run_name or "primary",
               "metrics": report, "calibration": calib, "temperature": temp,
               "per_language": per_lang,
               "class_distribution": {space.labels[c]: counts[c] for c in range(len(space))},
               "majority_baseline": {"predicted_class": space.labels[maj],
                                     "accuracy": baseline["accuracy"],
                                     "macro_f1": baseline["macro_f1"],
                                     "weighted_f1": baseline["weighted_f1"]},
               "evidence_used": ev_used,
               "evidence_languages": keep_langs,
               "claim_only": bool(args.claim_only), "oracle": bool(args.oracle),
               "reproducibility": {
                   "checkpoint_dir": str(ckpt_dir),
                   "checkpoint_file": str(cm.best if cm.best.is_file() else cm.latest),
                   "encoder": v["encoder"], "tokenizer": v["encoder"],
                   "max_length": int(v["max_length"]),
                   "batch_size": int(v["batch_size"]),
                   "evidence_per_claim": int(v["evidence_per_claim"]),
                   "seed": int(cfg["reproducibility"]["seed"]),
                   "ece_bins": int(cfg["calibration"]["ece_bins"]),
                   "temperature_fitted_on": "validation",
                   "device": detect_device(),
                   "python": sys.version.split()[0], "torch": torch.__version__},
               "note": ("FEVER (entailment) and X-FACT (veracity) are separate tasks "
                        "with separate label spaces; results are never combined.")}
    write_json(out / "verification_metrics.json", payload)
    write_manifest(out, {"stage": "evaluation", "dataset": args.dataset,
                         "split": args.split, "status": "complete",
                         "run_name": args.run_name, "variant": args.run_name or "primary",
                         "macro_f1": report["macro_f1"], "accuracy": report["accuracy"],
                         "weighted_f1": report["weighted_f1"],
                         "majority_baseline_accuracy": baseline["accuracy"],
                         "majority_baseline_macro_f1": baseline["macro_f1"],
                         "evidence_used": ev_used,
                         "ece_before": calib["ece_before"], "ece_after": calib["ece_after"]})

    assert_module5_untouched(before, root)
    print(f"  majority baseline: acc {baseline['accuracy']:.4f}  "
          f"macro-F1 {baseline['macro_f1']:.4f}")
    print(f"  accuracy {report['accuracy']:.4f}  macro-F1 {report['macro_f1']:.4f}"
          f"  weighted-F1 {report['weighted_f1']:.4f}")
    print(f"  ECE {calib['ece_before']:.4f} -> {calib['ece_after']:.4f} after scaling")


if __name__ == "__main__":
    main()
