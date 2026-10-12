#!/usr/bin/env python
"""Module 6 Stage 6 — train the verifier. GPU. Resumable.

Two distinct tasks, never merged:

  --dataset fever   3-way ENTAILMENT   SUPPORTS / REFUTES / NOT ENOUGH INFO
  --dataset x_fact  3-way VERACITY     false / true / partly true/misleading

X-FACT training starts from the FEVER-pretrained ENCODER via
`--init-from`, and always REPLACES the classification head. Reusing the FEVER
head would smuggle entailment semantics into veracity labels and make the
transfer number uninterpretable.

Compute scope (free Colab): FEVER is trained on a stratified 50,000-claim
subset for 3 epochs at max_length 256, not 228,090 x 10 at 512. Recorded in
the manifest and reported as a limitation.

    python scripts/m6_06_train_verifier.py --dataset fever
    python scripts/m6_06_train_verifier.py --dataset x_fact --init-from models/module6/fever
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m6_common import (  # noqa: E402
    PROJECT_ROOT, assert_module5_untouched, banner, detect_device, load_config,
    load_corpus, module5_fingerprint, stage_dir, write_manifest)
from src.verification.dataset import (  # noqa: E402
    build_examples, get_label_space, stratified_subset)
from src.verification.trainer import (  # noqa: E402
    CheckpointManager, VerifierTrainer, classification_report, set_seed)
from src.verification.verifier_model import VerifierConfig, build_verifier  # noqa: E402


def load_retrieved(cfg, root, dataset, split):
    """Reranked top-5 if present, else dense candidates, else nothing."""
    rr = stage_dir(cfg, "rerank", root) / f"{dataset}_{split}" / "reranked_top5.json"
    if rr.is_file():
        return json.loads(rr.read_text(encoding="utf-8")), "reranked_top5"
    dn = stage_dir(cfg, "retrieve", root) / f"{dataset}_{split}" / "dense_candidates.json"
    if dn.is_file():
        return json.loads(dn.read_text(encoding="utf-8")), "dense_candidates"
    return {}, "none"


def run_dirs(cfg, root, dataset: str, run_name: str | None):
    """(checkpoint dir, manifest dir) -- isolated per ablation variant.

    THE PRIMARY RUN AND EVERY ABLATION MUST NOT SHARE A CHECKPOINT DIRECTORY.
    With a shared directory and `resume=True`, launching an ablation after a
    finished primary run resumes that run's state: `range(state.epoch, epochs)`
    is empty, the training loop never executes, `best.pt` is overwritten, and
    the script reports the PREVIOUS run's `best_metric` as the ablation result
    -- exit code 0, no warning, a fabricated number. Hence separate trees.
    """
    if run_name:
        return (root / cfg["output"]["models_dir"] / "ablations" / dataset / run_name,
                stage_dir(cfg, "verifier_training", root) / "ablations" / dataset / run_name)
    return (root / cfg["output"]["models_dir"] / dataset,
            stage_dir(cfg, "verifier_training", root) / dataset)


def assert_evidence_present(source: str, stats: dict, claim_only: bool, dataset: str) -> None:
    """Refuse to train an evidence-aware verifier with no evidence attached.

    This is the guard for the defect that invalidated the first Module 6 run:
    `retrieve/{ds}_train/` did not exist, `load_retrieved()` returned
    ({}, "none") silently, and both verifiers trained on (claim, "") while
    Stage 7 evaluated them on (claim, 5 passages). Exit code was 0 and the
    manifest recorded `"evidence_source": "retrieved"`.

    A missing-evidence run is now only reachable by asking for it explicitly
    with --claim-only, which is a legitimate ablation and is recorded as such.
    """
    if claim_only:
        return
    if source == "none":
        raise SystemExit(
            f"REFUSING TO TRAIN: no training evidence for {dataset!r}.\n"
            f"  load_retrieved() found neither\n"
            f"    outputs/module6/rerank/{dataset}_train/reranked_top5.json\n"
            f"    outputs/module6/retrieve/{dataset}_train/dense_candidates.json\n"
            f"  Training would silently use (claim, \"\") while Stage 7 evaluates\n"
            f"  with retrieved evidence -- the train/eval mismatch this guard exists\n"
            f"  to prevent.\n"
            f"  Fix : python scripts/m6_04_retrieve.py --dataset {dataset} --split train\n"
            f"  Or  : pass --claim-only if a claim-only ablation is what you intended.")
    if stats.get("n_with_evidence", 0) == 0:
        raise SystemExit(
            f"REFUSING TO TRAIN: {dataset!r} loaded candidates from {source!r}, but "
            f"ZERO of {stats.get('n_built', 0):,} examples ended up with any evidence "
            f"text.\n  Likely cause: retrieved evidence_ids do not match the ids in "
            f"evidence.jsonl, or a language filter removed everything.\n"
            f"  Inspect the candidate file before retrying.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the fact verifier.")
    ap.add_argument("--dataset", required=True, choices=["fever", "x_fact"])
    ap.add_argument("--init-from", default=None,
                    help="Checkpoint dir whose ENCODER is reused. The head is always "
                         "re-initialised for the new label space.")
    ap.add_argument("--claim-only", action="store_true", help="Ablation: no evidence.")
    ap.add_argument("--oracle", action="store_true", help="Ablation: gold evidence.")
    ap.add_argument("--evidence-languages", default=None,
                    help="Ablation: comma-separated evidence languages to keep, "
                         "e.g. 'en' or 'en,ta'. Default: no filter.")
    ap.add_argument("--class-weights", action="store_true",
                    help="Ablation: inverse-frequency class weights in the loss, "
                         "computed on TRAIN ONLY.")
    ap.add_argument("--run-name", default=None,
                    help="Isolate this run under models/module6/ablations/{dataset}/"
                         "{run_name}/ and outputs/module6/verifier_training/ablations/"
                         "{dataset}/{run_name}/. REQUIRED for every ablation.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Start from scratch, ignoring any checkpoint in the target dir.")
    ap.add_argument("--resume", action="store_true",
                    help="Explicitly allow resuming from a populated target dir.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.no_resume and args.resume:
        raise SystemExit("--no-resume and --resume are mutually exclusive.")
    if args.claim_only and args.oracle:
        raise SystemExit("--claim-only and --oracle are mutually exclusive.")

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_config(args.config)
    v = cfg["verifier"]
    seed = int(args.seed or cfg["reproducibility"]["seed"])
    epochs = int(args.epochs or v["epochs"])
    banner(f"Stage 6 train verifier [{args.dataset}]", cfg)
    before = module5_fingerprint(root)
    set_seed(seed)

    space = get_label_space(args.dataset)
    print(f"  label space : {space.name} ({space.task}) -> {space.labels}")
    print(f"  NOTE: FEVER and X-FACT label spaces are never mapped onto each other.")

    corpus = load_corpus(cfg, args.dataset, root=root)
    ev_lang = {e["evidence_id"]: str(e.get("language", "unknown"))
               for e in corpus["evidence"]}
    keep_langs = ([s.strip() for s in args.evidence_languages.split(",") if s.strip()]
                  if args.evidence_languages else None)

    retrieved, source = load_retrieved(cfg, root, args.dataset, "train")
    val_retrieved, val_source = load_retrieved(cfg, root, args.dataset, "validation")
    print(f"  evidence source (train)     : {source}")
    print(f"  evidence source (validation): {val_source}")
    if keep_langs:
        print(f"  evidence languages kept     : {keep_langs}")

    def build(retr, split):
        ex, st = build_examples(
            corpus["claims"], corpus["evidence_texts"], retr, args.dataset,
            corpus["split_of"], max_evidence=int(v["evidence_per_claim"]),
            use_gold_evidence=bool(args.oracle), claim_only=bool(args.claim_only),
            evidence_lang=ev_lang, keep_languages=keep_langs)
        return [e for e in ex if e.split == split], st

    train_ex, stats = build(retrieved, "train")
    val_ex, val_stats = build(val_retrieved, "validation")

    # GUARD: refuse a silent claim-only run masquerading as evidence-aware.
    assert_evidence_present(source, stats, bool(args.claim_only), args.dataset)

    subset_size = (v.get(args.dataset) or {}).get("train_subset_size")
    n_before = len(train_ex)
    train_ex = stratified_subset(train_ex, subset_size, seed=seed)
    print(f"  train claims: {n_before:,} -> {len(train_ex):,}"
          + ("  (stratified subset, free-Colab scope)" if subset_size else ""))
    print(f"  val claims  : {len(val_ex):,}")
    print(f"  accounting balances: {stats['accounting_balances']}")

    # Statistics on the ACTUAL training set, after subsetting.
    n_tr_with_ev = sum(1 for e in train_ex if e.evidence_texts)
    tr_labels: dict[str, int] = {}
    for e in train_ex:
        tr_labels[e.label_str] = tr_labels.get(e.label_str, 0) + 1
    val_labels: dict[str, int] = {}
    for e in val_ex:
        val_labels[e.label_str] = val_labels.get(e.label_str, 0) + 1
    print(f"  train with evidence: {n_tr_with_ev:,}/{len(train_ex):,} "
          f"({100 * n_tr_with_ev / max(len(train_ex), 1):.1f}%)")
    print(f"  train label distribution: {tr_labels}")

    ckpt_dir, out = run_dirs(cfg, root, args.dataset, args.run_name)

    # ISOLATION: never resume implicitly into a populated directory.
    existing = (ckpt_dir / "latest.pt").is_file()
    if existing and not args.resume:
        raise SystemExit(
            f"REFUSING TO START: {ckpt_dir} already contains latest.pt.\n"
            f"  Resuming here would continue a DIFFERENT run's state: if that run\n"
            f"  finished, range(state.epoch, epochs) is empty, no training happens,\n"
            f"  best.pt is overwritten, and the previous run's best_metric is\n"
            f"  reported as this run's result -- with exit code 0.\n"
            f"  Pick one:\n"
            f"    --run-name <variant>   train an ablation in its own directory\n"
            f"    --resume               genuinely continue THIS interrupted run\n"
            f"    --no-resume            after moving the existing directory aside")
    resume = bool(args.resume) and not args.no_resume

    manifest = {"stage": "verifier_training", "dataset": args.dataset,
                "run_name": args.run_name, "variant": args.run_name or "primary",
                "label_space": space.labels, "task": space.task,
                "n_train": len(train_ex), "n_train_before_subset": n_before,
                "n_train_with_evidence": n_tr_with_ev,
                "n_train_without_evidence": len(train_ex) - n_tr_with_ev,
                "n_val": len(val_ex), "epochs": epochs, "seed": seed,
                "train_label_distribution": tr_labels,
                "val_label_distribution": val_labels,
                "evidence_source": source, "evidence_source_validation": val_source,
                "retrieval_split_train": "train", "retrieval_split_validation": "validation",
                "evidence_languages": keep_langs,
                "claim_only": bool(args.claim_only),
                "oracle_gold_evidence": bool(args.oracle),
                "class_weights": bool(args.class_weights),
                "example_stats": stats, "example_stats_validation": val_stats,
                "init_from": args.init_from, "resume": resume,
                "compute_scope": cfg["compute_scope"]["reductions"],
                "device": detect_device(),
                "reproducibility": {
                    "seed": seed, "encoder": v["encoder"], "tokenizer": v["encoder"],
                    "max_length": int(v["max_length"]), "batch_size": int(v["batch_size"]),
                    "learning_rate": float(v["learning_rate"]),
                    "weight_decay": float(v["weight_decay"]),
                    "warmup_ratio": float(v["warmup_ratio"]), "epochs": epochs,
                    "train_subset_size": subset_size,
                    "evidence_per_claim": int(v["evidence_per_claim"]),
                    "retrieval_top_k": int(cfg["retrieval"]["top_k_candidates"]),
                    "retrieval_method": "dense_labse_faiss_flat_ip",
                    "gradient_accumulation_steps_declared_but_unused":
                        v.get("gradient_accumulation_steps"),
                    "mixed_precision_declared_but_unused":
                        cfg["runtime"].get("use_mixed_precision"),
                    "checkpoint_dir": str(ckpt_dir)}}

    if args.dry_run:
        manifest["status"] = "dry_run"
        write_manifest(out, manifest)
        print("  DRY RUN: no model built, no training performed.")
        assert_module5_untouched(before, root)
        return

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    vcfg = VerifierConfig(encoder_name=v["encoder"], num_labels=len(space),
                          hidden_dim=int(v["hidden_dim"]), dropout=float(v["dropout"]),
                          max_length=int(v["max_length"]), label_space=space.name)
    model = build_verifier(vcfg)

    manifest["software"] = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "transformers": __import__("transformers").__version__}

    if args.init_from:
        init_dir = Path(args.init_from)
        cm = CheckpointManager(init_dir)
        # Prefer best.pt: latest.pt is the FINAL epoch, which is not necessarily
        # the best-validation model. They coincide only when best_step == the
        # last step. Silently transferring a worse encoder would be invisible.
        init_path = cm.best if cm.best.is_file() else cm.latest
        state = cm.load(model, path=init_path)
        if state is None:
            raise SystemExit(f"--init-from {init_dir}: no loadable checkpoint")
        swap = model.swap_head(len(space), space.name, int(v["hidden_dim"]))
        manifest["head_swap"] = swap
        manifest["init_checkpoint"] = {
            "path": str(init_path), "which": init_path.name,
            "bytes": init_path.stat().st_size,
            "source_epoch": state.epoch, "source_step": state.global_step,
            "source_best_metric": state.best_metric}
        print(f"  loaded encoder from {init_path}; head RE-INITIALISED "
              f"({swap['from']} -> {swap['to']})")
    else:
        manifest["head_swap"] = None
        manifest["init_checkpoint"] = None
        print("  no --init-from: encoder starts from pretrained "
              f"{v['encoder']} (NO transfer)")

    dev = detect_device()
    model.to(dev)
    tok = AutoTokenizer.from_pretrained(v["encoder"])

    def collate(batch):
        pairs = [e.to_text_pair(max_evidence=int(v["evidence_per_claim"])) for e in batch]
        enc = tok([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                  truncation=True, max_length=int(v["max_length"]), return_tensors="pt")
        enc["labels"] = torch.tensor([e.label_id for e in batch])
        return enc

    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ex, batch_size=int(v["batch_size"]), shuffle=True,
                          collate_fn=collate, generator=g)
    val_dl = DataLoader(val_ex, batch_size=int(v["batch_size"]), collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=float(v["learning_rate"]),
                            weight_decay=float(v["weight_decay"]))
    total = max(1, len(train_dl) * epochs)
    sched = get_linear_schedule_with_warmup(
        opt, int(total * float(v["warmup_ratio"])), total)

    # Inverse-frequency class weights, computed on TRAIN ONLY (never validation
    # or test). w_c = N / (K * n_c), the standard balanced form. Absent classes
    # get weight 0 rather than inf.
    cls_w = None
    if args.class_weights:
        counts = [sum(1 for e in train_ex if e.label_id == c) for c in range(len(space))]
        n, k = len(train_ex), len(space)
        w = [(n / (k * c) if c > 0 else 0.0) for c in counts]
        cls_w = torch.tensor(w, dtype=torch.float32, device=dev)
        manifest["class_weight_values"] = {space.labels[i]: w[i] for i in range(k)}
        manifest["class_weight_counts"] = {space.labels[i]: counts[i] for i in range(k)}
        print(f"  class weights (train-only): {manifest['class_weight_values']}")

    def step_fn(m, batch):
        m.train()
        batch = {k: t.to(dev) for k, t in batch.items()}
        labels = batch.pop("labels")
        loss = torch.nn.functional.cross_entropy(
            m(batch["input_ids"], batch["attention_mask"]), labels, weight=cls_w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        return float(loss.item())

    @torch.no_grad()
    def eval_fn(m):
        m.eval()
        yt, yp = [], []
        for batch in val_dl:
            batch = {k: t.to(dev) for k, t in batch.items()}
            labels = batch.pop("labels")
            preds = m(batch["input_ids"], batch["attention_mask"]).argmax(-1)
            yt.extend(labels.cpu().tolist()); yp.extend(preds.cpu().tolist())
        return classification_report(yt, yp, space.labels)

    trainer = VerifierTrainer(ckpt_dir, {**vcfg.to_dict(), "seed": seed, "epochs": epochs},
                              checkpoint_every_steps=int(v["checkpoint_every_steps"]),
                              early_stopping_patience=int(v["early_stopping_patience"]))
    result = trainer.train(model, opt, sched, lambda _e: train_dl, step_fn, eval_fn,
                           epochs=epochs, resume=resume)
    manifest.update({"status": "complete", "training": result,
                     "checkpoint_dir": str(ckpt_dir)})
    write_manifest(out, manifest)
    assert_module5_untouched(before, root)
    print(f"  best val macro_f1: {result['best_metric']:.4f}")
    print(f"  checkpoints -> {ckpt_dir}")


if __name__ == "__main__":
    main()
