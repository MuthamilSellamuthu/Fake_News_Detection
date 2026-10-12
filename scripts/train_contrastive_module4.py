#!/usr/bin/env python
"""Module 4 — CLI entry point: ablation sweep, final model, zero-shot.

Execution order is fixed and enforced:

  1. Pre-flight gate against the FROZEN Module 3 embeddings. Refuses to
     train if counts, dimensions, record_ids, languages or leakage
     disagree with Module 2's verified figures.
  2. Ablation sweep (seed 42): 2 poolings x 2 sampling strategies, each
     with SupCon and a capacity-matched CE-MLP, plus logistic-regression
     and majority-class baselines.
  3. Configuration selection on VALIDATION macro-F1 only.
  4. Final model: selected configuration re-run on seeds 42/43/44,
     reported as mean +/- std.
  5. Test set evaluated EXACTLY ONCE, after selection.
  6. Zero-shot: English-only training, Tamil test evaluation, reported
     separately from the multilingual results.

Every experiment is recorded, including ones that perform badly. A
sweep that silently drops its weak arms is not an ablation.

Usage
-----
  python scripts/train_contrastive_module4.py --preflight-only
  python scripts/train_contrastive_module4.py
  python scripts/train_contrastive_module4.py --config configs/module4_contrastive.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.contrastive.embedding_dataset import (  # noqa: E402
    load_all_splits,
    verify_frozen_inputs,
)
from src.contrastive.evaluation import (  # noqa: E402
    evaluate_predictions,
    fit_logistic_regression,
    majority_class_baseline,
    summarise_across_seeds,
)
from src.utils.config_loader import load_yaml  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("module4.train")


# --------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------

def run_preflight(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """Hard gate. Loads both poolings and cross-checks them.

    Returns the report; the caller STOPS if `passed` is False. Training
    on silently-wrong inputs is the exact failure this prevents -- it
    happened once in Module 3 and cost a full GPU run.
    """
    in_cfg = cfg["input"]
    emb_dir = root / in_cfg["embeddings_dir"]
    poolings = in_cfg["pooling_strategies"]

    reports: dict[str, Any] = {}
    reference_ids: dict[str, np.ndarray] | None = None

    for pooling in poolings:
        data = load_all_splits(emb_dir, pooling, in_cfg["splits"])
        report = verify_frozen_inputs(
            data=data,
            expected_counts=in_cfg["expected_counts"],
            expected_total=in_cfg["expected_total"],
            expected_dim=in_cfg["embedding_dim"],
            expected_tamil_rows=in_cfg["expected_tamil_rows"],
            reference_record_ids=reference_ids,
        )
        reports[pooling] = report
        if reference_ids is None:
            reference_ids = {s: e.record_ids for s, e in data.items()}

        for check in report["checks"]:
            level = logging.INFO if check["passed"] else logging.ERROR
            logger.log(level, "[preflight:%s] %-34s %s | %s", pooling, check["name"],
                       "PASS" if check["passed"] else "FAIL", check["detail"])

    passed = all(r["passed"] for r in reports.values())
    return {"passed": passed, "per_pooling": reports}


# --------------------------------------------------------------------
# Single experiment
# --------------------------------------------------------------------

def run_one_experiment(
    cfg: dict[str, Any],
    root: Path,
    pooling: str,
    sampling: str,
    seed: int,
    evaluate_test: bool = False,
) -> dict[str, Any]:
    """One (pooling, sampling, seed) cell: SupCon two-stage + CE-MLP.

    `evaluate_test` is False everywhere during the ablation. It is set
    True only for the final selected configuration, after selection.
    """
    from src.contrastive.trainer import ContrastiveTrainer, set_all_seeds

    set_all_seeds(seed, cfg["reproducibility"]["deterministic"])
    data = load_all_splits(root / cfg["input"]["embeddings_dir"], pooling, cfg["input"]["splits"])
    train, val, test = data["train"], data["validation"], data["test"]

    trainer = ContrastiveTrainer(cfg)
    tag = f"{pooling}/{sampling}/seed{seed}"
    logger.info("=== experiment %s (evaluate_test=%s) ===", tag, evaluate_test)

    t0 = time.perf_counter()
    head = trainer.train_projection_head(train, val, sampling, seed)
    clf = trainer.train_linear_classifier(head, train, val)
    supcon_val = trainer.evaluate(head, clf, val)
    logger.info("[%s] SupCon val macro-F1=%.4f (en=%.4f ta=%.4f)", tag,
                supcon_val["overall"]["macro_f1"],
                supcon_val["per_language"]["en"].get("macro_f1", float("nan")),
                supcon_val["per_language"]["ta"].get("macro_f1", float("nan")))

    ce_model = trainer.train_ce_mlp(train, val, sampling, seed)
    ce_val = trainer.evaluate_ce_mlp(ce_model, val)
    logger.info("[%s] CE-MLP val macro-F1=%.4f", tag, ce_val["overall"]["macro_f1"])

    result: dict[str, Any] = {
        "pooling": pooling,
        "sampling": sampling,
        "seed": seed,
        "supcon": {"validation": supcon_val},
        "ce_mlp": {"validation": ce_val},
        "history": trainer.history,
        "telemetry": trainer.telemetry,
        "gpu": trainer.gpu_telemetry(),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "test_evaluated": bool(evaluate_test),
    }

    if evaluate_test:
        # The single permitted test evaluation.
        logger.info("[%s] evaluating TEST (once, post-selection)", tag)
        result["supcon"]["test"] = trainer.evaluate(head, clf, test)
        result["ce_mlp"]["test"] = trainer.evaluate_ce_mlp(ce_model, test)
        result["_models"] = {"head": head, "clf": clf, "ce_mlp": ce_model}

    return result


# --------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------

def run_baselines(cfg: dict[str, Any], root: Path, seed: int) -> dict[str, Any]:
    """Majority class + logistic regression on each pooling.

    Validation only -- baselines are part of the comparison table, and
    the test set is reserved for the selected configuration.
    """
    out: dict[str, Any] = {}
    in_cfg = cfg["input"]

    first = in_cfg["pooling_strategies"][0]
    data = load_all_splits(root / in_cfg["embeddings_dir"], first, in_cfg["splits"])
    out["majority_class"] = {
        "validation": majority_class_baseline(data["train"].labels, data["validation"].labels,
                                              data["validation"].languages)
    }
    logger.info("[baseline] majority-class val macro-F1=%.4f",
                out["majority_class"]["validation"]["overall"]["macro_f1"])

    if cfg["baselines"]["logistic_regression"]:
        for pooling in in_cfg["pooling_strategies"]:
            d = load_all_splits(root / in_cfg["embeddings_dir"], pooling, in_cfg["splits"])
            clf = fit_logistic_regression(d["train"].embeddings, d["train"].labels, seed=seed)
            preds = clf.predict(d["validation"].embeddings)
            probs = clf.predict_proba(d["validation"].embeddings)[:, 1]
            res = evaluate_predictions(d["validation"].labels, preds, d["validation"].languages, probs)
            out[f"logistic_regression_{pooling}"] = {"validation": res}
            logger.info("[baseline] logreg(%s) val macro-F1=%.4f", pooling, res["overall"]["macro_f1"])
    return out


# --------------------------------------------------------------------
# Zero-shot
# --------------------------------------------------------------------

def run_zero_shot(cfg: dict[str, Any], root: Path, pooling: str, sampling: str, seed: int) -> dict[str, Any]:
    """English-only training, Tamil test evaluation.

    Reported separately: it answers a different question (does the
    shared XLM-R space transfer with zero Tamil supervision?) and must
    not be blended into the multilingual headline numbers.

    Uniform sampling is forced here -- language-aware sampling is
    meaningless when the training pool has no Tamil rows.
    """
    from src.contrastive.trainer import ContrastiveTrainer, set_all_seeds

    set_all_seeds(seed, cfg["reproducibility"]["deterministic"])
    data = load_all_splits(root / cfg["input"]["embeddings_dir"], pooling, cfg["input"]["splits"])

    train_en = data["train"].subset(data["train"].language_mask("en"))
    val_en = data["validation"].subset(data["validation"].language_mask("en"))
    test_ta = data["test"].subset(data["test"].language_mask("ta"))
    test_en = data["test"].subset(data["test"].language_mask("en"))

    logger.info("[zero-shot] train_en=%d val_en=%d -> test_ta=%d (test_en=%d for reference)",
                train_en.n, val_en.n, test_ta.n, test_en.n)

    trainer = ContrastiveTrainer(cfg)
    head = trainer.train_projection_head(train_en, val_en, "uniform", seed)
    clf = trainer.train_linear_classifier(head, train_en, val_en)

    return {
        "pooling": pooling,
        "sampling": "uniform (forced: no Tamil in the training pool)",
        "seed": seed,
        "train_rows_en": train_en.n,
        "val_rows_en": val_en.n,
        "test_rows_ta": test_ta.n,
        "validation_en": trainer.evaluate(head, clf, val_en),
        "test_tamil_zero_shot": trainer.evaluate(head, clf, test_ta),
        "test_english_reference": trainer.evaluate(head, clf, test_en),
        "history": trainer.history,
        "telemetry": trainer.telemetry,
    }


# --------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------

def select_best_configuration(ablation: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the winning (pooling, sampling, method) on VALIDATION macro-F1.

    Test results are not consulted -- they do not exist at this point in
    the run, by construction.
    """
    candidates = []
    for r in ablation:
        for method in ("supcon", "ce_mlp"):
            candidates.append({
                "pooling": r["pooling"],
                "sampling": r["sampling"],
                "method": method,
                "val_macro_f1": r[method]["validation"]["overall"]["macro_f1"],
                "val_macro_f1_en": r[method]["validation"]["per_language"]["en"].get("macro_f1"),
                "val_macro_f1_ta": r[method]["validation"]["per_language"]["ta"].get("macro_f1"),
            })
    candidates.sort(key=lambda c: c["val_macro_f1"], reverse=True)
    best = candidates[0]
    logger.info("Selected on validation macro-F1: %s/%s/%s (val macro-F1=%.4f)",
                best["pooling"], best["sampling"], best["method"], best["val_macro_f1"])
    return {"selected": best, "ranking": candidates, "selection_metric": "validation_macro_f1"}


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Module 4: supervised contrastive learning.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run the frozen-input gate and exit without training.")
    parser.add_argument("--skip-zero-shot", action="store_true")
    args = parser.parse_args()

    root = PROJECT_ROOT
    cfg_path = Path(args.config) if args.config else root / "configs" / "module4_contrastive.yaml"
    cfg = load_yaml(cfg_path)

    out_dir = root / cfg["output"]["reports_dir"]
    models_dir = root / cfg["output"]["models_dir"]
    figs_dir = root / cfg["output"]["figures_dir"]
    for d in (out_dir, models_dir, figs_dir):
        d.mkdir(parents=True, exist_ok=True)

    _attach_log_file(out_dir / cfg["logging"]["log_filename"])

    # ---- 1. pre-flight ----
    logger.info("Module 4 pre-flight: verifying FROZEN Module 3 embeddings...")
    preflight = run_preflight(cfg, root)
    (out_dir / "preflight_verification.json").write_text(
        json.dumps(preflight, indent=2, default=str), encoding="utf-8")

    if not preflight["passed"]:
        logger.error("PRE-FLIGHT FAILED - refusing to train. See preflight_verification.json")
        print("\nMODULE 4 PRE-FLIGHT FAILED - no training performed.")
        sys.exit(1)
    logger.info("Pre-flight PASSED for all poolings.")

    if args.preflight_only:
        print("\nPRE-FLIGHT PASSED. Exiting without training (--preflight-only).")
        return

    seed0 = cfg["reproducibility"]["seed"]
    run_started = time.perf_counter()

    # ---- 2. ablation sweep (seed 42) ----
    logger.info("=== ABLATION SWEEP (seed %d) ===", seed0)
    ablation: list[dict[str, Any]] = []
    for pooling in cfg["input"]["pooling_strategies"]:
        for sampling in cfg["sampling"]["ablation_strategies"]:
            ablation.append(run_one_experiment(cfg, root, pooling, sampling, seed0, evaluate_test=False))

    baselines = run_baselines(cfg, root, seed0)

    # ---- 3. selection on validation only ----
    selection = select_best_configuration(ablation)
    best = selection["selected"]

    # ---- 4. final configuration across seeds; test once ----
    logger.info("=== FINAL CONFIGURATION: %s/%s/%s over seeds %s ===",
                best["pooling"], best["sampling"], best["method"], cfg["reproducibility"]["seeds"])
    final_runs: list[dict[str, Any]] = []
    for seed in cfg["reproducibility"]["seeds"]:
        r = run_one_experiment(cfg, root, best["pooling"], best["sampling"], seed, evaluate_test=True)
        models = r.pop("_models", None)
        if models is not None and seed == cfg["reproducibility"]["seeds"][0]:
            _save_models(models, models_dir, best, seed)
        final_runs.append(r)

    method = best["method"]
    final_summary = {
        "selected_configuration": best,
        "seeds": cfg["reproducibility"]["seeds"],
        "validation_across_seeds": summarise_across_seeds(
            [r[method]["validation"]["overall"] for r in final_runs]),
        "test_across_seeds": summarise_across_seeds(
            [r[method]["test"]["overall"] for r in final_runs]),
        "test_per_language_across_seeds": {
            lang: summarise_across_seeds([r[method]["test"]["per_language"][lang] for r in final_runs])
            for lang in cfg["evaluation"]["languages"]
        },
        "per_seed_full": final_runs,
    }

    # ---- 5. zero-shot ----
    zero_shot = None
    if cfg["zero_shot"]["enabled"] and not args.skip_zero_shot:
        logger.info("=== ZERO-SHOT: English-only -> Tamil test ===")
        zero_shot = run_zero_shot(cfg, root, best["pooling"], best["sampling"], seed0)

    # ---- 6. persist everything ----
    total_runtime = round(time.perf_counter() - run_started, 2)
    _write_outputs(cfg, out_dir, cfg_path, preflight, ablation, baselines,
                   selection, final_summary, zero_shot, total_runtime)

    print("\n" + "=" * 72)
    print("MODULE 4 — TRAINING COMPLETE")
    print("=" * 72)
    print(json.dumps({
        "selected": best,
        "test_macro_f1_mean_std": final_summary["test_across_seeds"].get("macro_f1"),
        "total_runtime_seconds": total_runtime,
    }, indent=2))
    print("=" * 72)


def _attach_log_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    h = logging.FileHandler(path, encoding="utf-8")
    h.setFormatter(fmt)
    logging.getLogger("module4").addHandler(h)


def _save_models(models: dict[str, Any], models_dir: Path, best: dict[str, Any], seed: int) -> None:
    import torch

    tag = f"{best['pooling']}_{best['sampling']}_seed{seed}"
    torch.save(models["head"].state_dict(), models_dir / f"projection_head_{tag}.pt")
    torch.save(models["clf"].state_dict(), models_dir / f"linear_classifier_{tag}.pt")
    torch.save(models["ce_mlp"].state_dict(), models_dir / f"ce_mlp_{tag}.pt")
    logger.info("Saved model weights for %s", tag)


def _write_outputs(cfg, out_dir: Path, cfg_path: Path, preflight, ablation, baselines,
                   selection, final_summary, zero_shot, total_runtime) -> None:
    import shutil

    o = cfg["output"]

    # Full ablation record -- INCLUDING poorly-performing arms.
    (out_dir / o["ablation_filename"]).write_text(json.dumps({
        "note": "Every experiment is recorded, including weak results. "
                "Selection used validation macro-F1 only.",
        "seed": cfg["reproducibility"]["seed"],
        "experiments": [{k: v for k, v in r.items() if k != "history"} for r in ablation],
        "baselines": baselines,
        "selection": selection,
    }, indent=2, default=str), encoding="utf-8")

    (out_dir / o["history_filename"]).write_text(json.dumps({
        "ablation": [{"pooling": r["pooling"], "sampling": r["sampling"], "seed": r["seed"],
                      "history": r["history"]} for r in ablation],
        "final": [{"seed": r["seed"], "history": r["history"]} for r in final_summary["per_seed_full"]],
        "zero_shot": zero_shot["history"] if zero_shot else None,
    }, indent=2, default=str), encoding="utf-8")

    (out_dir / o["metrics_filename"]).write_text(json.dumps({
        "final": final_summary,
        "baselines": baselines,
        "zero_shot": zero_shot,
        "total_runtime_seconds": total_runtime,
        "config_used": cfg,
    }, indent=2, default=str), encoding="utf-8")

    shutil.copyfile(cfg_path, out_dir / "module4_config_used.yaml")
    _write_report(cfg, out_dir, ablation, baselines, selection, final_summary, zero_shot, total_runtime)


def _write_report(cfg, out_dir: Path, ablation, baselines, selection, final_summary,
                  zero_shot, total_runtime) -> None:
    best = selection["selected"]
    L = ["# Module 4 — Supervised Contrastive Learning: Results", "",
         f"Total runtime: {total_runtime}s · Selection metric: **validation macro-F1** "
         "(test never consulted for selection).", "",
         "## Ablation (seed 42)", "",
         "| Pooling | Sampling | Method | Val macro-F1 | Val F1 (en) | Val F1 (ta) |",
         "|---|---|---|---:|---:|---:|"]
    for c in selection["ranking"]:
        L.append(f"| {c['pooling']} | {c['sampling']} | {c['method']} | {c['val_macro_f1']:.4f} | "
                 f"{c['val_macro_f1_en']} | {c['val_macro_f1_ta']} |")

    L += ["", "## Baselines (validation)", "", "| Baseline | Macro-F1 |", "|---|---:|"]
    for name, b in baselines.items():
        L.append(f"| {name} | {b['validation']['overall']['macro_f1']:.4f} |")

    L += ["", "## Selected configuration", "",
          f"**{best['pooling']} pooling · {best['sampling']} sampling · {best['method']}**", "",
          f"Seeds: {final_summary['seeds']}", "",
          "### Test (evaluated once, after selection)", "",
          "| Metric | Mean | Std |", "|---|---:|---:|"]
    for k, v in final_summary["test_across_seeds"].items():
        if isinstance(v, dict) and "mean" in v:
            L.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} |")

    for lang in cfg["evaluation"]["languages"]:
        stats = final_summary["test_per_language_across_seeds"].get(lang, {})
        L += ["", f"### Test — {lang}", "", "| Metric | Mean | Std |", "|---|---:|---:|"]
        for k, v in stats.items():
            if isinstance(v, dict) and "mean" in v:
                L.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} |")

    if zero_shot:
        zt = zero_shot["test_tamil_zero_shot"]["overall"]
        ze = zero_shot["test_english_reference"]["overall"]
        L += ["", "## Zero-shot: English-only training → Tamil test", "",
              "Reported separately from the multilingual results — it answers a different question.", "",
              "| | Macro-F1 | Accuracy | ROC-AUC |", "|---|---:|---:|---:|",
              f"| Tamil (zero-shot) | {zt['macro_f1']:.4f} | {zt['accuracy']:.4f} | {zt.get('roc_auc')} |",
              f"| English (reference) | {ze['macro_f1']:.4f} | {ze['accuracy']:.4f} | {ze.get('roc_auc')} |"]

    L += ["", "## Integrity", "",
          "- Module 3 embeddings read-only; never regenerated or modified.",
          "- Splits unchanged (48,305 / 10,351 / 10,352).",
          "- Configuration selected on validation macro-F1 only.",
          "- Test set evaluated exactly once, after selection.",
          "- All experiments recorded, including weak arms.", ""]
    (out_dir / cfg["output"]["report_filename"]).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
