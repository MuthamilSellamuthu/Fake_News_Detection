#!/usr/bin/env python
"""Module 7 Stage 6 — evaluation + ablations + statistics. CPU.

Reuses Module 4's `evaluate_predictions()` (overall + per-language en/ta) so
Module 7 numbers are directly comparable with Module 4's published table --
the central comparison of the paper.

Ablations: A classifier-only, B verifier-only, C integrated,
D classifier+retrieval (no verifier).

    python scripts/m7_06_evaluate.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.m7_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m7_config, m7_dir, reproducibility_block,
    set_all_seeds, write_json, write_manifest)
from src.contrastive.evaluation import evaluate_predictions  # noqa: E402
from src.integration.error_analysis import (  # noqa: E402
    language_comparison, retrieval_effect_analysis)


def mcnemar(y, a, b, min_discordant=10):
    """Exact-ish McNemar on paired predictions over identical records."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    n01 = int(np.sum((a == y) & (b != y)))
    n10 = int(np.sum((a != y) & (b == y)))
    n = n01 + n10
    if n < min_discordant:
        return {"applicable": False, "n_discordant": n,
                "reason": f"only {n} discordant pairs; {min_discordant} required",
                "n01": n01, "n10": n10}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    try:
        from scipy.stats import chi2
        p = float(1 - chi2.cdf(stat, 1))
    except ImportError:
        p = None
    return {"applicable": True, "n_discordant": n, "n01": n01, "n10": n10,
            "statistic": float(stat), "p_value": p,
            "note": "continuity-corrected McNemar on paired predictions"}


def bootstrap_ci(y, pred, n_samples=1000, seed=42, alpha=0.05):
    from src.integration.fusion import _macro_f1_binary
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    n = len(y)
    if n == 0:
        return {"applicable": False}
    vals = [_macro_f1_binary(y[i], pred[i])
            for i in (rng.integers(0, n, n) for _ in range(int(n_samples)))]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"applicable": True, "macro_f1_mean": float(np.mean(vals)),
            "ci_lower": float(lo), "ci_upper": float(hi),
            "n_bootstrap": int(n_samples), "alpha": alpha}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Module 7.")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m7_config(args.config, root=root)
    banner(f"Stage 6 evaluate [{args.split}]")
    set_all_seeds(int(cfg["reproducibility"]["seed"]))

    fname = "integrated_predictions.json" if args.split == "test" else "fusion_predictions.json"
    p = m7_dir(cfg, args.split, root) / fname
    if not p.is_file():
        raise SystemExit(f"No predictions at {p}. Run m7_05 first.")
    rows = json.loads(p.read_text(encoding="utf-8"))["records"]

    y = np.array([r["gold_label"] for r in rows], dtype=int)
    langs = np.array([r["language"] for r in rows], dtype=str)
    m4_pred = np.array([r["module4_prediction"] for r in rows], dtype=int)
    m4_p = np.array([r["module4_p_fake"] for r in rows], dtype=float)
    fused_pred = np.array([r["final_prediction"] for r in rows], dtype=int)
    fused_p = np.array([r["fused_p_fake"] for r in rows], dtype=float)
    has_ver = np.array([r["has_verifier"] for r in rows], dtype=bool)

    stages: dict[str, dict] = {}
    stages["classifier_only"] = evaluate_predictions(y, m4_pred, langs, m4_p)
    stages["integrated"] = evaluate_predictions(y, fused_pred, langs, fused_p)

    if has_ver.any():
        vp = np.array([r["p_fake_verifier"] for r in rows], dtype=float)
        thr = 0.5
        stages["verifier_only"] = evaluate_predictions(
            y[has_ver], (vp[has_ver] >= thr).astype(int), langs[has_ver], vp[has_ver])
        stages["verifier_only"]["note"] = (
            f"Computed only over the {int(has_ver.sum())} records with a verifier "
            f"verdict; threshold 0.5 on P_fake_verifier. NOT comparable "
            f"record-for-record with the other rows unless coverage is 100%.")
    else:
        stages["verifier_only"] = {"applicable": False,
                                   "reason": "no verifier output present (Module 6 Stage 6 pending)"}

    has_ev = np.array([bool(r.get("has_evidence")) for r in rows], dtype=bool)
    stages["classifier_plus_retrieval"] = {
        "applicable": bool(has_ev.any()),
        "note": ("Ablation D isolates retrieval from verification: identical to "
                 "classifier-only by construction, reported on the evidence-bearing "
                 "subset to expose selection effects."),
        "on_records_with_evidence": (evaluate_predictions(
            y[has_ev], m4_pred[has_ev], langs[has_ev], m4_p[has_ev])
            if has_ev.any() else None)}

    st = cfg["evaluation"]["statistical_tests"]
    comparisons = {}
    if st.get("enabled"):
        comparisons["classifier_only_vs_integrated"] = mcnemar(
            y, m4_pred, fused_pred, int(st["min_discordant_pairs"]))
        if has_ver.any():
            vp = np.array([r["p_fake_verifier"] for r in rows], dtype=float)
            vpred = (vp >= 0.5).astype(int)
            comparisons["classifier_only_vs_verifier_only"] = mcnemar(
                y[has_ver], m4_pred[has_ver], vpred[has_ver], int(st["min_discordant_pairs"]))
            comparisons["verifier_only_vs_integrated"] = mcnemar(
                y[has_ver], vpred[has_ver], fused_pred[has_ver], int(st["min_discordant_pairs"]))
        if st.get("bootstrap_ci"):
            comparisons["bootstrap"] = {
                "classifier_only": bootstrap_ci(y, m4_pred, int(st["bootstrap_samples"]),
                                                int(cfg["reproducibility"]["seed"]),
                                                float(st["alpha"])),
                "integrated": bootstrap_ci(y, fused_pred, int(st["bootstrap_samples"]),
                                           int(cfg["reproducibility"]["seed"]),
                                           float(st["alpha"]))}

    lang_delta = language_comparison(stages, cfg["evaluation"]["languages"])
    retr = retrieval_effect_analysis(rows)

    out = m7_dir(cfg, args.split, root)
    write_json(out / "integrated_metrics.json",
               {"split": args.split, "stages": stages, "statistical_comparisons": comparisons,
                "language_comparison": lang_delta, "retrieval_effect": retr,
                "n_records": len(rows), "n_with_verifier": int(has_ver.sum())})
    adir = m7_dir(cfg, "ablations", root)
    for name in ("classifier_only", "verifier_only", "integrated",
                 "classifier_plus_retrieval"):
        write_json(adir / f"{name}.json", {"split": args.split, "metrics": stages[name]})
    write_json(adir / "ablation_results.json",
               {"split": args.split, "stages": stages, "comparisons": comparisons})
    ldir = m7_dir(cfg, "language_analysis", root)
    for lang in cfg["evaluation"]["languages"]:
        write_json(ldir / f"{lang}.json",
                   {"language": lang, "split": args.split,
                    "classifier_only": stages["classifier_only"]["per_language"].get(lang),
                    "integrated": stages["integrated"]["per_language"].get(lang),
                    "delta": lang_delta.get(lang)})
    write_manifest(out, {"stage": "m7_06_evaluate", "split": args.split,
                         "status": "complete", "n_records": len(rows),
                         "reproducibility": reproducibility_block(cfg)})

    print(f"\n  classifier-only  overall macro_f1: "
          f"{stages['classifier_only']['overall'].get('macro_f1')}")
    print(f"  integrated       overall macro_f1: "
          f"{stages['integrated']['overall'].get('macro_f1')}")
    for lang in cfg["evaluation"]["languages"]:
        d = lang_delta.get(lang, {}).get("macro_f1", {})
        print(f"  [{lang}] macro_f1 {d.get('classifier_only')} -> {d.get('integrated')} "
              f"(delta {d.get('delta')})")
    print(f"\n  wrote -> {out}")


if __name__ == "__main__":
    main()
