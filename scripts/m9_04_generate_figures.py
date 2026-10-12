#!/usr/bin/env python
"""Module 9 Stage 4 — figures, only where real data exists. CPU.

A figure is skipped, with a reason, when its underlying measurement is
missing. An empty or placeholder chart in a paper is worse than no chart.

    python scripts/m9_04_generate_figures.py --split test
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m9_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m9_config, m9_dir, manifest_block, read_json,
    write_json, write_manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Module 9 figures.")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m9_config(args.config, root=root)
    banner(f"Stage 4 figures [{args.split}]")

    try:
        import matplotlib
        matplotlib.use("Agg")               # deterministic, headless
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "matplotlib is not installed, so figures cannot be generated.\n"
            "  Install with: pip install matplotlib\n"
            "  Tables and metrics are unaffected; re-run this stage afterwards.")

    coll = read_json(m9_dir(cfg, "", root) / "collected_results.json")
    if coll is None:
        raise SystemExit("Run m9_01_collect_results.py first.")
    found = coll["found"]
    fdir = m9_dir(cfg, "figures", root)
    made, skipped = [], []
    dpi, fmt = int(cfg["figures"]["dpi"]), cfg["figures"]["format"]

    def save(fig, name):
        p = fdir / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        made.append(str(p.relative_to(root)))
        print(f"  wrote {p.name}")

    stages = (found.get("module7_metrics") or {}).get("stages", {})
    primary = cfg["metrics"]["primary"]

    if stages:
        names, vals = [], []
        for k, v in stages.items():
            val = (v.get("overall") or {}).get(primary) if isinstance(v, dict) else None
            if isinstance(val, (int, float)):
                names.append(k); vals.append(val)
        if names:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(names, vals)
            ax.set_ylabel(primary); ax.set_ylim(0, 1)
            ax.set_title(f"Model comparison ({primary}, {args.split})")
            ax.tick_params(axis="x", rotation=20)
            save(fig, "model_comparison")
        else:
            skipped.append({"figure": "model_comparison", "reason": "no numeric metrics"})

        langs = cfg["metrics"]["languages"]
        series = {}
        for k, v in stages.items():
            per = (v.get("per_language") or {}) if isinstance(v, dict) else {}
            vals = [(per.get(l) or {}).get(primary) for l in langs]
            if all(isinstance(x, (int, float)) for x in vals):
                series[k] = vals
        if series:
            fig, ax = plt.subplots(figsize=(7, 4))
            width = 0.8 / len(series)
            for i, (k, vals) in enumerate(series.items()):
                ax.bar([x + i * width for x in range(len(langs))], vals, width, label=k)
            ax.set_xticks([x + 0.4 - width / 2 for x in range(len(langs))])
            ax.set_xticklabels(langs); ax.set_ylabel(primary); ax.set_ylim(0, 1)
            ax.set_title(f"English vs Tamil ({primary}, {args.split})"); ax.legend()
            save(fig, "language_macro_f1")
        else:
            skipped.append({"figure": "language_macro_f1",
                            "reason": "per-language metrics unavailable"})
    else:
        skipped += [{"figure": f, "reason": "Module 7 metrics unavailable"}
                    for f in ("model_comparison", "language_macro_f1")]

    retr = {k.replace("retrieval_", ""): v for k, v in found.items()
            if k.startswith("retrieval_")}
    if retr:
        ks = [1, 5, 10, 50]
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = 0
        for name, res in sorted(retr.items()):
            ov = (res or {}).get("overall", {})
            ys = [ov.get(f"recall@{k}") for k in ks]
            if all(isinstance(y, (int, float)) for y in ys):
                ax.plot(ks, ys, marker="o", label=name); plotted += 1
        if plotted:
            ax.set_xlabel("k"); ax.set_ylabel("Recall@k"); ax.set_ylim(0, 1)
            ax.set_title("Retrieval Recall@k"); ax.legend(fontsize=7)
            save(fig, "retrieval_recall_at_k")
        else:
            plt.close(fig)
            skipped.append({"figure": "retrieval_recall_at_k", "reason": "no recall values"})
    else:
        skipped.append({"figure": "retrieval_recall_at_k", "reason": "no retrieval metrics"})

    preds = found.get("module7_predictions")
    if preds:
        import numpy as np
        recs = preds["records"]
        y = np.array([int(r["gold_label"]) for r in recs])
        p = np.array([int(r["final_prediction"]) for r in recs])
        cm = np.array([[int(((y == i) & (p == j)).sum()) for j in (0, 1)] for i in (0, 1)])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["real", "fake"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["real", "fake"])
        ax.set_xlabel("predicted"); ax.set_ylabel("gold")
        ax.set_title(f"Confusion matrix ({args.split})")
        save(fig, "confusion_matrix_integrated")

        pf = np.array([float(r["fused_p_fake"]) for r in recs])
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(pf, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 5:
                xs.append(float(pf[m].mean())); ys.append(float(y[m].mean()))
        if len(xs) >= 2:
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.plot(xs, ys, marker="o")
            ax.set_xlabel("mean predicted P(fake)"); ax.set_ylabel("observed fraction fake")
            ax.set_title(f"Reliability ({args.split})")
            save(fig, "reliability_curve")
        else:
            skipped.append({"figure": "reliability_curve",
                            "reason": "too few populated probability bins"})
    else:
        skipped += [{"figure": f, "reason": "Module 7 predictions unavailable"}
                    for f in ("confusion_matrix_integrated", "reliability_curve")]

    m8 = found.get("module8_metrics")
    if m8 and (m8.get("faithfulness") or {}).get("overall", {}).get("n_records"):
        fo = m8["faithfulness"]["overall"]
        conc = fo.get("attribution_concentration", {})
        ks = [k for k in ("top_1", "top_3", "top_5", "top_10") if conc.get(k, {}).get("mean")]
        if ks:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(ks, [conc[k]["mean"] for k in ks])
            ax.set_ylabel("mean attribution mass"); ax.set_ylim(0, 1)
            ax.set_title("Attribution concentration")
            save(fig, "explanation_faithfulness")
        else:
            skipped.append({"figure": "explanation_faithfulness",
                            "reason": "no concentration values"})
    else:
        skipped.append({"figure": "explanation_faithfulness",
                        "reason": "Module 8 faithfulness unavailable"})

    write_json(fdir / "figures_manifest.json",
               {"split": args.split, "generated": made, "skipped": skipped,
                "policy": "A figure is generated only from measured data."})
    write_manifest(m9_dir(cfg, "", root), manifest_block(
        cfg, "m9_04_generate_figures", args.split, ["collected_results.json"],
        made, "complete", {"n_generated": len(made), "n_skipped": len(skipped)}))
    print(f"\n  generated {len(made)}, skipped {len(skipped)}")
    for s in skipped:
        print(f"    skipped {s['figure']}: {s['reason']}")


if __name__ == "__main__":
    main()
