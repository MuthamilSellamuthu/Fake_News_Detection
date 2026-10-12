"""Module 9 — build the paper's tables from measured results only.

Every cell is either a measured number or the string "unavailable". A missing
experiment is never imputed, averaged over, or silently dropped -- an absent
ablation must be visibly absent.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

UNAVAILABLE = "unavailable"


def _get(d: dict | None, *path, default=UNAVAILABLE):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return default if cur is None else cur


def classification_table(results: dict[str, dict], metrics: Sequence[str],
                         scope: str = "overall") -> list[dict[str, Any]]:
    """One row per system. `scope` is 'overall' or a language code."""
    rows = []
    for system, res in results.items():
        block = (res.get("overall") if scope == "overall"
                 else _get(res, "per_language", scope, default=None))
        row = {"system": system, "scope": scope,
               "n": _get(block, "n") if isinstance(block, dict) else UNAVAILABLE}
        for m in metrics:
            row[m] = _get(block, m) if isinstance(block, dict) else UNAVAILABLE
        rows.append(row)
    return rows


def language_comparison_table(results: dict[str, dict], metric: str = "macro_f1",
                              languages: Sequence[str] = ("en", "ta")
                              ) -> list[dict[str, Any]]:
    rows = []
    for system, res in results.items():
        row = {"system": system, "metric": metric,
               "overall": _get(res, "overall", metric)}
        for lang in languages:
            row[lang] = _get(res, "per_language", lang, metric)
        en, ta = row.get("en"), row.get("ta")
        row["en_minus_ta"] = (float(en) - float(ta)
                              if isinstance(en, (int, float)) and isinstance(ta, (int, float))
                              else UNAVAILABLE)
        rows.append(row)
    return rows


def retrieval_table(retrieval_metrics: dict[str, dict]) -> list[dict[str, Any]]:
    rows = []
    for key, res in retrieval_metrics.items():
        overall = res.get("overall", {}) if isinstance(res, dict) else {}
        acc = res.get("accounting", {}) if isinstance(res, dict) else {}
        row = {"dataset_retriever": key,
               "n_scored": acc.get("n_scored", UNAVAILABLE),
               "n_excluded_no_gold": acc.get("n_excluded_no_gold_evidence", UNAVAILABLE)}
        for m in ("recall@1", "recall@5", "recall@10", "recall@50", "mrr", "ndcg@10"):
            row[m] = overall.get(m, UNAVAILABLE)
        rows.append(row)
    return rows


def ablation_table(stages: dict[str, dict], metric: str = "macro_f1",
                   baseline: str = "classifier_only") -> list[dict[str, Any]]:
    """Delta is always relative to the declared baseline, never to the best arm."""
    base = _get(stages.get(baseline), "overall", metric, default=None)
    rows = []
    for name, res in stages.items():
        val = _get(res, "overall", metric, default=None)
        rows.append({
            "ablation": name, "metric": metric,
            "value": val if val is not None else UNAVAILABLE,
            "baseline": baseline,
            "delta_vs_baseline": (float(val) - float(base)
                                  if isinstance(val, (int, float))
                                  and isinstance(base, (int, float)) else UNAVAILABLE),
            "applicable": bool(res.get("applicable", True)) if isinstance(res, dict) else False,
            "reason_if_unavailable": res.get("reason", "") if isinstance(res, dict) else "",
        })
    return rows


def statistical_table(tests: dict[str, dict],
                      corrected: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = []
    adj = {}
    if corrected and corrected.get("applicable"):
        keys = list(tests.keys())
        for r in corrected["results"]:
            if r["index"] < len(keys):
                adj[keys[r["index"]]] = r
    for name, t in tests.items():
        rows.append({
            "comparison": name,
            "test": t.get("test", UNAVAILABLE),
            "applicable": t.get("applicable", False),
            "reason_if_not": t.get("reason", ""),
            "n_records": t.get("n_records", UNAVAILABLE),
            "n_discordant": t.get("n_discordant", UNAVAILABLE),
            "statistic": t.get("statistic", UNAVAILABLE),
            "p_value": t.get("p_value", UNAVAILABLE),
            "p_adjusted_holm": adj.get(name, {}).get("p_adjusted", UNAVAILABLE),
            "significant_after_correction": adj.get(name, {}).get("rejected", UNAVAILABLE),
            "null_hypothesis": t.get("null_hypothesis", UNAVAILABLE),
        })
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Atomic CSV write. Union of keys as header so no column is silently lost."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        tmp = path.with_suffix(".tmp.csv")
        tmp.write_text("", encoding="utf-8")
        tmp.replace(path)
        return path
    header: list[str] = []
    for r in rows:
        for k in r:
            if k not in header:
                header.append(k)
    tmp = path.with_suffix(".tmp.csv")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, UNAVAILABLE) for k in header})
    tmp.replace(path)
    return path
