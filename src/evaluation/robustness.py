"""Module 9 — robustness slices.

Each slice is only computed when the data supports it. A slice with too few
records returns `applicable: False` with the count, rather than a metric
computed on a handful of examples.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

MIN_SLICE = 20


def _slice_metrics(y: np.ndarray, p: np.ndarray, metric_fn) -> dict[str, Any]:
    return {"n": int(len(y)), "value": float(metric_fn(y, p)) if len(y) else None}


def slice_by(records: Sequence[dict[str, Any]], key_fn: Callable[[dict], Any],
             metric_fn, y_key: str = "gold_label", pred_key: str = "final_prediction",
             min_slice: int = MIN_SLICE) -> dict[str, Any]:
    groups: dict[Any, list[dict]] = {}
    for r in records:
        groups.setdefault(key_fn(r), []).append(r)
    out: dict[str, Any] = {}
    for k in sorted(groups, key=lambda x: str(x)):
        rows = groups[k]
        if len(rows) < min_slice:
            out[str(k)] = {"applicable": False, "n": len(rows),
                           "reason": f"slice has {len(rows)} records; {min_slice} required"}
            continue
        y = np.array([int(r[y_key]) for r in rows])
        p = np.array([int(r[pred_key]) for r in rows])
        out[str(k)] = {"applicable": True, **_slice_metrics(y, p, metric_fn)}
    return out


def confidence_robustness(records: Sequence[dict[str, Any]], metric_fn,
                          threshold: float = 0.8, min_slice: int = MIN_SLICE
                          ) -> dict[str, Any]:
    """High- vs low-confidence predictions.

    Confidence is max(p, 1-p) on the fused probability -- distance from the
    decision boundary, not the raw fake probability.
    """
    def conf(r):
        p = float(r["fused_p_fake"])
        return "high" if max(p, 1 - p) >= threshold else "low"
    res = slice_by(records, conf, metric_fn, min_slice=min_slice)
    res["threshold"] = threshold
    res["definition"] = "confidence = max(P(fake), 1 - P(fake))"
    return res


def evidence_availability_robustness(records: Sequence[dict[str, Any]], metric_fn,
                                     min_slice: int = MIN_SLICE) -> dict[str, Any]:
    res = slice_by(records, lambda r: "with_evidence" if r.get("has_evidence")
                   else "without_evidence", metric_fn, min_slice=min_slice)
    res["note"] = ("Records without evidence fall back to the classifier by policy, "
                   "so a difference here reflects which records retrieval covered, "
                   "not a fusion effect.")
    return res


def error_breakdown(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """False positives / negatives, plus verifier disagreement and NEI counts."""
    fp = fn = tp = tn = 0
    disagree = nei = no_ev = 0
    for r in records:
        g, p = int(r["gold_label"]), int(r["final_prediction"])
        if g == 1 and p == 1: tp += 1
        elif g == 0 and p == 0: tn += 1
        elif g == 0 and p == 1: fp += 1
        else: fn += 1
        m4 = int(r.get("module4_prediction", p))
        if r.get("has_verifier"):
            vp = int(float(r.get("p_fake_verifier", 0.5)) >= 0.5)
            if vp != m4:
                disagree += 1
            probs = r.get("verifier_probs") or {}
            if probs:
                top = max(probs, key=lambda k: probs[k])
                if top in ("NOT ENOUGH INFO", "partly true/misleading"):
                    nei += 1
        if not r.get("has_evidence"):
            no_ev += 1
    n = len(records)
    return {"n_records": n, "true_positive": tp, "true_negative": tn,
            "false_positive": fp, "false_negative": fn,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else None,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) else None,
            "classifier_verifier_disagreements": disagree,
            "neutral_verdicts": nei, "records_without_evidence": no_ev,
            "counts_balance": (tp + tn + fp + fn) == n}
