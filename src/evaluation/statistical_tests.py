"""Module 9 — paired significance testing with multiple-comparison correction.

Predictions come from the same test records, so paired tests are the correct
family: McNemar for paired binary decisions. An unpaired test would ignore the
pairing and be less powerful and less valid.

Every test refuses to run when its requirements are unmet and says why, rather
than returning a p-value computed on inadequate data.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def mcnemar_test(y_true: Sequence[int], pred_a: Sequence[int], pred_b: Sequence[int],
                 min_discordant: int = 10, name_a: str = "A", name_b: str = "B"
                 ) -> dict[str, Any]:
    """Continuity-corrected McNemar on paired decisions.

    n01 = A right / B wrong; n10 = A wrong / B right. Only discordant pairs
    carry information, so a small discordant count means the test cannot
    distinguish the systems no matter how many records exist.
    """
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    if not (len(y) == len(a) == len(b)):
        return {"applicable": False, "reason": "prediction arrays have different lengths"}
    n01 = int(np.sum((a == y) & (b != y)))
    n10 = int(np.sum((a != y) & (b == y)))
    n = n01 + n10
    base = {"test": "mcnemar_continuity_corrected", "comparison": f"{name_a} vs {name_b}",
            "null_hypothesis": "the two systems have equal error rates on paired records",
            "alternative": "two-sided: the error rates differ",
            "n_records": int(len(y)), "n_discordant": n,
            f"{name_a}_right_{name_b}_wrong": n01, f"{name_a}_wrong_{name_b}_right": n10}
    if n < int(min_discordant):
        return {**base, "applicable": False,
                "reason": f"only {n} discordant pair(s); {min_discordant} required",
                "statistic": None, "p_value": None}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    try:
        from scipy.stats import chi2
        p = float(1 - chi2.cdf(stat, 1))
        method = "scipy chi2 survival"
    except ImportError:
        # Wilson-Hilferty approximation to the chi-square(1) tail; adequate for
        # reporting when scipy is unavailable, and labelled as such.
        p = float(math.erfc(math.sqrt(stat / 2.0)))
        method = "erfc approximation (scipy unavailable)"
    return {**base, "applicable": True, "statistic": float(stat), "p_value": p,
            "p_value_method": method,
            "effect": {"difference_in_errors": n10 - n01,
                       "odds_ratio": (n10 / n01) if n01 else None}}


def holm_correction(p_values: Sequence[float | None], alpha: float = 0.05
                    ) -> dict[str, Any]:
    """Holm-Bonferroni step-down correction.

    Holm rather than plain Bonferroni: it controls the same family-wise error
    rate while being uniformly more powerful. `None` p-values (inapplicable
    tests) are excluded from the family rather than treated as 1.0, since an
    untested hypothesis is not a failed one.
    """
    idx = [i for i, p in enumerate(p_values) if p is not None]
    m = len(idx)
    if m == 0:
        return {"applicable": False, "reason": "no valid p-values to correct",
                "n_hypotheses": 0, "results": []}
    order = sorted(idx, key=lambda i: p_values[i])
    results = [None] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(running, (m - rank) * float(p_values[i])))
        running = adj                       # enforce monotonicity
        results[i] = {"index": i, "p_raw": float(p_values[i]), "p_adjusted": adj,
                      "rejected": adj < alpha, "rank": rank + 1}
    return {"applicable": True, "method": "holm-bonferroni", "alpha": float(alpha),
            "n_hypotheses": m,
            "results": [r for r in results if r is not None],
            "excluded_inapplicable": len(p_values) - m,
            "note": ("Inapplicable tests are excluded from the family rather than "
                     "counted as non-significant.")}


def paired_bootstrap_difference(y_true: Sequence[int], pred_a: Sequence[int],
                                pred_b: Sequence[int], metric_fn,
                                n_resamples: int = 1000, seed: int = 42,
                                confidence: float = 0.95) -> dict[str, Any]:
    """Bootstrap CI for metric(A) - metric(B), resampling RECORDS not predictions.

    Resampling record indices keeps the pairing intact; resampling each system
    independently would destroy it and widen the interval artificially.
    """
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    n = len(y)
    if n == 0 or not (len(a) == len(b) == n):
        return {"applicable": False, "reason": "empty or mismatched inputs"}
    observed = float(metric_fn(y, a) - metric_fn(y, b))
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(n_resamples), dtype=np.float64)
    for i in range(int(n_resamples)):
        s = rng.integers(0, n, n)
        diffs[i] = metric_fn(y[s], a[s]) - metric_fn(y[s], b[s])
    lo_q, hi_q = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    lo, hi = np.percentile(diffs, [100 * lo_q, 100 * hi_q])
    return {"applicable": True, "observed_difference": observed,
            "ci_lower": float(lo), "ci_upper": float(hi),
            "confidence_level": float(confidence), "n_resamples": int(n_resamples),
            "seed": int(seed), "n_records": n,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "interpretation": ("A confidence interval excluding zero indicates a "
                               "difference; one containing zero does not establish "
                               "equivalence, only that a difference was not shown.")}
