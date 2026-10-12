"""Module 9 — bootstrap confidence intervals for single metrics."""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def bootstrap_metric(y_true: Sequence[int], y_pred: Sequence[int],
                     metric_fn: Callable[[np.ndarray, np.ndarray], float],
                     n_resamples: int = 1000, seed: int = 42,
                     confidence: float = 0.95) -> dict[str, Any]:
    """Percentile bootstrap CI. Records the seed so the interval is reproducible."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    n = len(y)
    if n == 0 or len(p) != n:
        return {"applicable": False, "reason": "empty or mismatched inputs"}
    point = float(metric_fn(y, p))
    rng = np.random.default_rng(int(seed))
    vals = np.empty(int(n_resamples), dtype=np.float64)
    for i in range(int(n_resamples)):
        s = rng.integers(0, n, n)
        vals[i] = metric_fn(y[s], p[s])
    lo, hi = np.percentile(vals, [100 * (1 - confidence) / 2,
                                  100 * (1 - (1 - confidence) / 2)])
    return {"applicable": True, "point_estimate": point,
            "ci_lower": float(lo), "ci_upper": float(hi),
            "confidence_level": float(confidence), "n_resamples": int(n_resamples),
            "seed": int(seed), "n_records": n,
            "bootstrap_mean": float(vals.mean()), "bootstrap_std": float(vals.std(ddof=0))}


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> dict[str, Any]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sensibly for small n or extreme proportions -- both common in the
    Tamil slices of this project.
    """
    if n <= 0:
        return {"applicable": False, "reason": "n must be positive"}
    import math
    z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(round(confidence, 2), 1.9600)
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return {"applicable": True, "proportion": float(phat),
            "ci_lower": float(max(0.0, centre - half)),
            "ci_upper": float(min(1.0, centre + half)),
            "confidence_level": float(confidence), "n": int(n),
            "method": "wilson_score"}
