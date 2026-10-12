"""Module 6 — calibration: ECE, reliability data, temperature scaling.

A fact-verification system that is confidently wrong is worse than one that
is unsure, because downstream the confidence is what a reader would act on.
Accuracy alone cannot show this; ECE can.

ECE here is the standard *confidence* ECE (Guo et al. 2017): bin predictions
by max softmax probability, and average |accuracy - confidence| per bin,
weighted by bin population.

Temperature scaling is fitted on VALIDATION only and then applied to test.
Fitting it on test would be tuning on the test set, which the project's
protocol forbids ("test evaluated once"). `fit_temperature()` refuses a
split named "test" as a guard against doing this by accident.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)          # stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: Sequence[int],
                               n_bins: int = 10) -> dict[str, Any]:
    """Confidence ECE with equal-width bins over [0, 1].

    Bin edges are half-open (lo, hi] with the first bin closed at 0, so every
    confidence lands in exactly one bin and the bin populations sum to N.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be (n, n_classes), got {probs.shape}")
    labels = np.asarray(labels, dtype=int)
    if len(labels) != probs.shape[0]:
        raise ValueError(f"{len(labels)} labels but {probs.shape[0]} rows")
    if probs.shape[0] == 0:
        return {"ece": 0.0, "n": 0, "n_bins": n_bins, "bins": []}

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin": i, "lo": float(lo), "hi": float(hi), "count": 0,
                         "accuracy": None, "confidence": None, "gap": None})
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        gap = abs(acc - avg_conf)
        ece += (count / n) * gap
        bins.append({"bin": i, "lo": float(lo), "hi": float(hi), "count": count,
                     "accuracy": acc, "confidence": avg_conf, "gap": gap})

    return {"ece": float(ece), "n": n, "n_bins": n_bins, "bins": bins,
            "mean_confidence": float(conf.mean()), "accuracy": float(correct.mean())}


def reliability_data(probs: np.ndarray, labels: Sequence[int],
                     n_bins: int = 10) -> dict[str, list]:
    """Points for a reliability diagram. Empty bins are omitted, not plotted at zero."""
    res = expected_calibration_error(probs, labels, n_bins)
    pts = [b for b in res["bins"] if b["count"] > 0]
    return {"confidence": [b["confidence"] for b in pts],
            "accuracy": [b["accuracy"] for b in pts],
            "counts": [b["count"] for b in pts],
            "ece": res["ece"]}


def fit_temperature(logits: np.ndarray, labels: Sequence[int], split_name: str = "validation",
                    lo: float = 0.05, hi: float = 10.0, tol: float = 1e-4) -> dict[str, Any]:
    """Fit a single temperature by minimising NLL via ternary search.

    The NLL of temperature scaling is convex in log-temperature, so ternary
    search converges without needing an optimiser or a gradient -- which keeps
    this deterministic and dependency-free.
    """
    if str(split_name).lower() == "test":
        raise ValueError(
            "Refusing to fit temperature on the test split: that is tuning on test. "
            "Fit on validation and apply the fitted value to test.")
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if logits.shape[0] == 0:
        return {"temperature": 1.0, "nll_before": 0.0, "nll_after": 0.0, "split": split_name}

    def nll(t: float) -> float:
        p = softmax(logits, t)
        return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())

    a, b = math.log(lo), math.log(hi)
    while b - a > tol:
        m1, m2 = a + (b - a) / 3.0, b - (b - a) / 3.0
        if nll(math.exp(m1)) < nll(math.exp(m2)):
            b = m2
        else:
            a = m1
    t = math.exp((a + b) / 2.0)
    return {"temperature": float(t), "nll_before": nll(1.0), "nll_after": nll(t),
            "split": split_name,
            "note": "Fitted on validation; apply this value unchanged to test."}


def calibrate_and_evaluate(logits: np.ndarray, labels: Sequence[int],
                           temperature: float = 1.0, n_bins: int = 10) -> dict[str, Any]:
    """ECE before and after applying an already-fitted temperature."""
    before = expected_calibration_error(softmax(logits, 1.0), labels, n_bins)
    after = expected_calibration_error(softmax(logits, temperature), labels, n_bins)
    return {
        "temperature": float(temperature),
        "ece_before": before["ece"], "ece_after": after["ece"],
        "accuracy": before["accuracy"],   # unchanged: temperature cannot alter argmax
        "reliability_before": reliability_data(softmax(logits, 1.0), labels, n_bins),
        "reliability_after": reliability_data(softmax(logits, temperature), labels, n_bins),
        "note": ("Temperature scaling is monotone, so accuracy and Macro-F1 are "
                 "identical before and after; only confidence changes."),
    }
