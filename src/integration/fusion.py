"""Module 7 — calibrated weighted fusion of classifier and verifier.

THE MAPPING. A three-way verification verdict has to become a fake/real
signal, and the mapping must not smuggle in an opinion:

    P_fake_verifier = 0.5 + 0.5 * (P_refute - P_support)

  * fully refuted claim  -> 1.0   (evidence contradicts it: fake)
  * fully supported      -> 0.0   (evidence corroborates it: real)
  * NOT ENOUGH INFO      -> 0.5   (uninformative, and stays uninformative)

The NEI behaviour is the point. Treating "no evidence found" as weak evidence
of fakeness is a common and invisible way to inflate results on a corpus where
retrieval is weak -- which is exactly this project's situation. Here NEI
contributes nothing in either direction.

X-FACT's veracity labels use the same algebra with `false`/`true` in the
refute/support roles and `partly true/misleading` as neutral. The two label
spaces are still never merged; each is mapped through its own role
assignment, declared in the config.

THE FUSION.

    P_final = alpha * P_fake(module4) + (1 - alpha) * P_fake(verifier)

`alpha` and the decision threshold are chosen by exhaustive grid search on the
VALIDATION split only, maximising macro-F1. A grid rather than an optimiser
because it is deterministic, inspectable, and reproducible from the config
alone -- no learning rate, no initialisation, no seed sensitivity.

Records with no verifier output fall back to alpha = 1 (classifier only) and
are counted. Imputing 0.5 for them would fabricate a verdict the system never
produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


class TestSplitLeakage(RuntimeError):
    """Raised when a fitting routine is pointed at the test split."""


@dataclass
class FusionParameters:
    """Everything needed to reproduce a decision. Serialised to JSON."""

    alpha: float
    threshold: float
    temperature: float = 1.0
    fit_split: str = "validation"
    select_on: str = "macro_f1"
    n_fit_records: int = 0
    n_fit_with_verifier: int = 0
    grid_size: int = 0
    selected_score: float = 0.0
    verifier_mapping: str = "symmetric_refute_minus_support"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": float(self.alpha), "threshold": float(self.threshold),
            "temperature": float(self.temperature), "fit_split": self.fit_split,
            "select_on": self.select_on, "n_fit_records": int(self.n_fit_records),
            "n_fit_with_verifier": int(self.n_fit_with_verifier),
            "grid_size": int(self.grid_size),
            "selected_score": float(self.selected_score),
            "verifier_mapping": self.verifier_mapping, "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FusionParameters":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def verifier_to_fake_prob(p_refute: float, p_support: float,
                          p_neutral: float | None = None) -> float:
    """Map a 3-way verdict to P(fake). See the module docstring.

    Validates that the probabilities are a distribution; a malformed verifier
    output must be rejected, not quietly rescaled.
    """
    for name, v in (("p_refute", p_refute), ("p_support", p_support)):
        if not (0.0 - 1e-6 <= v <= 1.0 + 1e-6):
            raise ValueError(f"{name}={v} outside [0, 1]")
    if p_neutral is not None:
        total = p_refute + p_support + p_neutral
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"verifier probabilities sum to {total:.6f}, expected 1.0")
    return float(min(1.0, max(0.0, 0.5 + 0.5 * (p_refute - p_support))))


def fuse(p_fake_m4: np.ndarray, p_fake_verifier: np.ndarray,
         has_verifier: np.ndarray, alpha: float) -> np.ndarray:
    """Weighted fusion; records without a verdict use the classifier alone."""
    p_fake_m4 = np.asarray(p_fake_m4, dtype=np.float64)
    p_fake_verifier = np.asarray(p_fake_verifier, dtype=np.float64)
    has = np.asarray(has_verifier, dtype=bool)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha={alpha} outside [0, 1]")
    fused = alpha * p_fake_m4 + (1.0 - alpha) * p_fake_verifier
    return np.where(has, fused, p_fake_m4)


def _macro_f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over both classes, so the minority class cannot be ignored."""
    f1s = []
    for c in (0, 1):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


SCORERS = {"macro_f1": _macro_f1_binary, "accuracy": _accuracy}


def fit_fusion(
    y_true: Sequence[int],
    p_fake_m4: Sequence[float],
    p_fake_verifier: Sequence[float],
    has_verifier: Sequence[bool],
    alpha_grid: Sequence[float],
    threshold_grid: Sequence[float],
    split_name: str = "validation",
    select_on: str = "macro_f1",
) -> tuple[FusionParameters, list[dict[str, Any]]]:
    """Grid-search alpha and the threshold on the fitting split.

    Refuses `split_name == "test"`. This is the single most important guard in
    Module 7: fitting here on test would make every reported test number
    optimistically biased in a way no downstream check could detect.
    """
    if str(split_name).lower() == "test":
        raise TestSplitLeakage(
            "Refusing to fit fusion parameters on the test split. Fit on "
            "validation and apply the frozen parameters to test.")
    if select_on not in SCORERS:
        raise ValueError(f"unknown selection metric {select_on!r}; known: {sorted(SCORERS)}")

    y = np.asarray(y_true, dtype=int)
    m4 = np.asarray(p_fake_m4, dtype=np.float64)
    ver = np.asarray(p_fake_verifier, dtype=np.float64)
    has = np.asarray(has_verifier, dtype=bool)
    if not (len(y) == len(m4) == len(ver) == len(has)):
        raise ValueError(f"length mismatch: {len(y)}/{len(m4)}/{len(ver)}/{len(has)}")

    scorer = SCORERS[select_on]
    trace: list[dict[str, Any]] = []
    best = None
    # Ties resolve to the HIGHEST alpha, i.e. towards the classifier alone.
    # This is deliberate and conservative. When several alphas score equally --
    # which happens whenever the classifier already separates the classes and
    # the threshold simply adapts -- selecting a low alpha would report that the
    # verifier contributed when it demonstrably did not. Preferring the highest
    # alpha means the verifier is credited only when it strictly improves the
    # score, so a reported alpha < 1 is real evidence that evidence helped.
    # Descending alpha + strict > gives that, deterministically.
    for alpha in sorted(alpha_grid, reverse=True):
        fused = fuse(m4, ver, has, alpha)
        for thr in sorted(threshold_grid):
            score = scorer(y, (fused >= thr).astype(int))
            trace.append({"alpha": float(alpha), "threshold": float(thr),
                          "score": float(score)})
            if best is None or score > best[0] + 1e-12:
                best = (score, float(alpha), float(thr))

    score, alpha, thr = best if best else (0.0, 1.0, 0.5)
    params = FusionParameters(
        alpha=alpha, threshold=thr, fit_split=split_name, select_on=select_on,
        n_fit_records=int(len(y)), n_fit_with_verifier=int(has.sum()),
        grid_size=len(trace), selected_score=score,
        notes=("alpha=1.0 means the verifier never helped on validation and the "
               "fusion reduced to the classifier."
               if alpha >= 1.0 - 1e-9 else
               "alpha<1 means verifier evidence contributed on validation."))
    return params, trace


def apply_fusion(p_fake_m4: Sequence[float], p_fake_verifier: Sequence[float],
                 has_verifier: Sequence[bool], params: FusionParameters
                 ) -> dict[str, np.ndarray]:
    """Apply FROZEN parameters. Contains no fitting of any kind."""
    fused = fuse(p_fake_m4, p_fake_verifier, has_verifier, params.alpha)
    if params.temperature and params.temperature != 1.0:
        fused = _temperature_scale_binary(fused, params.temperature)
    return {"p_fake": fused, "p_real": 1.0 - fused,
            "prediction": (fused >= params.threshold).astype(int)}


def _temperature_scale_binary(p: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature scaling in binary logit space, clipped for stability."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) / float(temperature))))


def fit_binary_temperature(y_true: Sequence[int], p_fake: Sequence[float],
                           split_name: str = "validation",
                           lo: float = 0.05, hi: float = 10.0,
                           tol: float = 1e-4) -> dict[str, Any]:
    """Fit a binary temperature by ternary search on NLL (convex in log T)."""
    if str(split_name).lower() == "test":
        raise TestSplitLeakage("Refusing to fit calibration temperature on test.")
    import math

    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p_fake, dtype=np.float64), 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return {"temperature": 1.0, "nll_before": 0.0, "nll_after": 0.0,
                "split": split_name}

    def nll(t: float) -> float:
        q = _temperature_scale_binary(p, t)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    a, b = math.log(lo), math.log(hi)
    while b - a > tol:
        m1, m2 = a + (b - a) / 3.0, b - (b - a) / 3.0
        if nll(math.exp(m1)) < nll(math.exp(m2)):
            b = m2
        else:
            a = m1
    t = math.exp((a + b) / 2.0)
    return {"temperature": float(t), "nll_before": nll(1.0), "nll_after": nll(t),
            "split": split_name}


def rank_normalise(scores: Sequence[float]) -> list[float]:
    """Per-query rank normalisation to [0, 1].

    Dense cosine lives in [-1, 1]; BM25 is unbounded and non-negative. They are
    not on a common scale and must never be summed raw. Ranking discards the
    magnitude but is comparable across retrievers, which is what fusion needs.
    Best-ranked item scores 1.0; a single candidate also scores 1.0.
    """
    s = list(scores)
    if not s:
        return []
    if len(s) == 1:
        return [1.0]
    order = sorted(range(len(s)), key=lambda i: (-float(s[i]), i))
    out = [0.0] * len(s)
    for rank, i in enumerate(order):
        out[i] = 1.0 - rank / (len(s) - 1)
    return out


@dataclass
class MetaClassifierResult:
    fitted: bool
    reason: str = ""
    coefficients: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    cv_scores: list[float] = field(default_factory=list)
    n_records: int = 0
    feature_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"fitted": self.fitted, "reason": self.reason,
                "coefficients": self.coefficients, "intercept": self.intercept,
                "cv_scores": self.cv_scores, "n_records": self.n_records,
                "feature_names": self.feature_names}


def fit_meta_classifier(features: np.ndarray, y_true: Sequence[int],
                        feature_names: Sequence[str], split_name: str = "validation",
                        min_records: int = 200, seed: int = 42,
                        cv_folds: int = 5) -> MetaClassifierResult:
    """Secondary ablation: logistic regression over the fusion inputs.

    Declines rather than forcing a fit when the sample is too small or a class
    is absent. A meta-classifier fitted on a handful of records would look like
    a result while being noise.
    """
    if str(split_name).lower() == "test":
        raise TestSplitLeakage("Refusing to fit the meta-classifier on test.")
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    if n < int(min_records):
        return MetaClassifierResult(False, f"only {n} records; {min_records} required",
                                    n_records=n, feature_names=list(feature_names))
    if len(np.unique(y)) < 2:
        return MetaClassifierResult(False, "only one class present in the fitting split",
                                    n_records=n, feature_names=list(feature_names))
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return MetaClassifierResult(False, "scikit-learn not available",
                                    n_records=n, feature_names=list(feature_names))

    clf = LogisticRegression(max_iter=1000, random_state=int(seed))
    folds = min(int(cv_folds), int(np.min(np.bincount(y))))
    cv = (cross_val_score(clf, X, y, cv=folds, scoring="f1_macro").tolist()
          if folds >= 2 else [])
    clf.fit(X, y)
    return MetaClassifierResult(
        True, "", {n_: float(c) for n_, c in zip(feature_names, clf.coef_[0])},
        float(clf.intercept_[0]), cv, n, list(feature_names))
