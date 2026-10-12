"""Module 8 — SHAP, applied honestly.

THE HONEST POSITION. Module 4's CE-MLP consumes a 768-d sentence embedding.
KernelSHAP over that input is perfectly valid and produces genuine Shapley
values -- but they attribute to EMBEDDING DIMENSIONS, and "dimension 412
contributed +0.03" is not an explanation a reader can use.

So this module:
  * computes SHAP where SHAP is actually applicable (embedding space), and
  * labels it `shap_embedding_space`, never plain "shap", and
  * refuses to present it as token attribution.

`is_shap_available()` and `explain_embedding()` fail loudly rather than
silently degrading, so a fallback can never be mislabelled. Token-level
explanation lives in `token_attribution.py` and is labelled `occlusion`.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np

SHAP_EMBEDDING_LABEL = "shap_embedding_space"


def is_shap_available() -> tuple[bool, str]:
    try:
        import shap  # noqa: F401
        return True, getattr(shap, "__version__", "unknown")
    except ImportError as exc:
        return False, str(exc)


def explain_embedding(
    embedding: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    n_samples: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """KernelSHAP over embedding dimensions.

    Raises if shap is missing. Callers must NOT catch this and substitute
    occlusion under the same label -- that is precisely the mislabelling this
    module exists to prevent.
    """
    ok, info = is_shap_available()
    if not ok:
        raise ImportError(
            "shap is not installed, so SHAP values cannot be computed.\n"
            f"  detail: {info}\n"
            "  Use the occlusion explainer instead -- and label it 'occlusion'.\n"
            "  Install with: pip install shap")
    import shap

    t0 = time.perf_counter()
    x = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
    bg = np.asarray(background, dtype=np.float64)
    if bg.ndim != 2 or bg.shape[1] != x.shape[1]:
        raise ValueError(f"background {bg.shape} incompatible with embedding {x.shape}")

    np.random.seed(int(seed))            # KernelSHAP samples coalitions
    explainer = shap.KernelExplainer(predict_fn, bg)
    values = explainer.shap_values(x, nsamples=int(n_samples), silent=True)
    vals = np.asarray(values[0] if isinstance(values, list) else values).ravel()

    order = np.argsort(-np.abs(vals))
    return {
        "method": SHAP_EMBEDDING_LABEL,
        "shap_version": info,
        "n_features": int(vals.size),
        "attributions": vals.astype(float).tolist(),
        "expected_value": float(np.asarray(explainer.expected_value).ravel()[0]),
        "top_dimensions": [{"dimension": int(i), "attribution": float(vals[i])}
                           for i in order[:20]],
        "seconds": round(time.perf_counter() - t0, 4),
        "background_size": int(bg.shape[0]),
        "n_samples": int(n_samples),
        "interpretability_caveat": (
            "These are Shapley values over XLM-R embedding dimensions, not over "
            "tokens. They are faithful to the classifier but are not directly "
            "human-readable; use the occlusion explanation for token-level "
            "reporting."),
    }


def select_background(embeddings: np.ndarray, size: int = 50,
                      seed: int = 42) -> np.ndarray:
    """Deterministic background sample for KernelSHAP.

    A fixed seed and sorted indices make the background reproducible, which
    matters because SHAP values are defined relative to it -- a different
    background silently changes every attribution.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.shape[0] <= size:
        return x
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(x.shape[0], size=int(size), replace=False))
    return x[idx]


def resolve_method(requested: str, allow_shap: bool = True) -> tuple[str, str]:
    """Decide the method and return (method_label, reason).

    Never returns the label "shap" for a non-SHAP computation.
    """
    if requested == "occlusion":
        return "occlusion", "configured primary method"
    if requested in ("shap", SHAP_EMBEDDING_LABEL):
        if not allow_shap:
            return "occlusion", "SHAP disabled in config; using occlusion (labelled as such)"
        ok, info = is_shap_available()
        if ok:
            return SHAP_EMBEDDING_LABEL, f"shap {info} available (embedding space)"
        return "occlusion", (
            "shap not installed; falling back to occlusion. The record is "
            "labelled 'occlusion', NOT 'shap'.")
    raise ValueError(f"unknown explanation method {requested!r}")
