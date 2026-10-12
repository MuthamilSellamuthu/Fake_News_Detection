"""Module 7 — Integrated Fake News Decision System.

Fuses Module 4's fake/real classifier with Module 6's evidence retrieval and
fact verification. Reads Modules 1-6; writes only under `outputs/module7/`.

    module4_inference  regenerate per-record P(fake) from the frozen checkpoint
    fusion             calibrated weighted fusion, fitted on validation only
    error_analysis     error categories, language deltas, retrieval effect

Every tunable parameter is fitted on the Module 2 validation split. The
fitting routines raise `TestSplitLeakage` if pointed at test.
"""

from src.integration.error_analysis import (  # noqa: F401
    CATEGORIES,
    categorise,
    categorise_all,
    language_comparison,
    retrieval_effect_analysis,
    sample_examples,
)
from src.integration.fusion import (  # noqa: F401
    FusionParameters,
    MetaClassifierResult,
    TestSplitLeakage,
    apply_fusion,
    fit_binary_temperature,
    fit_fusion,
    fit_meta_classifier,
    fuse,
    rank_normalise,
    verifier_to_fake_prob,
)
from src.integration.module4_inference import (  # noqa: F401
    build_prediction_records,
    file_sha256,
    load_split_arrays,
    predict_p_fake,
    summarise,
)

__all__ = [
    "FusionParameters", "MetaClassifierResult", "TestSplitLeakage",
    "verifier_to_fake_prob", "fuse", "fit_fusion", "apply_fusion",
    "fit_binary_temperature", "fit_meta_classifier", "rank_normalise",
    "load_split_arrays", "predict_p_fake", "build_prediction_records",
    "summarise", "file_sha256",
    "CATEGORIES", "categorise", "categorise_all", "sample_examples",
    "language_comparison", "retrieval_effect_analysis",
]
