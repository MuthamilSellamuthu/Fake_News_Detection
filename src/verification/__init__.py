"""Module 6 — fact verification.

FEVER and X-FACT are kept as SEPARATE tasks with separate label spaces and
separate reported results. The XLM-R encoder is shared so cross-lingual
transfer is measurable; the classification head is replaced, never reused.

    dataset          label spaces, (claim, evidence) pairing, stratified subsets
    verifier_model   XLM-R + replaceable head
    trainer          resumable training, checkpoints incl. RNG state
    calibration      ECE, reliability data, temperature scaling
    ablations        the six configured ablations
"""

from src.verification.calibration import (  # noqa: F401
    calibrate_and_evaluate,
    expected_calibration_error,
    fit_temperature,
    reliability_data,
    softmax,
)
from src.verification.dataset import (  # noqa: F401
    FEVER_LABELS,
    FEVER_SPACE,
    LABEL_SPACES,
    X_FACT_LABELS,
    X_FACT_SPACE,
    LabelSpace,
    VerifierExample,
    assert_label_spaces_disjoint,
    build_examples,
    get_label_space,
    stratified_subset,
)

__all__ = [
    "FEVER_LABELS", "X_FACT_LABELS", "FEVER_SPACE", "X_FACT_SPACE", "LABEL_SPACES",
    "LabelSpace", "VerifierExample", "build_examples", "get_label_space",
    "stratified_subset", "assert_label_spaces_disjoint",
    "expected_calibration_error", "reliability_data", "fit_temperature",
    "calibrate_and_evaluate", "softmax",
]
