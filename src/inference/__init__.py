"""Module 10 — inference and demo.

Read-only with respect to every trained artifact. This package imports the
Module 1-9 implementations that produced the research results; it does not
reimplement them. The single new model-facing component is `text_encoder`,
which exists because Modules 1-9 provide no single-raw-text -> embedding API.
"""

from src.inference.pipeline import (  # noqa: F401
    Dependencies, InferenceConfigError, build_default_dependencies,
    compute_confidence, detect_language, display_state, load_frozen_fusion,
    run_inference, select_claim, validate_input)
from src.inference.report import (  # noqa: F401
    ReasonSentence, build_reason_block, render_report, validate_provenance)
from src.inference.text_encoder import (  # noqa: F401
    EncoderProvenanceError, InferenceEncoder)

__all__ = [
    "Dependencies", "InferenceConfigError", "EncoderProvenanceError",
    "InferenceEncoder", "ReasonSentence", "build_default_dependencies",
    "build_reason_block", "compute_confidence", "detect_language",
    "display_state", "load_frozen_fusion", "render_report", "run_inference",
    "select_claim", "validate_input", "validate_provenance",
]
