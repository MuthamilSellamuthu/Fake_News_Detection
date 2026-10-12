"""Module 8 — Explainable AI.

Explains the integrated decision using only externally inspectable signals:
token attributions, retrieved evidence, verifier probabilities, and the
fusion's additive contributions.

    token_attribution      deterministic occlusion attribution (PRIMARY)
    shap_explainer         KernelSHAP over embedding dimensions, labelled honestly
    verification_explainer evidence + verdict + contribution accounting
    combined_explanation   assembles one record's explanation and text report
    explanation_metrics    aggregate statistics and faithfulness summaries

A fallback is never labelled "shap"; `resolve_method()` enforces that.
"""

from src.explainability.combined_explanation import (  # noqa: F401
    build_explanation_record, build_reason, render_text_report)
from src.explainability.explanation_metrics import (  # noqa: F401
    summarise_explanations, summarise_faithfulness)
from src.explainability.shap_explainer import (  # noqa: F401
    SHAP_EMBEDDING_LABEL, explain_embedding, is_shap_available, resolve_method,
    select_background)
from src.explainability.token_attribution import (  # noqa: F401
    OCCLUSION, TokenAttribution, attribution_concentration, deletion_test,
    insertion_test, occlusion_attribution, simple_tokenize, stability_check)
from src.explainability.verification_explainer import (  # noqa: F401
    contribution_breakdown, explain_verification)

__all__ = [
    "OCCLUSION", "SHAP_EMBEDDING_LABEL", "TokenAttribution",
    "occlusion_attribution", "simple_tokenize", "attribution_concentration",
    "deletion_test", "insertion_test", "stability_check",
    "is_shap_available", "explain_embedding", "select_background", "resolve_method",
    "explain_verification", "contribution_breakdown",
    "build_explanation_record", "build_reason", "render_text_report",
    "summarise_explanations", "summarise_faithfulness",
]
