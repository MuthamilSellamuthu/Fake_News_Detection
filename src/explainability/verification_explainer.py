"""Module 8 — explain the evidence/verification side of a decision.

Nothing here is a model. It surfaces signals the system actually produced:
which evidence was retrieved, how confident retrieval was, what the verifier
predicted, and how much each side moved the fused probability. No hidden
reasoning is exposed or invented -- only externally inspectable numbers.
"""

from __future__ import annotations

from typing import Any, Sequence


def explain_verification(verifier_probs: dict[str, float] | None,
                         evidence: Sequence[dict[str, Any]] | None,
                         retrieval_confidence: Sequence[float] | None = None,
                         max_evidence: int = 5,
                         truncate_chars: int = 300) -> dict[str, Any]:
    """Structured account of the evidence side for one record."""
    if not verifier_probs:
        return {"has_verification": False,
                "reason": "no verifier output for this record",
                "evidence": [], "n_evidence": 0}

    ranked = sorted(verifier_probs.items(), key=lambda kv: (-kv[1], kv[0]))
    verdict, top_p = ranked[0]

    items = []
    for i, ev in enumerate(list(evidence or [])[:max_evidence]):
        text = str(ev.get("text", ""))
        items.append({
            "rank": i + 1,
            "evidence_id": ev.get("evidence_id"),
            "text": text[:truncate_chars] + ("..." if len(text) > truncate_chars else ""),
            "retrieval_confidence": (float(retrieval_confidence[i])
                                     if retrieval_confidence and i < len(retrieval_confidence)
                                     else None),
        })

    return {
        "has_verification": True,
        "verdict": verdict,
        "verdict_probability": float(top_p),
        "probabilities": {k: float(v) for k, v in verifier_probs.items()},
        "margin": float(top_p - ranked[1][1]) if len(ranked) > 1 else None,
        "evidence": items,
        "n_evidence": len(items),
        "certainty": ("confident" if top_p >= 0.7 else
                      "weak" if top_p >= 0.5 else "very weak"),
        "note": ("Verdict is the argmax of the verifier's own probabilities. "
                 "Margin is the gap to the runner-up; a small margin means the "
                 "verdict should not be read as decisive."),
    }


def contribution_breakdown(p_fake_classifier: float, p_fake_verifier: float | None,
                           alpha: float, fused_p_fake: float,
                           has_verifier: bool) -> dict[str, Any]:
    """How much each side moved the final probability.

    Contributions are the actual additive terms of the fusion, so they sum to
    the fused probability (before any temperature scaling). Reporting anything
    else would be a narrative rather than an accounting.
    """
    if not has_verifier or p_fake_verifier is None:
        return {"classifier_contribution": float(p_fake_classifier),
                "verifier_contribution": 0.0, "alpha": float(alpha),
                "fused_p_fake": float(fused_p_fake), "verifier_used": False,
                "note": ("No verifier output; the fusion fell back to the "
                         "classifier alone by policy.")}
    c = float(alpha) * float(p_fake_classifier)
    v = (1.0 - float(alpha)) * float(p_fake_verifier)
    return {"classifier_contribution": c, "verifier_contribution": v,
            "alpha": float(alpha), "fused_p_fake": float(fused_p_fake),
            "verifier_used": True,
            "sum_of_contributions": c + v,
            "matches_fused": abs((c + v) - float(fused_p_fake)) < 1e-6,
            "note": ("Contributions are the literal additive terms of "
                     "alpha*P_clf + (1-alpha)*P_ver. If matches_fused is false, "
                     "temperature scaling was applied afterwards.")}
