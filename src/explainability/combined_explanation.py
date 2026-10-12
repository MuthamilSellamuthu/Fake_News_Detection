"""Module 8 — assemble one record's full explanation.

The decision sentence is GENERATED FROM THE ACTUAL NUMBERS. There is no
template that assumes a conclusion: the wording is chosen by comparing the
values that were really produced, so a weak or contradictory case reads as
weak or contradictory.
"""

from __future__ import annotations

from typing import Any, Sequence

from src.explainability.verification_explainer import (
    contribution_breakdown, explain_verification)


def build_reason(predicted_label: int, p_fake: float, classifier_p: float,
                 verification: dict[str, Any],
                 contributions: dict[str, Any]) -> str:
    """Compose the decision explanation from measured values only."""
    label = "FAKE" if int(predicted_label) == 1 else "REAL"
    parts = [f"Predicted {label} with fused P(fake) = {p_fake:.3f}."]

    strength = ("strongly" if classifier_p >= 0.8 or classifier_p <= 0.2 else
                "moderately" if classifier_p >= 0.65 or classifier_p <= 0.35 else
                "weakly")
    direction = "fake" if classifier_p >= 0.5 else "real"
    parts.append(f"The classifier {strength} favours {direction} "
                 f"(P(fake) = {classifier_p:.3f}).")

    if not verification.get("has_verification"):
        parts.append("No verifier output was available for this record, so the "
                     "decision rests on the classifier alone.")
        return " ".join(parts)

    verdict = verification["verdict"]
    vp = verification["verdict_probability"]
    parts.append(f"The verifier returned {verdict} at {vp:.3f} "
                 f"({verification['certainty']}) over "
                 f"{verification['n_evidence']} retrieved evidence item(s).")

    if contributions.get("verifier_used"):
        c, v = contributions["classifier_contribution"], contributions["verifier_contribution"]
        dominant = "classifier" if c >= v else "verification"
        parts.append(f"In fusion the {dominant} side contributed more "
                     f"(classifier {c:.3f} vs verification {v:.3f}, "
                     f"alpha = {contributions['alpha']}).")

    margin = verification.get("margin")
    if margin is not None and margin < 0.15:
        parts.append(f"The verifier's margin over its runner-up is only "
                     f"{margin:.3f}, so its verdict is not decisive.")
    return " ".join(parts)


def build_explanation_record(
    record: dict[str, Any],
    attribution: Any | None,
    verifier_probs: dict[str, float] | None,
    evidence: Sequence[dict[str, Any]] | None,
    retrieval_confidence: Sequence[float] | None,
    alpha: float,
    method_label: str,
    top_k: int = 10,
    max_evidence: int = 5,
    truncate_chars: int = 300,
) -> dict[str, Any]:
    """One explanation record, matching the schema declared in the config."""
    verification = explain_verification(verifier_probs, evidence, retrieval_confidence,
                                        max_evidence, truncate_chars)
    contributions = contribution_breakdown(
        float(record["module4_p_fake"]),
        record.get("p_fake_verifier"),
        float(alpha), float(record["fused_p_fake"]),
        bool(record.get("has_verifier")))

    tops = attribution.top_tokens(top_k) if attribution is not None else {
        "top_positive_tokens": [], "top_negative_tokens": []}

    out: dict[str, Any] = {
        "record_id": record["record_id"],
        "language": record.get("language", "unknown"),
        "text": record.get("text", ""),
        "gold_label": int(record["gold_label"]),
        "predicted_label": int(record["final_prediction"]),
        "p_fake": float(record["fused_p_fake"]),
        "explanation_method": method_label,
        "token_attributions": (
            [{"token": t, "attribution": float(a), "position": i}
             for i, (t, a) in enumerate(zip(attribution.tokens, attribution.attributions))]
            if attribution is not None else []),
        "top_positive_tokens": tops["top_positive_tokens"],
        "top_negative_tokens": tops["top_negative_tokens"],
        "evidence": verification.get("evidence", []),
        "verification_probabilities": verification.get("probabilities", {}),
        "fusion_contributions": contributions,
        "confidence": float(max(record["fused_p_fake"], 1.0 - record["fused_p_fake"])),
    }
    out["final_decision_reason"] = build_reason(
        out["predicted_label"], out["p_fake"], float(record["module4_p_fake"]),
        verification, contributions)
    if attribution is not None:
        out["attribution_diagnostics"] = {
            "n_tokens": attribution.n_tokens, "truncated": attribution.truncated,
            "seconds": round(attribution.seconds, 4), "error": attribution.error}
    return out


def render_text_report(explanation: dict[str, Any]) -> str:
    """Human-readable rendering, generated from the record's own values."""
    e = explanation
    lines = [f"Record: {e['record_id']}  [{e['language']}]",
             f"Prediction: {'FAKE' if e['predicted_label'] == 1 else 'REAL'}"
             f"   (gold: {'FAKE' if e['gold_label'] == 1 else 'REAL'})",
             f"Fused P(fake) = {e['p_fake']:.3f}   confidence = {e['confidence']:.3f}",
             "", f"Explanation method: {e['explanation_method']}"]
    if e["top_positive_tokens"]:
        lines.append("Tokens pushing towards FAKE:")
        lines += [f"  {t['token']!r}  {t['attribution']:+.4f}"
                  for t in e["top_positive_tokens"][:5]]
    if e["top_negative_tokens"]:
        lines.append("Tokens pushing towards REAL:")
        lines += [f"  {t['token']!r}  {t['attribution']:+.4f}"
                  for t in e["top_negative_tokens"][:5]]
    if e["verification_probabilities"]:
        lines += ["", "Verification:"]
        lines += [f"  {k:<20} {v:.3f}"
                  for k, v in sorted(e["verification_probabilities"].items(),
                                     key=lambda kv: -kv[1])]
    if e["evidence"]:
        lines += ["", "Evidence:"]
        lines += [f"  {ev['rank']}. {ev['text']}" for ev in e["evidence"]]
    fc = e.get("fusion_contributions", {})
    if fc.get("verifier_used"):
        lines += ["", "Fusion:",
                  f"  classifier contribution   = {fc['classifier_contribution']:.4f}",
                  f"  verification contribution = {fc['verifier_contribution']:.4f}",
                  f"  alpha                     = {fc['alpha']}"]
    lines += ["", "Primary reason:", "  " + e["final_decision_reason"]]
    return "\n".join(lines)
