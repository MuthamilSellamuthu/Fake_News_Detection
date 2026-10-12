"""Module 10 — reason and human-readable report.

SLOT FILLING ONLY. No language model writes any part of this. Every sentence
is a template whose slots come from a computed field, and every sentence
carries a `provenance` list naming the result paths it was built from. A
sentence that cannot name its source cannot be emitted -- `ReasonSentence`
raises on an empty provenance list.

Sentences are tagged by kind so the reader can tell them apart:
    MEASURED       a number the pipeline computed
    INTERPRETATION what the model's output means, stated as a model output
    LIMITATION     a constraint on how far the result can be read
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MEASURED, INTERPRETATION, LIMITATION = "MEASURED", "INTERPRETATION", "LIMITATION"


@dataclass(frozen=True)
class ReasonSentence:
    template_id: str
    kind: str
    text: str
    provenance: list[str]

    def __post_init__(self) -> None:
        if not self.provenance:
            raise ValueError(f"{self.template_id}: no provenance -> refusing to emit")
        if self.kind not in (MEASURED, INTERPRETATION, LIMITATION):
            raise ValueError(f"{self.template_id}: unknown kind {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"template_id": self.template_id, "kind": self.kind,
                "text": self.text, "provenance": list(self.provenance)}


def _get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def build_reason_block(result: dict[str, Any]) -> dict[str, Any]:
    """Compose the reason from measured values only."""
    f = result.get("fusion", {})
    c = result.get("classifier", {})
    v = result.get("verification", {})
    r = result.get("retrieval", {})
    x = result.get("explainability", {})
    s: list[ReasonSentence] = []

    state, fused, thr = f.get("display_state"), f.get("fused_p_fake"), f.get("threshold")
    s.append(ReasonSentence(
        "verdict", MEASURED,
        f"Final state {state}: fused P(fake) = {fused:.4f} against a frozen "
        f"decision threshold of {thr:.2f}.",
        ["fusion.display_state", "fusion.fused_p_fake", "fusion.threshold"]))

    if state == "UNCERTAIN":
        s.append(ReasonSentence(
            "uncertain", LIMITATION,
            f"The probability is within {f.get('uncertain_band')} of the threshold, "
            "so the decision is reported as UNCERTAIN. The underlying threshold "
            f"outcome was {f.get('final_label')}.",
            ["fusion.uncertain_band", "fusion.final_label"]))

    p = c.get("p_fake")
    strength = ("strongly" if p >= 0.8 or p <= 0.2 else
                "moderately" if p >= 0.65 or p <= 0.35 else "weakly")
    s.append(ReasonSentence(
        "classifier", MEASURED,
        f"The classifier assigned P(fake) = {p:.4f}, {strength} favouring "
        f"{'fake' if p >= 0.5 else 'real'}.",
        ["classifier.p_fake"]))

    if v.get("available"):
        s.append(ReasonSentence(
            "verifier", MEASURED,
            f"The {v.get('dataset')} verifier returned \"{v.get('verdict')}\" at "
            f"p = {v.get('verdict_probability'):.3f} over "
            f"{v.get('n_evidence_used')} candidate evidence item(s). Label space: "
            f"{', '.join(v.get('label_space', []))}.",
            ["verification.dataset", "verification.verdict",
             "verification.verdict_probability", "verification.n_evidence_used"]))
        s.append(ReasonSentence(
            "fusion_math", MEASURED,
            f"Fusion: {f.get('alpha')} x {p:.4f} (classifier) + "
            f"{1 - float(f.get('alpha', 1)):.2f} x {f.get('p_fake_verifier'):.4f} "
            f"(verifier) = {f.get('fused_p_fake_before_temperature'):.4f}, then "
            f"temperature T = {f.get('temperature')} -> {fused:.4f}.",
            ["fusion.alpha", "fusion.p_fake_verifier",
             "fusion.fused_p_fake_before_temperature", "fusion.temperature"]))
        clf_dir = "fake" if p >= 0.5 else "real"
        ver_fake = float(f.get("p_fake_verifier", 0.5))
        ver_dir = "fake" if ver_fake > 0.5 else "real" if ver_fake < 0.5 else "neither"
        agree = (clf_dir == ver_dir)
        s.append(ReasonSentence(
            "agreement", INTERPRETATION,
            ("The classifier and the evidence verifier point the same way."
             if agree else
             f"DISAGREEMENT: the classifier points toward {clf_dir} while the "
             f"verifier-derived probability points toward {ver_dir}. Because "
             f"alpha = {f.get('alpha')}, the classifier dominates the fused result."),
            ["classifier.p_fake", "fusion.p_fake_verifier", "fusion.alpha"]))
        if v.get("verdict") in ("NOT ENOUGH INFO", "partly true/misleading"):
            s.append(ReasonSentence(
                "neutral_verdict", LIMITATION,
                f"The verifier returned \"{v.get('verdict')}\", which is a statement "
                "about insufficient or mixed evidence. It is NOT evidence that the "
                "article is false.",
                ["verification.verdict"]))
    else:
        s.append(ReasonSentence(
            "no_verifier", MEASURED,
            f"No verifier output ({v.get('reason')}). The fusion fell back to the "
            "classifier alone, by Module 7's missing-verifier policy.",
            ["verification.available", "verification.reason"]))

    if r.get("status") == "NO EVIDENCE RETRIEVED":
        s.append(ReasonSentence(
            "no_evidence", MEASURED,
            f"NO EVIDENCE RETRIEVED from the {r.get('pool')} corpus for this claim.",
            ["retrieval.status", "retrieval.pool"]))
    elif r.get("n_retrieved"):
        s.append(ReasonSentence(
            "evidence", MEASURED,
            f"{r.get('n_retrieved')} candidate evidence item(s) were retrieved "
            f"automatically from the fixed {r.get('pool')} corpus using "
            f"{r.get('retriever').upper()}.",
            ["retrieval.n_retrieved", "retrieval.pool", "retrieval.retriever"]))

    pos = (x.get("top_positive_tokens") or [])[:3]
    if pos:
        s.append(ReasonSentence(
            "attribution", INTERPRETATION,
            "Text portions with the largest measured influence toward the "
            "classifier's FAKE direction: "
            + ", ".join(f"\"{t['token']}\" ({t['attribution']:+.4f})" for t in pos)
            + ". This explains the classifier only, not the verifier.",
            ["explainability.top_positive_tokens"]))

    return {"sentences": [x_.to_dict() for x_ in s],
            "kinds": {MEASURED: sum(1 for y in s if y.kind == MEASURED),
                      INTERPRETATION: sum(1 for y in s if y.kind == INTERPRETATION),
                      LIMITATION: sum(1 for y in s if y.kind == LIMITATION)},
            "generator": "slot-filling templates; no language model involved"}


def validate_provenance(result: dict[str, Any]) -> list[str]:
    """Every cited path must resolve in the result. Returns the broken ones."""
    broken = []
    for sent in result.get("reason", {}).get("sentences", []):
        for path in sent.get("provenance", []):
            if _get(result, path) is None:
                broken.append(f"{sent['template_id']} -> {path}")
    return broken


def render_report(result: dict[str, Any], width: int = 66) -> str:
    """Human-readable report. Tamil text passes through unmodified."""
    def rule(t: str = "") -> str:
        return ("-" * width) if not t else f"{'-' * 4} {t} " + "-" * max(0, width - 6 - len(t))

    if not result.get("input", {}).get("accepted", False):
        return f"INPUT REJECTED: {result.get('error')}"

    f, c = result["fusion"], result["classifier"]
    v, r, x = result["verification"], result["retrieval"], result.get("explainability", {})
    icon = {"FAKE": "[FAKE]", "REAL": "[REAL]", "UNCERTAIN": "[UNCERTAIN]"}
    L = ["=" * width, "NEWS VERIFICATION RESULT".center(width), "=" * width, "",
         f"VERDICT            : {icon.get(f['display_state'])} {f['display_state']}",
         f"Final P(fake)      : {100 * f['fused_p_fake']:.2f}%",
         f"Confidence         : {100 * f['confidence']:.2f}%   "
         f"({f['confidence_definition']})",
         f"Decision threshold : {100 * f['threshold']:.2f}%",
         f"Language           : {result['language']['code']} "
         f"(tamil_ratio={result['language']['tamil_ratio']})", "",
         rule("CLASSIFIER"),
         f"Classifier P(fake) : {100 * c['p_fake']:.2f}%",
         f"Classifier P(real) : {100 * c['p_real']:.2f}%",
         "Confidence above is NOT this number; see the fusion block.", "",
         rule("CLAIM USED FOR RETRIEVAL"),
         f"{result['claim']['text']}",
         f"(rule: {result['claim']['selection_rule']})", "",
         rule("VERIFICATION")]

    if v.get("available"):
        L += [f"Dataset            : {v['dataset']}",
              f"Label space        : {', '.join(v['label_space'])}",
              f"Verifier result    : {v['verdict']}  (p={v['verdict_probability']:.3f})",
              "Probabilities      : "
              + ", ".join(f"{k}={p:.3f}" for k, p in v["probs"].items())]
    else:
        L += [f"Not available      : {v.get('reason')}"]

    L += ["", rule("CANDIDATE EVIDENCE — RETRIEVED AUTOMATICALLY")]
    if r["status"] == "NO EVIDENCE RETRIEVED":
        L += ["NO EVIDENCE RETRIEVED"]
    else:
        L += [f"Pool {r['pool']} | retriever {r['retriever'].upper()} | "
              f"{r['n_retrieved']} of top_k {r['top_k']}"]
        for e in r["candidates"]:
            txt = e["text"][:300] + ("..." if len(e["text"]) > 300 else "")
            L += [f"  {e['rank']:>2}. [{e['corpus']}:{e['evidence_id'][:16]}] "
                  f"score={e['score']:.3f} conf={e['confidence']:.2f}",
                  f"      {txt}"]

    L += ["", rule("FUSION (Module 7 frozen parameters)"),
          f"alpha={f['alpha']}  threshold={f['threshold']}  T={f['temperature']}  "
          f"(fitted on {f['fit_split']})",
          f"pre-temperature fused P(fake) : {f['fused_p_fake_before_temperature']:.6f}",
          f"temperature-scaled P(fake)    : {f['fused_p_fake']:.6f}",
          "", rule("CLASSIFIER-FOCUSED EXPLANATION")]

    if x.get("method"):
        L += [f"Method: {x['method']}   target: {x['target']}   "
              f"base P(fake)={x['base_probability']:.4f}",
              "Pushed the classifier toward FAKE (removing lowered P(fake)):"]
        L += ["  " + "  ".join(f"{t['token']} ({t['attribution']:+.4f})"
                               for t in (x.get("top_positive_tokens") or [])[:8]) or "  none"]
        L += ["Pushed the classifier toward REAL (removing raised P(fake)):"]
        L += ["  " + "  ".join(f"{t['token']} ({t['attribution']:+.4f})"
                               for t in (x.get("top_negative_tokens") or [])[:8]) or "  none"]
        if "faithfulness" in x:
            L += [f"{x['faithfulness']['label']}"]
        L += ["", "This explains the CLASSIFIER's probability only. It does not",
              "explain the verifier, and it does not establish factual truth."]
    else:
        L += ["not computed"]

    L += ["", rule("REASON")]
    for s in result["reason"]["sentences"]:
        L += [f"  [{s['kind']}] {s['text']}"]

    L += ["", rule("LIMITATIONS")]
    L += [f"  - {t}" for t in result.get("limitations", [])]
    L += ["", "=" * width]
    return "\n".join(L)
