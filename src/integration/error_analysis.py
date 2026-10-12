"""Module 7 — error categorisation and language-wise comparison.

The categories are chosen to answer the question the paper actually asks:
did fact verification help, and where? `fusion_fixed_classifier` and
`fusion_broke_classifier` are the two that matter most -- they are the direct
evidence for or against the integration, and reporting only the first would be
a selective account.

Nothing here computes a metric that Module 4 or Module 6 already defines;
`evaluate_predictions()` and `compute_classification_metrics()` are reused by
the calling script.
"""

from __future__ import annotations

from typing import Any, Sequence

CATEGORIES = ("both_correct", "classifier_only_correct", "verifier_only_correct",
              "both_wrong", "fusion_fixed_classifier", "fusion_broke_classifier",
              "no_verifier_output")


def categorise(gold: int, m4_pred: int, verifier_pred: int | None,
               fused_pred: int) -> str:
    """Assign one record to exactly one category.

    The fusion-effect categories take precedence over agreement categories:
    once fusion has changed a decision, that IS the finding, and recording it
    merely as "both_wrong" would hide it.
    """
    m4_ok = m4_pred == gold
    fused_ok = fused_pred == gold

    if m4_ok and not fused_ok:
        return "fusion_broke_classifier"
    if (not m4_ok) and fused_ok:
        return "fusion_fixed_classifier"
    if verifier_pred is None:
        return "no_verifier_output"

    ver_ok = verifier_pred == gold
    if m4_ok and ver_ok:
        return "both_correct"
    if m4_ok and not ver_ok:
        return "classifier_only_correct"
    if (not m4_ok) and ver_ok:
        return "verifier_only_correct"
    return "both_wrong"


def categorise_all(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Categorise every record and verify the counts balance."""
    counts = {c: 0 for c in CATEGORIES}
    per_language: dict[str, dict[str, int]] = {}
    out: list[dict[str, Any]] = []

    for r in records:
        cat = categorise(int(r["gold_label"]), int(r["module4_prediction"]),
                         (None if r.get("verifier_prediction") is None
                          else int(r["verifier_prediction"])),
                         int(r["final_prediction"]))
        counts[cat] += 1
        lang = str(r.get("language", "unknown"))
        per_language.setdefault(lang, {c: 0 for c in CATEGORIES})[cat] += 1
        rec = dict(r)
        rec["category"] = cat
        out.append(rec)

    return {"records": out, "counts": counts, "per_language": per_language,
            "n_records": len(records),
            "counts_balance": sum(counts.values()) == len(records),
            "net_fusion_effect": counts["fusion_fixed_classifier"]
                                 - counts["fusion_broke_classifier"],
            "interpretation": (
                "net_fusion_effect > 0 means fusion corrected more records than it "
                "broke. A negative value is a legitimate finding and must be "
                "reported, not suppressed.")}


def sample_examples(categorised: dict[str, Any], max_per_category: int = 25,
                    include_fields: Sequence[str] | None = None,
                    truncate_chars: int = 300) -> dict[str, list[dict[str, Any]]]:
    """Deterministic examples per category: first N by record_id, no sampling.

    Long article text is truncated -- the reports are for reading, and full
    article bodies do not belong in a JSON artefact.
    """
    buckets: dict[str, list[dict[str, Any]]] = {c: [] for c in CATEGORIES}
    for r in sorted(categorised["records"], key=lambda x: str(x.get("record_id", ""))):
        cat = r["category"]
        if len(buckets[cat]) >= max_per_category:
            continue
        item = ({k: r.get(k) for k in include_fields if k in r}
                if include_fields else dict(r))
        for k, v in list(item.items()):
            if isinstance(v, str) and len(v) > truncate_chars:
                item[k] = v[:truncate_chars] + "..."
        buckets[cat].append(item)
    return buckets


def language_comparison(metrics_by_stage: dict[str, dict[str, Any]],
                        languages: Sequence[str] = ("en", "ta")) -> dict[str, Any]:
    """Per-language deltas from classifier-only to integrated.

    `metrics_by_stage` maps a stage name to the output of
    `evaluate_predictions()`, i.e. {"overall": {...}, "per_language": {...}}.
    """
    out: dict[str, Any] = {}
    base = metrics_by_stage.get("classifier_only", {})
    integ = metrics_by_stage.get("integrated", {})

    for lang in languages:
        b = (base.get("per_language", {}) or {}).get(lang, {}) or {}
        i = (integ.get("per_language", {}) or {}).get(lang, {}) or {}
        entry: dict[str, Any] = {"n": i.get("n", b.get("n"))}
        for metric in ("accuracy", "macro_f1", "f1", "precision", "recall", "roc_auc"):
            bv, iv = b.get(metric), i.get(metric)
            entry[metric] = {
                "classifier_only": bv, "integrated": iv,
                "delta": (None if bv is None or iv is None else float(iv) - float(bv)),
            }
        out[lang] = entry

    out["note"] = ("Deltas are integrated minus classifier_only. A negative Tamil "
                   "delta means fact verification hurt Tamil and must be reported "
                   "as such; this project must not assume Tamil improves.")
    return out


def retrieval_effect_analysis(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Did having evidence at all change anything?

    Separates the effect of *finding* evidence from the effect of *verifying*
    it -- the distinction ablation D is designed to isolate.
    """
    with_ev = [r for r in records if r.get("has_evidence")]
    without = [r for r in records if not r.get("has_evidence")]

    def acc(rows: Sequence[dict[str, Any]], key: str) -> float | None:
        if not rows:
            return None
        return float(sum(1 for r in rows if int(r[key]) == int(r["gold_label"]))
                     / len(rows))

    return {
        "n_with_evidence": len(with_ev),
        "n_without_evidence": len(without),
        "coverage": (len(with_ev) / len(records)) if records else 0.0,
        "with_evidence": {"classifier_accuracy": acc(with_ev, "module4_prediction"),
                          "integrated_accuracy": acc(with_ev, "final_prediction")},
        "without_evidence": {"classifier_accuracy": acc(without, "module4_prediction"),
                             "integrated_accuracy": acc(without, "final_prediction")},
        "note": ("Records without evidence fall back to the classifier by policy, so "
                 "their two accuracies should be identical. A difference indicates a "
                 "fusion bug."),
    }
