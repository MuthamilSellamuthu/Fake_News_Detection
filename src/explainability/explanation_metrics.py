"""Module 8 — aggregate explanation quality statistics.

Every metric is defined separately and reported separately. No composite
"explanation quality score" is produced: collapsing deletion, insertion,
concentration and stability into one number would hide which property failed,
and there is no accepted weighting for such a combination.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _agg(values: Sequence[float]) -> dict[str, float | None]:
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    a = np.asarray(v, dtype=np.float64)
    return {"n": len(v), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()),
            "std": float(a.std(ddof=0))}


def summarise_explanations(records: Sequence[dict[str, Any]],
                           languages: Sequence[str] = ("en", "ta")) -> dict[str, Any]:
    """Overall + per-language statistics over generated explanations."""

    def block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"n_records": 0}
        methods: dict[str, int] = {}
        for r in rows:
            methods[r.get("explanation_method", "unknown")] = \
                methods.get(r.get("explanation_method", "unknown"), 0) + 1
        diags = [r.get("attribution_diagnostics", {}) or {} for r in rows]
        failures = [r for r in rows if (r.get("attribution_diagnostics") or {}).get("error")]
        return {
            "n_records": len(rows),
            "explanation_methods": dict(sorted(methods.items())),
            "token_count": _agg([d.get("n_tokens") for d in diags]),
            "latency_seconds": _agg([d.get("seconds") for d in diags]),
            "n_truncated": sum(1 for d in diags if d.get("truncated")),
            "n_failures": len(failures),
            "failure_rate": len(failures) / len(rows),
            "n_with_verification": sum(1 for r in rows if r.get("verification_probabilities")),
            "n_with_evidence": sum(1 for r in rows if r.get("evidence")),
            "attribution_coverage": (
                sum(1 for r in rows if r.get("token_attributions")) / len(rows)),
        }

    out: dict[str, Any] = {"overall": block(list(records)), "per_language": {}}
    for lang in languages:
        out["per_language"][lang] = block([r for r in records
                                           if r.get("language") == lang])
    counted = sum(b["n_records"] for b in out["per_language"].values())
    out["language_accounting"] = {
        "sum_per_language": counted, "overall": out["overall"]["n_records"],
        "balances": counted == out["overall"]["n_records"],
        "note": "Records in an unlisted language are the difference, if any.",
    }
    return out


def summarise_faithfulness(faithfulness: Sequence[dict[str, Any]],
                           languages: Sequence[str] = ("en", "ta")) -> dict[str, Any]:
    """Aggregate deletion / insertion / concentration / stability."""

    def block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"n_records": 0}
        dele = [r["deletion"]["mean_absolute_delta"] for r in rows
                if (r.get("deletion") or {}).get("applicable")]
        ins = [abs(r["insertion"]["curve"][-1]["recovery"]) for r in rows
               if (r.get("insertion") or {}).get("applicable") and r["insertion"]["curve"]]
        conc = {}
        for k in ("top_1", "top_3", "top_5", "top_10"):
            conc[k] = _agg([(r.get("concentration") or {}).get(k) for r in rows])
        stab = [r["stability"]["identical"] for r in rows if r.get("stability")]
        return {
            "n_records": len(rows),
            "deletion_mean_absolute_delta": _agg(dele),
            "insertion_final_gap_to_full_prediction": _agg(ins),
            "attribution_concentration": conc,
            "stability_identical_rate": (float(np.mean(stab)) if stab else None),
            "n_stability_checked": len(stab),
        }

    out: dict[str, Any] = {"overall": block(list(faithfulness)), "per_language": {}}
    for lang in languages:
        out["per_language"][lang] = block([r for r in faithfulness
                                           if r.get("language") == lang])
    out["definitions"] = {
        "deletion_mean_absolute_delta": (
            "Mean |P(fake) after removing top-attribution tokens - P(fake) on full "
            "text|, averaged over the configured fractions. Larger means the "
            "identified tokens mattered more."),
        "insertion_final_gap_to_full_prediction": (
            "|P(fake) using only the top-attribution tokens - P(fake) on full text| "
            "at the largest fraction. Smaller means those tokens alone recover the "
            "prediction."),
        "attribution_concentration": (
            "Share of total |attribution| mass held by the top-k tokens."),
        "stability_identical_rate": (
            "Fraction of records whose attributions were bit-identical across "
            "repeated deterministic runs."),
        "no_composite_score": (
            "These are reported separately on purpose; no single explanation-quality "
            "score is defined."),
    }
    return out
