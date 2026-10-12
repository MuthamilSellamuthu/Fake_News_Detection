"""Module 9 tests — statistics, tables, robustness, reproducibility.

The statistical tests are checked against hand-computed values, and every
"cannot run" path is exercised: a test that silently returns a p-value on
inadequate data is worse than one that refuses.

    python tests/test_final_evaluation.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.comparison_tables import (  # noqa: E402
    UNAVAILABLE, ablation_table, classification_table, language_comparison_table,
    retrieval_table, statistical_table, write_csv)
from src.evaluation.confidence_intervals import (  # noqa: E402
    bootstrap_metric, wilson_interval)
from src.evaluation.reproducibility import (  # noqa: E402
    config_hash, hash_artefacts, software_versions)
from src.evaluation.robustness import (  # noqa: E402
    confidence_robustness, error_breakdown, evidence_availability_robustness, slice_by)
from src.evaluation.statistical_tests import (  # noqa: E402
    holm_correction, mcnemar_test, paired_bootstrap_difference)
from src.utils.config_loader import load_yaml  # noqa: E402

CONFIG = load_yaml(PROJECT_ROOT / "configs" / "module9_evaluation.yaml")


def macro_f1(y, p):
    y, p = np.asarray(y), np.asarray(p)
    f1s = []
    for c in (0, 1):
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return sum(f1s) / 2


# ------------------------------------------------------------- McNemar

def test_mcnemar_hand_computed():
    """n01 = 8, n10 = 2 -> chi2 = (|8-2|-1)^2 / 10 = 25/10 = 2.5"""
    y = np.array([1] * 20)
    a = np.array([1] * 12 + [0] * 8)          # A wrong on the last 8
    b = np.array([1] * 10 + [0] * 2 + [1] * 8)  # B wrong on 2
    r = mcnemar_test(y, a, b, min_discordant=1)
    assert r["applicable"] is True
    assert r["n_discordant"] == 10
    assert abs(r["statistic"] - 2.5) < 1e-12, r["statistic"]
    assert 0.0 <= r["p_value"] <= 1.0


def test_mcnemar_refuses_too_few_discordant_pairs():
    y = np.array([1] * 100)
    a = np.array([1] * 100)
    b = np.array([1] * 99 + [0])
    r = mcnemar_test(y, a, b, min_discordant=10)
    assert r["applicable"] is False
    assert "discordant" in r["reason"]
    assert r["p_value"] is None


def test_mcnemar_identical_systems_have_zero_discordant():
    y = np.array([0, 1, 0, 1] * 10)
    a = y.copy()
    r = mcnemar_test(y, a, a, min_discordant=1)
    assert r["n_discordant"] == 0 and r["applicable"] is False


def test_mcnemar_rejects_mismatched_lengths():
    r = mcnemar_test([1, 0], [1], [1, 0])
    assert r["applicable"] is False and "length" in r["reason"]


def test_mcnemar_records_hypotheses():
    r = mcnemar_test([1] * 30, [1] * 30, [0] * 30, min_discordant=1)
    assert r["null_hypothesis"] and r["alternative"]
    assert r["test"].startswith("mcnemar")


# --------------------------------------------------------------- Holm

def test_holm_correction_hand_computed():
    """p = [0.01, 0.04, 0.03]; m = 3.
    sorted: 0.01(x3)=0.03, 0.03(x2)=0.06, 0.04(x1)=0.04 -> monotone -> 0.06
    """
    r = holm_correction([0.01, 0.04, 0.03], alpha=0.05)
    by_idx = {x["index"]: x for x in r["results"]}
    assert abs(by_idx[0]["p_adjusted"] - 0.03) < 1e-12
    assert abs(by_idx[2]["p_adjusted"] - 0.06) < 1e-12
    assert abs(by_idx[1]["p_adjusted"] - 0.06) < 1e-12, "monotonicity not enforced"
    assert by_idx[0]["rejected"] is True
    assert by_idx[1]["rejected"] is False


def test_holm_excludes_inapplicable_tests_from_family():
    """A test that could not run is not a non-significant result."""
    r = holm_correction([0.01, None, 0.02], alpha=0.05)
    assert r["n_hypotheses"] == 2, "None p-values must not enter the family"
    assert r["excluded_inapplicable"] == 1
    assert abs(r["results"][0]["p_adjusted"] - 0.02) < 1e-12


def test_holm_with_no_valid_p_values():
    r = holm_correction([None, None])
    assert r["applicable"] is False and r["n_hypotheses"] == 0


def test_holm_is_less_conservative_than_bonferroni():
    r = holm_correction([0.02, 0.03, 0.04], alpha=0.05)
    largest = max(x["p_adjusted"] for x in r["results"])
    assert largest <= 3 * 0.04 + 1e-12


# ---------------------------------------------------------- bootstrap

def test_bootstrap_metric_is_reproducible():
    y = np.array([0, 1] * 50)
    p = np.array([0, 1] * 50)
    a = bootstrap_metric(y, p, macro_f1, n_resamples=100, seed=42)
    b = bootstrap_metric(y, p, macro_f1, n_resamples=100, seed=42)
    assert a["ci_lower"] == b["ci_lower"] and a["ci_upper"] == b["ci_upper"]
    assert a["seed"] == 42 and a["n_resamples"] == 100
    assert abs(a["point_estimate"] - 1.0) < 1e-12


def test_bootstrap_ci_ordering_and_bounds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.integers(0, 2, 200)
    r = bootstrap_metric(y, p, macro_f1, n_resamples=200, seed=7)
    assert r["ci_lower"] <= r["point_estimate"] + 1e-9
    assert r["ci_upper"] >= r["point_estimate"] - 1e-9
    assert 0.0 <= r["ci_lower"] <= 1.0 and 0.0 <= r["ci_upper"] <= 1.0


def test_bootstrap_rejects_empty_input():
    assert bootstrap_metric([], [], macro_f1)["applicable"] is False


def test_paired_bootstrap_preserves_pairing():
    """Identical systems must give a difference CI containing zero."""
    y = np.array([0, 1] * 60)
    a = y.copy()
    r = paired_bootstrap_difference(y, a, a, macro_f1, n_resamples=200, seed=3)
    assert r["applicable"] is True
    assert abs(r["observed_difference"]) < 1e-12
    assert r["ci_lower"] <= 0 <= r["ci_upper"]
    assert r["excludes_zero"] is False


def test_wilson_interval_hand_checked():
    r = wilson_interval(50, 100, 0.95)
    assert abs(r["proportion"] - 0.5) < 1e-12
    assert 0.39 < r["ci_lower"] < 0.41 and 0.59 < r["ci_upper"] < 0.61


def test_wilson_interval_stays_in_unit_range_at_extremes():
    r = wilson_interval(0, 10)
    assert r["ci_lower"] >= 0.0 and r["ci_upper"] <= 1.0
    r = wilson_interval(10, 10)
    assert r["ci_upper"] <= 1.0


# ------------------------------------------------------------- tables

def test_classification_table_marks_missing_as_unavailable():
    rows = classification_table(
        {"a": {"overall": {"macro_f1": 0.7, "n": 10}}, "b": {}},
        ["macro_f1", "roc_auc"])
    assert rows[0]["macro_f1"] == 0.7
    assert rows[0]["roc_auc"] == UNAVAILABLE
    assert rows[1]["macro_f1"] == UNAVAILABLE


def test_language_table_computes_en_minus_ta():
    rows = language_comparison_table(
        {"sys": {"overall": {"macro_f1": 0.6},
                 "per_language": {"en": {"macro_f1": 0.7}, "ta": {"macro_f1": 0.4}}}},
        "macro_f1", ("en", "ta"))
    assert abs(rows[0]["en_minus_ta"] - 0.3) < 1e-12


def test_ablation_delta_is_relative_to_declared_baseline():
    rows = ablation_table({"classifier_only": {"overall": {"macro_f1": 0.60}},
                           "integrated": {"overall": {"macro_f1": 0.55}}},
                          "macro_f1", "classifier_only")
    d = {r["ablation"]: r for r in rows}
    assert abs(d["integrated"]["delta_vs_baseline"] + 0.05) < 1e-12, "negative delta lost"
    assert d["classifier_only"]["delta_vs_baseline"] == 0.0


def test_ablation_records_unavailable_experiments():
    rows = ablation_table({"verifier_only": {"applicable": False,
                                             "reason": "experiment not run"}},
                          "macro_f1", "verifier_only")
    assert rows[0]["value"] == UNAVAILABLE
    assert rows[0]["reason_if_unavailable"] == "experiment not run"


def test_statistical_table_joins_holm_results():
    tests = {"a__vs__b": {"test": "mcnemar", "applicable": True, "p_value": 0.01,
                          "statistic": 6.0, "n_discordant": 30,
                          "null_hypothesis": "equal error rates"}}
    corrected = holm_correction([0.01])
    rows = statistical_table(tests, corrected)
    assert rows[0]["p_adjusted_holm"] is not None
    assert rows[0]["significant_after_correction"] in (True, False)


def test_write_csv_atomic_and_full_header():
    d = Path(tempfile.mkdtemp())
    p = write_csv(d / "t.csv", [{"a": 1}, {"a": 2, "b": 3}])
    text = p.read_text(encoding="utf-8")
    assert "a,b" in text.splitlines()[0]
    assert UNAVAILABLE in text
    assert not list(d.glob("*.tmp.csv")), "temp file left behind"


def test_write_csv_empty_rows():
    d = Path(tempfile.mkdtemp())
    p = write_csv(d / "e.csv", [])
    assert p.is_file() and p.read_text(encoding="utf-8") == ""


def test_retrieval_table_reports_exclusions():
    rows = retrieval_table({"x_fact_test_bm25": {
        "overall": {"recall@5": 0.67, "mrr": 0.96},
        "accounting": {"n_scored": 221, "n_excluded_no_gold_evidence": 0}}})
    assert rows[0]["recall@5"] == 0.67
    assert rows[0]["n_excluded_no_gold"] == 0
    assert rows[0]["recall@50"] == UNAVAILABLE


# --------------------------------------------------------- robustness

def _recs(n=60):
    return [{"record_id": f"r{i}", "language": "en" if i % 2 else "ta",
             "gold_label": i % 2, "final_prediction": i % 2,
             "module4_prediction": i % 2, "fused_p_fake": 0.9 if i % 2 else 0.1,
             "has_evidence": i % 3 == 0, "has_verifier": i % 3 == 0,
             "p_fake_verifier": 0.8 if i % 2 else 0.2,
             "verifier_probs": {"REFUTES": 0.8, "SUPPORTS": 0.1,
                                "NOT ENOUGH INFO": 0.1}}
            for i in range(n)]


def test_slice_refuses_tiny_slices():
    out = slice_by(_recs(10), lambda r: r["language"], macro_f1, min_slice=20)
    assert all(v["applicable"] is False for v in out.values())
    assert all("required" in v["reason"] for v in out.values())


def test_slice_computes_when_large_enough():
    out = slice_by(_recs(80), lambda r: r["language"], macro_f1, min_slice=20)
    assert out["en"]["applicable"] is True and out["en"]["n"] == 40


def test_confidence_robustness_definition_recorded():
    out = confidence_robustness(_recs(80), macro_f1, threshold=0.8, min_slice=10)
    assert out["threshold"] == 0.8
    assert "max(P(fake)" in out["definition"]


def test_evidence_availability_notes_the_fallback():
    out = evidence_availability_robustness(_recs(90), macro_f1, min_slice=10)
    assert "fall back to the classifier" in out["note"]


def test_error_breakdown_counts_balance():
    out = error_breakdown(_recs(60))
    assert out["counts_balance"] is True
    assert out["true_positive"] + out["true_negative"] + \
           out["false_positive"] + out["false_negative"] == 60
    assert out["neutral_verdicts"] >= 0


# ---------------------------------------------------- reproducibility

def test_software_versions_never_fabricates():
    v = software_versions()
    assert "python" in v
    for val in v.values():
        assert isinstance(val, str)
    assert all(val == "unavailable" or val for val in v.values())


def test_hash_artefacts_marks_missing_files():
    out = hash_artefacts(PROJECT_ROOT, ["configs/module9_evaluation.yaml",
                                        "does/not/exist.json"])
    assert out["configs/module9_evaluation.yaml"]["exists"] is True
    assert out["does/not/exist.json"]["exists"] is False
    assert out["does/not/exist.json"]["sha256"] == "unavailable"


def test_config_hash_is_stable_and_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


# ---------------------------------------------------------------- config

def test_config_declares_holm_and_paired_test():
    st = CONFIG["statistics"]
    assert st["paired_test"] == "mcnemar"
    assert st["multiple_comparison_correction"] == "holm"
    assert st["bootstrap"]["seed"] == 42


def test_config_claim_policy_requires_evidence():
    pol = CONFIG["reporting"]["claim_policy"].lower()
    assert "superiority" in pol and "holm" in pol
    assert set(CONFIG["reporting"]["result_kinds"]) == {
        "measured", "derived", "statistical_inference", "limitation"}


def test_config_leakage_audit_covers_every_fitted_parameter():
    checks = set(CONFIG["leakage_audit"]["checks"])
    for needed in ("fusion_parameters_fitted_on_validation",
                   "calibration_temperature_fitted_on_validation",
                   "threshold_selected_on_validation",
                   "test_labels_never_used_for_fitting"):
        assert needed in checks, f"leakage audit missing {needed}"


def test_config_systems_marked_required_or_optional():
    for name, spec in CONFIG["systems"].items():
        assert "required" in spec, f"system {name} does not declare `required`"


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"[FAIL] {t.__name__}: {exc}")
            failed.append(t.__name__)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {t.__name__}: {type(exc).__name__}: {exc}")
            failed.append(t.__name__)
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("FAILED:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
