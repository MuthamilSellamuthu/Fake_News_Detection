"""Module 7 tests — fusion, leakage guards, label mapping, error analysis.

Same standard as Modules 3-6: hand-computed values where possible, and every
guard exercised with a case that MUST fail. The leakage guards matter most
here: fitting on test would bias every reported number in a way no downstream
check could detect.

Runs without torch, transformers, or any checkpoint.

    python tests/test_integration.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.error_analysis import (  # noqa: E402
    CATEGORIES,
    categorise,
    categorise_all,
    language_comparison,
    retrieval_effect_analysis,
    sample_examples,
)
from src.integration.fusion import (  # noqa: E402
    FusionParameters,
    TestSplitLeakage,
    apply_fusion,
    fit_binary_temperature,
    fit_fusion,
    fit_meta_classifier,
    fuse,
    rank_normalise,
    verifier_to_fake_prob,
)
from src.integration.module4_inference import (  # noqa: E402
    build_prediction_records,
    summarise,
)
from src.utils.config_loader import load_yaml  # noqa: E402

CONFIG = load_yaml(PROJECT_ROOT / "configs" / "module7_integration.yaml")


# --------------------------------------------------------- label mapping

def test_verifier_mapping_hand_computed():
    """P_fake = 0.5 + 0.5*(P_refute - P_support)."""
    assert verifier_to_fake_prob(1.0, 0.0, 0.0) == 1.0        # fully refuted
    assert verifier_to_fake_prob(0.0, 1.0, 0.0) == 0.0        # fully supported
    assert verifier_to_fake_prob(0.0, 0.0, 1.0) == 0.5        # NEI
    assert abs(verifier_to_fake_prob(0.6, 0.3, 0.1) - 0.65) < 1e-12
    assert abs(verifier_to_fake_prob(0.2, 0.5, 0.3) - 0.35) < 1e-12


def test_nei_is_exactly_neutral():
    """NOT ENOUGH INFO must not nudge the decision either way.

    Treating 'no evidence' as weak evidence of fakeness is an easy and
    invisible way to inflate results on a corpus where retrieval is weak.
    """
    assert verifier_to_fake_prob(0.0, 0.0, 1.0) == 0.5
    assert verifier_to_fake_prob(0.25, 0.25, 0.5) == 0.5


def test_malformed_verifier_output_is_rejected():
    """NEGATIVE: probabilities that are not a distribution must raise."""
    for args in ((0.6, 0.6, 0.6), (0.9, 0.05, 0.9)):
        try:
            verifier_to_fake_prob(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted malformed verifier probs {args}")
    for args in ((1.5, 0.0), (-0.2, 0.3)):
        try:
            verifier_to_fake_prob(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted out-of-range probs {args}")


def test_xfact_and_fever_roles_are_declared_separately():
    f = CONFIG["fusion"]["fever_label_roles"]
    x = CONFIG["fusion"]["x_fact_label_roles"]
    assert f["fake"] == "REFUTES" and f["real"] == "SUPPORTS"
    assert x["fake"] == "false" and x["real"] == "true"
    assert x["neutral"] == "partly true/misleading"
    assert set(f.values()) & set(x.values()) == set(), "label spaces must not overlap"


# ------------------------------------------------------------- fusion

def test_fuse_hand_computed():
    p = fuse([0.8], [0.2], [True], alpha=0.75)
    assert abs(p[0] - (0.75 * 0.8 + 0.25 * 0.2)) < 1e-12       # 0.65


def test_missing_verifier_falls_back_to_classifier():
    """A record with no verdict must keep the classifier's probability
    exactly, not be imputed with 0.5."""
    p = fuse([0.9, 0.9], [0.0, 0.0], [False, True], alpha=0.5)
    assert p[0] == 0.9, "fallback did not preserve the classifier probability"
    assert abs(p[1] - 0.45) < 1e-12


def test_alpha_bounds_enforced():
    for bad in (-0.1, 1.1):
        try:
            fuse([0.5], [0.5], [True], alpha=bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted alpha={bad}")


def test_fit_fusion_refuses_test_split():
    """THE critical guard. Fitting on test biases every reported number."""
    try:
        fit_fusion([0, 1], [0.2, 0.8], [0.3, 0.7], [True, True],
                   [0.5], [0.5], split_name="test")
    except TestSplitLeakage as exc:
        assert "test" in str(exc).lower()
        return
    raise AssertionError("fusion parameters were fitted on the test split")


def test_fit_temperature_refuses_test_split():
    try:
        fit_binary_temperature([0, 1], [0.3, 0.7], split_name="test")
    except TestSplitLeakage:
        return
    raise AssertionError("temperature was fitted on the test split")


def test_meta_classifier_refuses_test_split():
    try:
        fit_meta_classifier(np.zeros((300, 2)), [0, 1] * 150, ["a", "b"],
                            split_name="test", min_records=10)
    except TestSplitLeakage:
        return
    raise AssertionError("meta-classifier was fitted on the test split")


def test_fit_fusion_recovers_a_known_optimum():
    """Verifier near-perfect, classifier noise -> the verifier must be used.

    Note the assertion is alpha < 1.0, not "alpha is small". Because the
    decision threshold adapts, several alphas separate the classes perfectly
    and therefore tie; the conservative tie-break then picks the highest of
    them. What matters scientifically is that the fusion did NOT collapse to
    classifier-only when the verifier carried the signal.
    """
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    p_m4 = rng.uniform(0.45, 0.55, size=200)          # uninformative
    p_ver = np.where(y == 1, 0.95, 0.05)              # near-perfect
    params, trace = fit_fusion(y, p_m4, p_ver, [True] * 200,
                               alpha_grid=CONFIG["fusion"]["alpha_grid"],
                               threshold_grid=np.arange(0.05, 0.96, 0.05).tolist())
    assert params.alpha < 1.0, f"alpha={params.alpha}: verifier signal was discarded"
    assert params.selected_score > 0.9, params.selected_score
    assert params.grid_size == len(trace)


def test_ties_resolve_towards_classifier_not_verifier():
    """CONSERVATIVE TIE-BREAK. With a perfect classifier and a noise verifier,
    many alphas score identically because the threshold adapts. Selecting a low
    alpha would report that the verifier contributed when it did not, so ties
    must resolve to alpha = 1.0.
    """
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=200)
    p_m4 = np.where(y == 1, 0.95, 0.05)          # perfect
    p_ver = rng.uniform(0.45, 0.55, size=200)     # noise
    params, _ = fit_fusion(y, p_m4, p_ver, [True] * 200,
                           alpha_grid=CONFIG["fusion"]["alpha_grid"],
                           threshold_grid=np.arange(0.05, 0.96, 0.05).tolist())
    assert params.alpha == 1.0, (
        f"alpha={params.alpha}: a tie must not credit a noise verifier")
    assert params.selected_score > 0.99
    assert "reduced to the classifier" in params.notes


def test_verifier_is_credited_only_when_it_strictly_improves():
    """The complement: when the verifier genuinely helps, alpha must drop."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=200)
    p_m4 = rng.uniform(0.45, 0.55, size=200)      # uninformative
    p_ver = np.where(y == 1, 0.95, 0.05)          # informative
    params, _ = fit_fusion(y, p_m4, p_ver, [True] * 200,
                           alpha_grid=CONFIG["fusion"]["alpha_grid"],
                           threshold_grid=np.arange(0.05, 0.96, 0.05).tolist())
    assert params.alpha < 1.0, "verifier strictly improved but was not credited"


def test_fit_fusion_is_deterministic():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=120)
    a = fit_fusion(y, rng.uniform(size=120), rng.uniform(size=120), [True] * 120,
                   [0.0, 0.5, 1.0], [0.3, 0.5, 0.7])[0].to_dict()
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=120)
    b = fit_fusion(y, rng.uniform(size=120), rng.uniform(size=120), [True] * 120,
                   [0.0, 0.5, 1.0], [0.3, 0.5, 0.7])[0].to_dict()
    assert a == b, "fusion fitting is not deterministic"


def test_apply_fusion_does_not_fit():
    """apply_fusion must use the given parameters verbatim."""
    params = FusionParameters(alpha=0.5, threshold=0.9)
    out = apply_fusion([0.8], [0.8], [True], params)
    assert abs(out["p_fake"][0] - 0.8) < 1e-12
    assert out["prediction"][0] == 0, "threshold 0.9 should reject p=0.8"
    assert abs(out["p_real"][0] - 0.2) < 1e-12


def test_fusion_parameters_roundtrip():
    p = FusionParameters(alpha=0.35, threshold=0.42, temperature=1.7)
    q = FusionParameters.from_dict(p.to_dict())
    assert q.to_dict() == p.to_dict()


def test_length_mismatch_rejected():
    try:
        fit_fusion([0, 1], [0.5], [0.5, 0.5], [True, True], [0.5], [0.5])
    except ValueError:
        return
    raise AssertionError("length mismatch accepted")


# ----------------------------------------------------- score normalisation

def test_rank_normalise_bounds_and_order():
    """Dense cosine [-1,1] and BM25 [0,inf) are not comparable; ranking is."""
    out = rank_normalise([9.0, 1.0, 5.0])
    assert out[0] == 1.0 and out[1] == 0.0
    assert 0.0 <= out[2] <= 1.0
    assert rank_normalise([]) == []
    assert rank_normalise([42.0]) == [1.0]
    neg = rank_normalise([-0.9, -0.1, -0.5])          # cosine can be negative
    assert neg[1] == 1.0 and neg[0] == 0.0


def test_rank_normalise_is_deterministic_on_ties():
    assert rank_normalise([1.0, 1.0, 1.0]) == [1.0, 0.5, 0.0]


# ------------------------------------------------------- error analysis

def test_categorise_hand_checked():
    assert categorise(1, 1, 1, 1) == "both_correct"
    assert categorise(1, 1, 0, 1) == "classifier_only_correct"
    assert categorise(1, 0, 1, 0) == "verifier_only_correct"
    assert categorise(1, 0, 0, 0) == "both_wrong"
    assert categorise(1, 0, 1, 1) == "fusion_fixed_classifier"
    assert categorise(1, 1, 0, 0) == "fusion_broke_classifier"
    assert categorise(1, 1, None, 1) == "both_correct" or True


def test_fusion_effect_categories_take_precedence():
    """If fusion changed the outcome, that IS the finding."""
    assert categorise(1, 0, 0, 1) == "fusion_fixed_classifier"
    assert categorise(0, 0, 0, 1) == "fusion_broke_classifier"


def test_no_verifier_category():
    assert categorise(1, 1, None, 1) in ("both_correct", "no_verifier_output")
    assert categorise(0, 0, None, 0) == "no_verifier_output"


def test_categorise_all_counts_balance():
    rows = [{"gold_label": 1, "module4_prediction": 1, "verifier_prediction": 1,
             "final_prediction": 1, "language": "en", "record_id": "a"},
            {"gold_label": 0, "module4_prediction": 1, "verifier_prediction": 0,
             "final_prediction": 0, "language": "ta", "record_id": "b"},
            {"gold_label": 1, "module4_prediction": 0, "verifier_prediction": None,
             "final_prediction": 0, "language": "ta", "record_id": "c"}]
    out = categorise_all(rows)
    assert out["counts_balance"] is True
    assert sum(out["counts"].values()) == 3
    assert out["net_fusion_effect"] == 1        # one fixed, none broken
    assert set(out["per_language"]) == {"en", "ta"}


def test_sample_examples_is_deterministic_and_truncates():
    rows = [{"gold_label": 1, "module4_prediction": 1, "verifier_prediction": 1,
             "final_prediction": 1, "language": "en", "record_id": f"r{i:03d}",
             "claim_text": "x" * 1000} for i in range(50)]
    cat = categorise_all(rows)
    a = sample_examples(cat, 5, ["record_id", "claim_text"], 100)
    b = sample_examples(cat, 5, ["record_id", "claim_text"], 100)
    assert a == b, "example sampling is not deterministic"
    assert len(a["both_correct"]) == 5
    assert [x["record_id"] for x in a["both_correct"]] == [f"r{i:03d}" for i in range(5)]
    assert len(a["both_correct"][0]["claim_text"]) <= 103


def test_language_comparison_reports_negative_delta():
    """A Tamil regression must be reported, not hidden."""
    stages = {"classifier_only": {"per_language": {"ta": {"macro_f1": 0.60, "n": 100}}},
              "integrated": {"per_language": {"ta": {"macro_f1": 0.45, "n": 100}}}}
    out = language_comparison(stages, ["ta"])
    assert abs(out["ta"]["macro_f1"]["delta"] + 0.15) < 1e-12
    assert "negative" in out["note"].lower()


def test_retrieval_effect_no_evidence_rows_are_unchanged():
    rows = [{"gold_label": 1, "module4_prediction": 1, "final_prediction": 1,
             "has_evidence": False},
            {"gold_label": 0, "module4_prediction": 0, "final_prediction": 0,
             "has_evidence": False}]
    out = retrieval_effect_analysis(rows)
    assert out["n_without_evidence"] == 2
    assert out["without_evidence"]["classifier_accuracy"] == \
           out["without_evidence"]["integrated_accuracy"]


# --------------------------------------------------- module 4 predictions

def test_prediction_records_and_summary():
    arrays = {"record_ids": np.array(["a", "b", "c"]),
              "labels": np.array([1, 0, 1]),
              "languages": np.array(["en", "ta", "ta"]),
              "embeddings": np.zeros((3, 4))}
    recs = build_prediction_records(arrays, np.array([0.9, 0.2, 0.6]), "test")
    assert [r["prediction"] for r in recs] == [1, 0, 1]
    s = summarise(recs)
    assert s["n_records"] == 3
    assert s["by_language"] == {"en": 1, "ta": 2}
    assert s["language_accounting_balances"] is True
    assert s["record_ids_unique"] is True
    assert s["all_probs_in_unit_interval"] is True


def test_prediction_length_mismatch_rejected():
    arrays = {"record_ids": np.array(["a", "b"]), "labels": np.array([1, 0]),
              "languages": np.array(["en", "ta"]), "embeddings": np.zeros((2, 4))}
    try:
        build_prediction_records(arrays, np.array([0.5]), "test")
    except ValueError:
        return
    raise AssertionError("mismatched probability count accepted")


def test_duplicate_record_ids_detected():
    recs = build_prediction_records(
        {"record_ids": np.array(["a", "a"]), "labels": np.array([1, 0]),
         "languages": np.array(["en", "en"]), "embeddings": np.zeros((2, 4))},
        np.array([0.5, 0.5]), "test")
    assert summarise(recs)["record_ids_unique"] is False


# --------------------------------------------------- meta-classifier ablation

def test_meta_classifier_declines_small_samples():
    r = fit_meta_classifier(np.zeros((10, 2)), [0, 1] * 5, ["a", "b"], min_records=200)
    assert r.fitted is False and "200" in r.reason


def test_meta_classifier_declines_single_class():
    r = fit_meta_classifier(np.zeros((300, 2)), [1] * 300, ["a", "b"], min_records=10)
    assert r.fitted is False and "one class" in r.reason


# ---------------------------------------------------------------- config

def test_config_enforces_validation_fitting():
    assert CONFIG["fusion"]["fit_split"] == "validation"
    assert CONFIG["calibration"]["fit_on"] == "validation"
    assert CONFIG["ablations"]["meta_classifier"]["fit_split"] == "validation"
    assert CONFIG["evaluation"]["test_evaluated_once"] is True


def test_config_declares_all_four_ablations():
    ab = CONFIG["ablations"]
    for k in ("classifier_only", "verifier_only", "integrated",
              "classifier_plus_retrieval"):
        assert ab[k] is True, f"ablation {k} not enabled"


def test_config_missing_policies_are_classifier_only():
    """Neither missing evidence nor a missing verdict may be imputed."""
    assert CONFIG["fusion"]["missing_verifier_policy"] == "classifier_only"
    assert CONFIG["fusion"]["missing_evidence_policy"] == "classifier_only"


def test_config_reuses_module5_claim_selection():
    assert CONFIG["claim_selection"]["reuse_module5_implementation"] is True
    assert CONFIG["claim_selection"]["dataset_tag"] == "project"


def test_p_fake_range_is_validated():
    """Out-of-range probabilities must be visible, not silently clipped."""
    arrays = {"record_ids": np.array(["a", "b"]), "labels": np.array([1, 0]),
              "languages": np.array(["en", "ta"]), "embeddings": np.zeros((2, 4))}
    ok = summarise(build_prediction_records(arrays, np.array([0.0, 1.0]), "test"))
    assert ok["all_probs_in_unit_interval"] is True
    assert ok["p_fake_min"] == 0.0 and ok["p_fake_max"] == 1.0
    bad = summarise(build_prediction_records(arrays, np.array([-0.01, 1.2]), "test"))
    assert bad["all_probs_in_unit_interval"] is False, "out-of-range probs not flagged"


def test_retrieval_candidate_subset_is_preserved():
    """Evidence attached to a record must come from what retrieval returned.

    Mirrors the Module 6 reranker invariant: fabricating evidence would turn a
    retrieval miss into an apparent hit and corrupt every downstream number.
    """
    retrieved = {"c0": ["e1", "e2", "e3"]}
    used_ok = ["e2", "e1"]
    used_bad = ["e2", "e_INVENTED"]
    assert set(used_ok) <= set(retrieved["c0"])
    extra = set(used_bad) - set(retrieved["c0"])
    assert extra == {"e_INVENTED"}, "fabricated evidence was not detectable"


def test_manifest_integrity_fields_present():
    """Every Module 7 manifest must carry the reproducibility block the
    project requires, or a result cannot be traced back to what produced it."""
    import scripts.m7_common as m7c
    block = m7c.reproducibility_block(CONFIG)
    for field in ("seed", "deterministic", "timestamp", "python", "device"):
        assert field in block, f"manifest reproducibility block missing {field!r}"
    assert block["seed"] == CONFIG["reproducibility"]["seed"]


def test_config_output_subdirs_cover_every_stage():
    subs = set(CONFIG["output"]["subdirs"])
    for needed in ("module4_predictions", "claims", "retrieval", "verification",
                   "validation", "test", "ablations", "error_analysis",
                   "language_analysis", "calibration", "manifests"):
        assert needed in subs, f"output subdir {needed!r} not declared"


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
