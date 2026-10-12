"""Module 8 tests — attribution, faithfulness, honest method labelling.

The labelling tests matter most: calling an occlusion fallback "SHAP" would be
a false methods claim in the paper, not merely a bug.

Runs without torch, transformers, shap, or any checkpoint.

    python tests/test_explainability.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.combined_explanation import (  # noqa: E402
    build_explanation_record, build_reason, render_text_report)
from src.explainability.explanation_metrics import (  # noqa: E402
    summarise_explanations, summarise_faithfulness)
from src.explainability.shap_explainer import (  # noqa: E402
    SHAP_EMBEDDING_LABEL, explain_embedding, is_shap_available, resolve_method,
    select_background)
from src.explainability.token_attribution import (  # noqa: E402
    OCCLUSION, TokenAttribution, attribution_concentration, deletion_test,
    insertion_test, occlusion_attribution, simple_tokenize, stability_check)
from src.explainability.verification_explainer import (  # noqa: E402
    contribution_breakdown, explain_verification)
from src.utils.config_loader import load_yaml  # noqa: E402

CONFIG = load_yaml(PROJECT_ROOT / "configs" / "module8_explainability.yaml")
TAMIL = "பஹல்காம் தாக்குதல் குறித்த தவறான கூற்று"


def keyword_predict(keyword="fake", base=0.3, boost=0.6):
    """Deterministic stand-in classifier: P(fake) rises when `keyword` is present."""
    def _p(texts):
        return np.array([min(1.0, base + boost) if keyword in t else base
                         for t in texts], dtype=np.float64)
    return _p


# ------------------------------------------------------------ attribution

def test_attribution_length_equals_token_length():
    text = "the senator said fake things today"
    att = occlusion_attribution(text, keyword_predict(), max_tokens=128)
    assert len(att.attributions) == len(att.tokens) == len(text.split())
    assert att.n_tokens == len(att.tokens)


def test_occlusion_identifies_the_causal_token():
    """Removing 'fake' should drop P(fake), giving it the top attribution."""
    att = occlusion_attribution("the senator said fake things", keyword_predict())
    top = att.top_tokens(3)["top_positive_tokens"]
    assert top and top[0]["token"] == "fake", top
    assert top[0]["attribution"] > 0.5


def test_no_nan_or_inf_attributions():
    att = occlusion_attribution("a b c d e", keyword_predict())
    assert all(math.isfinite(a) for a in att.attributions)


def test_nan_producing_predictor_is_rejected():
    """NEGATIVE: an unstable predictor must raise, not emit NaN attributions."""
    def bad(texts):
        return np.array([float("nan")] * len(texts))
    try:
        occlusion_attribution("a b c", bad)
    except ValueError as exc:
        assert "nan" in str(exc).lower()
        return
    raise AssertionError("NaN attributions were not rejected")


def test_deterministic_explanation():
    text = "the senator said fake things today"
    a = occlusion_attribution(text, keyword_predict())
    b = occlusion_attribution(text, keyword_predict())
    assert a.attributions == b.attributions, "occlusion is not deterministic"


def test_stability_check_reports_identical():
    text = "the senator said fake things"
    res = stability_check(text, lambda t: occlusion_attribution(t, keyword_predict()),
                          n_repeats=3)
    assert res["identical"] is True
    assert res["max_absolute_difference"] == 0.0


def test_empty_text_handled_not_crashed():
    att = occlusion_attribution("   ", keyword_predict())
    assert att.tokens == [] and att.error


def test_truncation_is_recorded():
    att = occlusion_attribution(" ".join(["w"] * 50), keyword_predict(), max_tokens=10)
    assert att.truncated is True and att.n_tokens == 10


def test_tamil_tokens_preserved():
    """Whitespace tokenisation must keep whole Tamil words intact.

    Module 6 hit a real bug where a regex shattered Tamil at every vowel sign;
    this guards the same failure here.
    """
    att = occlusion_attribution(TAMIL, keyword_predict(keyword="never"))
    assert att.tokens == TAMIL.split()
    joined = "".join(att.tokens)
    assert all(ch in joined for ch in TAMIL if "஀" <= ch <= "௿")
    assert "�" not in joined


def test_simple_tokenize_is_whitespace_only():
    assert simple_tokenize("a b  c") == ["a", "b", "c"]
    assert simple_tokenize(TAMIL) == TAMIL.split()


# ---------------------------------------------------------- method labels

def test_fallback_is_never_labelled_shap():
    """THE methods-integrity test."""
    method, reason = resolve_method("occlusion", allow_shap=True)
    assert method == "occlusion"
    method, reason = resolve_method("shap", allow_shap=False)
    assert method == "occlusion", "SHAP disabled must not yield a shap label"
    assert "occlusion" in reason.lower()
    ok, _ = is_shap_available()
    method, reason = resolve_method("shap", allow_shap=True)
    if ok:
        assert method == SHAP_EMBEDDING_LABEL
    else:
        assert method == "occlusion" and "not installed" in reason.lower()
    assert method != "shap", "plain 'shap' must never be used as a label"


def test_shap_label_is_embedding_space():
    assert SHAP_EMBEDDING_LABEL == "shap_embedding_space"
    assert CONFIG["explanation"]["shap_label"] == SHAP_EMBEDDING_LABEL


def test_unknown_method_rejected():
    try:
        resolve_method("magic")
    except ValueError:
        return
    raise AssertionError("unknown explanation method accepted")


def test_explain_embedding_raises_when_shap_missing():
    ok, _ = is_shap_available()
    if ok:
        return
    try:
        explain_embedding(np.zeros(8), lambda x: np.zeros(len(x)), np.zeros((4, 8)))
    except ImportError as exc:
        assert "occlusion" in str(exc).lower()
        return
    raise AssertionError("missing shap did not raise")


def test_select_background_is_deterministic():
    x = np.random.default_rng(0).normal(size=(100, 8))
    assert np.array_equal(select_background(x, 10, seed=42),
                          select_background(x, 10, seed=42))
    assert select_background(x, 200).shape[0] == 100


# --------------------------------------------------------- faithfulness

def test_deletion_test_hand_checked():
    """Deleting the causal token must move the probability."""
    att = occlusion_attribution("the senator said fake things", keyword_predict())
    res = deletion_test("the senator said fake things", att, keyword_predict(),
                        fractions=[0.2])
    assert res["applicable"] is True
    assert res["curve"][0]["delta"] < -0.5, res["curve"]


def test_insertion_test_hand_checked():
    """The top token alone should already recover most of the prediction."""
    att = occlusion_attribution("the senator said fake things", keyword_predict())
    res = insertion_test("the senator said fake things", att, keyword_predict(),
                         fractions=[0.2])
    assert res["applicable"] is True
    assert abs(res["curve"][0]["recovery"]) < 1e-9, res["curve"]


def test_attribution_concentration_hand_computed():
    """|attr| = [3,1,0,0]; total 4 -> top_1 = 0.75, top_3 = 1.0"""
    c = attribution_concentration([3.0, -1.0, 0.0, 0.0], top_k=(1, 3))
    assert abs(c["top_1"] - 0.75) < 1e-12
    assert abs(c["top_3"] - 1.0) < 1e-12


def test_concentration_handles_all_zero():
    c = attribution_concentration([0.0, 0.0], top_k=(1,))
    assert c["top_1"] == 0.0


def test_faithfulness_on_empty_attribution():
    empty = TokenAttribution([], [], OCCLUSION, 0.5)
    assert deletion_test("", empty, keyword_predict())["applicable"] is False
    assert insertion_test("", empty, keyword_predict())["applicable"] is False


# ------------------------------------------------- verification explanation

def test_explain_verification_reports_verdict_and_margin():
    out = explain_verification({"SUPPORTS": 0.08, "REFUTES": 0.78, "NOT ENOUGH INFO": 0.14},
                               [{"evidence_id": "e1", "text": "some evidence"}],
                               [1.0])
    assert out["has_verification"] is True
    assert out["verdict"] == "REFUTES"
    assert abs(out["verdict_probability"] - 0.78) < 1e-12
    assert abs(out["margin"] - (0.78 - 0.14)) < 1e-12
    assert out["certainty"] == "confident"
    assert out["evidence"][0]["evidence_id"] == "e1"


def test_missing_verifier_output_handled():
    out = explain_verification(None, None)
    assert out["has_verification"] is False and out["n_evidence"] == 0


def test_evidence_text_truncated():
    out = explain_verification({"A": 1.0}, [{"evidence_id": "e", "text": "x" * 999}],
                               truncate_chars=50)
    assert len(out["evidence"][0]["text"]) <= 53


def test_contribution_breakdown_sums_to_fused():
    c = contribution_breakdown(0.8, 0.4, alpha=0.75, fused_p_fake=0.7, has_verifier=True)
    assert abs(c["classifier_contribution"] - 0.6) < 1e-12
    assert abs(c["verifier_contribution"] - 0.1) < 1e-12
    assert c["matches_fused"] is True


def test_contribution_without_verifier():
    c = contribution_breakdown(0.8, None, alpha=1.0, fused_p_fake=0.8, has_verifier=False)
    assert c["verifier_used"] is False and c["verifier_contribution"] == 0.0


# --------------------------------------------------- combined explanation

def _record(**kw):
    base = {"record_id": "r1", "language": "en", "text": "the senator said fake things",
            "gold_label": 1, "final_prediction": 1, "fused_p_fake": 0.87,
            "module4_p_fake": 0.81, "p_fake_verifier": 0.9, "has_verifier": True}
    base.update(kw)
    return base


def test_explanation_record_schema():
    att = occlusion_attribution("the senator said fake things", keyword_predict())
    row = build_explanation_record(
        _record(), att, {"SUPPORTS": 0.08, "REFUTES": 0.78, "NOT ENOUGH INFO": 0.14},
        [{"evidence_id": "e1", "text": "evidence"}], [1.0], alpha=0.6,
        method_label=OCCLUSION)
    for field in CONFIG["evaluation"]["record_fields"]:
        assert field in row, f"explanation record missing required field {field!r}"
    assert row["explanation_method"] == OCCLUSION
    assert 0.0 <= row["confidence"] <= 1.0


def test_reason_is_generated_from_values_not_templated():
    strong = build_reason(1, 0.95, 0.95, {"has_verification": False}, {})
    weak = build_reason(1, 0.55, 0.55, {"has_verification": False}, {})
    assert "strongly" in strong and "weakly" in weak
    assert strong != weak


def test_reason_flags_indecisive_verifier_margin():
    v = {"has_verification": True, "verdict": "REFUTES", "verdict_probability": 0.36,
         "certainty": "very weak", "n_evidence": 2, "margin": 0.02,
         "probabilities": {}}
    r = build_reason(1, 0.6, 0.6, v, {"verifier_used": False})
    assert "not decisive" in r


def test_reason_states_when_no_verifier():
    r = build_reason(0, 0.2, 0.2, {"has_verification": False}, {})
    assert "classifier alone" in r


def test_render_text_report_contains_key_sections():
    att = occlusion_attribution("the senator said fake things", keyword_predict())
    row = build_explanation_record(_record(), att, {"REFUTES": 0.8, "SUPPORTS": 0.2},
                                   [{"evidence_id": "e1", "text": "ev"}], [1.0],
                                   0.6, OCCLUSION)
    txt = render_text_report(row)
    for section in ("Prediction:", "Explanation method:", "Verification:",
                    "Evidence:", "Primary reason:"):
        assert section in txt, f"report missing {section!r}"


def test_tamil_record_unicode_preserved_end_to_end():
    att = occlusion_attribution(TAMIL, keyword_predict(keyword="never"))
    row = build_explanation_record(_record(record_id="t1", language="ta", text=TAMIL),
                                   att, None, None, None, 1.0, OCCLUSION)
    assert row["text"] == TAMIL
    assert "�" not in row["text"]
    assert [c for c in row["text"] if "஀" <= c <= "௿"] == \
           [c for c in TAMIL if "஀" <= c <= "௿"]


# ------------------------------------------------------------- metrics

def test_summarise_explanations_language_accounting():
    rows = [{"record_id": "a", "language": "en", "explanation_method": "occlusion",
             "token_attributions": [1], "evidence": [], "verification_probabilities": {},
             "attribution_diagnostics": {"n_tokens": 5, "seconds": 0.1}},
            {"record_id": "b", "language": "ta", "explanation_method": "occlusion",
             "token_attributions": [], "evidence": [], "verification_probabilities": {},
             "attribution_diagnostics": {"n_tokens": 0, "seconds": 0.2, "error": "x"}}]
    s = summarise_explanations(rows, ["en", "ta"])
    assert s["overall"]["n_records"] == 2
    assert s["language_accounting"]["balances"] is True
    assert s["overall"]["n_failures"] == 1
    assert abs(s["overall"]["failure_rate"] - 0.5) < 1e-12


def test_summarise_faithfulness_defines_every_metric():
    f = summarise_faithfulness([{"language": "en",
                                 "deletion": {"applicable": True, "mean_absolute_delta": 0.4},
                                 "insertion": {"applicable": True,
                                               "curve": [{"recovery": -0.05}]},
                                 "concentration": {"top_1": 0.5, "top_3": 0.9},
                                 "stability": {"identical": True}}], ["en", "ta"])
    assert f["overall"]["n_records"] == 1
    assert f["overall"]["stability_identical_rate"] == 1.0
    for key in ("deletion_mean_absolute_delta", "insertion_final_gap_to_full_prediction",
                "attribution_concentration", "stability_identical_rate",
                "no_composite_score"):
        assert key in f["definitions"], f"metric {key!r} undefined"


# ---------------------------------------------------------------- config

def test_config_forbids_fitting_and_marks_test_read_only():
    assert CONFIG["leakage"]["no_parameters_fitted"] is True
    assert CONFIG["leakage"]["test_is_read_only"] is True
    assert CONFIG["leakage"]["fit_split"] == "validation"


def test_config_declares_sampling_and_cost_ceiling():
    s = CONFIG["explanation"]["sample"]
    assert s["validation"] > 0 and s["test"] > 0
    assert CONFIG["explanation"]["max_tokens_per_record"] > 0
    assert CONFIG["explanation"]["deterministic"] is True


def test_config_primary_method_is_labelled_honestly():
    assert CONFIG["explanation"]["primary_method"] in ("occlusion", "shap")
    assert CONFIG["explanation"]["occlusion_label"] == "occlusion"
    assert CONFIG["explanation"]["shap_label"] != "shap"


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
