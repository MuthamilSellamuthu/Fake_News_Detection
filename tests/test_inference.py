"""Module 10 — inference tests.

Runs without torch, without checkpoints and without a GPU. The heavy stages are
injected through `Dependencies`, the same pattern `VerifierTrainer` uses for
`step_fn`/`eval_fn`, so the orchestration and every invariant is testable in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.pipeline import (  # noqa: E402
    Dependencies, InferenceConfigError, compute_confidence, detect_language,
    display_state, load_frozen_fusion, run_inference, validate_input)
from src.inference.report import (  # noqa: E402
    ReasonSentence, build_reason_block, render_report, validate_provenance)

ENGLISH = ("Scientists at the national research institute announced on Monday that "
           "the new vaccine trial reached its primary endpoint after eighteen months "
           "of follow up across four hospitals.")
TAMIL = ("தமிழ்நாடு அரசு புதிய கல்வித் திட்டத்தை அறிவித்தது. இந்த திட்டம் "
         "அடுத்த கல்வியாண்டில் அமல்படுத்தப்படும் என்று அதிகாரிகள் தெரிவித்தனர்.")


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def cfg(tmp_path):
    """Real Module 10 config, with ONLY the frozen fusion artifact redirected.

    Everything else resolves against the real repo, so the tests exercise the
    actual config chain (module6 -> module5 claim-selection rules) rather than
    a stub that could drift from it.
    """
    from src.utils.config_loader import load_yaml
    c = load_yaml(ROOT / "configs/module10_inference.yaml")
    fp = tmp_path / "fusion_parameters.json"
    fp.write_text(json.dumps({
        "parameters": {"alpha": 0.95, "threshold": 0.56, "temperature": 0.7079,
                       "fit_split": "validation"}}), encoding="utf-8")
    c["inputs"]["fusion_parameters"] = str(fp)      # absolute -> root / abs == abs
    return c


@pytest.fixture
def fake_root():
    return ROOT


def make_deps(p_fake=0.9, evidence=None, verifier=None):
    ev = [] if evidence is None else evidence

    def encode_fn(text):
        v = np.ones(768, dtype=np.float32)
        return v / np.linalg.norm(v)

    return Dependencies(
        encode_fn=encode_fn,
        classify_fn=lambda vec: float(p_fake),
        retrieve_fn=lambda claim, lang: list(ev),
        verify_fn=lambda claim, e, lang: verifier,
        encoder_provenance={"encoder": "xlm-roberta-base", "pooling": "mean",
                            "normalized": True, "max_length": 512,
                            "self_check": {"passed": True, "cosine": 0.99999,
                                           "abs_p_fake_delta": 2.0e-06}})


def ev_item(rank=1, eid="ev1", corpus="fever", text="evidence text"):
    return {"rank": rank, "evidence_id": eid, "corpus": corpus, "pool_size": 100,
            "retriever": "bm25", "score": 12.5, "confidence": 1.0, "text": text}


FEVER_V = {"dataset": "fever", "label_space": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
           "probs": {"SUPPORTS": 0.1, "REFUTES": 0.7, "NOT ENOUGH INFO": 0.2},
           "verdict": "REFUTES", "verdict_probability": 0.7, "n_evidence_used": 1,
           "evidence_ids": ["ev1"], "checkpoint": "models/module6/fever/best.pt"}
XFACT_V = {"dataset": "x_fact",
           "label_space": ["false", "true", "partly true/misleading"],
           "probs": {"false": 0.6, "true": 0.2, "partly true/misleading": 0.2},
           "verdict": "false", "verdict_probability": 0.6, "n_evidence_used": 1,
           "evidence_ids": ["ev1"], "checkpoint": "models/module6/x_fact/best.pt"}


# ------------------------------------------------------------ 1. validation
def test_empty_input_rejected(cfg):
    assert validate_input("", cfg)["accepted"] is False
    assert validate_input("   ", cfg)["accepted"] is False
    assert validate_input(None, cfg)["accepted"] is False


def test_short_input_rejected_with_reason(cfg):
    r = validate_input("too short", cfg)
    assert r["accepted"] is False and "too short" in r["reason"]


def test_valid_input_accepted_and_unaltered(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg, deps=make_deps(), explain=False)
    assert res["input"]["accepted"] is True
    assert res["input"]["text"] == ENGLISH          # never silently altered


# ------------------------------------------------------ 2. language detection
def test_language_detection_english():
    assert detect_language(ENGLISH)["code"] == "en"


def test_language_detection_tamil():
    d = detect_language(TAMIL)
    assert d["code"] == "ta" and d["tamil_ratio"] > 0.3


def test_tamil_unicode_survives(cfg, fake_root):
    res = run_inference(TAMIL, fake_root, cfg, deps=make_deps(), explain=False)
    assert "�" not in res["input"]["text"]
    assert "தமிழ்நாடு" in res["input"]["text"]
    assert "�" not in json.dumps(res, ensure_ascii=False)


# --------------------------------------------------------- 3/4. claim + encoder
def test_claim_selection_records_rule_and_text(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg, title="Vaccine trial reaches endpoint",
                        deps=make_deps(), explain=False)
    assert res["claim"]["text"]
    assert res["claim"]["selection_rule"]
    assert res["claim"]["claim_id"].startswith("inference:")


def test_encoder_output_shape_and_norm():
    deps = make_deps()
    v = deps.encode_fn(ENGLISH)
    assert np.asarray(v).shape == (768,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_pipeline_records_embedding_norm(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg, deps=make_deps(), explain=False)
    assert res["classifier"]["embedding_dim"] == 768
    assert abs(res["classifier"]["embedding_l2_norm"] - 1.0) < 1e-5


def test_encoder_rejects_wrong_model_name(tmp_path, cfg, monkeypatch):
    """The self-check must refuse a non-XLM-R encoder before loading anything."""
    from src.inference.text_encoder import EncoderProvenanceError, InferenceEncoder
    from src.utils import config_loader

    bad = {"model": {"name": "sentence-transformers/LaBSE"},
           "tokenization": {"max_length": 512, "padding": "longest"},
           "extraction": {"normalize_embeddings": True},
           "input": {}, "output": {}}
    monkeypatch.setattr(config_loader, "load_yaml", lambda p: bad)
    monkeypatch.setattr("src.utils.config_loader.load_yaml", lambda p: bad)
    with pytest.raises(EncoderProvenanceError, match="xlm-roberta-base"):
        InferenceEncoder(tmp_path, cfg, classify_fn=lambda v: 0.5, self_check=False)


def test_encoder_rejects_unnormalised_config(tmp_path, cfg, monkeypatch):
    from src.inference.text_encoder import EncoderProvenanceError, InferenceEncoder

    bad = {"model": {"name": "xlm-roberta-base"},
           "tokenization": {"max_length": 512, "padding": "longest"},
           "extraction": {"normalize_embeddings": False},
           "input": {}, "output": {}}
    monkeypatch.setattr("src.utils.config_loader.load_yaml", lambda p: bad)
    with pytest.raises(EncoderProvenanceError, match="normalize_embeddings"):
        InferenceEncoder(tmp_path, cfg, classify_fn=lambda v: 0.5, self_check=False)


# --------------------------------------------------------------- 5. classifier
@pytest.mark.parametrize("p", [0.0, 0.001, 0.5, 0.9713, 1.0])
def test_classifier_probability_in_range(cfg, fake_root, p):
    res = run_inference(ENGLISH, fake_root, cfg, deps=make_deps(p_fake=p), explain=False)
    c = res["classifier"]
    assert 0.0 <= c["p_fake"] <= 1.0 and 0.0 <= c["p_real"] <= 1.0
    assert abs(c["p_fake"] + c["p_real"] - 1.0) < 1e-12


# ------------------------------------------------------------------ 6/7. fusion
def test_fusion_parameters_loaded_from_artifact(fake_root, cfg):
    p = load_frozen_fusion(fake_root, cfg)
    assert p.alpha == 0.95 and p.threshold == 0.56
    assert abs(p.temperature - 0.7079) < 1e-12
    assert p.fit_split == "validation"


def test_missing_fusion_artifact_fails_loudly(tmp_path, cfg):
    missing = dict(cfg)
    missing["inputs"] = {**cfg["inputs"],
                         "fusion_parameters": str(tmp_path / "does_not_exist.json")}
    with pytest.raises(InferenceConfigError, match="Frozen fusion parameters not found"):
        load_frozen_fusion(tmp_path, missing)


def test_fusion_params_fitted_on_test_are_rejected(tmp_path, cfg):
    """A leakage guard: parameters not fitted on validation must be refused."""
    bad = tmp_path / "leaky.json"
    bad.write_text(json.dumps({"parameters": {
        "alpha": 0.95, "threshold": 0.56, "temperature": 0.7079,
        "fit_split": "test"}}), encoding="utf-8")
    leaky = dict(cfg)
    leaky["inputs"] = {**cfg["inputs"], "fusion_parameters": str(bad)}
    with pytest.raises(InferenceConfigError, match="not 'validation'"):
        load_frozen_fusion(tmp_path, leaky)


def test_fusion_params_not_hardcoded_in_source():
    """The decision numbers must not appear as literals in src/inference/."""
    for f in (ROOT / "src/inference").glob("*.py"):
        src = f.read_text(encoding="utf-8")
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("#"))
        for lit in ("0.95", "0.56", "0.7079"):
            assert lit not in code, f"{f.name} hard-codes {lit}"


def test_parameters_reach_the_result_unchanged(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg, deps=make_deps(0.9),
                        explain=False)
    f = res["fusion"]
    assert (f["alpha"], f["threshold"]) == (0.95, 0.56)
    assert f["fit_split"] == "validation"


def test_no_fitting_functions_called_during_inference(cfg, fake_root, monkeypatch):
    import src.integration.fusion as fusion_mod

    def boom(*a, **k):
        raise AssertionError("a fitting function was called during inference")

    monkeypatch.setattr(fusion_mod, "fit_fusion", boom)
    monkeypatch.setattr(fusion_mod, "fit_binary_temperature", boom)
    monkeypatch.setattr(fusion_mod, "fit_meta_classifier", boom)
    res = run_inference(ENGLISH, fake_root, cfg, deps=make_deps(), explain=False)
    assert res["fusion"]["fused_p_fake"] is not None


# ------------------------------------------------------- 8. confidence + state
def test_confidence_is_not_p_fake():
    assert compute_confidence(0.56, 0.56) == 0.0
    assert compute_confidence(1.0, 0.56) == pytest.approx(1.0)
    assert compute_confidence(0.0, 0.56) == pytest.approx(1.0)
    assert compute_confidence(0.9713, 0.56) != pytest.approx(0.9713)


def test_confidence_bounded():
    for p in np.linspace(0, 1, 41):
        assert 0.0 <= compute_confidence(float(p), 0.56) <= 1.0


def test_display_state_rule():
    assert display_state(0.90, 0.56, 0.05) == "FAKE"
    assert display_state(0.10, 0.56, 0.05) == "REAL"
    assert display_state(0.58, 0.56, 0.05) == "UNCERTAIN"
    assert display_state(0.54, 0.56, 0.05) == "UNCERTAIN"


def test_threshold_decides_even_when_state_is_uncertain(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg, deps=make_deps(p_fake=0.57),
                        explain=False)
    f = res["fusion"]
    assert f["final_prediction"] == int(f["fused_p_fake"] >= f["threshold"])


# ------------------------------------------------------------- 9/10. evidence
def test_no_evidence_reported_not_fabricated(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg, deps=make_deps(evidence=[]),
                        explain=False)
    assert res["retrieval"]["status"] == "NO EVIDENCE RETRIEVED"
    assert res["retrieval"]["candidates"] == []
    assert res["verification"]["available"] is False


def test_evidence_provenance_preserved(cfg, fake_root):
    ev = [ev_item(1, "ev1"), ev_item(2, "ev2")]
    res = run_inference(ENGLISH, fake_root, cfg,
                        deps=make_deps(evidence=ev, verifier=FEVER_V), explain=False)
    for e in res["retrieval"]["candidates"]:
        assert {"evidence_id", "corpus", "score", "rank", "retriever", "text"} <= set(e)
    assert res["retrieval"]["label"].startswith("CANDIDATE EVIDENCE")


def test_missing_verifier_falls_back_to_classifier(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg,
                        deps=make_deps(0.8, evidence=[ev_item()], verifier=None),
                        explain=False)
    assert res["fusion"]["has_verifier"] is False
    assert res["fusion"]["p_fake_verifier"] is None
    assert res["fusion"]["fused_p_fake_before_temperature"] == pytest.approx(0.8)


# --------------------------------------------------------- 11. verifier labels
def test_english_uses_fever_label_space(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg,
                        deps=make_deps(evidence=[ev_item()], verifier=FEVER_V),
                        explain=False)
    v = res["verification"]
    assert v["dataset"] == "fever"
    assert set(v["label_space"]) == {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}


def test_tamil_uses_xfact_label_space_never_fever(cfg, fake_root):
    res = run_inference(TAMIL, fake_root, cfg,
                        deps=make_deps(evidence=[ev_item(corpus="x_fact")],
                                       verifier=XFACT_V), explain=False)
    v = res["verification"]
    assert v["dataset"] == "x_fact"
    assert set(v["label_space"]) == {"false", "true", "partly true/misleading"}
    assert v["verdict"] == "false"
    blob = json.dumps(res, ensure_ascii=False)
    for fever_label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
        assert fever_label not in blob, f"X-FACT result leaked FEVER label {fever_label}"


def test_neutral_verdict_is_not_read_as_fake(cfg, fake_root):
    nei = {**FEVER_V, "verdict": "NOT ENOUGH INFO",
           "probs": {"SUPPORTS": 0.2, "REFUTES": 0.2, "NOT ENOUGH INFO": 0.6},
           "verdict_probability": 0.6}
    res = run_inference(ENGLISH, fake_root, cfg,
                        deps=make_deps(0.4, evidence=[ev_item()], verifier=nei),
                        explain=False)
    txt = " ".join(s["text"] for s in res["reason"]["sentences"])
    assert "NOT evidence that the article is false" in txt


# ------------------------------------------------------------ 12. attribution
def test_attribution_sign_semantics():
    from src.explainability.token_attribution import occlusion_attribution

    def predict_fn(texts):
        # "hoax" raises P(fake); removing it lowers P(fake) -> positive attribution
        return np.array([0.9 if "hoax" in t else 0.2 for t in texts], dtype=float)

    attr = occlusion_attribution("this hoax spread widely", predict_fn, mask_token="")
    i = attr.tokens.index("hoax")
    assert attr.attributions[i] > 0
    assert len(attr.tokens) == len(attr.attributions)
    top = attr.top_tokens(5)
    assert all(t["attribution"] > 0 for t in top["top_positive_tokens"])
    assert all(t["attribution"] < 0 for t in top["top_negative_tokens"])


def test_attribution_preserves_tamil_tokens():
    from src.explainability.token_attribution import simple_tokenize
    toks = simple_tokenize(TAMIL)
    assert "தமிழ்நாடு" in toks
    assert all("�" not in t for t in toks)


# ------------------------------------------------------------- 13. reason/report
def test_reason_sentence_requires_provenance():
    with pytest.raises(ValueError, match="no provenance"):
        ReasonSentence("t", "MEASURED", "text", [])


def test_reason_provenance_all_resolve(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg,
                        deps=make_deps(evidence=[ev_item()], verifier=FEVER_V),
                        explain=False)
    assert validate_provenance(res) == []


def test_reason_has_no_language_model():
    assert "no language model" in build_reason_block({
        "fusion": {"display_state": "REAL", "fused_p_fake": 0.1, "threshold": 0.56,
                   "alpha": 0.95, "uncertain_band": 0.05, "final_label": "REAL"},
        "classifier": {"p_fake": 0.1}, "verification": {"available": False,
                                                        "reason": "none"},
        "retrieval": {"status": "NO EVIDENCE RETRIEVED", "pool": "fever"},
        "explainability": {}})["generator"]


def test_disagreement_is_reported(cfg, fake_root):
    supports = {**FEVER_V, "verdict": "SUPPORTS",
                "probs": {"SUPPORTS": 0.8, "REFUTES": 0.1, "NOT ENOUGH INFO": 0.1}}
    res = run_inference(ENGLISH, fake_root, cfg,
                        deps=make_deps(0.95, evidence=[ev_item()], verifier=supports),
                        explain=False)
    txt = " ".join(s["text"] for s in res["reason"]["sentences"])
    assert "DISAGREEMENT" in txt


def test_output_schema_is_stable(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg,
                        deps=make_deps(evidence=[ev_item()], verifier=FEVER_V),
                        explain=False)
    for k in ("schema_version", "input", "language", "claim", "classifier",
              "retrieval", "verification", "fusion", "reason", "limitations",
              "provenance"):
        assert k in res, f"missing top-level field {k}"
    for k in ("alpha", "threshold", "temperature", "fused_p_fake", "confidence",
              "display_state", "final_prediction"):
        assert k in res["fusion"], f"missing fusion.{k}"


def test_report_renders_and_shows_threshold(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg,
                        deps=make_deps(evidence=[ev_item()], verifier=FEVER_V),
                        explain=False)
    txt = render_report(res)
    assert "Decision threshold" in txt and "56.00%" in txt
    assert "CANDIDATE EVIDENCE" in txt
    assert "LIMITATIONS" in txt


def test_limitations_state_alpha_and_threshold(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg, deps=make_deps(), explain=False)
    blob = " ".join(res["limitations"])
    assert "0.95" in blob and "0.56" in blob
    assert "fixed corpus" in blob.lower() or "FIXED corpus" in blob


# ------------------------------------------------------------ 14. end-to-end
def test_end_to_end_with_mocked_dependencies(cfg, fake_root):
    res = run_inference(ENGLISH, fake_root, cfg, title="Headline here",
                        deps=make_deps(0.97, evidence=[ev_item(1), ev_item(2, "ev2")],
                                       verifier=FEVER_V), explain=False)
    assert res["fusion"]["display_state"] in ("FAKE", "REAL", "UNCERTAIN")
    assert res["fusion"]["final_prediction"] in (0, 1)
    assert validate_provenance(res) == []
    assert render_report(res)


def test_deterministic_for_identical_input(cfg, fake_root):
    a = run_inference(ENGLISH, fake_root, cfg, deps=make_deps(0.9), explain=False)
    b = run_inference(ENGLISH, fake_root, cfg, deps=make_deps(0.9), explain=False)
    assert a["fusion"]["fused_p_fake"] == b["fusion"]["fused_p_fake"]
    assert a["claim"]["claim_id"] == b["claim"]["claim_id"]
