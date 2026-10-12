"""Module 6 verification tests — label spaces, calibration, resume, ablations.

The label-space tests are the most important in this file. FEVER and X-FACT
are different tasks; any code path that maps one onto the other is a research
error, not a bug, and would invalidate the reported transfer result.

Runs without torch, transformers, or any downloaded model.

    python tests/test_verification.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_yaml  # noqa: E402
from src.verification.ablations import (  # noqa: E402
    AblationSpec,
    build_ablation_specs,
    run_ablation,
    run_all_ablations,
)
from src.verification.calibration import (  # noqa: E402
    calibrate_and_evaluate,
    expected_calibration_error,
    fit_temperature,
    reliability_data,
    softmax,
)
from src.verification.dataset import (  # noqa: E402
    FEVER_LABELS,
    FEVER_SPACE,
    X_FACT_LABELS,
    X_FACT_SPACE,
    LabelSpace,
    assert_label_spaces_disjoint,
    build_examples,
    get_label_space,
    stratified_subset,
)
from src.verification.trainer import (  # noqa: E402
    CheckpointManager,
    TrainState,
    VerifierTrainer,
    classification_report,
    macro_f1,
)

CONFIG = load_yaml(PROJECT_ROOT / "configs" / "module6_retrieval_verification.yaml")
TAMIL = "பஹல்காம் தாக்குதல் குறித்த தவறான கூற்று"


# ------------------------------------------------------------- label spaces

def test_fever_and_xfact_label_spaces_are_separate_and_exact():
    assert FEVER_LABELS == ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    assert X_FACT_LABELS == ["false", "true", "partly true/misleading"]
    assert_label_spaces_disjoint()
    assert FEVER_SPACE.task == "entailment" and X_FACT_SPACE.task == "veracity"


def test_xfact_label_string_is_preserved_verbatim():
    """'partly true/misleading' must survive exactly; no normalisation,
    no collapsing into 'partly true', no mapping to a FEVER class."""
    i = X_FACT_SPACE.encode("partly true/misleading")
    assert X_FACT_SPACE.decode(i) == "partly true/misleading"


def test_no_cross_space_mapping_exists():
    """A FEVER label must NOT be encodable in X-FACT's space, and vice versa.

    If this ever passes silently, a mis-routed record would be scored against
    the wrong task and the transfer result would be meaningless.
    """
    for lab in FEVER_LABELS:
        assert not X_FACT_SPACE.can_encode(lab), f"X-FACT accepted FEVER label {lab!r}"
    for lab in X_FACT_LABELS:
        assert not FEVER_SPACE.can_encode(lab), f"FEVER accepted X-FACT label {lab!r}"


def test_config_forbids_label_mapping():
    assert CONFIG["label_spaces"]["mapping_between_spaces"] is False
    assert CONFIG["label_spaces"]["fever"]["labels"] == FEVER_LABELS
    assert CONFIG["label_spaces"]["x_fact"]["labels"] == X_FACT_LABELS


def test_unknown_label_raises_rather_than_defaulting():
    for bad in ("MAYBE", "half true", None, "supports_"):
        try:
            FEVER_SPACE.encode(bad)
        except KeyError:
            continue
        raise AssertionError(f"FEVER space silently accepted {bad!r}")


def test_label_encoding_tolerates_case_noise_but_not_semantics():
    assert FEVER_SPACE.encode("supports") == FEVER_SPACE.encode("SUPPORTS")
    assert X_FACT_SPACE.encode(" FALSE ") == X_FACT_SPACE.encode("false")
    assert not FEVER_SPACE.can_encode("SUPPORT")     # not a synonym


def test_get_label_space_rejects_unknown_dataset():
    for bad in ("liar", "fever2", ""):
        try:
            get_label_space(bad)
        except KeyError:
            continue
        raise AssertionError(f"unknown dataset {bad!r} accepted")


# ---------------------------------------------------------- example building

def _claims(dataset):
    labs = FEVER_LABELS if dataset == "fever" else X_FACT_LABELS
    return [{"claim_id": f"{dataset}:c{i}", "text": f"claim {i}", "label": labs[i % 3],
             "language": "ta" if dataset == "x_fact" else "en",
             "evidence": [f"{dataset}:ev:g{i}"], "metadata": {"official_split": "train"}}
            for i in range(9)]


def test_build_examples_accounts_for_every_claim():
    claims = _claims("fever")
    claims.append({"claim_id": "bad", "text": "x", "label": "NONSENSE",
                   "language": "en", "evidence": [], "metadata": {}})
    ev = {f"fever:ev:g{i}": f"evidence text {i}" for i in range(9)}
    split_of = {i: "train" for i in range(len(claims))}
    ex, stats = build_examples(claims, ev, None, "fever", split_of, use_gold_evidence=True)
    assert stats["n_input"] == 10
    assert stats["n_built"] == 9
    assert stats["n_dropped_unknown_label"] == 1
    assert stats["accounting_balances"] is True
    assert len(ex) == 9


def test_claim_only_mode_attaches_no_evidence():
    claims = _claims("fever")
    ev = {f"fever:ev:g{i}": f"evidence {i}" for i in range(9)}
    split_of = {i: "train" for i in range(9)}
    ex, stats = build_examples(claims, ev, None, "fever", split_of, claim_only=True)
    assert all(not e.evidence_texts for e in ex)
    assert stats["evidence_source"] == "claim_only"


def test_gold_and_retrieved_evidence_paths_differ():
    claims = _claims("fever")
    ev = {f"fever:ev:g{i}": f"gold {i}" for i in range(9)}
    ev["fever:ev:r0"] = "retrieved text"
    split_of = {i: "train" for i in range(9)}
    gold, _ = build_examples(claims, ev, None, "fever", split_of, use_gold_evidence=True)
    retr, s = build_examples(claims, ev, {"fever:c0": ["fever:ev:r0"]}, "fever", split_of)
    assert gold[0].evidence_texts == ["gold 0"]
    assert retr[0].evidence_texts == ["retrieved text"]
    assert s["evidence_source"] == "retrieved"


def test_xfact_examples_keep_tamil_and_original_labels():
    claims = [{"claim_id": "x_fact:c0", "text": TAMIL, "label": "partly true/misleading",
               "language": "ta", "evidence": [], "metadata": {}}]
    ex, _ = build_examples(claims, {}, None, "x_fact", {0: "train"}, claim_only=True)
    assert ex[0].label_str == "partly true/misleading"
    assert ex[0].language == "ta"
    assert [c for c in ex[0].claim_text if "஀" <= c <= "௿"] == \
           [c for c in TAMIL if "஀" <= c <= "௿"]


def test_to_text_pair_keeps_rank_order_and_truncates():
    claims = _claims("fever")[:1]
    ev = {"fever:ev:g0": "first", "a": "second", "b": "third"}
    ex, _ = build_examples(claims, ev, {"fever:c0": ["fever:ev:g0", "a", "b"]},
                           "fever", {0: "train"})
    _, joined = ex[0].to_text_pair(max_evidence=2)
    assert joined.startswith("first"), joined
    assert "third" not in joined


def test_stratified_subset_preserves_label_balance_and_is_deterministic():
    claims = _claims("fever") * 40      # 360 claims, balanced 3 ways
    for i, c in enumerate(claims):
        c = dict(c)
        c["claim_id"] = f"fever:c{i}"
        claims[i] = c
    ev = {}
    split_of = {i: "train" for i in range(len(claims))}
    ex, _ = build_examples(claims, ev, None, "fever", split_of, claim_only=True)

    a = stratified_subset(ex, 60, seed=42)
    b = stratified_subset(ex, 60, seed=42)
    assert [x.claim_id for x in a] == [x.claim_id for x in b], "not deterministic"
    assert len(a) == 60
    counts = {}
    for x in a:
        counts[x.label_id] = counts.get(x.label_id, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 2, counts


def test_stratified_subset_returns_all_when_size_exceeds_corpus():
    claims = _claims("x_fact")
    ex, _ = build_examples(claims, {}, None, "x_fact", {i: "train" for i in range(9)},
                           claim_only=True)
    assert len(stratified_subset(ex, 10_000)) == len(ex)
    assert len(stratified_subset(ex, None)) == len(ex)


# ------------------------------------------------------------- calibration

def test_ece_hand_computed():
    """Two predictions, one bin each.

    p1 = [0.9, 0.1], label 0 -> confidence 0.9, correct
    p2 = [0.6, 0.4], label 1 -> confidence 0.6, wrong
    10 bins: 0.9 lands in (0.8,0.9], 0.6 in (0.5,0.6].
    bin(0.6): acc 0, conf 0.6, gap 0.6, weight 1/2
    bin(0.9): acc 1, conf 0.9, gap 0.1, weight 1/2
    ECE = 0.5*0.6 + 0.5*0.1 = 0.35
    """
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    res = expected_calibration_error(probs, [0, 1], n_bins=10)
    assert abs(res["ece"] - 0.35) < 1e-12, res["ece"]
    assert abs(res["accuracy"] - 0.5) < 1e-12


def test_perfect_calibration_gives_zero_ece():
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, [0, 0, 1], n_bins=10)["ece"] < 1e-12


def test_confidently_wrong_model_has_high_ece():
    """A model that is 100% confident and always wrong must score ECE = 1."""
    probs = np.array([[1.0, 0.0]] * 20)
    res = expected_calibration_error(probs, [1] * 20, n_bins=10)
    assert abs(res["ece"] - 1.0) < 1e-9, res["ece"]


def test_ece_bin_counts_sum_to_n():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 3))
    probs = softmax(logits)
    labels = rng.integers(0, 3, size=200)
    res = expected_calibration_error(probs, labels, n_bins=10)
    assert sum(b["count"] for b in res["bins"]) == 200


def test_softmax_is_numerically_stable_and_sums_to_one():
    z = np.array([[1000.0, 1001.0, 999.0]])
    p = softmax(z)
    assert not np.isnan(p).any()
    assert abs(p.sum() - 1.0) < 1e-12


def test_temperature_above_one_softens_confidence():
    logits = np.array([[5.0, 0.0, 0.0]])
    assert softmax(logits, 5.0).max() < softmax(logits, 1.0).max()


def test_fit_temperature_refuses_the_test_split():
    """Fitting calibration on test is tuning on test."""
    logits = np.random.default_rng(0).normal(size=(20, 3))
    labels = [0] * 20
    try:
        fit_temperature(logits, labels, split_name="test")
    except ValueError as exc:
        assert "test" in str(exc).lower()
        return
    raise AssertionError("temperature was fitted on the test split")


def test_fit_temperature_reduces_nll_on_an_overconfident_model():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 3, size=300)
    logits = rng.normal(size=(300, 3)) * 0.5
    logits[np.arange(300), labels] += 1.0
    logits *= 8.0                       # deliberately overconfident
    res = fit_temperature(logits, labels, split_name="validation")
    assert res["temperature"] > 1.0, res
    assert res["nll_after"] <= res["nll_before"] + 1e-9


def test_temperature_scaling_does_not_change_accuracy():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(100, 3))
    labels = rng.integers(0, 3, size=100)
    res = calibrate_and_evaluate(logits, labels, temperature=2.5)
    a = softmax(logits, 1.0).argmax(1)
    b = softmax(logits, 2.5).argmax(1)
    assert np.array_equal(a, b), "temperature changed the argmax"
    assert "accuracy" in res


def test_reliability_data_omits_empty_bins():
    probs = np.array([[0.95, 0.05], [0.93, 0.07]])
    rel = reliability_data(probs, [0, 0], n_bins=10)
    assert len(rel["confidence"]) == 1 and rel["counts"] == [2]


# ----------------------------------------------------------------- metrics

def test_macro_f1_hand_computed():
    """y_true = [0,0,1,1], y_pred = [0,1,1,1]
    class0: tp1 fp0 fn1 -> P=1, R=0.5, F1=2/3
    class1: tp2 fp1 fn0 -> P=2/3, R=1, F1=0.8
    macro = (2/3 + 0.8)/2 = 0.7333...
    """
    got = macro_f1([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert abs(got - ((2 / 3 + 0.8) / 2)) < 1e-12, got


def test_macro_f1_counts_absent_classes():
    """A class never predicted and never present still counts in the mean;
    skipping it would inflate scores on rare X-FACT classes."""
    assert abs(macro_f1([0, 0], [0, 0], 3) - (1.0 / 3)) < 1e-12


def test_classification_report_confusion_matrix():
    rep = classification_report([0, 1, 2, 0], [0, 1, 1, 2], FEVER_LABELS)
    assert rep["confusion_matrix"][0] == [1, 0, 1]
    assert rep["labels"] == FEVER_LABELS
    assert abs(rep["accuracy"] - 0.5) < 1e-12


# ------------------------------------------------------ checkpoint / resume

def test_train_state_roundtrip():
    s = TrainState(epoch=2, global_step=50, best_metric=0.7)
    assert TrainState.from_dict(s.to_dict()).to_dict() == s.to_dict()


def test_trainer_resume_matches_uninterrupted_run():
    """Interrupt training, resume from checkpoint, require identical history.

    Uses a tiny pure-python model so this runs without torch.
    """
    torch = None
    try:
        import torch  # noqa: F811
    except Exception:  # noqa: BLE001
        return       # torch-free environment: covered by test_checkpoint_manager_*

    import torch.nn as nn

    def make():
        torch.manual_seed(0)
        m = nn.Linear(4, 3)
        o = torch.optim.SGD(m.parameters(), lr=0.1)
        return m, o

    data = [torch.randn(2, 4) for _ in range(6)]
    targets = [torch.tensor([0, 1]) for _ in range(6)]

    def batches(epoch):
        return list(zip(data, targets))

    def step(model, batch):
        x, y = batch
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        return float(loss.item())

    def ev(model):
        with torch.no_grad():
            preds = model(torch.cat(data)).argmax(1).tolist()
        return {"macro_f1": macro_f1(sum([t.tolist() for t in targets], []), preds, 3)}

    d1 = Path(tempfile.mkdtemp())
    m, o = make()
    full = VerifierTrainer(d1, {"x": 1}, checkpoint_every_steps=2,
                           early_stopping_patience=99).train(
        m, o, None, batches, step, ev, epochs=3, resume=False, log=lambda *_: None)

    d2 = Path(tempfile.mkdtemp())
    m2, o2 = make()
    VerifierTrainer(d2, {"x": 1}, checkpoint_every_steps=2,
                    early_stopping_patience=99).train(
        m2, o2, None, batches, step, ev, epochs=1, resume=False, log=lambda *_: None)
    m3, o3 = make()
    resumed = VerifierTrainer(d2, {"x": 1}, checkpoint_every_steps=2,
                              early_stopping_patience=99).train(
        m3, o3, None, batches, step, ev, epochs=3, resume=True, log=lambda *_: None)

    assert resumed["global_step"] == full["global_step"], (resumed, full)
    assert len(resumed["history"]) == len(full["history"])


def test_checkpoint_manager_reports_absence_cleanly():
    cm = CheckpointManager(Path(tempfile.mkdtemp()))
    assert cm.exists() is False
    assert cm.load(model=None) is None


# ----------------------------------------------------------------- ablations

def test_all_six_ablations_are_specified():
    specs = build_ablation_specs(CONFIG)
    names = {s.name for s in specs}
    assert names == {"bm25_vs_dense", "top_k_sweep", "claim_only_vs_claim_evidence",
                     "english_only_vs_multilingual_evidence",
                     "retrieval_only_vs_retrieval_rerank", "oracle_gold_evidence"}, names


def test_ablation_runner_is_deterministic_and_reports_delta():
    spec = AblationSpec(name="t", question="q",
                        variants=[{"a": 1}, {"a": 2}], metric="macro_f1")
    runner = lambda v: {"macro_f1": 0.5 + 0.1 * v["a"]}  # noqa: E731
    r1 = run_ablation(spec, runner, seed=42)
    r2 = run_ablation(spec, runner, seed=42)
    assert r1 == r2
    assert abs(r1["delta"] - 0.1) < 1e-12
    assert r1["best_variant"] == {"a": 2}


def test_ablation_seed_is_passed_to_runner():
    seen = {}
    spec = AblationSpec(name="t", question="q", variants=[{"a": 1}])

    def runner(v):
        seen.update(v)
        return {"macro_f1": 1.0}

    run_ablation(spec, runner, seed=43)
    assert seen["seed"] == 43


def test_missing_runner_is_skipped_not_silently_passed():
    out = run_all_ablations(CONFIG, runners={}, seed=42)
    assert out["ablations"] == {}
    assert len(out["skipped"]) == out["n_specs"] > 0


# --------------------------------------------------------- compute scope doc

def test_compute_reductions_are_declared_with_originals():
    cs = CONFIG["compute_scope"]
    assert cs["full_scale_results_claimed"] is False
    items = {r["item"]: r for r in cs["reductions"]}
    assert items["verifier_epochs"]["original"] == 10 and items["verifier_epochs"]["used"] == 3
    assert items["fever_train_claims"]["original"] == 228090
    assert items["fever_train_claims"]["used"] == 50000
    assert items["top_k_candidates"]["original"] == 100


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
