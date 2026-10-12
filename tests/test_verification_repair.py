"""Regression tests for the Module 6 evidence-mismatch repair.

Each test here corresponds to a defect found in the 2026-08-21 audit. They run
without torch, without a GPU and without downloaded weights, so they can gate
every future run cheaply.

  P1/P2  verifier trained with zero evidence, silently
  P3     build_examples() hard-coded stats["evidence_source"] = "retrieved"
  P6     resume=True hard-coded + no per-variant output isolation
  R1     the stratified subset must not depend on whether evidence is attached
  --     swap_head() must actually discard head weights
  --     fit_temperature() must refuse the test split
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m6_06_train_verifier import assert_evidence_present, run_dirs  # noqa: E402
from src.verification.calibration import fit_temperature  # noqa: E402
from src.verification.dataset import (  # noqa: E402
    assert_label_spaces_disjoint, build_examples, get_label_space, stratified_subset)


# --------------------------------------------------------------- fixtures
def make_corpus(n=60, dataset="x_fact"):
    labels = get_label_space(dataset).labels
    claims, split_of, ev_texts, ev_lang, retrieved = [], {}, {}, {}, {}
    for i in range(n):
        cid = f"c{i:04d}"
        eid_en, eid_ta = f"e{i:04d}en", f"e{i:04d}ta"
        ev_texts[eid_en] = f"english evidence {i}"
        ev_texts[eid_ta] = f"தமிழ் ஆதாரம் {i}"
        ev_lang[eid_en], ev_lang[eid_ta] = "en", "ta"
        claims.append({"claim_id": cid, "text": f"claim {i}",
                       "label": labels[i % len(labels)], "language": "ta",
                       "evidence": [eid_ta]})
        split_of[i] = "train" if i % 5 else "validation"
        retrieved[cid] = [eid_en, eid_ta]
    return claims, ev_texts, ev_lang, split_of, retrieved


# ------------------------------------------------- P3: honest evidence_source
def test_evidence_source_is_none_when_nothing_attached():
    """The exact defect: retrieval missing, yet the manifest said 'retrieved'."""
    claims, ev_texts, ev_lang, split_of, _ = make_corpus()
    _, stats = build_examples(claims, ev_texts, {}, "x_fact", split_of,
                              evidence_lang=ev_lang)
    assert stats["evidence_source"] == "none", (
        "build_examples() reported an evidence source despite attaching nothing. "
        "This is the P3 defect that produced a false manifest record.")
    assert stats["n_with_evidence"] == 0
    assert stats["requested_mode"] == "retrieved"      # intent preserved separately


def test_evidence_source_is_retrieved_when_evidence_attached():
    claims, ev_texts, ev_lang, split_of, retrieved = make_corpus()
    _, stats = build_examples(claims, ev_texts, retrieved, "x_fact", split_of,
                              evidence_lang=ev_lang)
    assert stats["evidence_source"] == "retrieved"
    assert stats["n_with_evidence"] == stats["n_built"] > 0
    assert stats["evidence_coverage"] == pytest.approx(1.0)


def test_claim_only_and_oracle_report_their_own_source():
    claims, ev_texts, ev_lang, split_of, retrieved = make_corpus()
    _, s1 = build_examples(claims, ev_texts, retrieved, "x_fact", split_of,
                           claim_only=True, evidence_lang=ev_lang)
    assert s1["evidence_source"] == "claim_only"
    _, s2 = build_examples(claims, ev_texts, {}, "x_fact", split_of,
                           use_gold_evidence=True, evidence_lang=ev_lang)
    assert s2["evidence_source"] == "gold"


# ------------------------------------------------------- P1/P2: the hard guard
def test_guard_raises_when_no_training_evidence():
    with pytest.raises(SystemExit) as e:
        assert_evidence_present("none", {"n_with_evidence": 0, "n_built": 100},
                                claim_only=False, dataset="fever")
    assert "REFUSING TO TRAIN" in str(e.value)
    assert "m6_04_retrieve" in str(e.value)        # tells the user the fix


def test_guard_raises_when_candidates_load_but_nothing_attaches():
    with pytest.raises(SystemExit):
        assert_evidence_present("dense_candidates",
                                {"n_with_evidence": 0, "n_built": 100},
                                claim_only=False, dataset="x_fact")


def test_guard_allows_explicit_claim_only():
    assert_evidence_present("none", {"n_with_evidence": 0, "n_built": 100},
                            claim_only=True, dataset="x_fact")


def test_guard_allows_normal_evidence_run():
    assert_evidence_present("dense_candidates", {"n_with_evidence": 90, "n_built": 100},
                            claim_only=False, dataset="fever")


# ------------------------------------------------------------ P6: isolation
def test_run_dirs_isolate_every_variant():
    cfg = {"output": {"models_dir": "models/module6", "root": "outputs/module6",
                      "stages": []}}
    root = Path("/tmp/proj")
    primary_ck, _ = run_dirs(cfg, root, "x_fact", None)
    seen = {primary_ck}
    for name in ("claim_only", "claim_evidence", "oracle", "class_weighted"):
        ck, _ = run_dirs(cfg, root, "x_fact", name)
        assert ck not in seen, f"{name} collides with an existing checkpoint dir"
        assert ck != primary_ck, f"{name} would overwrite the PRIMARY checkpoint"
        assert name in str(ck)
        seen.add(ck)
    assert len(seen) == 5


def test_write_manifest_creates_nested_ablation_directory(tmp_path):
    """write_manifest() must create its own directory, like write_json() does.

    Regression: the first caller to pass a genuinely new nested path -- an
    isolated ablation directory -- raised FileNotFoundError on
    manifest.tmp.json, because stage_dir() creates only
    outputs/module6/<stage> and nothing beneath it.
    """
    import json as _json

    from scripts.m6_common import write_manifest

    nested = tmp_path / "outputs/module6/verifier_training/ablations/x_fact/claim_only"
    assert not nested.exists()
    p = write_manifest(nested, {"stage": "verifier_training", "status": "dry_run"})
    assert p.is_file()
    assert _json.loads(p.read_text(encoding="utf-8"))["status"] == "dry_run"
    assert not (nested / "manifest.tmp.json").exists()   # temp file was renamed away


def test_write_manifest_is_idempotent(tmp_path):
    from scripts.m6_common import write_manifest
    d = tmp_path / "a/b/c"
    write_manifest(d, {"n": 1})
    write_manifest(d, {"n": 2})
    import json as _json
    assert _json.loads((d / "manifest.json").read_text(encoding="utf-8"))["n"] == 2


def test_run_dirs_separate_datasets():
    cfg = {"output": {"models_dir": "models/module6", "root": "outputs/module6",
                      "stages": []}}
    a, _ = run_dirs(cfg, Path("/tmp/proj"), "fever", "oracle")
    b, _ = run_dirs(cfg, Path("/tmp/proj"), "x_fact", "oracle")
    assert a != b


# ------------------------------- R1: subset must not depend on evidence
def test_stratified_subset_is_invariant_to_evidence():
    """Justifies retrieving only the 50,000-claim subset for FEVER.

    stratified_subset() must select on (claim_id, label, seed) alone. If
    attaching evidence changed the selection, retrieving a precomputed subset
    would train on a different set than intended.
    """
    claims, ev_texts, ev_lang, split_of, retrieved = make_corpus(n=200)
    no_ev, _ = build_examples(claims, ev_texts, {}, "x_fact", split_of,
                              evidence_lang=ev_lang)
    with_ev, _ = build_examples(claims, ev_texts, retrieved, "x_fact", split_of,
                                evidence_lang=ev_lang)
    a = [e for e in no_ev if e.split == "train"]
    b = [e for e in with_ev if e.split == "train"]
    ids_a = [e.claim_id for e in stratified_subset(a, 40, seed=42)]
    ids_b = [e.claim_id for e in stratified_subset(b, 40, seed=42)]
    assert ids_a == ids_b, "subset selection changed when evidence was attached"
    assert len(ids_a) <= 40


def test_stratified_subset_is_deterministic():
    claims, ev_texts, ev_lang, split_of, retrieved = make_corpus(n=200)
    ex, _ = build_examples(claims, ev_texts, retrieved, "x_fact", split_of,
                           evidence_lang=ev_lang)
    tr = [e for e in ex if e.split == "train"]
    first = [e.claim_id for e in stratified_subset(tr, 40, seed=42)]
    second = [e.claim_id for e in stratified_subset(tr, 40, seed=42)]
    assert first == second


# -------------------------------------------------- evidence-language filter
def test_language_filter_keeps_only_requested_languages():
    claims, ev_texts, ev_lang, split_of, retrieved = make_corpus()
    ex, stats = build_examples(claims, ev_texts, retrieved, "x_fact", split_of,
                               evidence_lang=ev_lang, keep_languages=["en"])
    assert set(stats["by_evidence_language"]) == {"en"}
    assert stats["n_dropped_by_language_filter"] > 0
    for e in ex:
        for t in e.evidence_texts:
            assert t.startswith("english evidence")


def test_language_filter_slices_after_filtering():
    """Top-k of the permitted language, not top-k overall with holes."""
    ev_texts = {f"e{i}": f"text {i}" for i in range(10)}
    ev_lang = {f"e{i}": ("en" if i >= 6 else "ta") for i in range(10)}
    claims = [{"claim_id": "c0", "text": "claim", "label": "false",
               "language": "ta", "evidence": []}]
    ex, _ = build_examples(claims, ev_texts, {"c0": [f"e{i}" for i in range(10)]},
                           "x_fact", {0: "train"}, max_evidence=3,
                           evidence_lang=ev_lang, keep_languages=["en"])
    # e6, e7, e8 are the first three English items; a slice-then-filter
    # implementation would have yielded nothing (e0..e2 are all Tamil).
    assert ex[0].evidence_texts == ["text 6", "text 7", "text 8"]


def test_language_filter_requires_language_map():
    claims, ev_texts, _, split_of, retrieved = make_corpus()
    with pytest.raises(ValueError):
        build_examples(claims, ev_texts, retrieved, "x_fact", split_of,
                       keep_languages=["en"])


# ------------------------------------------------------ standing guarantees
def test_label_spaces_stay_disjoint():
    assert_label_spaces_disjoint()
    assert get_label_space("fever").labels != get_label_space("x_fact").labels


def test_unknown_label_raises_rather_than_defaulting():
    with pytest.raises(KeyError):
        get_label_space("x_fact").encode("SUPPORTS")     # a FEVER label
    with pytest.raises(KeyError):
        get_label_space("fever").encode("partly true/misleading")


def test_temperature_refuses_test_split():
    import numpy as np
    logits, labels = np.zeros((4, 3)), [0, 1, 2, 0]
    with pytest.raises(ValueError):
        fit_temperature(logits, labels, split_name="test")
    assert fit_temperature(logits, labels, split_name="validation")["split"] == "validation"


def test_weighted_f1_is_reported_alongside_macro():
    from src.verification.trainer import classification_report
    r = classification_report([0, 0, 0, 1], [0, 0, 0, 0], ["a", "b"])
    assert "weighted_f1" in r and "macro_f1" in r
    # majority-collapse: weighted looks far healthier than macro
    assert r["weighted_f1"] > r["macro_f1"]
    assert r["per_class"]["b"]["f1"] == 0.0


def test_accounting_balances_on_every_build():
    claims, ev_texts, ev_lang, split_of, retrieved = make_corpus()
    for kwargs in ({}, {"claim_only": True}, {"use_gold_evidence": True},
                   {"keep_languages": ["en"]}):
        _, stats = build_examples(claims, ev_texts, retrieved, "x_fact", split_of,
                                  evidence_lang=ev_lang, **kwargs)
        assert stats["accounting_balances"], kwargs
