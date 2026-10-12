"""Module 5 foundation tests — claim selection, dedup/grouping, leakage gate.

Same standard as Modules 3/4: values checked against hand-computed
results where possible, and every guard tested with a case that MUST
fail. A leakage gate that cannot fail is worse than no gate, because it
manufactures false confidence.

Runs without torch, sklearn, or any external corpus.

    python tests/test_fact_verification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fact_verification.claim_selection import (  # noqa: E402
    Claim,
    hash_key,
    normalize_claim,
    select_claim_from_record,
    select_claims,
    tamil_ratio,
)
from src.fact_verification.leakage_gate import (  # noqa: E402
    build_index_scope,
    candidate_pairs,
    char_ngrams,
    enforce_group_safe_splits,
    group_aware_split,
    group_claims,
    jaccard,
    verify_no_leakage,
)
from src.utils.config_loader import load_yaml  # noqa: E402

CONFIG = load_yaml(PROJECT_ROOT / "configs" / "module5_fact_verification.yaml")

TAMIL_CLAIM = "பஹல்காம் தாக்குதல்: கடற்படை அதிகாரியின் இறுதிக் காட்சி என்ற தவறான கூற்று"
ENGLISH_CLAIM = "Senator claims the new bill will eliminate all federal income tax by 2027"


# ---------------------------------------------------------- normalisation

def test_normalisation_preserves_tamil_script():
    """Tamil codepoints must survive normalisation exactly.

    Module 2 established that Tamil combining vowel signs are easy to
    destroy with the wrong normalisation form. NFC is required; NFKC
    would collapse linguistically distinct characters.
    """
    out = normalize_claim(TAMIL_CLAIM, CONFIG["normalization"])
    tamil_in = [c for c in TAMIL_CLAIM if "஀" <= c <= "௿"]
    tamil_out = [c for c in out if "஀" <= c <= "௿"]
    assert tamil_in == tamil_out, "Tamil codepoints altered by normalisation"
    assert "�" not in out


def test_normalisation_lowercases_latin_only():
    s = normalize_claim("BREAKING: Senator SAYS " + TAMIL_CLAIM, CONFIG["normalization"])
    assert "BREAKING" not in s and "breaking" in s
    assert all(c in s for c in TAMIL_CLAIM if "஀" <= c <= "௿")


def test_normalisation_strips_urls_and_collapses_whitespace():
    s = normalize_claim("Claim   text https://example.com/x?y=1   here", CONFIG["normalization"])
    assert "http" not in s
    assert "  " not in s
    assert s == "claim text here"


def test_tamil_ratio_bounded_and_correct():
    assert tamil_ratio(TAMIL_CLAIM) > 0.7
    assert tamil_ratio(ENGLISH_CLAIM) == 0.0
    assert tamil_ratio("") == 0.0
    for s in (TAMIL_CLAIM, ENGLISH_CLAIM, "mixed தமிழ் text"):
        assert 0.0 <= tamil_ratio(s) <= 1.0


def test_hash_key_collapses_punctuation_variants():
    assert hash_key("The claim is false!") == hash_key("the claim is false")
    assert hash_key('"Quoted claim."') == hash_key("Quoted claim")
    assert hash_key("A claim") != hash_key("A different claim")


# ------------------------------------------------------- claim selection

def _project_record(**kw):
    base = {"record_id": "welfake_1", "title": ENGLISH_CLAIM, "text": ENGLISH_CLAIM + " Body...",
            "body_text": "Body...", "language": "en", "label": 1, "source": "welfake", "url": None}
    base.update(kw)
    return base


def test_tamil_headline_is_used_directly_as_claim():
    """Tamil rows are headline-only; selection must be a no-op."""
    c = select_claim_from_record(
        _project_record(record_id="aanirudh_1", title=TAMIL_CLAIM, text=TAMIL_CLAIM,
                        body_text=None, language="ta", label=0), CONFIG)
    assert c is not None
    assert c.selection_rule.startswith("title_as_claim")
    assert c.language == "ta"
    assert [x for x in c.text if "஀" <= x <= "௿"] == [x for x in TAMIL_CLAIM if "஀" <= x <= "௿"]


def test_english_title_selected_not_body():
    c = select_claim_from_record(_project_record(), CONFIG)
    assert c is not None and c.selection_rule.startswith("title_as_claim")
    assert "Body" not in c.text and c.text.startswith("senator claims")


def test_fallback_when_title_missing():
    c = select_claim_from_record(
        _project_record(title=None, text="A sufficiently long fallback claim sentence."), CONFIG)
    assert c is not None and c.selection_rule.startswith("fallback_text_prefix")


def test_claims_below_minimum_are_dropped_not_silently_kept():
    assert select_claim_from_record(_project_record(title="Hi", text="Hi"), CONFIG) is None


def test_long_titles_truncated_to_max():
    long_title = "word " * 300
    c = select_claim_from_record(_project_record(title=long_title), CONFIG)
    assert c is not None
    assert len(c.text) <= CONFIG["claim_selection"]["max_claim_chars"]
    assert "truncated" in c.selection_rule


def test_selection_summary_reports_drops_and_languages():
    recs = [_project_record(record_id=f"e{i}") for i in range(5)]
    recs += [_project_record(record_id=f"t{i}", title=TAMIL_CLAIM, language="ta") for i in range(3)]
    recs += [_project_record(record_id="short", title="Hi", text="Hi")]
    claims, summary = select_claims(recs, CONFIG)
    assert summary["n_input_records"] == 9
    assert summary["n_claims_selected"] == 8
    assert summary["n_dropped_below_min_chars"] == 1
    assert summary["by_language"] == {"en": 5, "ta": 3}
    assert len(claims) == 8


def test_dataset_identity_is_preserved():
    """Corpora must stay distinguishable end-to-end; merging would hide
    domain mismatch, which the design requires reporting."""
    c = select_claim_from_record(_project_record(), CONFIG, dataset="x_fact")
    assert c.dataset == "x_fact" and c.claim_id.startswith("x_fact:")


# ------------------------------------------------------------- similarity

def test_jaccard_and_ngrams_hand_computed():
    a, b = {"ab", "bc", "cd"}, {"bc", "cd", "de"}
    assert abs(jaccard(a, b) - 2 / 4) < 1e-9      # |∩|=2, |∪|=4
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"x"}, set()) == 0.0
    assert char_ngrams("abcdef", 3) == {"abc", "bcd", "cde", "def"}
    assert char_ngrams("ab", 5) == {"ab"}          # shorter than n


def test_near_duplicate_similarity_is_high_for_rewordings():
    a = "the prime minister announced a new tax policy yesterday"
    b = "the prime minister announced a new tax policy yesterday!"
    assert jaccard(char_ngrams(a), char_ngrams(b)) > 0.9
    c = "a completely unrelated statement about football results"
    assert jaccard(char_ngrams(a), char_ngrams(c)) < 0.3


# -------------------------------------------------------------- grouping

def _mk(cid, text, lang="en", dataset="project", label=0):
    return Claim(claim_id=cid, text=normalize_claim(text, CONFIG["normalization"]),
                 raw_text=text, language=lang, dataset=dataset, label=label)


def test_grouping_collapses_exact_duplicates():
    claims = [_mk("a", "The claim is false"), _mk("b", "the claim is false!"),
              _mk("c", "A totally different statement about economics")]
    g = group_claims(claims, near_duplicate=False)
    assert g.group_ids[0] == g.group_ids[1]
    assert g.group_ids[2] != g.group_ids[0]
    assert g.n_groups == 2 and g.n_exact_duplicate_pairs == 1


def test_grouping_collapses_near_duplicates():
    claims = [_mk("a", "The prime minister announced a new tax policy yesterday in parliament"),
              _mk("b", "The prime minister announced a new tax policy yesterday in parliament."),
              _mk("c", "Scientists discovered a new species of deep sea fish this week")]
    g = group_claims(claims, near_duplicate=True, threshold=0.90)
    assert g.group_ids[0] == g.group_ids[1], "near-duplicate not grouped"
    assert g.group_ids[2] != g.group_ids[0]


def test_grouping_is_transitive():
    """A~B and B~C must place A, B, C together even if A and C are not
    directly similar enough — otherwise a chain of rewordings leaks."""
    base = "the government announced a major infrastructure investment programme today"
    claims = [_mk("a", base), _mk("b", base + " ."), _mk("c", base + " . .")]
    g = group_claims(claims, near_duplicate=True, threshold=0.85)
    assert len(set(g.group_ids)) == 1


def test_grouping_handles_tamil():
    claims = [_mk("a", TAMIL_CLAIM, lang="ta"), _mk("b", TAMIL_CLAIM + "!", lang="ta"),
              _mk("c", "முற்றிலும் வேறுபட்ட ஒரு தமிழ் செய்தி தலைப்பு", lang="ta")]
    g = group_claims(claims, near_duplicate=True, threshold=0.90)
    assert g.group_ids[0] == g.group_ids[1]
    assert g.group_ids[2] != g.group_ids[0]


# --------------------------------------------------------------- splitting

def _corpus(n_groups=60, per_group=2):
    claims, expected = [], []
    for gi in range(n_groups):
        lang = "ta" if gi % 5 == 0 else "en"
        base = f"claim number {gi} about topic {gi} with enough characters to pass the minimum"
        for k in range(per_group):
            claims.append(_mk(f"c{gi}_{k}", base + ("" if k == 0 else " " * k + "."), lang=lang))
            expected.append(gi)
    return claims, expected


def test_group_aware_split_never_splits_a_group():
    claims, _ = _corpus()
    g = group_claims(claims, near_duplicate=True, threshold=0.85)
    splits = group_aware_split(claims, g.group_ids, CONFIG["leakage"]["split_ratios"], seed=42)
    where: dict[int, set] = {}
    for name, idxs in splits.items():
        for i in idxs:
            where.setdefault(g.group_ids[i], set()).add(name)
    assert all(len(v) == 1 for v in where.values()), "a group spans multiple splits"


def test_group_aware_split_covers_every_claim_once():
    claims, _ = _corpus()
    g = group_claims(claims, near_duplicate=True, threshold=0.85)
    splits = group_aware_split(claims, g.group_ids, CONFIG["leakage"]["split_ratios"], seed=42)
    allidx = [i for v in splits.values() for i in v]
    assert sorted(allidx) == list(range(len(claims)))
    assert len(allidx) == len(set(allidx))


def test_group_aware_split_is_deterministic():
    claims, _ = _corpus()
    g = group_claims(claims, near_duplicate=True, threshold=0.85)
    r = CONFIG["leakage"]["split_ratios"]
    a = group_aware_split(claims, g.group_ids, r, seed=42)
    b = group_aware_split(claims, g.group_ids, r, seed=42)
    c = group_aware_split(claims, g.group_ids, r, seed=43)
    assert a == b
    assert a != c


# ------------------------------------------------------------- the gate

def test_gate_passes_on_a_correct_split():
    claims, _ = _corpus()
    g = group_claims(claims, near_duplicate=True, threshold=0.85)
    splits = group_aware_split(claims, g.group_ids, CONFIG["leakage"]["split_ratios"], seed=42)
    r = verify_no_leakage(claims, splits, g.group_ids, near_duplicate_threshold=0.85)
    assert r["passed"] is True, [c for c in r["checks"] if not c["passed"]]


def test_gate_catches_a_group_split_across_train_and_test():
    """The exact failure the gate exists to prevent."""
    claims, _ = _corpus()
    g = group_claims(claims, near_duplicate=True, threshold=0.85)
    splits = group_aware_split(claims, g.group_ids, CONFIG["leakage"]["split_ratios"], seed=42)
    victim = splits["train"].pop()
    splits["test"].append(victim)
    r = verify_no_leakage(claims, splits, g.group_ids, near_duplicate_threshold=0.85)
    assert r["passed"] is False
    assert any(c["name"] == "no_group_spans_multiple_splits" and not c["passed"] for c in r["checks"])


def test_gate_catches_exact_duplicate_across_splits():
    dup = "an identical claim text appearing in two different splits entirely"
    claims = [_mk(f"x{i}", f"unique claim number {i} with plenty of characters here") for i in range(20)]
    claims += [_mk("dup_a", dup), _mk("dup_b", dup)]
    # Force them apart, pretending they are separate groups.
    gids = list(range(len(claims)))
    splits = {"train": list(range(0, 10)) + [20], "validation": list(range(10, 15)),
              "test": list(range(15, 20)) + [21]}
    r = verify_no_leakage(claims, splits, gids, near_duplicate_threshold=0.90)
    assert r["passed"] is False
    assert any(c["name"] == "no_exact_claim_overlap_across_splits" and not c["passed"] for c in r["checks"])


def test_gate_catches_near_duplicate_across_splits():
    base = "the health ministry confirmed a new outbreak in the northern districts today"
    claims = [_mk(f"u{i}", f"distinct claim {i} with sufficient length for the minimum") for i in range(20)]
    claims += [_mk("near_a", base), _mk("near_b", base + ".")]
    gids = list(range(len(claims)))
    splits = {"train": list(range(0, 10)) + [20], "validation": list(range(10, 15)),
              "test": list(range(15, 20)) + [21]}
    r = verify_no_leakage(claims, splits, gids, near_duplicate_threshold=0.90)
    assert r["passed"] is False
    assert any(c["name"] == "no_near_duplicate_claims_across_splits" and not c["passed"] for c in r["checks"])


def test_gate_catches_missing_and_double_assignment():
    claims, _ = _corpus(n_groups=10)
    g = group_claims(claims, near_duplicate=False)
    splits = group_aware_split(claims, g.group_ids, CONFIG["leakage"]["split_ratios"], seed=42)
    splits["train"] = splits["train"][:-1]                       # drop one
    r = verify_no_leakage(claims, splits, g.group_ids, check_cross_split_similarity=False)
    assert r["passed"] is False

    splits2 = group_aware_split(claims, g.group_ids, CONFIG["leakage"]["split_ratios"], seed=42)
    splits2["test"].append(splits2["train"][0])                  # assign twice
    r2 = verify_no_leakage(claims, splits2, g.group_ids, check_cross_split_similarity=False)
    assert r2["passed"] is False


# -------------------------------------------------------- index scoping

def test_index_scope_prevents_training_seeing_test_evidence():
    splits = {"train": [0, 1, 2], "validation": [3, 4], "test": [5, 6]}
    scope = CONFIG["leakage"]["index_scope"]
    train_idx = build_index_scope(splits, "train", scope)
    val_idx = build_index_scope(splits, "validation", scope)
    test_idx = build_index_scope(splits, "test", scope)

    assert set(train_idx) == {0, 1, 2}
    assert not (set(train_idx) & set(splits["test"])), "training index exposes test evidence"
    assert not (set(val_idx) & set(splits["test"])), "validation index exposes test evidence"
    assert set(test_idx) == {0, 1, 2, 3, 4, 5, 6}


def test_index_scope_rejects_unknown_phase_and_split():
    splits = {"train": [0], "validation": [1], "test": [2]}
    for bad in ({"phase": "nonsense"},):
        try:
            build_index_scope(splits, bad["phase"], CONFIG["leakage"]["index_scope"])
            raise AssertionError("should reject unknown phase")
        except ValueError:
            pass
    try:
        build_index_scope({"train": [0]}, "validation", CONFIG["leakage"]["index_scope"])
        raise AssertionError("should reject scope referencing a missing split")
    except ValueError:
        pass


# ----------------------------------------------------------- config gate

def test_config_locks_download_behind_approval():
    """Neither corpus may be fetched until explicitly approved."""
    for name in ("x_fact", "fever"):
        assert CONFIG["corpora"][name]["download_approved"] is False, f"{name} download not gated"
        assert CONFIG["corpora"][name]["licence_verified"] is True


def test_config_preserves_dataset_separation_and_excludes_settled_ablation():
    assert CONFIG["corpora"]["merge_datasets"] is False
    assert CONFIG["corpora"]["preserve_source_identity"] is True
    assert CONFIG["corpora"]["project_corpus"]["read_only"] is True
    # CLS-vs-mean was settled in Module 4; re-running it answers nothing.
    assert "cls_vs_mean" not in CONFIG["ablations"]
    assert CONFIG["leakage"]["gold_evidence_in_primary_path"] is False
    assert CONFIG["ablations"]["oracle_gold_evidence"] is True


# ------------------------------------------------ Stage 3: corpus builder

from src.fact_verification.corpus_builder import (  # noqa: E402
    build_corpus_from_files,
    canonical_split,
    detect_language,
    evidence_hash,
    extract_evidence as cb_extract_evidence,
    extract_evidence_ids as cb_extract_evidence_ids,
    first_present as cb_first_present,
    splits_from_claims,
    write_corpus,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    import json as _json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(_json.dumps(r, ensure_ascii=False) + "\n")


def test_split_alias_mapping_unifies_dev_and_validation():
    """X-FACT ships 'dev', FEVER ships 'validation'. Downstream code
    needs one vocabulary, but the corpus's own name must survive."""
    assert canonical_split("dev") == "validation"
    assert canonical_split("validation") == "validation"
    assert canonical_split("Dev") == "validation"
    assert canonical_split("train") == "train"
    assert canonical_split("test") == "test"


def test_detect_language_prefers_declared_over_script():
    """A declared language must never be silently overridden."""
    assert detect_language("Paris is in France", "ta") == "ta"
    assert detect_language(TAMIL_CLAIM, None) == "ta"
    assert detect_language("Paris is in France", None) == "en"
    assert detect_language("", None) == "unknown"


def test_corpus_builder_helpers_agree_with_audit_script():
    """Guard against drift between the two copies of the field-discovery
    helpers. `scripts/audit_module5_corpora.py` is a verified, passing
    artefact and was deliberately not edited, so the builder carries its
    own copy — this test makes that duplication safe by proving both
    implementations agree on every observed record shape.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit_m5_drift", PROJECT_ROOT / "scripts" / "audit_module5_corpora.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    shapes = [
        {"language": "ta", "claim": TAMIL_CLAIM, "label": "false",
         "evidence_1": "Tamil evidence", "evidence_2": "English evidence", "evidence_3": ""},
        {"id": 1, "claim": "Paris is in France.", "label": "SUPPORTS",
         "evidence": [["Paris", 0, "Paris is the capital."], ["France", 3, "France is in Europe."]]},
        {"id": 2, "claim": "X", "label": "REFUTES",
         "evidence": [{"page": "Foo", "sentence_id": 1, "text": "Foo is a bar."}]},
        {"claim": "no evidence", "label": "NOT ENOUGH INFO"},
    ]
    for row in shapes:
        assert cb_extract_evidence(row) == audit.extract_evidence(row)
        assert cb_extract_evidence_ids(row) == audit.extract_evidence_ids(row)
        for fields in (audit.CLAIM_FIELDS, audit.LABEL_FIELDS, audit.LANG_FIELDS, audit.ID_FIELDS):
            assert cb_first_present(row, fields) == audit.first_present(row, fields)


def test_builder_accounts_for_every_input_row(tmp_dir: Path | None = None):
    """No silent row loss: built + dropped must equal input."""
    import tempfile

    d = Path(tmp_dir or tempfile.mkdtemp())
    rows = [
        {"claim": "A sufficiently long factual claim about economics here", "label": "false", "language": "ta"},
        {"claim": TAMIL_CLAIM, "label": "true", "language": "ta",
         "evidence_1": "ஆதாரம் ஒன்று", "evidence_2": "English supporting evidence"},
        {"claim": "Hi", "label": "false", "language": "ta"},        # below min -> dropped
        {"claim": "   ", "label": "true", "language": "ta"},         # no text -> dropped
    ]
    _write_jsonl(d / "x_fact_train.jsonl", rows)
    claims, pool, stats = build_corpus_from_files([(d / "x_fact_train.jsonl", "train")], "x_fact", CONFIG)

    assert stats["n_input_rows"] == 4
    assert stats["n_claims_built"] == 2
    assert stats["n_dropped_below_min_chars"] == 1
    assert stats["n_dropped_no_claim_text"] == 1
    assert stats["row_accounting_balances"] is True
    assert len(claims) == 2


def test_builder_preserves_dataset_identity_and_official_split():
    import tempfile

    d = Path(tempfile.mkdtemp())
    _write_jsonl(d / "x_fact_dev.jsonl",
                 [{"claim": "A claim long enough to survive the minimum length filter", "label": "true"}])
    claims, _, _ = build_corpus_from_files([(d / "x_fact_dev.jsonl", "dev")], "x_fact", CONFIG)
    c = claims[0]
    assert c.dataset == "x_fact"
    assert c.claim_id.startswith("x_fact:dev:")
    assert c.metadata["official_split"] == "dev"          # corpus's own name kept
    assert c.metadata["canonical_split"] == "validation"  # unified for downstream


def test_builder_deduplicates_evidence_pool():
    """Identical evidence text must collapse to one pool entry, or the
    retrieval index would contain the same sentence many times."""
    import tempfile

    d = Path(tempfile.mkdtemp())
    shared = "The ministry published the official figures last week."
    _write_jsonl(d / "fever_gold_train.jsonl", [
        {"id": 1, "claim": "First distinct claim with sufficient length", "label": "SUPPORTS",
         "evidence": [["Doc", 0, shared]]},
        {"id": 2, "claim": "Second distinct claim with sufficient length", "label": "REFUTES",
         "evidence": [["Doc", 0, shared]]},
    ])
    claims, pool, stats = build_corpus_from_files(
        [(d / "fever_gold_train.jsonl", "train")], "fever", CONFIG)

    assert len(claims) == 2
    assert stats["n_evidence_mentions"] == 2
    assert stats["n_unique_evidence_items"] == 1, "identical evidence not de-duplicated"
    only = next(iter(pool.values()))
    assert only.n_claims_referencing == 2
    assert claims[0].evidence == claims[1].evidence


def test_evidence_ids_differ_for_shared_64_char_prefix():
    """REGRESSION. `evidence_id` was once `hash_key(text)[:64]`, and
    `hash_key()` returns the normalized string — so the id was literally
    the sentence's first 64 characters. Two distinct evidence sentences
    sharing a long boilerplate lead collided and were merged into one
    pool entry, deleting real evidence and inflating the reference count.
    """
    import tempfile

    lead = "According to a statement released by the finance ministry on Monday morning, "
    assert len(lead) >= 64, "shared prefix must exceed the 64-char truncation point"
    a = lead + "exports rose by four percent last quarter."
    b = lead + "imports fell by nine percent last quarter."
    assert a[:64] == b[:64], "test premise broken: prefixes must collide"
    assert a != b

    ha, hb = evidence_hash(a), evidence_hash(b)
    assert ha != hb, "SHA-256 ids collided on a shared 64-char prefix"
    assert len(ha) == 64 and all(c in "0123456789abcdef" for c in ha), "not a hex digest"
    assert ha != a[:64], "id is a truncated copy of the text, not a digest"
    assert evidence_hash(a) == ha, "digest is not deterministic"
    assert evidence_hash(TAMIL_CLAIM) == evidence_hash(TAMIL_CLAIM)
    assert len(evidence_hash(TAMIL_CLAIM)) == 64, "Tamil text must hash cleanly under UTF-8"

    # End-to-end: both sentences must survive as separate pool entries.
    d = Path(tempfile.mkdtemp())
    _write_jsonl(d / "x_fact_train.jsonl", [
        {"claim": "A claim long enough to survive the minimum filter one", "label": "true",
         "evidence_1": a},
        {"claim": "A claim long enough to survive the minimum filter two", "label": "false",
         "evidence_1": b},
    ])
    claims, pool, stats = build_corpus_from_files(
        [(d / "x_fact_train.jsonl", "train")], "x_fact", CONFIG)
    assert stats["n_unique_evidence_items"] == 2, "prefix-colliding evidence was merged"
    assert claims[0].evidence != claims[1].evidence
    assert all(it.n_claims_referencing == 1 for it in pool.values())


def test_builder_records_claims_without_evidence():
    """FEVER's NOT ENOUGH INFO class has no gold evidence by design —
    that must be counted, not treated as an extraction failure."""
    import tempfile

    d = Path(tempfile.mkdtemp())
    _write_jsonl(d / "fever_gold_train.jsonl", [
        {"id": 1, "claim": "A claim with no evidence attached at all here", "label": "NOT ENOUGH INFO"},
    ])
    _, _, stats = build_corpus_from_files([(d / "fever_gold_train.jsonl", "train")], "fever", CONFIG)
    assert stats["n_claims_with_no_evidence"] == 1
    assert "none_found" in stats["evidence_extraction_methods"]


def test_splits_from_claims_uses_official_splits_not_reshuffle():
    import tempfile

    d = Path(tempfile.mkdtemp())
    for suffix, official, n in (("train", "train", 5), ("dev", "dev", 2), ("test", "test", 3)):
        _write_jsonl(d / f"x_fact_{suffix}.jsonl",
                     [{"claim": f"Official split claim number {i} with enough characters", "label": "true"}
                      for i in range(n)])
    files = [(d / "x_fact_train.jsonl", "train"), (d / "x_fact_dev.jsonl", "dev"),
             (d / "x_fact_test.jsonl", "test")]
    claims, _, _ = build_corpus_from_files(files, "x_fact", CONFIG)
    splits = splits_from_claims(claims)
    assert {k: len(v) for k, v in splits.items()} == {"train": 5, "validation": 2, "test": 3}
    allidx = sorted(i for v in splits.values() for i in v)
    assert allidx == list(range(10)), "claims lost or duplicated across splits"


def test_written_corpus_roundtrips_and_carries_group_ids():
    import json as _json
    import tempfile

    d = Path(tempfile.mkdtemp())
    _write_jsonl(d / "x_fact_train.jsonl", [
        {"claim": f"Distinct claim number {i} with plenty of characters here", "label": "false",
         "language": "ta" if i % 2 else "en", "evidence_1": f"Evidence text {i}"} for i in range(6)])
    claims, pool, stats = build_corpus_from_files([(d / "x_fact_train.jsonl", "train")], "x_fact", CONFIG)
    splits = splits_from_claims(claims)
    g = group_claims(claims, near_duplicate=True,
                     ngram=CONFIG["leakage"]["near_duplicate_ngram"],
                     threshold=CONFIG["leakage"]["near_duplicate_threshold"])

    out = d / "built"
    paths = write_corpus(claims, pool, splits, g.group_ids, out, "claims.jsonl")
    lines = [_json.loads(x) for x in Path(paths["claims"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == len(claims)
    assert all("claim_group_id" in r for r in lines)
    assert all(r["dataset"] == "x_fact" for r in lines)
    ev_lines = [_json.loads(x) for x in Path(paths["evidence"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(ev_lines) == len(pool)
    sp = _json.loads(Path(paths["splits"]).read_text(encoding="utf-8"))
    assert sp["counts"]["train"] == len(claims)
    assert "not reshuffled" in sp["note"]


def test_builder_output_feeds_leakage_gate_cleanly():
    """End-to-end: built corpus -> grouping -> gate, with disjoint splits."""
    import tempfile

    d = Path(tempfile.mkdtemp())
    for suffix, n, off in (("train", 12, 0), ("dev", 4, 100), ("test", 4, 200)):
        _write_jsonl(d / f"x_fact_{suffix}.jsonl",
                     [{"claim": f"Unique claim {off+i} concerning distinct topic {off+i} entirely",
                       "label": "true"} for i in range(n)])
    files = [(d / "x_fact_train.jsonl", "train"), (d / "x_fact_dev.jsonl", "dev"),
             (d / "x_fact_test.jsonl", "test")]
    claims, _, _ = build_corpus_from_files(files, "x_fact", CONFIG)
    splits = splits_from_claims(claims)
    g = group_claims(claims, near_duplicate=True, threshold=0.90)
    r = verify_no_leakage(claims, splits, g.group_ids, near_duplicate_threshold=0.90)
    assert r["passed"] is True, [c for c in r["checks"] if not c["passed"]]


def test_candidate_filter_is_exact_vs_brute_force():
    """The prefix/length filter must lose NO pair above the threshold.

    A filter that drops a genuine near-duplicate would silently let a
    leaked claim through the gate, which is the exact failure the gate
    exists to prevent. Brute force is the ground truth here.
    """
    import random

    rng = random.Random(7)
    words = ["ministry", "reported", "exports", "rose", "percent", "quarter", "official",
             "statement", "released", "monday", "figures", "revised", "budget", "growth"]
    texts = []
    for i in range(120):
        texts.append(" ".join(rng.choice(words) for _ in range(rng.randint(6, 18))) + f" {i}")
    # Seed guaranteed near-duplicates: tiny edits to existing texts.
    for i in range(0, 40, 4):
        texts.append(texts[i] + ".")
        texts.append(texts[i].replace("percent", "per cent"))
    texts.append(TAMIL_CLAIM)
    texts.append(TAMIL_CLAIM + " ")

    grams = [char_ngrams(t, 5) for t in texts]
    langs = ["ta" if any("஀" <= c <= "௿" for c in t) else "en" for t in texts]
    t = 0.90

    brute = {(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))
             if langs[i] == langs[j] and jaccard(grams[i], grams[j]) >= t}
    cand = set(candidate_pairs(grams, langs, t))
    fast = {(i, j) for i, j in cand if jaccard(grams[i], grams[j]) >= t}

    assert brute, "test data produced no near-duplicates; the test would be vacuous"
    missed = brute - fast
    assert not missed, f"candidate filter LOST {len(missed)} true pair(s): {sorted(missed)[:5]}"
    assert fast == brute
    # It must also actually filter, or it is just brute force with extra steps.
    total = len(texts) * (len(texts) - 1) // 2
    assert len(cand) < total * 0.75, f"filter examined {len(cand)}/{total} pairs — no pruning"


def test_grouping_identical_with_and_without_fast_filter():
    """End-to-end: group ids must match a known-correct small-corpus grouping."""
    specs = [
        ("The senator said the bill eliminates federal income tax by 2027", "en"),
        ("The senator said the bill eliminates federal income tax by 2027.", "en"),
        ("A completely unrelated statement regarding maritime shipping law", "en"),
        (TAMIL_CLAIM, "ta"),
        (TAMIL_CLAIM + ".", "ta"),
    ]
    claims = [_mk(f"c{i}", t, lang=l) for i, (t, l) in enumerate(specs)]
    g = group_claims(claims, near_duplicate=True, ngram=5, threshold=0.90)
    assert g.group_ids[0] == g.group_ids[1], "punctuation-only variant not grouped"
    assert g.group_ids[3] == g.group_ids[4], "Tamil variant not grouped"
    assert g.group_ids[2] not in (g.group_ids[0], g.group_ids[3]), "unrelated claim merged"
    assert g.n_groups == 3


def test_group_safety_repair_is_a_noop_when_already_safe():
    """X-FACT already passes 5/5; the repair must not touch it."""
    claims = [_mk(f"c{i}", f"Distinct claim number {i} about a separate topic entirely")
              for i in range(9)]
    g = group_claims(claims, near_duplicate=True, threshold=0.90)
    splits = {"train": [0, 1, 2, 3, 4], "validation": [5, 6], "test": [7, 8]}
    new, rep = enforce_group_safe_splits(claims, splits, g.group_ids)
    assert new == splits, "repair altered an already group-safe split"
    assert rep["n_groups_repaired"] == 0 and rep["n_claims_moved"] == 0


def test_group_safety_repair_moves_whole_group_to_majority_split():
    """The FEVER case: a duplicate claim straddling train and test."""
    dup = "Warren Beatty co-wrote Reds."
    claims = [_mk("t0", dup), _mk("t1", dup + " "), _mk("t2", dup.upper()),
              _mk("x0", "A wholly unrelated statement about maritime shipping law"),
              _mk("e0", dup)]
    g = group_claims(claims, near_duplicate=True, threshold=0.90)
    assert g.group_ids[0] == g.group_ids[4], "test premise: duplicates must share a group"
    splits = {"train": [0, 1, 2, 3], "test": [4]}
    new, rep = enforce_group_safe_splits(claims, splits, g.group_ids)

    assert rep["n_groups_repaired"] == 1
    assert rep["n_claims_moved"] == 1
    assert rep["claims_deleted"] == 0
    assert new["test"] == [], "minority member was not moved out of test"
    assert sorted(new["train"]) == [0, 1, 2, 3, 4]
    assert rep["moves"][0]["from_split"] == "test" and rep["moves"][0]["to_split"] == "train"
    # every claim still present exactly once
    assert sorted(i for v in new.values() for i in v) == list(range(5))


def test_contested_group_always_goes_to_train_even_when_test_is_the_majority():
    """POLICY: evaluation sets only ever shrink.

    Here TEST holds 3 of the 4 duplicates and TRAIN only 1. A
    majority-wins rule would pull the train claim INTO test, letting a
    claim the corpus designated for training influence a reported score.
    The required policy moves the whole group to train instead.
    """
    dup = "Floyd Mayweather Jr. is a professional boxer."
    claims = [_mk(f"d{i}", dup + " " * i) for i in range(4)] + \
             [_mk("u", "An unrelated claim concerning agricultural subsidies")]
    g = group_claims(claims, near_duplicate=True, threshold=0.90)
    splits = {"train": [0], "test": [1, 2, 3], "validation": [4]}
    new, rep = enforce_group_safe_splits(claims, splits, g.group_ids)

    assert rep["policy"] == "contested_groups_to_train"
    assert sorted(new["train"]) == [0, 1, 2, 3], "group did not land wholly in train"
    assert new["test"] == [], "test must shrink, not absorb the group"
    assert all(m["to_split"] == "train" for m in rep["moves"])
    assert all(m["from_split"] == "test" for m in rep["moves"])
    assert new["validation"] == [4], "unrelated claim must not move"


def test_group_safety_repair_never_grows_an_evaluation_split():
    """Invariant across a mixed corpus: train may grow; val/test never do."""
    dup_a, dup_b, dup_c = "Claim alpha duplicated", "Claim beta duplicated", "Claim gamma dup"
    claims = [_mk(f"u{i}", f"Unique claim {i} on wholly separate subject matter {i}")
              for i in range(10)]
    claims += [_mk("a0", dup_a), _mk("a1", dup_a + "!"),
               _mk("b0", dup_b), _mk("b1", dup_b + " "),
               _mk("c0", dup_c), _mk("c1", dup_c + ".")]
    g = group_claims(claims, near_duplicate=True, threshold=0.90)
    splits = {"train": [0, 1, 2, 3, 10, 12], "validation": [4, 5, 14],
              "test": [6, 7, 8, 9, 11, 13, 15]}
    new, rep = enforce_group_safe_splits(claims, splits, g.group_ids)
    for s in ("validation", "test"):
        assert len(new[s]) <= len(splits[s]), f"{s} grew during repair"
    assert len(new["train"]) >= len(splits["train"])
    assert rep["delta"]["train"] == -(rep["delta"]["validation"] + rep["delta"]["test"])
    assert sorted(i for v in new.values() for i in v) == list(range(len(claims)))


def test_group_safety_repair_is_deterministic():
    dup = "An evenly split duplicated claim about federal tax policy"
    claims = [_mk(f"d{i}", dup + " " * i) for i in range(4)]
    g = group_claims(claims, near_duplicate=True, threshold=0.90)
    splits = {"train": [0, 1], "test": [2, 3]}
    a, ra = enforce_group_safe_splits(claims, splits, g.group_ids)
    b, rb = enforce_group_safe_splits(claims, splits, g.group_ids)
    assert a == b and ra["moves"] == rb["moves"], "repair is not deterministic"
    assert sorted(a["train"]) == [0, 1, 2, 3]
    assert a["test"] == []


def test_group_safety_repair_guarantees_the_three_failing_gate_checks():
    """End-to-end: after repair the gate must report no spanning group,
    no exact cross-split overlap and no cross-split near-duplicate."""
    dup_a, dup_b = "Warren Beatty co-wrote Reds.", "Floyd Mayweather is a professional boxer."
    claims = [_mk(f"u{i}", f"Unique claim {i} regarding distinct subject matter {i}")
              for i in range(12)]
    claims += [_mk("a0", dup_a), _mk("a1", dup_a + "!"), _mk("b0", dup_b), _mk("b1", dup_b + " ")]
    g = group_claims(claims, near_duplicate=True, threshold=0.90)
    splits = {"train": list(range(0, 8)) + [12, 14],
              "validation": [8, 9],
              "test": [10, 11, 13, 15]}

    before = verify_no_leakage(claims, splits, g.group_ids, near_duplicate_threshold=0.90)
    failed_before = {c["name"] for c in before["checks"] if not c["passed"]}
    assert "no_group_spans_multiple_splits" in failed_before, "fixture must leak first"

    new, rep = enforce_group_safe_splits(claims, splits, g.group_ids)
    after = verify_no_leakage(claims, new, g.group_ids, near_duplicate_threshold=0.90)
    assert after["passed"], [c for c in after["checks"] if not c["passed"]]
    assert rep["n_claims_moved"] > 0
    assert sorted(i for v in new.values() for i in v) == list(range(len(claims)))
    assert sum(len(v) for v in new.values()) == sum(len(v) for v in splits.values()), \
        "repair changed the total claim count"


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
