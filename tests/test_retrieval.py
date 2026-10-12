"""Module 6 retrieval tests — scope, index exactness, metrics, reranking.

Same standard as Modules 3-5: values checked against hand-computed results,
and every guard tested with a case that MUST fail. A leakage gate that cannot
fail is worse than no gate.

Runs without torch, faiss, sentence-transformers, or any downloaded model.

    python tests/test_retrieval.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_baseline import BM25, BM25Retriever, tokenize  # noqa: E402
from src.retrieval.encoders import ShardedEncoder, text_fingerprint  # noqa: E402
from src.retrieval.faiss_index import (  # noqa: E402
    EvidenceIndex,
    brute_force_search,
    faiss_available,
    l2_normalize,
)
from src.retrieval.index_scope import (  # noqa: E402
    assert_index_within_scope,
    build_evidence_scope,
    claim_evidence_ids,
    verify_phase_isolation,
)
from src.retrieval.reranker import (  # noqa: E402
    CandidateSetViolation,
    CrossEncoderReranker,
    assert_subset_of_candidates,
    rerank_candidates,
)
from src.retrieval.retrieval_metrics import (  # noqa: E402
    dcg,
    evaluate_retrieval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

TAMIL = "பஹல்காம் தாக்குதல் குறித்த தவறான கூற்று"
SCOPE_CFG = {"train": ["train"], "validation": ["train", "validation"],
             "test": ["train", "validation", "test"]}


def _claim(cid, ev, text="a claim", lang="en", label="SUPPORTS"):
    return {"claim_id": cid, "text": text, "evidence": list(ev), "language": lang,
            "label": label, "metadata": {}}


def _corpus():
    """train 0-2, validation 3-4, test 5-6, with deliberately shared evidence."""
    claims = [
        _claim("c0", ["e_tr1", "e_shared"]),
        _claim("c1", ["e_tr2"]),
        _claim("c2", []),                      # no gold evidence (NEI-like)
        _claim("c3", ["e_va1", "e_shared"]),
        _claim("c4", ["e_va2"]),
        _claim("c5", ["e_te1"]),
        _claim("c6", ["e_te2", "e_va1"]),
    ]
    splits = {"train": [0, 1, 2], "validation": [3, 4], "test": [5, 6]}
    return claims, splits


# ------------------------------------------------------------- index scope

def test_train_scope_excludes_validation_and_test_exclusive_evidence():
    claims, splits = _corpus()
    scope = build_evidence_scope(claims, splits, "train", SCOPE_CFG)
    assert scope.evidence_ids == {"e_tr1", "e_tr2", "e_shared"}
    for forbidden in ("e_va1", "e_va2", "e_te1", "e_te2"):
        assert forbidden not in scope.evidence_ids, f"{forbidden} leaked into train index"


def test_shared_evidence_stays_in_scope_when_reachable_from_in_scope_claim():
    """e_shared is cited by a train claim AND a validation claim. It is
    legitimately reachable from train, so excluding it would distort the
    corpus. This asymmetry is deliberate."""
    claims, splits = _corpus()
    scope = build_evidence_scope(claims, splits, "train", SCOPE_CFG)
    assert "e_shared" in scope.evidence_ids
    assert scope.n_evidence_shared_with_excluded_claims >= 1


def test_validation_scope_excludes_test_exclusive_evidence():
    claims, splits = _corpus()
    scope = build_evidence_scope(claims, splits, "validation", SCOPE_CFG)
    assert {"e_tr1", "e_tr2", "e_shared", "e_va1", "e_va2"} <= scope.evidence_ids
    assert "e_te1" not in scope.evidence_ids
    assert "e_te2" not in scope.evidence_ids


def test_test_scope_sees_whole_corpus():
    claims, splits = _corpus()
    scope = build_evidence_scope(claims, splits, "test", SCOPE_CFG)
    all_ids = set()
    for c in claims:
        all_ids.update(claim_evidence_ids(c))
    assert scope.evidence_ids == all_ids


def test_phase_isolation_passes_on_a_correct_corpus():
    claims, splits = _corpus()
    res = verify_phase_isolation(claims, splits, SCOPE_CFG)
    assert res["passed"], [c for c in res["checks"] if not c["passed"]]


def test_phase_isolation_CATCHES_a_misconfigured_scope():
    """NEGATIVE TEST. The realistic failure is a scope config that hands the
    training phase access to test claims -- the "just index everything"
    mistake. The isolation check must fail loudly.

    Note what is NOT leakage: adding a test-cited evidence id to a train claim
    makes that evidence legitimately reachable from train, so it stops being
    test-exclusive. An earlier version of this test injected exactly that and
    passed, which is correct behaviour, not a hole in the gate.
    """
    claims, splits = _corpus()
    broken = {"train": ["train", "test"],            # <- the bug
              "validation": ["train", "validation"],
              "test": ["train", "validation", "test"]}
    res = verify_phase_isolation(claims, splits, broken)
    assert not res["passed"], "misconfigured scope was not detected"
    failed = [c["name"] for c in res["checks"] if not c["passed"]]
    assert "train_index_excludes_test_exclusive_evidence" in failed, failed


def test_phase_isolation_CATCHES_validation_seeing_test():
    claims, splits = _corpus()
    broken = {"train": ["train"],
              "validation": ["train", "validation", "test"],   # <- the bug
              "test": ["train", "validation", "test"]}
    res = verify_phase_isolation(claims, splits, broken)
    assert not res["passed"]
    assert "validation_index_excludes_test_exclusive_evidence" in \
        [c["name"] for c in res["checks"] if not c["passed"]]


def test_assert_index_within_scope_fires_on_extra_evidence():
    claims, splits = _corpus()
    scope = build_evidence_scope(claims, splits, "train", SCOPE_CFG)
    assert_index_within_scope(["e_tr1", "e_shared"], scope)      # fine
    try:
        assert_index_within_scope(["e_tr1", "e_te1"], scope)
    except AssertionError as exc:
        assert "LEAKAGE" in str(exc)
    else:
        raise AssertionError("out-of-scope evidence was not detected")


def test_unknown_phase_raises():
    claims, splits = _corpus()
    for bad in ("训练", "predict", ""):
        try:
            build_evidence_scope(claims, splits, bad, SCOPE_CFG)
        except ValueError:
            continue
        raise AssertionError(f"unknown phase {bad!r} did not raise")


# --------------------------------------------------------------- FAISS index

def test_index_matches_brute_force_cosine():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(40, 16)).astype(np.float32)
    queries = rng.normal(size=(7, 16)).astype(np.float32)
    ids = [f"e{i}" for i in range(40)]

    idx = EvidenceIndex(ids, vecs, normalize=True)
    got_ids, got_scores = idx.search(queries, top_k=5)
    exp_ids, exp_scores = brute_force_search(queries, vecs, ids, top_k=5)

    assert got_ids == exp_ids, "index ranking differs from brute-force cosine"
    assert np.allclose(got_scores, exp_scores, atol=1e-5)


def test_numpy_and_faiss_backends_agree():
    if not faiss_available():
        return  # nothing to compare against in this environment
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(50, 32)).astype(np.float32)
    q = rng.normal(size=(5, 32)).astype(np.float32)
    ids = [f"e{i}" for i in range(50)]
    a, sa = EvidenceIndex(ids, vecs, use_faiss=True).search(q, 10)
    b, sb = EvidenceIndex(ids, vecs, use_faiss=False).search(q, 10)
    assert a == b and np.allclose(sa, sb, atol=1e-5)


def test_normalized_inner_product_equals_cosine():
    a = np.array([[3.0, 4.0]], dtype=np.float32)      # |a| = 5
    b = np.array([[4.0, 3.0]], dtype=np.float32)      # |b| = 5
    # cos = (3*4 + 4*3) / 25 = 24/25 = 0.96
    idx = EvidenceIndex(["b"], b, normalize=True)
    _, scores = idx.search(a, top_k=1)
    assert abs(float(scores[0][0]) - 0.96) < 1e-6, scores


def test_index_rejects_duplicate_ids_and_length_mismatch():
    v = np.zeros((2, 4), dtype=np.float32)
    for bad_args in (({"evidence_ids": ["a", "a"], "vectors": v}),
                     ({"evidence_ids": ["a"], "vectors": v})):
        try:
            EvidenceIndex(**bad_args)
        except ValueError:
            continue
        raise AssertionError(f"bad index construction accepted: {list(bad_args)}")


def test_zero_vector_does_not_produce_nan():
    v = np.zeros((3, 8), dtype=np.float32)
    assert not np.isnan(l2_normalize(v)).any()


def test_top_k_larger_than_corpus_is_clamped():
    v = np.eye(3, dtype=np.float32)
    idx = EvidenceIndex(["a", "b", "c"], v)
    ids, _ = idx.search(np.eye(1, 3, dtype=np.float32), top_k=99)
    assert len(ids[0]) == 3


def test_index_save_load_roundtrip():
    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(2)
    vecs = rng.normal(size=(20, 8)).astype(np.float32)
    ids = [f"e{i}" for i in range(20)]
    idx = EvidenceIndex(ids, vecs)
    idx.save(d)
    back = EvidenceIndex.load(d)
    q = rng.normal(size=(3, 8)).astype(np.float32)
    a, sa = idx.search(q, 5)
    b, sb = back.search(q, 5)
    assert a == b and np.allclose(sa, sb, atol=1e-6)


# ------------------------------------------------------------------- BM25

def test_bm25_score_matches_hand_computation():
    """Hand-check one score end to end.

    corpus: d0="the cat sat", d1="the dog sat", d2="a bird flew away now"
    query "cat": df=1, N=3 -> idf = log((3-1+0.5)/(1+0.5) + 1) = log(2.6667)
    d0: f=1, |d0|=3, avgdl=(3+3+5)/3=3.6667
        denom = 1 + 1.5*(1-0.75+0.75*3/3.6667) = 1 + 1.5*0.86364 = 2.29545
        score = idf * (1*2.5)/2.29545
    """
    import math
    corpus = [tokenize("the cat sat"), tokenize("the dog sat"),
              tokenize("a bird flew away now")]
    bm = BM25(corpus, k1=1.5, b=0.75)
    idf = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1.0)
    avgdl = (3 + 3 + 5) / 3
    denom = 1.0 * 1.5 * (1 - 0.75 + 0.75 * 3 / avgdl) + 1.0
    expected = idf * (1 * (1.5 + 1)) / denom
    assert abs(bm.score(tokenize("cat"), 0) - expected) < 1e-9, \
        (bm.score(tokenize("cat"), 0), expected)


def test_bm25_idf_never_negative():
    """A term in most documents must not penalise a document for containing it."""
    corpus = [tokenize("common term here")] * 9 + [tokenize("rare word")]
    bm = BM25(corpus)
    assert all(v >= 0.0 for v in bm.idf.values()), bm.idf
    assert bm.score(tokenize("common"), 0) >= 0.0


def test_bm25_tokenizer_preserves_tamil_and_lowercases_latin():
    toks = tokenize("BREAKING News " + TAMIL)
    assert "breaking" in toks and "BREAKING" not in toks
    joined = "".join(toks)
    assert all(ch in joined for ch in TAMIL if "஀" <= ch <= "௿")


def test_bm25_retriever_ranks_relevant_first_and_is_deterministic():
    ids = ["e0", "e1", "e2"]
    texts = ["the prime minister announced a new tax policy",
             "football results from yesterday evening",
             "a new tax policy was announced by the minister"]
    r = BM25Retriever(ids, texts)
    got, scores = r.retrieve("new tax policy announced", top_k=3)
    assert got[0] in {"e0", "e2"} and "e1" == got[-1], got
    assert r.retrieve("new tax policy announced", top_k=3)[0] == got, "not deterministic"


# ---------------------------------------------------------------- metrics

def test_recall_mrr_ndcg_hand_computed():
    retrieved = ["e9", "e1", "e7", "e2", "e5"]
    gold = ["e1", "e2", "e3"]
    # e1 at rank 2, e2 at rank 4, e3 never.
    assert recall_at_k(retrieved, gold, 1) == 0.0
    assert abs(recall_at_k(retrieved, gold, 2) - 1 / 3) < 1e-12
    assert abs(recall_at_k(retrieved, gold, 5) - 2 / 3) < 1e-12
    assert abs(reciprocal_rank(retrieved, gold) - 0.5) < 1e-12   # first hit at rank 2

    import math
    # DCG = 1/log2(3) + 1/log2(5); IDCG = 1/log2(2)+1/log2(3)+1/log2(4)
    exp_dcg = 1 / math.log2(3) + 1 / math.log2(5)
    exp_idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert abs(ndcg_at_k(retrieved, gold, 5) - exp_dcg / exp_idcg) < 1e-12


def test_perfect_and_empty_retrieval_edges():
    assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0
    assert dcg([1.0]) == 1.0     # 1/log2(2) = 1


def test_ndcg_not_penalised_when_gold_exceeds_k():
    """3 gold items but k=2: a perfect top-2 must score 1.0."""
    assert abs(ndcg_at_k(["a", "b", "z"], ["a", "b", "c"], 2) - 1.0) < 1e-12


def test_claims_without_gold_evidence_excluded_but_counted():
    """The 427 FEVER NEI claims must not sit in a recall denominator."""
    results = {"c1": ["e1"], "c2": ["e9"], "c3": ["e2"], "c4": ["e3"]}
    gold = {"c1": ["e1"], "c2": ["e2"], "c3": [], "c4": ["e3"]}   # c3 has no gold
    res = evaluate_retrieval(results, gold, k_values=(1,), ndcg_k=1)
    acc = res["accounting"]
    assert acc["n_queries"] == 4
    assert acc["n_scored"] == 3
    assert acc["n_excluded_no_gold_evidence"] == 1
    assert acc["balances"] is True
    # scored: c1 hit, c2 miss, c4 hit -> 2/3
    assert abs(res["overall"]["recall@1"] - 2 / 3) < 1e-12


def test_recall_raises_on_claim_with_no_gold():
    try:
        recall_at_k(["a"], [], 1)
    except ValueError:
        return
    raise AssertionError("recall_at_k accepted a claim with no gold evidence")


def test_per_language_breakdown():
    results = {"c1": ["e1"], "c2": ["e2"]}
    gold = {"c1": ["e1"], "c2": ["e9"]}
    langs = {"c1": "ta", "c2": "en"}
    res = evaluate_retrieval(results, gold, k_values=(1,), ndcg_k=1, languages=langs)
    assert res["per_language"]["ta"]["recall@1"] == 1.0
    assert res["per_language"]["en"]["recall@1"] == 0.0
    assert res["per_language_counts"] == {"ta": 1, "en": 1}


# --------------------------------------------------------------- reranking

def test_reranker_reorders_and_truncates():
    ids = ["a", "b", "c", "d"]
    out, scores = rerank_candidates(ids, [0.1, 0.9, 0.5, 0.2], top_n=2)
    assert out == ["b", "c"] and scores == [0.9, 0.5]


def test_reranker_never_introduces_new_evidence():
    """The invariant: reranking is a permutation of the candidate set."""
    ids = ["a", "b", "c"]
    out, _ = rerank_candidates(ids, [1.0, 2.0, 3.0], top_n=3)
    assert set(out) <= set(ids)
    assert len(set(out)) == len(out)


def test_candidate_set_violation_is_detected_end_to_end():
    """NEGATIVE TEST: a reranker subclass that invents evidence must be caught
    by the pipeline, not merely by the reranker's own bookkeeping."""
    class BadReranker(CrossEncoderReranker):
        def rerank_one(self, claim_text, candidate_ids, evidence_texts):
            return list(candidate_ids) + ["e_INVENTED"], [1.0] * (len(candidate_ids) + 1)

    bad = BadReranker(score_fn=lambda pairs: np.ones(len(pairs)), top_n=5)
    try:
        bad.rerank_all({"c0": "claim"}, {"c0": ["a", "b"]}, {"a": "x", "b": "y"},
                       Path(tempfile.mkdtemp()), log=lambda *_: None)
    except CandidateSetViolation as exc:
        assert "e_INVENTED" in str(exc), exc
        return
    raise AssertionError("invented evidence went undetected by rerank_all")


def test_assert_subset_of_candidates_guard():
    assert_subset_of_candidates(["a"], ["a", "b"])            # fine
    for bad in (["a", "z"], ["a", "a"]):
        try:
            assert_subset_of_candidates(bad, ["a", "b"], "c0")
        except CandidateSetViolation:
            continue
        raise AssertionError(f"guard missed {bad}")


def test_reranker_ties_break_by_original_order():
    ids = ["a", "b", "c"]
    out, _ = rerank_candidates(ids, [0.5, 0.5, 0.5], top_n=3)
    assert out == ["a", "b", "c"], out


def test_reranker_end_to_end_with_injected_scorer():
    """Score = length of evidence text, so the expected order is known."""
    ev = {"a": "x", "b": "xxxxx", "c": "xxx"}
    rr = CrossEncoderReranker(score_fn=lambda pairs: np.array([len(p[1]) for p in pairs],
                                                             dtype=np.float32),
                              top_n=2, batch_size=2)
    ids, scores = rr.rerank_one("claim", ["a", "b", "c"], ev)
    assert ids == ["b", "c"] and scores == [5.0, 3.0]


def test_reranker_empty_candidates():
    rr = CrossEncoderReranker(score_fn=lambda p: np.zeros(len(p)), top_n=5)
    assert rr.rerank_one("claim", [], {}) == ([], [])


# ----------------------------------------------------- sharding and resume

def test_sharded_encoder_resume_matches_uninterrupted_run():
    """Kill the encoder mid-way, resume, and require identical output."""
    texts = [f"text number {i}" for i in range(25)]

    def enc(batch):
        return np.array([[float(len(t)), float(hash(t) % 97)] for t in batch],
                        dtype=np.float32)

    full_dir = Path(tempfile.mkdtemp())
    ShardedEncoder(full_dir, shard_size=10, encode_fn=enc).encode(texts, log=lambda *_: None)
    full = ShardedEncoder(full_dir, shard_size=10, encode_fn=enc).load_all()

    part_dir = Path(tempfile.mkdtemp())
    calls = {"n": 0}

    def flaky(batch):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated Colab disconnect")
        return enc(batch)

    try:
        ShardedEncoder(part_dir, shard_size=10, encode_fn=flaky).encode(texts, log=lambda *_: None)
    except RuntimeError:
        pass
    man = ShardedEncoder(part_dir, shard_size=10, encode_fn=enc).load_manifest()
    assert man.get("status") == "in_progress" and len(man.get("completed_shards", [])) == 1

    ShardedEncoder(part_dir, shard_size=10, encode_fn=enc).encode(texts, log=lambda *_: None)
    resumed = ShardedEncoder(part_dir, shard_size=10, encode_fn=enc).load_all()

    assert resumed.shape == full.shape
    assert np.array_equal(resumed, full), "resumed output differs from uninterrupted run"


def test_manifest_rejects_a_changed_corpus():
    """Reusing a manifest against different texts would mix two corpora."""
    d = Path(tempfile.mkdtemp())
    enc = lambda b: np.zeros((len(b), 3), dtype=np.float32)  # noqa: E731
    ShardedEncoder(d, shard_size=5, encode_fn=enc).encode(["a", "b"], log=lambda *_: None)
    try:
        ShardedEncoder(d, shard_size=5, encode_fn=enc).encode(["a", "c"], log=lambda *_: None)
    except RuntimeError as exc:
        assert "fingerprint" in str(exc).lower()
        return
    raise AssertionError("manifest accepted a different corpus")


def test_fingerprint_is_order_sensitive():
    assert text_fingerprint(["a", "b"]) != text_fingerprint(["b", "a"])
    assert text_fingerprint(["a", "b"]) == text_fingerprint(["a", "b"])


def test_shard_temp_suffix_is_tmp_npy_not_npy_tmp():
    """np.save appends .npy unless the name already ends with it, so a
    `.npy.tmp` temp file becomes `.npy.tmp.npy` and every rename fails.
    This exact bug was found in Module 3."""
    d = Path(tempfile.mkdtemp())
    enc = lambda b: np.ones((len(b), 2), dtype=np.float32)  # noqa: E731
    ShardedEncoder(d, shard_size=4, encode_fn=enc).encode(["a"] * 8, log=lambda *_: None)
    assert (d / "shard_00000.npy").is_file()
    assert not list(d.glob("*.npy.tmp*")), list(d.glob("*"))
    assert not list(d.glob("*.tmp.npy")), "temp files were left behind"


def test_batched_search_equals_unbatched_and_bruteforce():
    """REGRESSION for the FEVER OOM (exit -9).

    The NumPy fallback used to build the whole (n_queries x n_evidence)
    similarity matrix. At FEVER scale that is 15,916 x 67,388 = 1.07e9 float32
    (4.3 GB) plus an 8.6 GB int64 argsort -- 12.87 GB, which SIGKILLs a Colab
    runtime with no traceback. Batching must not change any result.
    """
    rng = np.random.default_rng(11)
    vecs = rng.normal(size=(500, 32)).astype(np.float32)
    queries = rng.normal(size=(97, 32)).astype(np.float32)
    ids = [f"e{i}" for i in range(500)]
    idx = EvidenceIndex(ids, vecs, use_faiss=False)

    unbatched_ids, unbatched_scores = idx.search(queries, 10, batch_size=10_000)
    batched_ids, batched_scores = idx.search(queries, 10, batch_size=7)
    exp_ids, exp_scores = brute_force_search(queries, vecs, ids, 10)

    assert batched_ids == unbatched_ids, "batching changed the ranking"
    assert batched_ids == exp_ids, "batched ranking differs from brute force"
    assert np.allclose(batched_scores, unbatched_scores, atol=1e-6), "scores changed"
    assert np.allclose(batched_scores, exp_scores, atol=1e-5)


def test_batch_size_never_changes_topk_count():
    rng = np.random.default_rng(12)
    vecs = rng.normal(size=(200, 16)).astype(np.float32)
    q = rng.normal(size=(53, 16)).astype(np.float32)
    ids = [f"e{i}" for i in range(200)]
    idx = EvidenceIndex(ids, vecs, use_faiss=False)
    for bs in (1, 5, 53, 128, 5000):
        got, scores = idx.search(q, 10, batch_size=bs)
        assert len(got) == 53, f"batch_size={bs} lost queries"
        assert all(len(r) == 10 for r in got), f"batch_size={bs} changed top-k count"
        assert scores.shape == (53, 10)


def test_batching_preserves_tie_breaking_by_index():
    """All-equal scores must still resolve to ascending index, in every batch."""
    vecs = np.ones((9, 4), dtype=np.float32)
    idx = EvidenceIndex([f"t{i}" for i in range(9)], vecs, use_faiss=False)
    q = np.ones((5, 4), dtype=np.float32)
    for bs in (1, 2, 9, 1000):
        got, _ = idx.search(q, 4, batch_size=bs)
        assert all(r == ["t0", "t1", "t2", "t3"] for r in got), (bs, got[0])


def test_large_query_set_never_builds_the_full_similarity_matrix():
    """THE memory guard. Records the largest array handed to np.argsort and
    asserts it is bounded by batch_size x n_evidence, not n_queries x
    n_evidence. Without batching this test fails by ~40x on these sizes."""
    rng = np.random.default_rng(13)
    n_ev, n_q, bs = 400, 1000, 25
    vecs = rng.normal(size=(n_ev, 8)).astype(np.float32)
    q = rng.normal(size=(n_q, 8)).astype(np.float32)
    idx = EvidenceIndex([f"e{i}" for i in range(n_ev)], vecs, use_faiss=False)

    import src.retrieval.faiss_index as fx
    real_argsort = fx.np.argsort
    seen = {"max_elems": 0, "calls": 0}

    def spy(a, *args, **kwargs):
        seen["max_elems"] = max(seen["max_elems"], int(np.asarray(a).size))
        seen["calls"] += 1
        return real_argsort(a, *args, **kwargs)

    fx.np.argsort = spy
    try:
        idx.search(q, 5, batch_size=bs)
    finally:
        fx.np.argsort = real_argsort

    full_matrix = n_q * n_ev
    allowed = bs * n_ev
    assert seen["calls"] >= n_q // bs, "batching loop did not run"
    assert seen["max_elems"] <= allowed, (
        f"largest array {seen['max_elems']} exceeds the {allowed}-element batch "
        f"bound; the full {full_matrix}-element matrix was materialised")
    assert seen["max_elems"] < full_matrix / 10, (
        f"peak {seen['max_elems']} is not meaningfully below the full matrix "
        f"{full_matrix}")


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
