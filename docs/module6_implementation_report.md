# Module 6 — Evidence Retrieval + Fact Verification: Implementation Report

Status: **implemented and locally tested. No full-corpus results exist yet.**
Nothing in this report is a measurement on the real FEVER/X-FACT corpora; the
pipeline has only been run on a synthetic fixture. Real numbers require the
Colab notebook to execute.

Module 5 is frozen and was verified byte-identical before and after this work.

---

## 1. Architecture

```
Module 5 corpus (READ-ONLY)
  claims.jsonl / evidence.jsonl / claim_splits.json
        |
  build_index_scope()          <- Module 5's own function, imported not copied
        |
  permitted CLAIM indices  ->  permitted EVIDENCE ids
        |
  LaBSE encoding (sharded, resumable)
        |
  FAISS IndexFlatIP over L2-normalised vectors  ==  exact cosine
        |
  top-50 candidates          BM25 baseline (same scope)
        |
  XLM-R cross-encoder rerank  ->  top-5
        |
  XLM-R verifier
        |
  FEVER : SUPPORTS / REFUTES / NOT ENOUGH INFO      (entailment)
  X-FACT: false / true / partly true/misleading     (veracity)
        |
  temperature scaling (fitted on validation) -> ECE, reliability, ablations
```

## 2. Label-space handling

FEVER and X-FACT are **different tasks** and are never merged, mapped, or
jointly scored.

- `LabelSpace` is a closed vocabulary: an unknown label raises rather than
  defaulting. `FEVER_SPACE.can_encode("false")` is `False` and
  `X_FACT_SPACE.can_encode("SUPPORTS")` is `False`, both asserted by tests.
- `"partly true/misleading"` is stored verbatim. It has no FEVER counterpart;
  forcing it into SUPPORTS or REFUTES would fabricate a judgement the
  annotator never made.
- `mapping_between_spaces: false` in the config, asserted by a test.
- The XLM-R **encoder is shared** so FEVER -> Tamil transfer is measurable.
  The **head is always re-initialised** by `swap_head()`, which returns a
  record of the swap. Both spaces have three classes, which makes reusing the
  FEVER head tempting and wrong: it would smuggle entailment semantics into
  veracity labels and make the transfer result uninterpretable.
- Results are reported separately, per dataset, per language.

## 3. Leakage prevention

Module 5's `build_index_scope()` returns **claim** indices; a retrieval index
stores **evidence**. That translation is the one place a leak could hide, so
it is isolated in `src/retrieval/index_scope.py` and tested hardest.

Rule: *permitted evidence = union of evidence cited by permitted claims.*
Evidence cited only by out-of-scope claims is excluded. Evidence cited by both
an in-scope and an out-of-scope claim is included — it is legitimately
reachable from the in-scope side. That asymmetry is deliberate and tested.

Three enforcement layers:

1. Stage 1 encodes only in-scope evidence, so an out-of-scope item never gets
   a vector.
2. `assert_index_within_scope()` audits the **built** index (after
   construction, not before) and raises on any extra id.
3. `verify_phase_isolation()` independently recomputes split-exclusive
   evidence from the claims and checks each phase, so a bug in the scope
   builder cannot hide behind its own bookkeeping.

Negative-tested: a scope config granting train access to test is caught, as is
one granting validation access to test.

## 4. Compute reduction (free Colab, 0 compute units)

| Item | Config original | Used | Reason |
|---|---:|---:|---|
| top_k_candidates | 100 | 50 | reranking cost |
| reranker max_length | 512 | 256 | ~2x speedup; evidence sentences are short |
| verifier max_length | 512 | 256 | as above |
| verifier epochs | 10 | 3 | ~12 h -> ~50 min |
| FEVER train claims | 228,090 | 50,000 (stratified) | as above |
| rerank train subset | all | 20,000 | 26M pairs ~ 60 h on T4 |

`full_scale_results_claimed: false`. The reductions are in the config, in
every stage manifest, and asserted by a test — they cannot be quietly dropped
from the write-up.

## 5. Files created

**Config (1):** `configs/module6_retrieval_verification.yaml`

**`src/retrieval/` (7):** `__init__`, `index_scope`, `encoders`, `faiss_index`,
`bm25_baseline`, `reranker`, `retrieval_metrics`

**`src/verification/` (6):** `__init__`, `dataset`, `verifier_model`,
`trainer`, `calibration`, `ablations`

**`scripts/` (10):** `m6_common`, `m6_01_encode_evidence`, `m6_02_build_index`,
`m6_03_encode_claims`, `m6_04_retrieve`, `m6_05_rerank`,
`m6_06_train_verifier`, `m6_07_evaluate_verifier`, `m6_08_run_ablations`,
`verify_module6_outputs`

**`tests/` (2):** `test_retrieval.py`, `test_verification.py`

**`notebooks/` (1):** `module6_colab.ipynb`

**Files modified: none.**

## 6. Checkpointing

Every stage: 10,000-item shards, `.tmp.npy` written then atomically renamed,
manifest listing completed shards, resume skips them. The `.tmp.npy` suffix is
deliberate — `np.save` appends `.npy` unless the name already ends with it, so
`shard.npy.tmp` becomes `shard.npy.tmp.npy` and every rename fails. That exact
bug was found in Module 3 and is regression-tested here.

Manifests carry a fingerprint of the input texts; encoding a different corpus
into an existing stage directory raises rather than silently mixing
embeddings.

Verifier training checkpoints model, optimizer, scheduler, epoch, step, config
**and RNG state** (python/numpy/torch/cuda). Without the RNG state a resumed
run would see a different shuffle, so "resume" would silently mean "restart
from these weights".

## 7. Tests

`tests/test_retrieval.py` **37/37**, `tests/test_verification.py` **35/35**.
Existing suites unchanged: 49/49, 21/21, 22/22, 5/5.

Hand-computed: BM25 score, Recall@k, MRR, nDCG, DCG, ECE, macro-F1.
Negative-tested: misconfigured scope (both directions), out-of-scope index,
invented evidence from a reranker subclass, lossy candidate filter, changed
corpus fingerprint, temperature fitted on test, unknown labels, cross-space
label encoding.

Two bugs were found by these tests during implementation:

1. **Tamil tokenisation.** `\w+` follows `str.isalnum()`, which is `False` for
   combining marks — and Tamil vowel signs and the virama are exactly that.
   Measured: `"தாக்குதல்"` tokenised as `['த','க','க','தல']`, dropping U+0BBE,
   U+0BBF, U+0BC1, U+0BC2, U+0BCD. BM25 over Tamil would have matched on
   meaningless consonant fragments. Fixed by adding the Tamil block and
   combining diacritics to the word pattern.
2. **Index reload changed the metric.** `EvidenceIndex.load()` passed
   `normalize=False` because stored vectors are already unit-length, which
   also stopped *queries* being normalised — silently returning inner products
   instead of cosines. Split into `normalize` (queries and vectors) and
   `vectors_prenormalized`.

## 8. Known limitations

- **No real results.** The pipeline has run only on a synthetic fixture.
- Cross-encoder reranking needs a fine-tuned checkpoint; with the stock
  `xlm-roberta-base` head the scores are arbitrary, and Stage 5 prints a
  warning saying so.
- Three ablations (`claim_only`, `english_only_vs_multilingual`,
  `oracle_gold_evidence`) require a trained verifier per variant and are
  reported as **pending**, not estimated.
- X-FACT has 1,096 Tamil training claims. Fine-tuning a 278M-parameter encoder
  on that will be high-variance; the three-seed protocol matters here.
- 2,792 FEVER test claims (17.41%) cite evidence also used by train. This is a
  corpus property, not label leakage, and is disclosed rather than corrected.
- `src/evaluation/retrieval_metrics.py` (Module 4's same-class proxy) is left
  untouched and is **not** used for Module 6 reporting.
