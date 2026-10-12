# Module 6 — Repair Plan (APPROVAL REQUIRED BEFORE EXECUTION)

**Date:** 2026-08-21
**Status:** 🛑 **NOTHING HAS BEEN EXECUTED, MODIFIED, RENAMED OR DELETED.**
This document is inspection + plan only, per your FINAL REQUIREMENT.

Everything below is derived from reading actual code, configs, Drive manifests, JSON outputs and directory listings — not notebook output.

---

## PART A — What is actually completed (verified against files)

| Stage | Script | Dataset | Train / Val / Test usage | Checkpoint used | Evidence source | Output dir | Exists? | Methodologically valid? | Rerun? |
|---|---|---|---|---|---|---|---|---|---|
| 1 Encode evidence | `m6_01_encode_evidence.py` | fever, x_fact | phase = validation, test | — | — | `outputs/module6/encode_evidence/{ds}_{validation,test}` | ✅ (4 dirs) | ✅ valid | ❌ **no — reusable** |
| 2 Build index | `m6_02_build_index.py` | fever, x_fact | phase = validation, test | — | — | `outputs/module6/build_index/{ds}_{validation,test}` | ✅ (4 dirs) | ✅ valid, `scope_audit: PASS` | ⏳ add `--phase train` only |
| 3 Encode claims | `m6_03_encode_claims.py` | fever, x_fact | split = validation, test | — | — | `outputs/module6/encode_claims/{ds}_{validation,test}` | ✅ (4 dirs) | ✅ valid | ⏳ add `train` only |
| 4 Retrieve | `m6_04_retrieve.py` | fever, x_fact | split = validation, test | — | dense + BM25, top-50 | `outputs/module6/retrieve/{ds}_{validation,test}` | ✅ (4 dirs) | ✅ valid, accounting balances | ⏳ add `train` only |
| 5 Rerank | `m6_05_rerank.py` | x_fact | test only | **`checkpoint: null`** | dense candidates | `outputs/module6/rerank/x_fact_test` | ⚠️ exists | 🔴 **INVALID — untrained cross-encoder** | ❌ do not rerun (no trainer exists) |
| **6 FEVER verifier** | `m6_06_train_verifier.py` | fever | train 50,000 / val 15,916 / test **not used** ✅ | wrote `models/module6/fever/{best,latest}.pt` | 🔴 **`"none"` — ZERO evidence** | `outputs/module6/verifier_training/fever` | ✅ | 🔴 **INVALID (train/eval mismatch)** | ✅ **YES** |
| **6b X-FACT verifier** | `m6_06_train_verifier.py` | x_fact | train ~1,096 / val 145 / test **not used** ✅ | wrote `models/module6/x_fact/{best,latest}.pt` | 🔴 **unrecorded — manifest missing** | `outputs/module6/verifier_training/x_fact` | ❌ **DOES NOT EXIST** | 🔴 **INVALID + NOT REPRODUCIBLE** | ✅ **YES** |
| 7 FEVER eval | `m6_07_evaluate_verifier.py` | fever | val → temperature; test → scoring only ✅ | `fever/best.pt` | dense_candidates (val + test) | `outputs/module6/evaluation/fever_test` | ✅ | ⚠️ valid protocol, invalid model | ✅ after Stage 6 |
| 7 X-FACT eval | `m6_07_evaluate_verifier.py` | x_fact | val → temperature; test → scoring only ✅ | `x_fact/best.pt` | ⚠️ **probably `reranked_top5.json` (untrained)** | `outputs/module6/evaluation/x_fact_test` | ✅ | ⚠️ probably contaminated | ✅ after Stage 6b |
| 8 Retrieval ablations | `m6_08_run_ablations.py` | fever, x_fact | test candidates only | **none loaded** | dense / bm25 / reranked | `outputs/module6/ablations/{ds}_test` | ✅ | ⚠️ **valid as retrieval analysis, mislabelled as ablation** | ⚠️ relabel only |
| 8 Verifier ablations | — | — | — | — | — | — | ❌ | ⏳ **never run** | ✅ **YES** |
| 9 Verify outputs | `verify_module6_outputs.py` | — | — | — | — | — | ❌ no report found | ⏳ pending | ✅ last |

**Test-label discipline is clean at every stage.** `m6_06` contains no `"test"` string. `fit_temperature()` raises on `split_name == "test"`. Checkpoint selection uses validation macro-F1 only. This part of the project is correct and should be described in the paper.

---

## PART B — Why training retrieval is absent (root cause, verified)

### Confirmed from `verifier_training/fever/manifest.json`

```json
"evidence_source": "none",
"claim_only": false,
"oracle_gold_evidence": false,
"example_stats": {
  "n_input": 260032, "n_built": 260032,
  "n_with_no_evidence": 260032,        ← 100 % of examples, ZERO evidence
  "evidence_source": "retrieved"       ← FALSE, hard-coded string
}
```

`n_with_no_evidence == n_built == 260,032`. Confirmed.

### Retrieval-output existence check (Drive directory listings, not filenames)

| Directory | fever_train | fever_validation | fever_test | x_fact_train | x_fact_validation | x_fact_test |
|---|---|---|---|---|---|---|
| `outputs/module6/encode_evidence/` | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| `outputs/module6/build_index/` | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| `outputs/module6/encode_claims/` | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| `outputs/module6/retrieve/` | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| `outputs/module6/rerank/` | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ untrained |

**No `_train` directory exists at any retrieval stage, for either dataset.** I verified each parent folder's children individually rather than inferring from validation/test.

### Root cause — a configuration decision, not a crash

`configs/module6_retrieval_verification.yaml`:

```yaml
retrieval:
  evaluate_splits: [validation, test]     # ← train deliberately excluded
reranking:
  rerank_splits: [validation, test]       # ← train deliberately excluded
```

Stages 1–4 were only ever driven for `validation` and `test`. Then in `m6_06_train_verifier.py`:

```python
retrieved, source = load_retrieved(cfg, root, args.dataset, "train")
#   → rerank/{ds}_train/reranked_top5.json     — missing
#   → retrieve/{ds}_train/dense_candidates.json — missing
#   → returns ({}, "none")                      ← silently
```

Then in `src/verification/dataset.py`:

```python
ev_ids = (retrieved or {}).get(cid, [])   # → [] for every claim
texts  = []                               # → to_text_pair() second segment = ""
```

**The failure is silent by design.** `load_retrieved()` returns an empty dict instead of raising, so training proceeded normally and exited 0. Meanwhile `m6_07` calls the *same* function for `validation` and `test`, where the files *do* exist — producing the mismatch: **trained on `(claim, "")`, evaluated on `(claim, ev1 </s> … </s> ev5)`.**

### Why this matters more than a missing ablation

The model was never trained to attend to the second tokenizer segment. Feeding it five passages at test time is out-of-distribution input. This biases results **downward** — so it did not inflate anything — but it means the published numbers do not measure the system the paper describes. It is a plausible contributor to FEVER test accuracy (0.3844) landing **below its own majority-class baseline** (0.4028).

### ✅ Good news: the leakage machinery for train retrieval already exists and is correct

`m6_04_retrieve.py` **already accepts `--split train`** and already enforces scope:

```python
scope = build_evidence_scope(corpus["claims"], corpus["splits"], phase, cfg["leakage"]["index_scope"])
...
assert_index_within_scope(index.evidence_ids, scope)   # raises on leakage
```

with `leakage.index_scope.train: [train]` — **train claims may only retrieve evidence cited by train claims.** No test or validation evidence can enter training. This satisfies your PART C leakage requirement **without any new leakage code**.

### ✅ Second piece of good news: evidence re-encoding is NOT needed

`m6_02_build_index.py` already contains a subset-derivation fallback:

```python
FALLBACK_ORDER = {"train": ["train", "validation", "test"], ...}
...
keep = [i for i, e in enumerate(ev_ids) if e in scope.evidence_ids]
```

Because train-scope evidence ⊂ test-scope evidence, and vectors are identical (same model, same text), the train index can be **derived by subset selection from the already-computed test-phase embeddings**. `assert_index_within_scope()` still audits the result. **This removes the single largest GPU cost from the repair (~1 h of LaBSE evidence encoding per dataset).**

---

## PART E — Transfer learning: what can and cannot be proven

**Design in code is correct.** `m6_06` does: build model with X-FACT `num_labels` → `CheckpointManager(init_from).load(model)` (loads FEVER encoder *and* head; shapes coincide at 3 classes) → `model.swap_head(...)` which unconditionally constructs a fresh `nn.Sequential`. No FEVER head weight survives. The optimizer is created *after* the swap and `cm.load` was called with `optimizer=None`, so FEVER optimizer state is not inherited. Correct.

**Whether it actually happened: UNPROVEN.**

- `manifest["init_from"]` and `manifest["head_swap"]` are written **only** to `verifier_training/x_fact/manifest.json`, which **does not exist**.
- `models/module6/x_fact/checkpoint_meta.json` records `label_space: "x_fact"` but has **no `init_from` field**.
- A Drive full-text search for `init-from` did not surface `module6_colab.ipynb`. Drive's full-text indexing of `.ipynb` is unreliable, so this is **not** evidence of absence.
- Timestamps are *consistent with* transfer (FEVER `latest.pt` finalised 03:48:15Z; `models/module6/x_fact/` created 03:54:08Z) but consistency is not proof.

**I will not claim transfer occurred because a command may have been executed.** The definitive test is a direct tensor comparison (step C1 below), which is cheap, non-destructive and conclusive. Until it passes, **all FEVER→Tamil transfer claims must be dropped from the paper.**

Note also defect **D3**: `--init-from` loads `latest.pt`, not `best.pt` (`cm.load(model)` defaults to `path=None` → `self.latest`). Harmless for the FEVER run *only* because FEVER's `best_step` (14125) equals its final step. It should still be fixed.

---

## PART F — Reranker file check (verified now)

| File | Status |
|---|---|
| `outputs/module6/rerank/x_fact_test/reranked_top5.json` | ✅ **CONFIRMED ABSENT** |
| `outputs/module6/rerank/x_fact_test/reranked_top5.UNTRAINED_DO_NOT_USE.json` | ✅ present, 97,574 B |

**No stop condition triggered.** The original file does not exist. `load_retrieved()` will now correctly fall through to `dense_candidates.json`.

⚠️ **But the rename came too late for the recorded results.** Drive `modifiedTime` shows the rename at **2026-08-21T04:29:14Z**, whereas X-FACT Stage 7 completed at **04:27:22Z** and Stage 8 at ~04:27:57Z. Stage 8 demonstrably read it (`ablation_results.json` records `rerank: true → recall@5 = 0.07503770739064858`, byte-identical to `rerank_metrics.json`). Stage 7 very probably did too, but neither `verification_metrics.json` nor its manifest records `evidence_source`, so I cannot prove it. Step **C3** settles this in ~3 minutes.

---

## PART G — Correction to my previous audit (evidence languages)

In the previous audit I stated that ablations 3/4 (English-only vs multilingual evidence) were "unimplementable" partly because there is no mixed-language evidence pool. **I am revising that.** `src/fact_verification/corpus_builder.py:269` shows evidence records DO carry a `language` field, assigned by `detect_language(e_norm)`.

So the correct position is: **feasibility depends on the actual per-language composition of the X-FACT evidence pool, which has not been measured.** Step **C2b** below measures it (seconds, CPU). Decision rule stated in advance, so it cannot be chosen after seeing results:

- If the minority language is **≥ 5 %** of X-FACT evidence items → run ablations 3/4.
- If **< 5 %** → the ablation is degenerate; **drop it and report why**, rather than reporting a comparison between "all evidence" and "99 % of evidence".

FEVER evidence is expected to be ~100 % `en`, so ablations 3/4 are almost certainly degenerate for FEVER regardless.

---

# 🛑 FINAL REQUIREMENT — the eight-point report

## 1. What is currently CORRECT

| Item | Evidence |
|---|---|
| Label-space separation (FEVER vs X-FACT) | No mapping function exists; `LabelSpace.encode()` raises on unknown; `mapping_between_spaces: false`; `swap_head()` always re-initialises; checkpoints stamped with their own space; results never combined |
| Test-label discipline | `m6_06` has no `"test"` string; `fit_temperature()` raises on `split_name=="test"`; checkpoint selection on validation macro-F1 only; test used only for final scoring |
| Retrieval leakage control | `build_evidence_scope()` + `assert_index_within_scope()` enforced in `m6_02` and `m6_04`; `index_scope.train: [train]` already correct |
| Stage 1–4 outputs for validation/test | 16 directories verified present, `scope_audit: PASS`, accounting balances, `n_excluded_no_gold` reported |
| Retrieval comparison results | Verified exact; BM25 > dense on both corpora (FEVER R@5 0.549 vs 0.323; X-FACT 0.671 vs 0.535) |
| Module 5 protection | `assert_module5_untouched()` brackets every Module 6 stage |
| Calibration machinery | ECE, reliability bins, ternary-search temperature fit — all correct; FEVER ECE 0.335 → 0.073 |
| Batched FAISS search | The earlier `exit -9` OOM fix is in place (`faiss_index.py`, `batch_size=128`) |

## 2. What is INCORRECT

| # | Problem | Severity |
|---|---|---|
| P1 | **Verifiers trained with zero evidence, evaluated with evidence.** `evidence_source: "none"`, `n_with_no_evidence == n_built == 260,032` | 🔴 CRITICAL |
| P2 | **`load_retrieved()` fails silently** — returns `({}, "none")` instead of raising when training evidence is absent | 🔴 CRITICAL |
| P3 | **`build_examples()` hard-codes `evidence_source: "retrieved"`** regardless of reality; already produced one false record | 🔴 HIGH |
| P4 | **X-FACT training manifest missing entirely** — `init_from`, `head_swap`, `n_train`, `n_val`, `claim_only`, `oracle` all unrecorded; transfer unprovable | 🔴 HIGH |
| P5 | **Stage 8 runs no verifier ablations.** All three runners call `evaluate_retrieval()` only; `m6_08` never imports torch. Values are byte-identical to Stage 4 `dense_metrics.json` | 🔴 CRITICAL (mislabelling) |
| P6 | **`resume=True` hard-coded + no per-variant output isolation.** An ablation run today would resume the base checkpoint, skip training entirely (`range(3,3)` is empty), overwrite `latest.pt`, and report the *previous* run's `best_metric` — **with exit code 0** | 🔴 CRITICAL (blocker) |
| P7 | **X-FACT Stage 7 probably consumed the untrained reranker** (evaluated 04:27:22, file renamed 04:29:14); `evidence_source` is not recorded anywhere, so it cannot be confirmed from artefacts | 🔴 HIGH |
| P8 | **FEVER test accuracy 0.3844 < majority baseline 0.4028** (McNemar χ²=8.527, **p=0.0035** — significantly worse than a constant predictor) | ⚠️ must be reported |
| P9 | **X-FACT gain over majority not significant** (0.5928 vs 0.5792; χ²=0.308, **p=0.579**, n=221) | ⚠️ must be reported |
| P10 | **X-FACT calibration regressed** (ECE 0.0531 → 0.0661) | ⚠️ must be reported |
| P11 | `gradient_accumulation_steps: 2` and `use_mixed_precision: true` are **never read** — effective batch is 16, training was fp32 | ⚠️ do not claim |
| P12 | **Top-k sweep is vacuous** — recall is monotone in k, so k=50 wins by construction | ⚠️ relabel |
| P13 | Unweighted cross-entropy under 6.2 : 1 imbalance; `partly true/misleading` never predicted once in 366 eval items | ⚠️ |
| P14 | Single seed (42) despite `seeds: [42, 43, 44]`; no confidence intervals at n=221 | ⚠️ |
| P15 | `--init-from` loads `latest.pt`, not `best.pt` (D3) | ⚠️ LOW |
| P16 | No reranker training script exists; `m6_05` ran with `checkpoint: null` | ❌ scope decision |

## 3. What must be CHANGED

**Code changes (8 items) — all additive; no existing behaviour is removed:**

| # | Change | Rationale |
|---|---|---|
| R1 | **New** `scripts/m6_00_export_train_subset.py` — deterministically reproduce `stratified_subset(train_ex, size, seed)` and write claim ids to `outputs/module6/subsets/{ds}_train_ids.json` | Lets us encode/retrieve only the 50,000 FEVER claims that training actually uses. Provably identical selection: `stratified_subset` depends only on (claim_ids, labels, seed) — **not** on evidence — so attaching evidence cannot change it. Cuts FEVER repair cost ~4×. |
| R2 | `m6_03_encode_claims.py` — add `--claim-ids-file` | Encode only the subset |
| R3 | `m6_04_retrieve.py` — add `--claim-ids-file`, and `--top-k 5` for train | Retrieve only the subset; caps `dense_candidates.json` at ~20 MB instead of **938 MB** (at 4,111 B/claim measured from `fever_validation`, 228,090 claims × 50 candidates would be ~938 MB and would need 3–5 GB RAM in `json.loads`) |
| R4 | `m6_06_train_verifier.py` — add `--run-name`, `--no-resume`, `--evidence-languages`; **raise `SystemExit` if `evidence_source == "none"` and `--claim-only` was not passed**; refuse to start if a `--run-name` checkpoint dir is non-empty and `--resume` was not explicitly given; write the full PART C + PART K manifest | Fixes P1, P2, P6; makes the silent failure impossible to repeat |
| R5 | `m6_07_evaluate_verifier.py` — add `--run-name`; **record `evidence_source` for validation and test**; add weighted-F1; record the full reproducibility block | Fixes P7 (unanswerable today only because this field is absent) and P6 |
| R6 | `src/verification/dataset.py` — set `stats["evidence_source"]` from what was *actually attached*; add `evidence_languages` filter | Fixes P3; enables ablations 3/4 |
| R7 | `src/verification/trainer.py` — add `weighted_f1` to `classification_report`; optional `class_weights` support | PART F requirement; P13 remedy as an extra arm |
| R8 | `m6_08_run_ablations.py` — add a **collector mode** that reads per-variant `verification_metrics.json` and assembles a real verifier-ablation table; relabel the existing three as `stage: retrieval` in the report | Fixes P5, P12 |

**Not changed:** `configs/module6_retrieval_verification.yaml` values stay as-is (the `compute_scope` record must remain accurate). Module 5 is untouched. `src/verification/calibration.py`, `verifier_model.py`, `faiss_index.py`, `index_scope.py` need no changes.

## 4. Which scripts must be modified

```
NEW      scripts/m6_00_export_train_subset.py          (~60 lines)
MODIFY   scripts/m6_03_encode_claims.py                (+8 lines,  --claim-ids-file)
MODIFY   scripts/m6_04_retrieve.py                     (+8 lines,  --claim-ids-file)
MODIFY   scripts/m6_06_train_verifier.py               (~+70 lines, run-name/no-resume/guard/manifest)
MODIFY   scripts/m6_07_evaluate_verifier.py            (~+35 lines, run-name/evidence_source/weighted-F1)
MODIFY   scripts/m6_08_run_ablations.py                (~+50 lines, verifier-ablation collector)
MODIFY   src/verification/dataset.py                   (~+20 lines, honest stats + language filter)
MODIFY   src/verification/trainer.py                   (~+15 lines, weighted_f1 + optional class weights)
NEW      tests/test_verification_repair.py             (~120 lines, guards below)
UNCHANGED  everything under src/fact_verification/ (Module 5, FROZEN)
```

**New regression tests (must pass before any GPU run):**

1. `m6_06` raises when `evidence_source == "none"` and `--claim-only` is not set.
2. `m6_06` refuses to start when a `--run-name` checkpoint dir is non-empty without explicit `--resume`.
3. `stats["evidence_source"] == "none"` when nothing was attached (regression on P3).
4. `stratified_subset` returns an identical id list with and without evidence attached (validates R1).
5. `swap_head()` changes at least one head weight tensor (guards the latent risk in PART 4 of the previous audit).
6. `fit_temperature(split_name="test")` still raises.

## 5. Estimated GPU runtime for each rerun

Anchored on measured data where available: FEVER verifier training took **1,407.79 s** (recorded in its manifest); claim encoding measured at **102–170 claims/s** (`last_shard_seconds: 58.85`); X-FACT training window 03:54:08 → 04:08:12 ≈ **14 min**; X-FACT evaluation 04:24 → 04:27 ≈ **3 min**.

| Step | Device | Estimate | Basis |
|---|---|---|---|
| C0 checkpoint integrity | CPU | **5 min** | 4 × zip header read |
| C1 transfer verification | CPU | **10 min** | 2 × `torch.load` on CPU |
| C2a export train subsets | CPU | **< 2 min** | pure Python over corpus |
| C2b evidence language census | CPU | **< 1 min** | count `by_language` in `evidence.jsonl` |
| C3 re-run X-FACT Stage 7 | GPU | **3 min** | measured |
| C4a `m6_02 --phase train` ×2 | CPU | **5–10 min** | derived by subset selection, **no re-encoding** |
| C4b `m6_03` x_fact train (1,096) | GPU | **~1 min** | 102–170 claims/s |
| C4c `m6_03` fever train (50,000 subset) | GPU | **5–9 min** | 50,000 ÷ (102–170/s) |
| C4c′ *if all 228,090 instead* | GPU | *22–38 min* | same rate |
| C4d `m6_04` x_fact train | CPU | **~1 min** | 1,096 queries × ~7k evidence |
| C4e `m6_04` fever train (50k, top-k 5) | CPU | **15–25 min** | scaled from ~5 min / 15,916 queries |
| C4e′ *if all 228,090 instead* | CPU | *60–90 min* | same scaling |
| C5 retrain FEVER **with evidence** | GPU | **25–40 min** | 23.5 min measured claim-only; longer sequences |
| C6 retrain X-FACT **with evidence** | GPU | **10–15 min** | ~14 min measured |
| C7 evaluate FEVER | GPU | **5–8 min** | 15,916 val + 16,026 test forwards |
| C8 evaluate X-FACT | GPU | **~2 min** | measured |
| C9 ablation variant, X-FACT (train+eval) | GPU | **~15 min each** | |
| C9′ ablation variant, FEVER (train+eval) | GPU | **~35 min each** | |
| C10 Stage 8 collector + verify | CPU | **5 min** | |

**Totals for the recommended scope** (FEVER on the 50,000 subset; 3 ablation variants on X-FACT, 3 on FEVER):
**≈ 1 h diagnostics + prerequisites, ≈ 1 h base retraining + evaluation, ≈ 2.5 h ablations → ~4.5 h GPU.**
Estimates marked *italic* are the more expensive alternative if you choose full-corpus FEVER retrieval (adds ~1.5 h).

⚠️ These are engineering estimates from measured rates, not guarantees. Colab throttling and Drive I/O can double any of them.

## 6. Exact commands I intend to run

**Phase 0 — diagnostics (no writes to any existing artefact)**

```bash
# C0  four-checkpoint zip integrity  (expect sizes 3,336,812,189 / 3,336,813,433 /
#     3,336,811,549 / 3,336,813,561)
for f in models/module6/{fever,x_fact}/{best,latest}.pt; do
  echo "=== $f"; ls -l "$f"
  python -c "import sys,zipfile;p=sys.argv[1]
try: print('  zip OK,',len(zipfile.ZipFile(p).namelist()),'members')
except Exception as e: print('  ZIP FAILED:',type(e).__name__,e)" "$f"
done

# C1  did FEVER -> X-FACT transfer actually happen?
python - <<'PY'
import torch
a=torch.load("models/module6/fever/latest.pt",map_location="cpu",weights_only=False)["model_state"]
b=torch.load("models/module6/x_fact/best.pt", map_location="cpu",weights_only=False)["model_state"]
k="encoder.embeddings.word_embeddings.weight"
print("encoder mean|delta| :", (a[k]-b[k]).abs().mean().item())
print("head keys           :", [n for n in a if n.startswith("head.")])
PY

# C2b  evidence language census (decides whether ablations 3/4 are meaningful)
python -c "
import json,collections
for ds in ('fever','x_fact'):
    c=collections.Counter(json.loads(l)['language']
        for l in open(f'datasets/fact_verification/{ds}/evidence.jsonl',encoding='utf-8'))
    print(ds, dict(c))"
```

**Phase 1 — settle the reranker question (backup first)**

```bash
# C3
cp -r outputs/module6/evaluation/x_fact_test \
      outputs/module6/evaluation/x_fact_test.PRE_REPAIR_AUDIT_EVIDENCE
python scripts/m6_07_evaluate_verifier.py --dataset x_fact --split test
```

**Phase 2 — code repair + tests (CPU, no training)**

```bash
python -m pytest tests/test_verification.py tests/test_verification_repair.py \
                 tests/test_retrieval.py -q
```

**Phase 3 — build the missing TRAIN retrieval**

```bash
# C2a
python scripts/m6_00_export_train_subset.py --dataset fever
python scripts/m6_00_export_train_subset.py --dataset x_fact

# C4a  CPU, derives from existing test-phase embeddings — no re-encoding
python scripts/m6_02_build_index.py --dataset x_fact --phase train
python scripts/m6_02_build_index.py --dataset fever  --phase train

# C4b/C4c  GPU
python scripts/m6_03_encode_claims.py --dataset x_fact --split train
python scripts/m6_03_encode_claims.py --dataset fever  --split train \
       --claim-ids-file outputs/module6/subsets/fever_train_ids.json

# C4d/C4e  CPU
python scripts/m6_04_retrieve.py --dataset x_fact --split train --retriever both --top-k 5
python scripts/m6_04_retrieve.py --dataset fever  --split train --retriever both --top-k 5 \
       --claim-ids-file outputs/module6/subsets/fever_train_ids.json
```

**Phase 4 — retrain the primary verifiers (evidence-aware), preserving the old ones**

```bash
# C5/C6  the mv preserves the claim-only checkpoints AS THE ABLATION-1 ARM
mv models/module6/fever  models/module6/fever.CLAIMONLY_PREREPAIR
mv models/module6/x_fact models/module6/x_fact.CLAIMONLY_PREREPAIR
cp -r outputs/module6/verifier_training/fever \
      outputs/module6/verifier_training/fever.CLAIMONLY_PREREPAIR
cp -r outputs/module6/evaluation/fever_test \
      outputs/module6/evaluation/fever_test.CLAIMONLY_PREREPAIR

python scripts/m6_06_train_verifier.py --dataset fever  --no-resume
python scripts/m6_06_train_verifier.py --dataset x_fact --no-resume \
       --init-from models/module6/fever

# C7/C8
python scripts/m6_07_evaluate_verifier.py --dataset fever  --split test
python scripts/m6_07_evaluate_verifier.py --dataset x_fact --split test
```

**Phase 5 — real verifier ablations (independent dir per variant, no resume)**

```bash
for V in "claim_only:--claim-only" "claim_evidence:" "oracle:--oracle"; do
  NAME=${V%%:*}; FLAG=${V#*:}
  python scripts/m6_06_train_verifier.py --dataset x_fact --run-name $NAME \
         --no-resume --init-from models/module6/fever $FLAG
  python scripts/m6_07_evaluate_verifier.py --dataset x_fact --split test \
         --run-name $NAME $FLAG
done
# identical loop for --dataset fever (without --init-from)
# ablations 3/4 added only if C2b shows the minority language is >= 5 %
```

**Phase 6 — collect, verify, report**

```bash
python scripts/m6_08_run_ablations.py --dataset fever  --split test --collect-verifier-ablations
python scripts/m6_08_run_ablations.py --dataset x_fact --split test --collect-verifier-ablations
python scripts/verify_module6_outputs.py --dataset fever  --split test
python scripts/verify_module6_outputs.py --dataset x_fact --split test
```

Then write `docs/module6/module6_final_audit.md` with all 12 required sections.

## 7. Which existing outputs will be PRESERVED

**Nothing is deleted. Nothing is overwritten in place. Every pre-repair result is retained as audit evidence.**

| Existing artefact | Fate |
|---|---|
| `models/module6/fever/{best,latest}.pt` | → renamed `models/module6/fever.CLAIMONLY_PREREPAIR/` — **kept, becomes the Ablation-1 (claim-only) arm** |
| `models/module6/x_fact/{best,latest}.pt` | → renamed `models/module6/x_fact.CLAIMONLY_PREREPAIR/` — kept, same role |
| `outputs/module6/verifier_training/fever/manifest.json` | → copied to `…/fever.CLAIMONLY_PREREPAIR/` — kept as the primary evidence of P1 |
| `outputs/module6/evaluation/fever_test/` | → copied to `…fever_test.CLAIMONLY_PREREPAIR/` — kept |
| `outputs/module6/evaluation/x_fact_test/` | → copied to `…x_fact_test.PRE_REPAIR_AUDIT_EVIDENCE/` **before** C3 — kept |
| `outputs/module6/ablations/{fever,x_fact}_test/` | **untouched** — still valid as retrieval analyses, only relabelled in prose |
| `outputs/module6/retrieve/*_{validation,test}/` | **untouched** — valid and reused |
| `outputs/module6/{encode_evidence,build_index,encode_claims}/*` | **untouched** — reused; train index derives from these |
| `outputs/module6/rerank/x_fact_test/*` | **untouched**, including `…UNTRAINED_DO_NOT_USE.json` — kept as evidence of P7 |
| `datasets/fact_verification/**` (Module 5) | **FROZEN — never opened for writing**; `assert_module5_untouched()` verifies before/after every stage |
| `docs/module6_audit_2026-08-21.md` | **untouched** — the pre-repair audit of record |

## 8. Which NEW output directories will be created

```
outputs/module6/subsets/                                   fever_train_ids.json, x_fact_train_ids.json
outputs/module6/encode_claims/{fever,x_fact}_train/
outputs/module6/build_index/{fever,x_fact}_train/
outputs/module6/retrieve/{fever,x_fact}_train/
outputs/module6/verifier_training/{fever,x_fact}/          (rewritten — old copied aside first)
outputs/module6/verifier_training/ablations/{fever,x_fact}/{claim_only,claim_evidence,oracle}/
outputs/module6/evaluation/ablations/{fever,x_fact}_test/{claim_only,claim_evidence,oracle}/
outputs/module6/ablations/{fever,x_fact}_test/verifier_ablation_results.json   (new file, same dir)

models/module6/fever/                                      (new evidence-aware checkpoints)
models/module6/x_fact/                                     (new evidence-aware checkpoints)
models/module6/ablations/{fever,x_fact}/{claim_only,claim_evidence,oracle}/

models/module6/fever.CLAIMONLY_PREREPAIR/                  (preserved originals)
models/module6/x_fact.CLAIMONLY_PREREPAIR/
outputs/module6/verifier_training/fever.CLAIMONLY_PREREPAIR/
outputs/module6/evaluation/fever_test.CLAIMONLY_PREREPAIR/
outputs/module6/evaluation/x_fact_test.PRE_REPAIR_AUDIT_EVIDENCE/

docs/module6/module6_repair_plan.md                        (this file)
docs/module6/module6_final_audit.md                        (written after execution)
```

---

## Baselines and statistics already computed (PART I / PART J preview)

Recomputed independently from the stored confusion matrices; both reproduce the stored `accuracy` and `macro_f1` to 4 dp, so the JSON files are internally consistent.

| Dataset | n | Class distribution | Majority acc | Majority macro-F1 | Model acc | Model macro-F1 | Paired test vs majority |
|---|---|---|---|---|---|---|---|
| FEVER test | 16,026 | S 4,682 / R 4,888 / NEI 6,456 | **0.4028** | 0.1914 | **0.3844** | 0.3227 | McNemar b=4,921 c=5,216, **χ²=8.527, p=0.0035** — model significantly **worse** on accuracy |
| X-FACT test | 221 | false 128 / true 62 / partly 31 | 0.5792 | 0.2445 | 0.5928 | 0.3146 | McNemar b=8 c=5, **χ²=0.308, p=0.579** — **not significant** |
| X-FACT validation | 145 | false 92 / true 38 / partly 15 | 0.6345 | 0.2588 | 0.6828 | 0.3796 | not tested (used for model selection — a test here would be circular) |

These are the comparison rows the repaired ablation table will extend with **claim-only / claim+evidence / oracle** arms. **X-FACT train class counts remain unverified** — the training manifest does not exist; `m6_00_export_train_subset.py` will record them.

---

## 🛑 STOP — approval required

No training, no evaluation, no file modification will begin until you approve. Please confirm the three scope decisions raised alongside this document.
