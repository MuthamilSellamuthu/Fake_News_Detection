# Module 6 — Read-Only Rigorous Audit

**Date:** 2026-08-21
**Scope:** Module 6 (evidence retrieval + fact verification), Stages 6 / 7 / 8
**Mode:** READ → INSPECT → VERIFY → REPORT. No file was created, edited, renamed, deleted or executed inside the project. No model was loaded or retrained.
**Project root inspected:** local mirror `D:\SEM7\Fake_News_Detection` (source code) + Drive project owned by `s.muthamil05@gmail.com` (outputs, manifests, checkpoints).

### Auditor limitations — stated up front

I could not `torch.load()` any checkpoint. My sandbox has no `torch`, no GPU, ~2.9 GB RAM, and the checkpoints are ~3.34 GB each. **Every statement about checkpoint binary integrity below is therefore explicitly marked as unverified**, and I give you the exact command to verify it yourself.

I could not read `datasets/fact_verification/x_fact/claims.jsonl` in full (2.5 MB, would have to be streamed; the local mirror's copy is empty). X-FACT *train* class counts are therefore marked unverified. X-FACT *validation* and *test* class counts ARE verified, because they are recoverable exactly from the confusion-matrix supports.

---

## PART 1 — The actual Module 6 pipeline, read from code

### Files inspected

| File | Bytes | Role |
|---|---|---|
| `scripts/m6_06_train_verifier.py` | 9,119 | Stage 6 — verifier training |
| `scripts/m6_07_evaluate_verifier.py` | — | Stage 7 — evaluation + calibration |
| `scripts/m6_08_run_ablations.py` | — | Stage 8 — ablations |
| `src/verification/dataset.py` | — | label spaces, `build_examples`, `to_text_pair` |
| `src/verification/verifier_model.py` | — | encoder + swappable head, `swap_head()` |
| `src/verification/trainer.py` | — | `CheckpointManager`, `VerifierTrainer`, metrics |
| `src/verification/calibration.py` | — | ECE, `fit_temperature` |
| `src/verification/ablations.py` | — | `build_ablation_specs`, `run_all_ablations` |
| `configs/module6_retrieval_verification.yaml` | — | all Module 6 configuration |

### Stage 6 → Stage 7 → Stage 8, as the code actually behaves

**Stage 6 (`m6_06_train_verifier.py`)**

1. Loads Module 5's frozen corpus for one dataset (`fever` or `x_fact`).
2. Calls `load_retrieved(cfg, root, dataset, "train")`. This resolves in strict order:
   `rerank/{ds}_train/reranked_top5.json` → `retrieve/{ds}_train/dense_candidates.json` → `({}, "none")`.
3. Builds train examples via `build_examples(...)`, filters to `e.split == "train"`.
4. Independently builds validation examples with `load_retrieved(..., "validation")`, filters to `e.split == "validation"`.
5. Optionally applies `stratified_subset` (FEVER: 50,000).
6. Builds `Verifier` = `AutoModel(xlm-roberta-base)` + `Dropout→Linear(768,512)→Tanh→Dropout→Linear(512,3)`, mean-pooled.
7. If `--init-from DIR`: `CheckpointManager(DIR).load(model)` then `model.swap_head(...)` — head always re-initialised.
8. Trains with AdamW, linear warmup schedule, `cross_entropy`, grad-clip 1.0, **fp32, batch size 16, no gradient accumulation** (see PART 1 defects).
9. After each epoch: `eval_fn` on validation → `macro_f1` → `is_best` → `CheckpointManager.save(..., is_best=)`.
10. Writes `outputs/module6/verifier_training/{ds}/manifest.json`.

**Stage 7 (`m6_07_evaluate_verifier.py`)**

1. `ckpt_dir = --checkpoint or models/module6/{ds}` (a **directory**, not a file).
2. `cm.load(model, path=cm.best if cm.best.is_file() else None)` → prefers `best.pt`, falls back to `latest.pt`. Exits if neither loads.
3. `logits_for("validation")` — **re-runs the validation set through the model, using `load_retrieved(..., "validation")` evidence.**
4. `fit_temperature(val_logits, val_labels, split_name="validation")`.
5. `logits_for(args.split)` — test set, using `load_retrieved(..., "test")` evidence.
6. `preds = softmax(logits, T).argmax(1)` → `classification_report` → `calibrate_and_evaluate` → per-language breakdown.
7. Writes `evaluation/{ds}_{split}/verification_metrics.json` + `manifest.json`.

**Stage 8 (`m6_08_run_ablations.py`)**

1. Loads `dense_candidates.json`, `bm25_candidates.json`, `reranked_top5.json` for `{ds}_{split}`.
2. Registers exactly **three** runners, all of which call only `evaluate_retrieval(...)`:
   `bm25_vs_dense`, `top_k_sweep`, `retrieval_only_vs_retrieval_rerank`.
3. Every spec in `build_ablation_specs(cfg)` **without** a registered runner is appended to `pending`.
4. Writes `ablations/{ds}_{split}/ablation_results.json`.

**No verifier model is ever constructed or loaded in Stage 8.** Verified by reading the whole file; there is no `build_verifier`, no `CheckpointManager`, no `torch` import.

### Defects found in the workflow itself

| # | Defect | Evidence | Severity |
|---|---|---|---|
| D1 | `verifier.gradient_accumulation_steps: 2` is declared in config but **read by no Python file in the project**. `grep -rn "gradient_accumulation" scripts/ src/ tests/` → 0 hits. Effective batch size is **16**, not 32. | config line 79 vs grep | Medium — the paper must not claim effective batch 32 |
| D2 | `runtime.use_mixed_precision: true` is likewise never used in Module 6 (only `src/embeddings/feature_extraction.py` reads it). Training ran fp32. | grep | Low — but do not claim AMP |
| D3 | `--init-from` loads `CheckpointManager.latest` (`latest.pt`), **not** `best.pt`, because `cm.load(model)` is called with `path=None`. | `m6_06` line ~131 vs `trainer.py` `load()` default | Medium — harmless here only by coincidence (FEVER's best_step == final step) |
| D4 | Neither `m6_06` nor `m6_07` has an `--output-dir` / `--run-name` flag. Checkpoint dir is hard-wired to `models/module6/{dataset}` and evaluation dir to `evaluation/{dataset}_{split}`. | argparse blocks | **Critical — blocks Stage 8 verifier ablations, see PART 12** |
| D5 | `VerifierTrainer.train(..., resume=True)` is hard-coded. Re-running `m6_06` for the same dataset with *different flags* silently **resumes** the previous run rather than training a fresh variant, and overwrites `best.pt`/`latest.pt`. | `m6_06` final call + `trainer.py` resume block | **Critical — see PART 12** |
| D6 | `build_examples()` hard-codes `stats["evidence_source"] = "retrieved"` whenever `claim_only=False and use_gold_evidence=False`, *regardless of whether a single evidence item was actually attached*. | `dataset.py`, `stats` dict initialiser | **High — this field is actively misleading and already produced a false record (PART 6)** |

---

## PART 2 — Checkpoint audit

All four files exist in Drive. Sizes and modification times (Drive metadata, authoritative):

| Path | Exists | Bytes | Drive modifiedTime |
|---|---|---|---|
| `models/module6/fever/best.pt` | ✅ | **3,336,812,189** | 2026-08-21T03:47:37Z |
| `models/module6/fever/latest.pt` | ✅ | **3,336,813,433** | 2026-08-21T03:48:15Z |
| `models/module6/x_fact/best.pt` | ✅ | **3,336,811,549** | 2026-08-21T04:04:47Z |
| `models/module6/x_fact/latest.pt` | ✅ | **3,336,813,561** | 2026-08-21T04:08:12Z |

`models/module6/fever/checkpoint_meta.json` (325 B) and `models/module6/x_fact/checkpoint_meta.json` (323 B) both exist and were read in full.

### Metadata actually recorded

| Field | fever | x_fact |
|---|---|---|
| `epoch` | 3 | 3 |
| `global_step` | 14125 | 207 |
| `best_metric` | 0.3452466539131494 | **0.37955704864999373** |
| `best_step` | 14125 | **138** |
| `has_best` | true | true |
| `config.encoder_name` | xlm-roberta-base | xlm-roberta-base |
| `config.num_labels` | 3 | 3 |
| `config.hidden_dim` | 512 | 512 |
| `config.max_length` | 256 | 256 |
| `config.label_space` | **"fever"** | **"x_fact"** |
| `config.seed` / `epochs` | 42 / 3 | 42 / 3 |

**Label-space configuration is correct in both checkpoints.** Each checkpoint is stamped with its own label space; there is no cross-contamination at the metadata level.

### `torch.load()` — CANNOT INDEPENDENTLY VERIFY

I have no `torch` and cannot hold 3.34 GB. **I did not confirm and did not refute** the reported
`PytorchStreamReader failed reading zip archive: failed finding central directory`.

What I *can* say from evidence:

1. The size is consistent with a **complete** payload. XLM-R base ≈ 278 M params × 4 B = 1.11 GB. `CheckpointManager.save()` stores `model_state` + AdamW `optimizer_state` (two moment tensors) + scheduler + RNG + config ≈ 3 × 1.11 GB ≈ **3.34 GB**. All four files match this to within ~2 KB of each other. A truncated file would almost certainly be shorter.
2. **The FEVER checkpoint demonstrably loaded successfully at 04:24 on 2026-08-21.** `evaluation/fever_test/manifest.json` has `"status": "complete"`, `written_at: 2026-08-21T04:24:40`, and `m6_07` raises `SystemExit` if the checkpoint does not load. FEVER test metrics could not exist otherwise.
3. Therefore the "corruption" you observed **appeared after 04:24:40**, i.e. after a working load.

**Most probable cause (hypothesis, not established fact):** the Google Drive FUSE mount served a partially-materialised copy of a 3.3 GB file. `failed finding central directory` is exactly the signature of reading a zip whose tail bytes have not been streamed down yet. This is a *mount* failure, not necessarily an *on-disk* failure.

**Non-destructive verification you should run before concluding anything (PART 17, C0).**

### Is each checkpoint usable by Stage 7?

- `x_fact/best.pt` — **YES, empirically.** `evaluation/x_fact_test/manifest.json` exists, `status: complete`, written 04:27:22.
- `fever/best.pt` — **WAS usable at 04:24:40.** Current usability **unknown**.
- `latest.pt` for both — Stage 7 only touches `latest.pt` if `best.pt` is missing, so neither was exercised. Usability **unknown**.

---

## PART 3 — X-FACT training result

### What is recorded, and what is NOT

**⚠️ `outputs/module6/verifier_training/x_fact/manifest.json` DOES NOT EXIST.**

I listed the children of `outputs/module6/verifier_training/` (Drive folder `1vLniMPkGl07gAXezv4jCXwdBQNXVkYTI`). It contains exactly one child: `fever`. There is **no `x_fact` subdirectory at all**.

Consequences — all of the following are **NOT RECORDED ANYWHERE** for the X-FACT run:

- `n_train`, `n_val` (your stated 1,096 / 145 are unverified)
- `evidence_source`
- `claim_only` / `oracle_gold_evidence` flags
- **`init_from`** — so transfer learning is unverifiable (PART 5)
- **`head_swap`** — so head re-initialisation is unverifiable (PART 5)
- full `history` (per-epoch confusion matrices)
- wall-clock seconds

This is a **reproducibility failure**, not merely a missing file. Either `m6_06` did not reach `write_manifest()` (i.e. it crashed after the final checkpoint save but before the manifest write), or the manifest was never synced to Drive.

### What IS verified

From `models/module6/x_fact/checkpoint_meta.json`:

- `best_metric = 0.37955704864999373` — **matches your figure 0.379557 exactly.**
- `best_step = 138`, `global_step = 207`, `epoch = 3`.

**Does `best.pt` correspond to the best validation checkpoint? YES.** Reading `trainer.py`: `best.pt` is written only inside `if improved:` via `self.ckpt.save(..., is_best=improved)`, using the same `payload` as `latest.pt` at that instant. `best_step = 138 ≠ global_step = 207`, and Drive shows `best.pt` modified at **04:04:47** while `latest.pt` was modified at **04:08:12** — 3½ minutes later. The two files are genuinely different snapshots and `best.pt` is the earlier, better one.

**Step arithmetic is consistent** with your stated data: 1,096 train / batch 16 = 68.5 → 69 steps/epoch. 69 × 2 = 138 (end of epoch index 1, i.e. the 2nd epoch). 69 × 3 = 207. Best was at the end of epoch 1; epoch 2 did not improve; `patience = 2` so no early stop; 3 epochs ran. All internally consistent. **Your "epoch = 1" figure is correct** (0-indexed).

### Bias toward the `false` class — quantified

Validation confusion matrix `[[91,1,0],[30,8,0],[15,0,0]]` (rows = gold false / true / partly; supports 92 / 38 / 15, total 145). I recomputed metrics from it:

| | model | majority-class baseline |
|---|---|---|
| accuracy | **0.6828** | 0.6345 |
| macro-F1 | **0.3796** | 0.2588 |

Predicted-class distribution: **false 136, true 9, `partly true/misleading` 0.** The third class is never predicted, at any point, on either split.

### Diagnosis — it is a combination, in this order of magnitude

1. **Insufficient training data — dominant.** 1,096 examples fine-tuning a 278 M-parameter encoder is roughly 4,000 gradient updates' worth of signal spread over 278 M weights. With `partly true/misleading` at ~10–14 % of the data, the third class has on the order of **110–150 training examples**. That is below the practical floor for a class this semantically subtle in a cross-encoder setting.
2. **Class imbalance — strong contributor.** Plain `cross_entropy` with no class weighting and no balanced sampler (verified: `m6_06` uses `torch.nn.functional.cross_entropy(logits, labels)` with no `weight=` argument, and `DataLoader(..., shuffle=True)` with no sampler). Under imbalance this converges to the majority prior for the rare class.
3. **Evidence is absent from training — structural, and the most damaging.** See PART 6. The model had *no evidence at all* during training, so it can only learn claim-side surface priors. Under that constraint, "predict `false`" is very close to the Bayes-optimal claim-only rule.
4. **Optimisation failure — no evidence of this.** Loss decreased, validation macro-F1 rose from the majority baseline (0.2588) to 0.3796, and a non-trivial number of `true` predictions were made. The optimiser worked; the problem is the learning signal, not the optimiser.
5. **Overfitting — mild and secondary.** Best at epoch 1 of 3, no improvement after. Consistent with early saturation, but the gap is not dramatic. Not the primary cause.

**Nothing here was improved, smoothed, or reframed. The result stands as measured.**

---

## PART 4 — Label-space correctness ✅ VERIFIED CORRECT

This is the strongest part of Module 6. Every requested property holds.

| Check | Result | Evidence |
|---|---|---|
| No function maps FEVER ↔ X-FACT labels | ✅ | `dataset.py` contains no mapping function of any kind; `LABEL_SPACES` is a flat dict of two independent `LabelSpace` objects |
| Unknown labels raise instead of defaulting | ✅ | `LabelSpace.encode()` raises `KeyError` with an explicit message; no `.get(label, 0)` fallback anywhere |
| X-FACT stored verbatim | ✅ | `X_FACT_LABELS = ["false", "true", "partly true/misleading"]` — the literal published string, including the slash |
| Config forbids mapping | ✅ | `label_spaces.mapping_between_spaces: false` |
| Disjointness assertable | ✅ | `assert_label_spaces_disjoint()` exists and raises on overlap |
| Head is never reused across spaces | ✅ | `swap_head()` unconditionally calls `_make_head()`; there is no code path that keeps the old head |
| Results never combined | ✅ | Both `verification_metrics.json` files carry: *"FEVER (entailment) and X-FACT (veracity) are separate tasks with separate label spaces; results are never combined."* Separate output directories, separate manifests, no aggregate file exists |
| Checkpoints stamped with their space | ✅ | `checkpoint_meta.config.label_space` = `"fever"` / `"x_fact"` respectively |

**One latent risk to keep in mind (not a current defect):** both spaces have exactly 3 classes and identical `hidden_dim`, so a FEVER head loads into an X-FACT-shaped model **without a shape error**. The only thing preventing FEVER entailment semantics from surviving into X-FACT is the `swap_head()` call in `m6_06`. There is no assertion that would fail if that call were ever removed. `tests/test_verification.py` should carry a test that a head-weight tensor differs before/after `swap_head`. (Reporting only — I did not add it.)

---

## PART 5 — Initialization / transfer learning

### What the code does — CORRECT BY DESIGN

```python
if args.init_from:
    cm = CheckpointManager(Path(args.init_from))
    state = cm.load(model)                                   # loads ENTIRE state dict
    swap = model.swap_head(len(space), space.name, hidden_dim) # head thrown away
    manifest["head_swap"] = swap
```

Sequence, precisely:

1. `model` is built with `num_labels = 3` for the **X-FACT** space.
2. `cm.load(model)` calls `model.load_state_dict(payload["model_state"])` — this loads **both** the FEVER encoder **and** the FEVER head. It succeeds silently because the shapes coincide (3 classes, 512 hidden).
3. `swap_head()` then replaces `self.head` with a **freshly constructed `nn.Sequential`**. New `nn.Linear` layers get PyTorch default (Kaiming-uniform) initialisation. **No FEVER head weight survives.**
4. `optimizer` is constructed *after* this, on `model.parameters()`, so the new head is optimised from scratch. `cm.load(model)` was called with `optimizer=None`, so **FEVER optimizer state is not carried over** — correct.

Encoder weights: carried over. Label space: correct. Number of labels: correct. **The design is exactly what you specified and the implementation matches it.**

### ⚠️ BUT — this was NOT VERIFIED FOR THE ACTUAL X-FACT RUN

`manifest["init_from"]` and `manifest["head_swap"]` are written **only** to `verifier_training/x_fact/manifest.json`, **which does not exist**. `checkpoint_meta.json` does not record `init_from`.

**I therefore cannot confirm that `--init-from models/module6/fever` was actually passed.** The X-FACT model may have been trained from the raw HuggingFace `xlm-roberta-base` checkpoint. Circumstantial support for the intended sequence: the `x_fact` checkpoint directory was created at 2026-08-21T03:54:08Z, ~6 minutes after FEVER's `latest.pt` was finalised at 03:48:15Z. That is consistent with, but does not prove, transfer.

**Consequence: you cannot currently claim FEVER → X-FACT transfer learning in the paper.** It is unverified.

### Did the corrupted FEVER checkpoint affect the X-FACT model?

**Almost certainly not**, and here is the reasoning:

- If `--init-from` was passed and the FEVER checkpoint had been unreadable at 03:54, `torch.load` would have raised `PytorchStreamReader ...` and `m6_06` would have **crashed immediately**, before creating any X-FACT checkpoint. X-FACT checkpoints exist. So at 03:54 the FEVER checkpoint was readable — consistent with the FEVER evaluation succeeding later at 04:24.
- A partially-corrupted-but-loadable `torch.save` payload is not a realistic failure mode: the zip either opens or it doesn't.

**What the X-FACT checkpoint contains (from the code, high confidence):** an XLM-R base encoder (either FEVER-fine-tuned or HF-pretrained — unresolved), plus a 3-class X-FACT head trained from random init for 2 epochs, plus its own AdamW/scheduler/RNG state, stamped `label_space: "x_fact"`.

**How to resolve this definitively (non-destructive, ~2 min):** load `models/module6/fever/latest.pt` and `models/module6/x_fact/best.pt` on CPU and compare a single encoder tensor, e.g. `encoder.embeddings.word_embeddings.weight`. If X-FACT was initialised from FEVER *and* the encoder was fine-tuned for 3 epochs, the tensors will be **close but not identical**. If X-FACT started from raw HF weights, compare against a freshly downloaded `xlm-roberta-base` instead — an exact match on a *frozen* tensor would be diagnostic. Command given in PART 17 (C1).

---

## PART 6 — Evidence usage: **THE CENTRAL FINDING OF THIS AUDIT** 🔴

### The retrieval directory listing

`outputs/module6/retrieve/` (Drive folder `1JwL9oa7Dyy1DFwC8b6QOO_gQxGekIbTv`) contains **exactly four** subdirectories:

```
fever_test          dense_candidates.json  bm25_candidates.json  (+ metrics, manifest)
fever_validation    dense_candidates.json  bm25_candidates.json  (+ metrics, manifest)
x_fact_test         dense_candidates.json  bm25_candidates.json  (+ metrics, manifest)
x_fact_validation   dense_candidates.json  bm25_candidates.json  (+ metrics, manifest)
```

**There is no `fever_train`. There is no `x_fact_train`.**

`outputs/module6/rerank/` contains **exactly one** subdirectory: `x_fact_test`.

This is not an accident of a failed run — it is what the configuration instructs:

```yaml
retrieval:
  evaluate_splits: [validation, test]     # train is NOT retrieved
reranking:
  rerank_splits: [validation, test]       # train is NOT reranked
```

### Therefore, resolved per (dataset, split) — from the actual filesystem, not filenames

| Dataset | Split | `load_retrieved()` resolves to | Evidence actually attached |
|---|---|---|---|
| FEVER | **train** | `("none")` — no file at either path | ❌ **NONE — claim only** |
| FEVER | validation | `dense_candidates.json` | ✅ dense (LaBSE top-5) |
| FEVER | test | `dense_candidates.json` (no rerank dir) | ✅ dense (LaBSE top-5) |
| X-FACT | **train** | `("none")` — no file at either path | ❌ **NONE — claim only** |
| X-FACT | validation | `dense_candidates.json` | ✅ dense (LaBSE top-5) |
| X-FACT | test | `reranked_top5.json` **if it existed at run time** — see PART 7 | ⚠️ **probably UNTRAINED-reranked, see below** |

### The manifest proves it, twice over

`outputs/module6/verifier_training/fever/manifest.json`, read in full:

```json
"evidence_source": "none",
"claim_only": false,
"oracle_gold_evidence": false,
"example_stats": {
  "n_input": 260032,
  "n_built": 260032,
  "n_with_no_evidence": 260032,        ←  100 % of examples had ZERO evidence
  "evidence_source": "retrieved"       ←  FALSE. Hard-coded string (defect D6).
}
```

`n_with_no_evidence == n_built == 260032`. **Every single example the FEVER verifier ever saw during training was claim-text-only.** The top-level `"evidence_source": "none"` is the honest field; the nested `"retrieved"` is the hard-coded lie from `dataset.py` (defect D6). Do not cite the nested field anywhere.

### What this means

**The FEVER verifier was trained claim-only, and then evaluated with dense-retrieved evidence appended.** This is a **train/test input-distribution mismatch**, not merely a missing ablation:

- At training time the tokenizer received `(claim, "")` — `to_text_pair()` returns `separator.join([])` = `""` for the second segment.
- At evaluation time it received `(claim, "ev1 </s> ev2 </s> ... ev5")` — up to 5 evidence passages, truncated at 256 tokens.

The model was never trained to read the second segment. Feeding it 5 passages at test time is out-of-distribution input, and it very plausibly **degrades** performance rather than helping. This is a strong candidate explanation for why FEVER test accuracy (0.3844) is **below the majority-class baseline** (0.4028) — see PART 9.

The same applies to X-FACT: trained claim-only, evaluated with evidence.

**Answering your question directly:** the trained verifier used **claim only** during training. Not retrieved evidence, not reranked evidence, not dense evidence, not BM25 evidence, not oracle evidence. Evidence appears **only at evaluation time**, and only in a form the model was never trained to consume.

---

## PART 7 — Reranker

### Verified state of the files

`outputs/module6/rerank/x_fact_test/` (Drive folder `1YuGnOyXPBWAkq8x_WC3HrUeu2-AlrtoL`) contains exactly:

| File | Bytes | Drive modifiedTime |
|---|---|---|
| `reranked_top5.UNTRAINED_DO_NOT_USE.json` | 97,574 | **2026-08-21T04:29:14Z** |
| `rerank_00000.json` | 120,982 | 2026-08-18T17:57:34Z |
| `rerank_metrics.json` | 769 | 2026-08-18T17:57:34Z |
| `manifest.json` | 621 | 2026-08-20T16:31:17Z |

- ✅ **`reranked_top5.json` DOES NOT exist.** Confirmed.
- ✅ **`reranked_top5.UNTRAINED_DO_NOT_USE.json` DOES exist.** Confirmed.
- ✅ **Dense candidates are therefore the fallback — from now on.** `load_retrieved()` will miss the renamed file and fall through to `dense_candidates.json`.

### Was a reranker ever trained? NO.

`outputs/module6/rerank/x_fact_test/manifest.json`, read in full:

```json
"stage": "rerank", "dataset": "x_fact", "split": "test",
"candidates_from": "dense", "checkpoint": null,       ←  NO TRAINED RERANKER
"metrics": { "recall@1": 0.0160633, "recall@5": 0.0750377,
             "mrr": 0.1463801, "ndcg@5": 0.0748922 },
"total_seconds": 0.0
```

`"checkpoint": null` — the cross-encoder was an **untrained** `xlm-roberta-base` with a randomly-initialised scoring head. There is **no reranker checkpoint anywhere in the project**; I searched Drive for `.pt` files and only the four Module 6 verifier checkpoints (and old Module 3/4 artefacts owned by the other account) exist. **`scripts/m6_05_rerank.py` has no training counterpart — there is no `m6_05_train_reranker.py`.**

The damage is quantified: the untrained reranker took dense candidates at **Recall@5 = 0.5347** and reordered them down to **Recall@5 = 0.0750** — a **7.1× degradation**, which is what random reordering of a 50-candidate list into a top-5 would produce.

### 🔴 The rename came TOO LATE

Timeline reconstructed from Drive `modifiedTime` (all 2026-08-21):

| Time | Event |
|---|---|
| 03:48:15 | FEVER Stage 6 training manifest written |
| 03:54:08 | `models/module6/x_fact/` created — X-FACT Stage 6 begins |
| 04:04:47 | X-FACT `best.pt` written (step 138) |
| 04:08:12 | X-FACT `latest.pt` written (step 207) — Stage 6 ends |
| 04:24:40 | **FEVER Stage 7 evaluation complete** |
| 04:27:22 | **X-FACT Stage 7 evaluation complete** |
| 04:27:52 | Stage 8 ablations begin (`ablations/fever_test` created) |
| 04:27:57 | `ablations/x_fact_test` created |
| **04:29:14** | **`reranked_top5.json` renamed to `...UNTRAINED_DO_NOT_USE.json`** |

**The rename happened ~2 minutes AFTER X-FACT Stage 7 finished and AFTER Stage 8 ran.**

Two consequences:

1. **X-FACT Stage 7 test evaluation very probably consumed the untrained reranker's top-5** (Recall@5 = 0.075), not dense candidates. `load_retrieved()` checks `reranked_top5.json` **first**, and at 04:27:22 that file still had its original name.
2. **X-FACT Stage 8 `retrieval_only_vs_retrieval_rerank` definitively consumed it** — `ablation_results.json` records `rerank: true → recall@5 = 0.07503770739064858`, byte-identical to the rerank manifest. Proof it read the file.

**Confidence on point 1: high, but not certain.** Drive's `modifiedTime` is updated by a rename, but I cannot rule out that the field reflects some other metadata touch. Neither `evaluation/x_fact_test/manifest.json` nor `verification_metrics.json` records `evidence_source` — a logging gap that should be fixed (`m6_07` should write the `source` string returned by `load_retrieved`).

**Resolution is cheap:** re-run `m6_07 --dataset x_fact --split test` now. `load_retrieved` will fall back to dense. If the metrics change, the original run used the untrained reranker and must be discarded. See PART 17 (C3).

**Nothing was renamed, deleted or modified by me.**

---

## PART 9 — Actual test results (verified against the JSON, not the notebook)

Both files read in full from Drive. **Every value you quoted is confirmed exactly.**

### FEVER — `outputs/module6/evaluation/fever_test/verification_metrics.json` (4,025 B)

```
dataset fever · split test · task entailment · n = 16,026
label_space  ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
claim_only false · oracle false
```

| Metric | Value |
|---|---|
| accuracy | **0.3844377885935355** |
| macro-F1 | **0.3226643340307262** |
| precision_macro | 0.577359124951867 |
| recall_macro | 0.41280276031997415 |

| Class | P | R | F1 | support |
|---|---|---|---|---|
| SUPPORTS | 0.3337 | **0.9393** | 0.4924 | 4,682 |
| REFUTES | **0.8408** | **0.1070** | 0.1898 | 4,888 |
| NOT ENOUGH INFO | 0.5576 | 0.1921 | 0.2857 | 6,456 |

Confusion matrix (rows gold, cols pred):
```
              S      R    NEI
SUPPORTS   4398     32    252
REFUTES    3633    523    732
NEI        5149     67   1240
```
Predicted distribution: **SUPPORTS 13,180 / REFUTES 622 / NEI 2,224.** The model dumps 82 % of all predictions into SUPPORTS.

Calibration: temperature **3.921345356549542** (fitted on validation, `nll 1.3541 → 1.0475`), **ECE 0.3345828656872451 → 0.07253440164936235**.
Per-language: `en` only, identical to overall (FEVER is monolingual English).

### X-FACT — `outputs/module6/evaluation/x_fact_test/verification_metrics.json` (3,769 B)

```
dataset x_fact · split test · task veracity · n = 221
label_space  ["false", "true", "partly true/misleading"]
claim_only false · oracle false
```

| Metric | Value |
|---|---|
| accuracy | **0.5927601809954751** |
| macro-F1 | **0.3146223146223146** |
| precision_macro | 0.3666666666666667 |
| recall_macro | 0.363323252688172 |

| Class | P | R | F1 | support |
|---|---|---|---|---|
| false | 0.6000 | **0.9609** | 0.7387 | 128 |
| true | 0.5000 | 0.1290 | 0.2051 | 62 |
| partly true/misleading | **0.0000** | **0.0000** | **0.0000** | 31 |

Confusion matrix:
```
                    false  true  partly
false                 123     5       0
true                   54     8       0
partly true/mislead    28     3       0
```
Predicted distribution: **false 205 / true 16 / partly true/misleading 0.**

Calibration: temperature **0.8407645736244652** (`nll 0.8122 → 0.8064`), **ECE 0.05307184303697499 → 0.06608538995717593.**
Per-language: `ta` only, identical to overall (Tamil subset only).

### ⚠️ Three things you must state in the paper

**(a) FEVER is below the majority-class baseline on accuracy.** I recomputed both from the confusion matrix:

| FEVER test | model | majority-class (always NEI) |
|---|---|---|
| accuracy | **0.3844** | **0.4028** |
| macro-F1 | 0.3227 | 0.1914 |

McNemar (continuity-corrected) on model vs. majority: b = 4,921, c = 5,216, **χ² = 8.527, p = 0.0035.** The model is **significantly worse than a constant predictor** on accuracy, while being better on macro-F1. Both facts must be reported.

**(b) X-FACT's gain over the majority baseline is not statistically significant.**

| X-FACT test | model | majority-class (always `false`) |
|---|---|---|
| accuracy | 0.5928 | 0.5792 |
| macro-F1 | 0.3146 | 0.2445 |

McNemar: b = 8, c = 5, **χ² = 0.308, p = 0.579.** On n = 221 the accuracy improvement is indistinguishable from noise. The macro-F1 gain (+0.070) comes entirely from the 8 correctly-predicted `true` items and needs a bootstrap CI before being reported as an improvement.

**(c) X-FACT's calibration got WORSE.** ECE 0.0531 → **0.0661**. The fitted temperature is 0.8408 < 1, i.e. the validation set said the model was *under*-confident, so scaling *sharpened* it — and that sharpening did not transfer to test. This is a legitimate, reportable negative result: **temperature scaling helped FEVER dramatically (4.6× ECE reduction) and hurt X-FACT (1.24× ECE increase).** Do not report "calibration improved" as a general Module 6 finding.

All arithmetic above was recomputed independently from the stored confusion matrices and reproduces the stored `accuracy`/`macro_f1` fields to 4 decimal places. **The JSON files are internally consistent.**

---

## PART 10 — Stage 8 ablation audit

Both `ablation_results.json` files read in full. **Every value you quoted is confirmed exactly.**

### FEVER — `outputs/module6/ablations/fever_test/ablation_results.json` (5,922 B)

| Ablation | Variant | recall@k | mrr | n_scored | n_excluded_no_gold |
|---|---|---|---|---|---|
| bm25_vs_dense | bm25 | **0.5490603768859842** | 0.4242277 | 15,995 | 31 |
| bm25_vs_dense | dense | 0.3226920615073163 | 0.2581291 | 15,995 | 31 |
| | **best = bm25, delta = 0.22636831537866792** | | | | |
| top_k_sweep | k=1 | 0.182363238512035 | | | |
| | k=5 | 0.3226920615073163 | | | |
| | k=10 | 0.37927335923428457 | | | |
| | k=20 | 0.433775435776061 | | | |
| | k=50 | **0.5066390538710014** | | | |
| | **best = k:50, delta = 0.3242758153589664** | | | | |
| retrieval_only_vs_rerank | rerank=false | 0.3226920615073163 | | | |
| | rerank=true | **`{"error": "candidates not available; run the earlier stage"}`** | | | |
| | best_variant = **null**, best_value = **null**, delta = 0.0 | | | | |

Note `"macro_f1": null` on every `bm25_vs_dense` result. The code writes it explicitly.

### X-FACT — `outputs/module6/ablations/x_fact_test/ablation_results.json` (5,970 B)

| Ablation | Variant | recall@k | mrr | n_scored |
|---|---|---|---|---|
| bm25_vs_dense | bm25 | **0.6705882352941176** | 0.9684874 | 221 |
| bm25_vs_dense | dense | 0.5346907993966817 | 0.9621041 | 221 |
| | **best = bm25, delta = 0.13589743589743586** | | | |
| top_k_sweep | k=1 | 0.2064102564102564 | | |
| | k=5 | 0.5346907993966817 | | |
| | k=10 | 0.6111613876319758 | | |
| | k=20 | 0.6747360482654601 | | |
| | k=50 | **0.736500754147813** | | |
| | **best = k:50, delta = 0.5300904977375567** | | | |
| retrieval_only_vs_rerank | rerank=false | 0.5346907993966817 | | |
| | rerank=true | **0.07503770739064858** (untrained reranker — PART 7) | | |
| | best_variant = null, best_value = null, delta = 0.0 | | | |

### Status of the four ablations you asked about

| # | Ablation | Status | Why |
|---|---|---|---|
| 1 | claim-only vs claim+evidence | ⏳ **PENDING** | Recorded in `skipped` as *"no runner registered for this ablation"* and in `pending` as *"requires a trained verifier checkpoint per variant; run m6_06/m6_07 with the corresponding flags"*. `m6_08` registers only 3 runners; this is not one of them. **Blocked by defects D4/D5 — see PART 12.** |
| 2 | English-only vs multilingual evidence | ⏳ **PENDING — and currently unimplementable** | Same `skipped`/`pending` entries. Worse: **there is no `--evidence-languages` flag in `m6_06` or `m6_07`**, and `build_examples()` has no language filter on evidence. Requires new code. And it is not meaningful on the current corpora anyway — FEVER evidence is 100 % `en`, X-FACT Tamil evidence is 100 % `ta`; there is no mixed-language evidence pool to ablate over. |
| 3 | Oracle gold evidence | ⏳ **PENDING** | Same `skipped`/`pending` entries. The `--oracle` flag exists and `build_examples(use_gold_evidence=True)` works, so this one is *runnable* once D4/D5 are fixed. |
| 4 | Retrieval-only vs retrieval+rerank | ❌ **FAILED / INVALID for both datasets** | FEVER: `rerank=true` errored — `rerank/fever_test/` does not exist, reranking was never run for FEVER. X-FACT: it *ran*, but against an **untrained** reranker (`checkpoint: null`), so the "result" (0.0750 vs 0.5347) measures random reordering, not reranking. `best_variant` is `null` in both files — the code correctly declined to pick a winner. **This number must not appear in the paper as a reranking result.** |

---

## PART 11 — 🔴 ABLATION METHODOLOGY: these are NOT verifier ablations

**This is the most consequential methodological finding after PART 6.**

### Proof from the code

All three registered runners in `m6_08_run_ablations.py` funnel through one function:

```python
def score(cands, k):
    if not cands:
        return {"error": "candidates not available; run the earlier stage"}
    m = evaluate_retrieval(cands, gold, (k,), ndcg_k=min(k, 10))
    return {f"recall@{k}": ..., "mrr": ..., "n_scored": ..., "n_excluded_no_gold": ...}

runners = {
    "bm25_vs_dense":                      lambda v: {**score(...), "macro_f1": None, ...},
    "top_k_sweep":                        lambda v: {...score(dense, k)...},
    "retrieval_only_vs_retrieval_rerank": lambda v: score(reranked if v["rerank"] else dense, 5),
}
```

`m6_08` **never imports `torch`, never imports `build_verifier`, never imports `CheckpointManager`, and never opens a `.pt` file.** No model is loaded. No inference is run. No verifier is retrained. The script reads Stage 4/5 JSON candidate lists and scores them against the gold-evidence map.

### Proof from the numbers

I read `outputs/module6/retrieve/fever_test/dense_metrics.json` (Stage 4 output, written 2026-08-20T15:29:37, **two days before** the ablations ran):

```json
"overall": { "recall@1": 0.182363238512035,   "recall@5": 0.3226920615073163,
             "recall@10": 0.37927335923428457, "recall@50": 0.5066390538710014,
             "mrr": 0.25812912255842085 }
```

Compare to the "ablation" results: **k=1 → 0.182363238512035, k=5 → 0.3226920615073163, k=10 → 0.37927335923428457, k=50 → 0.5066390538710014.** Byte-for-byte identical. The "top-k ablation" is a re-print of the Stage 4 retrieval metrics file. Same for the dense arm of `bm25_vs_dense`.

Every ablation result also carries `"stage": "retrieval"` (from `build_ablation_specs`) and `"macro_f1": null`.

### What each Stage 8 result actually measures

| Result | What it measures | What it does NOT measure |
|---|---|---|
| `bm25_vs_dense` | Recall@5 of two **candidate generators** against gold-evidence ids | Any effect on verification accuracy or macro-F1 |
| `top_k_sweep` | Recall@k of the **dense retriever** at k ∈ {1,5,10,20,50} | Any effect on the verifier |
| `retrieval_only_vs_retrieval_rerank` | Recall@5 before vs after an **untrained** cross-encoder shuffle | Whether reranking helps verification, or whether a *trained* reranker helps anything |

### The top-k sweep is additionally **vacuous as a design comparison**

Recall@k is **monotone non-decreasing in k** by construction — a larger candidate list is a superset. `k=50` will *always* win. The script's own note says so: *"Recall is monotone non-decreasing in k."* Yet the result is reported as `best_variant: {"top_k": 50}, delta: 0.324`, which invites the reader to conclude that k=50 is a validated design choice. **It is a tautology, not a finding.** The informative quantity is the *shape* of the curve (marginal gain per k) and, crucially, the downstream effect on verification — neither of which was measured.

### 🔴 Correct terminology for the paper

**Do NOT write:** "verifier ablation", "ablation study", "component ablation", "we ablate the retriever and observe a drop in F1", "removing dense retrieval degrades verification performance".

**DO write:** **"retrieval-quality analysis"** or **"retriever comparison"**. Specifically:

- "We compare candidate generators by Recall@5 on the test claim set (BM25 vs. LaBSE dense). This is a **retrieval-stage** comparison and does not measure downstream verification quality."
- "We report the retrieval recall curve over k ∈ {1, 5, 10, 20, 50}. Recall is monotone in k by construction; we report the curve to show the **marginal** return of candidate depth, not to select k by performance."
- "**Verifier-level ablations (claim-only, oracle-evidence, evidence-language) are not reported: they require a separately trained verifier per variant, which was not run.**"

And the headline retrieval finding, stated plainly and without spin:

> **BM25 outperforms LaBSE dense retrieval on both corpora** — FEVER Recall@5 0.549 vs 0.323 (Δ = 0.226); X-FACT (Tamil) Recall@5 0.671 vs 0.535 (Δ = 0.136). The multilingual dense bi-encoder does not justify its cost on either corpus at this scale.

That is a genuine, defensible, negative-but-publishable result. Report it as such.

---

## PART 12 — Claim-only / evidence ablation design

### How the flags work

`--claim-only` → `build_examples(claim_only=True)`:
```python
if claim_only:
    ev_ids = []                     # evidence ignored entirely
...
texts = [...]                       # → []
if not texts and not claim_only:    # not counted as "missing" — it is intentional
    stats["n_with_no_evidence"] += 1
```
→ `to_text_pair()` returns `(claim_text, "")`. The tokenizer gets an empty second segment.

`--oracle` → `build_examples(use_gold_evidence=True)`:
```python
elif use_gold_evidence:
    ev_ids = c.get("evidence") or []   # gold ids straight from the Module 5 corpus
```
→ `to_text_pair()` returns `(claim, gold_ev_1 </s> ... </s> gold_ev_5)`, capped at `evidence_per_claim = 5`.

Both flags exist on **both** `m6_06` and `m6_07`, and `m6_07` records them in `verification_metrics.json` as `claim_only` / `oracle`. **Both current runs correctly record `false` / `false`.**

### 🔴 BLOCKER — you cannot run these ablations with the current scripts

Two defects (D4 + D5) combine into a hard blocker:

**D4 — no output isolation.** `m6_06` hard-codes `ckpt_dir = root / cfg["output"]["models_dir"] / args.dataset`. `m6_07` hard-codes `out = evaluation/{dataset}_{split}`. Neither takes an override. **Every variant writes to the same two places.**

**D5 — `resume=True` is hard-coded.** `trainer.train(..., resume=True)`. If `models/module6/x_fact/latest.pt` exists with `state.epoch = 3` and you launch `--epochs 3 --claim-only`:

```python
if resume and self.ckpt.exists():
    loaded = self.ckpt.load(model, optimizer, scheduler)   # restores epoch=3, best_metric=0.3796
for epoch in range(3, 3):                                  # ← empty range, body NEVER executes
    ...
else:
    state.epoch = 3
    self.ckpt.save(...)                                    # ← OVERWRITES latest.pt
return {"best_metric": 0.3796, ...}                        # ← the OLD run's number
```

**Result: zero training happens, `latest.pt` is overwritten, and the script reports the previous run's `best_metric` as if it were the claim-only result.** You would get a fabricated ablation number and lose a checkpoint, with **exit code 0**. This is precisely the failure mode your instruction *"do not assume exit code 0 means the methodology is correct"* anticipates.

`m6_07` would then overwrite `evaluation/x_fact_test/verification_metrics.json`, destroying your only Stage 7 result.

### Feasibility of each experiment

| | Experiment | Feasible today? | Blocker |
|---|---|---|---|
| **A** | Claim-only verifier | ❌ | D4 + D5. Also: **the current "full" runs ARE already claim-only at training time** (PART 6), so A and B are currently the *same experiment*. |
| **B** | Claim + retrieved evidence verifier | ❌ | Requires train-split retrieval, which **does not exist** (`retrieve/{ds}_train/`). Must run `m6_04` on the train split first. |
| **C** | Claim + oracle gold evidence verifier | ❌ | D4 + D5. Otherwise implemented and correct. |
| **D** | English-only evidence verifier | ❌❌ | **No flag exists.** No `--evidence-languages`. `build_examples()` has no language filter. Requires new code. |
| **E** | Multilingual evidence verifier | ❌❌ | Same, plus there is **no mixed-language evidence pool** — FEVER evidence is all `en`, X-FACT Tamil evidence is all `ta`. Not meaningful on the current corpora. |

### The commands you would use — **after** D4/D5 are fixed

These are written for a future `m6_06`/`m6_07` that accepts `--run-name` (isolating both `models/module6/{dataset}__{run}` and `evaluation/{dataset}_{split}__{run}`) and `--no-resume`. **Do not run them against the current scripts.**

```bash
# ---- Prerequisite: retrieve evidence for the TRAIN split (currently missing) ----
python scripts/m6_04_retrieve.py --dataset x_fact --split train
python scripts/m6_04_retrieve.py --dataset fever  --split train

# ---- A. Claim-only ----
python scripts/m6_06_train_verifier.py --dataset x_fact --claim-only \
       --init-from models/module6/fever --run-name claim_only --no-resume
python scripts/m6_07_evaluate_verifier.py --dataset x_fact --split test \
       --claim-only --run-name claim_only

# ---- B. Claim + retrieved evidence (the real primary system) ----
python scripts/m6_06_train_verifier.py --dataset x_fact \
       --init-from models/module6/fever --run-name retrieved --no-resume
python scripts/m6_07_evaluate_verifier.py --dataset x_fact --split test --run-name retrieved

# ---- C. Claim + oracle gold evidence (upper bound) ----
python scripts/m6_06_train_verifier.py --dataset x_fact --oracle \
       --init-from models/module6/fever --run-name oracle --no-resume
python scripts/m6_07_evaluate_verifier.py --dataset x_fact --split test \
       --oracle --run-name oracle

# ---- D / E. Not implementable without new code. Do not attempt. ----
```

**Critical protocol note:** the `--claim-only` / `--oracle` flag must be passed **identically to both `m6_06` and `m6_07`**. Passing it to only one reproduces the exact train/eval mismatch documented in PART 6.

**I did not run any of these.**

---

## PART 13 — Data leakage / test leakage audit

| # | Risk | Verdict | Evidence |
|---|---|---|---|
| 1 | Test labels used during training | ✅ **CLEAN** | `m6_06` contains the string `"test"` **nowhere**. Train filter is `e.split == "train"`, validation filter `e.split == "validation"`. |
| 2 | Test labels used for temperature fitting | ✅ **CLEAN — guarded in code** | `m6_07` calls `fit_temperature(val_logits, val_labels, split_name="validation")`. `calibration.py` opens with `if str(split_name).lower() == "test": raise ValueError(...)`. Both metrics files record `"temperature": {"split": "validation"}`. |
| 3 | Test used for checkpoint selection | ✅ **CLEAN** | `VerifierTrainer` computes `score = eval_fn(model)["macro_f1"]` where `eval_fn` iterates `val_dl` only. `is_best` derives solely from validation. |
| 4 | Test used for model selection / threshold tuning | ✅ **CLEAN** | No threshold exists — prediction is `softmax(logits, T).argmax(1)`. No grid search over test. No test-conditioned branching. |
| 5 | Validation/test contamination (shared claims) | ⚠️ **NOT RE-VERIFIED HERE** | Delegated to Module 5's `enforce_group_safe_splits()` (exact dedup + char-5-gram Jaccard ≥ 0.90 + union-find closure, contested groups → train). That gate was audited previously and Module 5 is frozen; `assert_module5_untouched()` runs before and after every Module 6 stage. But **I did not re-run the duplicate check in this audit.** |
| 6 | Duplicate claims across train/val/test | ⚠️ **NOT RE-VERIFIED HERE** | Same as #5. |
| 7 | Evidence leakage (test evidence visible at train time) | ✅ **VACUOUSLY CLEAN** | Training used **no evidence at all** (PART 6). Nothing could leak. Config `leakage.index_scope` is `train:[train]`, `validation:[train,validation]`, `test:[train,validation,test]` — the standard nested-scope design. Test claims retrieving from the test evidence pool is **not** label leakage: evidence texts carry no verdict. |
| 8 | Oracle evidence accidentally used in a primary run | ✅ **CLEAN for FEVER; ⚠️ UNVERIFIED for X-FACT** | FEVER training manifest: `"oracle_gold_evidence": false`. Both `verification_metrics.json`: `"oracle": false`. But the **X-FACT training manifest does not exist**, so the X-FACT training-time flag is unrecorded. |
| 9 | Retrieval files generated using test labels | ✅ **CLEAN** | `gold_evidence_map()` reads `c["evidence"]` — evidence ids, not labels. It is used only by `m6_08` to compute recall denominators, i.e. **after** retrieval, for scoring. `m6_04` does not consult labels. |
| 10 | Language info leaking into labels | ✅ **CLEAN** | `language` is stored on `VerifierExample` and used only for per-language *reporting* slices in `m6_07`. `to_text_pair()` returns `(claim_text, evidence)` — **no language token is prepended**. The model never sees the language field. |
| 11 | Checkpoint selection using test performance | ✅ **CLEAN** | Covered by #3. `m6_07` selects `best.pt` by file existence, not by comparing test scores across checkpoints. |

### 🔴 The genuine methodological problems are NOT leakage

To be unambiguous: **I found no test-set leakage.** The protocol discipline in this code is good. The problems are different in kind and, for the paper, more serious:

- **P1 — Train/eval evidence mismatch (PART 6).** Trained on `(claim, "")`, evaluated on `(claim, 5 evidence passages)`. Not leakage — it is an *invalid experimental setup*. It biases results **downward**, so it does not inflate your numbers, but it means the reported numbers do not measure the system you describe.
- **P2 — Probable untrained-reranker contamination of X-FACT Stage 7 (PART 7).** Also biases downward.
- **P3 — The X-FACT run is not reproducible** (missing manifest, PART 3/5).

---

## PART 14 — Class imbalance audit (X-FACT)

### Verified distributions

Recovered exactly from the `support` fields / confusion-matrix row sums:

| Split | false | true | partly true/misleading | n |
|---|---|---|---|---|
| **validation** | **92** (63.4 %) | **38** (26.2 %) | **15** (10.3 %) | 145 |
| **test** | **128** (57.9 %) | **62** (28.1 %) | **31** (14.0 %) | 221 |
| **train** | *unverified* | *unverified* | *unverified* | *1,096 (your figure, unverified)* |

⚠️ **Train counts are UNVERIFIED.** The X-FACT training manifest does not exist (PART 3), and I could not stream the 2.5 MB `claims.jsonl` from Drive within my limits. Extrapolating the val/test rate, `partly true/misleading` in train is likely **110–150 examples**, but I will not state that as a measurement.

### Majority-class baselines (recomputed from the confusion matrices)

| | acc | macro-F1 |
|---|---|---|
| **Validation** — always predict `false` | **0.6345** | **0.2588** |
| Validation — actual model | 0.6828 | **0.3796** |
| **Test** — always predict `false` | **0.5792** | **0.2445** |
| Test — actual model | 0.5928 | **0.3146** |

### Comparison and verdict

| Comparison | Δ accuracy | Δ macro-F1 | Significance |
|---|---|---|---|
| Validation model vs. majority | +0.0483 | **+0.1208** | not tested (used for model selection; a significance test here would be circular) |
| Test model vs. majority | +0.0136 | **+0.0701** | **McNemar χ² = 0.308, p = 0.579 — NOT SIGNIFICANT** |

The model does beat the majority baseline on macro-F1 by a visible margin. But on test, the accuracy improvement rests on **13 items total** (8 gained, 5 lost), and it is not statistically distinguishable from chance at n = 221.

### Is imbalance severe enough to explain the third class scoring 0.0?

**Partially — but it is not the whole story, and it is not the main story.**

Arguments that imbalance is the cause:
- The `partly true/misleading` class is 10–14 % of the data, i.e. an imbalance ratio of roughly **6.2 : 1** against the majority class.
- Training uses unweighted `cross_entropy` with a plain shuffled `DataLoader` — no class weights, no `WeightedRandomSampler`, no focal loss. This is the standard recipe for collapsing a rare class.
- The class is never predicted **once** across 145 + 221 = 366 evaluation items. Total collapse, not degraded performance.

Arguments that imbalance is **not sufficient** as an explanation:
- 6.2 : 1 is **moderate**, not severe. Published work routinely learns minority classes at 20 : 1 and worse. A ratio of 6 : 1 with adequate data and signal should yield a non-zero F1.
- The `true` class (26–28 % of data, only 2.1 : 1 against majority) also nearly collapsed — **recall 0.129, F1 0.205**. Imbalance alone cannot explain a near-majority class failing this badly.
- **`partly true/misleading` is intrinsically the hardest label.** It is a graded, fact-checker-specific editorial judgement. Distinguishing it from `false` requires *degree* reasoning over evidence — exactly the capability the model was denied.

**The dominant cause is PART 6: the verifier had no evidence during training.** "Partly true" is definitionally a statement about the *relationship* between a claim and the facts. Without any evidence in the input, that relationship is unobservable. The model could only learn claim-surface priors, under which `partly true/misleading` has no distinguishing signal at all. Absolute data scarcity (≈120 examples for the class) then makes recovery impossible.

**Ranking:** (1) no evidence at training time, (2) absolute data scarcity, (3) class imbalance amplified by unweighted CE, (4) intrinsic label difficulty. Imbalance is real and contributes, but fixing imbalance alone would not fix this.

### Research-valid remedies (NOT applied, NOT run)

**Fix these first, in this order — the later items are pointless without the earlier ones:**

1. **Retrieve evidence for the train split and retrain with evidence.** This is the prerequisite for everything else. Without it you are tuning a claim-only classifier.
2. **Run the claim-only vs. claim+evidence ablation** (PART 12 A vs B) to *measure* how much evidence contributes. If the gap is near zero, that is itself the paper's most important finding and must be reported.

**Then, standard imbalance treatments (report every one you try, including failures):**

3. Class-weighted cross-entropy, `weight = N / (K · n_c)` — a one-line change, fitted on train only.
4. `WeightedRandomSampler` for balanced batch composition.
5. Focal loss (γ ≈ 2) as an alternative to (3); report both, do not cherry-pick.
6. Report **macro-F1 and per-class F1 as primary metrics** and de-emphasise accuracy — you are already doing this correctly.
7. **Multiple seeds.** `reproducibility.seeds: [42, 43, 44]` is configured but only seed 42 was run. At n = 221 with a 3-class problem, single-seed results carry very wide uncertainty. Run all three and report mean ± std.
8. Report **bootstrap 95 % CIs** on macro-F1 (percentile bootstrap over test items, ≥ 1,000 resamples). At n = 221 the CI on 0.3146 will be wide, and that honesty strengthens the paper.

**Explicitly NOT recommended:** SMOTE or synthetic oversampling in embedding space, or merging `partly true/misleading` into `false`. The first fabricates training signal in a space where interpolation is not meaningful; the second destroys the label your project claims to handle.

---

## PART 15 — Research novelty check (strict)

Your stated claim: *"multilingual fact verification using retrieved evidence."*

**Verdict on the current Module 6 code and outputs: the claim is NOT SUPPORTED. Do not make it.**

| Sub-claim | Supported? | Strict finding |
|---|---|---|
| "using retrieved evidence" | ❌ **NO** | **Evidence retrieval is NOT connected to verifier training.** Stated explicitly, as you asked: `retrieve/fever_train/` and `retrieve/x_fact_train/` do not exist; the FEVER training manifest records `evidence_source: "none"` and `n_with_no_evidence == n_built == 260,032`. Both verifiers were trained on claim text alone. Evidence appears only at evaluation time, in a form the model was never trained to read. |
| "multilingual" (one model, two languages) | ❌ **NO** | Two **separate** models, two **separate** label spaces, two **separate** evaluations. FEVER `per_language` contains only `en`; X-FACT `per_language` contains only `ta`. **No single model is evaluated on both languages.** The word "multilingual" here describes the *encoder's pretraining*, not any measured capability. |
| "multilingual evidence" | ❌ **NO** | **Multilingual evidence is NOT evaluated.** Stated explicitly, as you asked. The ablation is in `pending`; no `--evidence-languages` flag exists; `build_examples()` has no language filter; and there is no mixed-language evidence pool — FEVER evidence is entirely `en`, X-FACT Tamil evidence entirely `ta`. |
| "cross-language support / FEVER→Tamil transfer" | ⚠️ **UNVERIFIED** | The mechanism (`--init-from` + `swap_head`) is implemented correctly (PART 5). But the X-FACT training manifest is missing, so **there is no record that `--init-from` was passed.** Unverified ≠ false — but it cannot be claimed. |
| "fact verification" | ⚠️ **OVERSTATED** | The system performs **3-way claim classification**. It does not verify claims against evidence, because it was never trained to consume evidence. Calling the current numbers "fact verification" overstates what the code demonstrates. |
| "explainability / SHAP / LIME" | — | Module 8. Out of scope for this audit. **Note:** attribution over an evidence segment the model was never trained to use would be uninterpretable. Module 8 should not be run on these checkpoints until PART 6 is fixed. |
| **BM25 vs. dense retrieval comparison** | ✅ **YES — genuinely supported** | The one solid empirical contribution here. Verified numbers, correct accounting (`n_excluded_no_gold` reported), and a clean negative result on both corpora. |
| **Retrieval quality on Tamil (X-FACT)** | ✅ **YES** | Recall@1 0.206 / @5 0.535 / @10 0.611 / @20 0.675 / @50 0.737 for dense; Recall@5 0.671 for BM25, n = 221, 0 excluded. Legitimately reportable. |
| **Calibration analysis** | ✅ **YES, with the negative result included** | Temperature fitted on validation with a hard code-level guard against test. FEVER ECE 0.335 → 0.073; X-FACT ECE 0.053 → **0.066 (worse)**. Both must be reported. |

### What you may honestly claim today

> "We construct a leakage-controlled Tamil–English claim corpus and evaluate multilingual dense retrieval (LaBSE) against a BM25 lexical baseline on FEVER (English) and the Tamil subset of X-FACT. **BM25 outperforms dense retrieval on both corpora.** We separately train XLM-RoBERTa claim classifiers for FEVER entailment and X-FACT veracity, keeping the two label spaces strictly disjoint, and analyse their calibration. **In the current configuration the classifiers are trained on claim text only; connecting retrieved evidence to verifier training is left to the next stage, and we therefore do not claim evidence-grounded verification.**"

That is defensible, honest, and still a contribution. What you may **not** claim is evidence-grounded verification, multilingual evidence, or a cross-lingual transfer result.

---

## PART 16 — Final status table

| Stage | Task | Status | Evidence | Problem | Next action |
|---|---|---|---|---|---|
| Stage 6 | FEVER verifier training | 🔴 **METHODOLOGICALLY INCORRECT** | `verifier_training/fever/manifest.json`: `evidence_source:"none"`, `n_with_no_evidence:260032 == n_built`, 3 epochs, best 0.3452 @ step 14125, 1407.79 s, cuda | **Trained claim-only.** Evidence never reached training. Nested `example_stats.evidence_source:"retrieved"` is a hard-coded false record (D6). | Run `m6_04` on `fever/train`, then retrain with evidence (C4) |
| Stage 6 | X-FACT verifier training | 🔴 **METHODOLOGICALLY INCORRECT + ❌ NOT REPRODUCIBLE** | `checkpoint_meta.json`: best 0.37955704864999373 @ step 138, `label_space:"x_fact"`. **`verifier_training/x_fact/manifest.json` DOES NOT EXIST** | Same claim-only defect. Plus: `init_from`, `head_swap`, `n_train`, `n_val`, `claim_only`, `oracle` are **all unrecorded**. Transfer learning unverifiable. | Same as above; and never run Stage 6 again without confirming the manifest was written |
| Stage 7 | FEVER evaluation | ⚠️ **COMPLETE BUT HAS LIMITATION** | `evaluation/fever_test/verification_metrics.json` (4,025 B), n=16,026, acc 0.3844, macro-F1 0.3227, T=3.9213, ECE 0.3346→0.0725, manifest `status:complete` @04:24:40 | Evaluated **with** evidence a claim-only model never saw. **Accuracy 0.3844 is BELOW the majority baseline 0.4028** (McNemar χ²=8.53, p=0.0035). | Re-evaluate after retraining with evidence; report the current number as a claim-only baseline |
| Stage 7 | X-FACT evaluation | ⚠️ **COMPLETE BUT PROBABLY CONTAMINATED** | `evaluation/x_fact_test/verification_metrics.json` (3,769 B), n=221, acc 0.5928, macro-F1 0.3146, T=0.8408, ECE 0.0531→0.0661, manifest `status:complete` @04:27:22 | Ran at 04:27:22; `reranked_top5.json` renamed at **04:29:14** — so this run likely consumed the **untrained** reranker (R@5 = 0.075). `partly true/misleading` never predicted. ECE got worse. | **Re-run C3 immediately** — free, and settles it |
| Stage 8 | Retrieval ablations | ⚠️ **COMPLETE BUT MISLABELLED** | Both `ablation_results.json` read; all six values match your figures exactly; FEVER dense values byte-identical to `retrieve/fever_test/dense_metrics.json` | These are **retrieval-quality analyses, not verifier ablations** (PART 11). Top-k sweep is **vacuous** — recall is monotone in k. | Rename in the paper to "retrieval-quality analysis". Do not call them ablations. |
| Stage 8 | Verifier ablations | ⏳ **PENDING (3) / ❌ FAILED (1)** | `skipped: ["claim_only_vs_claim_evidence","english_only_vs_multilingual_evidence","oracle_gold_evidence"]` (reason: *no runner registered*); `retrieval_only_vs_retrieval_rerank` → FEVER `error`, X-FACT untrained (0.0750) | Blocked by D4 (no output isolation) + D5 (hard-coded `resume=True`). **Running them now would fabricate numbers and destroy checkpoints.** English/multilingual has **no flag at all**. | Fix D4+D5 first (C5), then run A/B/C. D/E need new code + a mixed-language corpus. |
| — | Checkpoint integrity | ⚠️ **UNRESOLVED — cannot verify** | 4 files, 3,336,811,549–3,336,813,561 B, sizes consistent with model+AdamW+scheduler. FEVER **demonstrably loaded at 04:24:40**. | Reported `failed finding central directory` **not reproduced and not refuted**. Likely a Drive FUSE materialisation failure, not on-disk corruption. | Run C0 (zip header + `torch.load` on a local copy) before concluding anything |
| — | Calibration | ✅ **VERIFIED COMPLETE** (with a negative result) | `fit_temperature(split_name="validation")`; `calibration.py` raises on `"test"`; both files record `"split":"validation"` | None. But **X-FACT ECE got worse** (0.0531→0.0661). | Report the FEVER improvement and the X-FACT regression side by side |
| — | Leakage checks | ✅ **VERIFIED COMPLETE** | 11 checks in PART 13; `m6_06` contains no `"test"` string; `assert_module5_untouched()` brackets every stage | No test-set leakage found. Duplicate-claim check delegated to frozen Module 5 and **not re-run here**. | Optionally re-run the Module 5 duplicate check for the paper's appendix |
| — | Final Module 6 verification | ❌ **NOT PASSED** | `scripts/verify_module6_outputs.py` exists but no Module 6 verification report was found in Drive | Stage 6 is methodologically invalid, so a verification pass would be meaningless right now. | Run last, after C3–C5 |

---

## PART 17 — Exact next steps (minimum sequence)

Ordered cheapest-and-most-informative first. **Nothing here retrains FEVER unnecessarily, and nothing overwrites a valid checkpoint.**

### C0 — Non-destructive checkpoint integrity test (~5 min, CPU)

```bash
cd /content/drive/MyDrive/Fake_News_Detection
for f in models/module6/fever/best.pt models/module6/fever/latest.pt \
         models/module6/x_fact/best.pt models/module6/x_fact/latest.pt; do
  echo "=== $f"; ls -l "$f"
  python - "$f" <<'PY'
import sys, zipfile
p = sys.argv[1]
try:
    z = zipfile.ZipFile(p); print("  zip OK, members:", len(z.namelist()))
except Exception as e:
    print("  ZIP FAILED:", type(e).__name__, e)
PY
done
```

Expected sizes to compare against: fever/best 3,336,812,189 · fever/latest 3,336,813,433 · x_fact/best 3,336,811,549 · x_fact/latest 3,336,813,561.

- **Why:** distinguishes real on-disk corruption from a Drive mount that served a truncated stream. `torch.save` writes a zip; if `zipfile` opens it, `torch.load` will too.
- **Success:** all four report `zip OK` and sizes match. The "corruption" was a mount artefact — **no retraining needed**.
- **Failure:** if a file genuinely fails, first `cp` it to local `/content/` and retry (Drive FUSE is the usual culprit). Only if the local copy also fails is FEVER Stage 6 lost — and it costs just **23.5 min** to redo (`seconds: 1407.79` in the manifest).

**Do not delete or overwrite anything based on this test.**

### C1 — Resolve whether FEVER→X-FACT transfer actually happened (~10 min, CPU)

```python
import torch
a = torch.load("models/module6/fever/latest.pt", map_location="cpu", weights_only=False)["model_state"]
b = torch.load("models/module6/x_fact/best.pt",  map_location="cpu", weights_only=False)["model_state"]
k = "encoder.embeddings.word_embeddings.weight"
print("cosine-ish delta:", (a[k]-b[k]).abs().mean().item())
print("head keys differ:", [n for n in a if n.startswith("head.")])
del a, b
```

- **Why:** PART 5 — the only record that would prove `--init-from` was used is missing.
- **Success:** a **small but non-zero** mean delta (≈1e-4 to 1e-2) ⇒ X-FACT started from FEVER and fine-tuned. You may then claim transfer.
- **Failure:** a **large** delta (comparable to comparing two independently-initialised models) ⇒ X-FACT started from raw HF weights. **You must then drop all transfer-learning claims** — or re-run X-FACT Stage 6 with `--init-from` (cheap: 1,096 claims × 3 epochs ≈ **10–15 min** on a T4, judging by the 03:54→04:08 window).

### C2 — Retrieve evidence for the TRAIN splits (X-FACT ~5 min; FEVER ~1.5–3 h, GPU)

```bash
python scripts/m6_04_retrieve.py --dataset x_fact --split train
python scripts/m6_04_retrieve.py --dataset fever  --split train
```

- **Why:** this is **the** fix for PART 6, and the prerequisite for every remaining experiment. Without it the project has no evidence-grounded verifier and no claim-only ablation to compare against.
- **Output:** `outputs/module6/retrieve/{x_fact,fever}_train/{dense,bm25}_candidates.json` + metrics + manifest.
- **Runtime (estimates, not measured):** X-FACT train = 1,096 claims over ~6,968 evidence items — **a few minutes**. FEVER train = 228,090 claims over 67,388 evidence items — LaBSE encoding dominates; **~1.5–3 h on a T4**. The batched `EvidenceIndex.search()` fix keeps peak RAM near `batch_size × n_evidence × 12 B` (~100 MB at bs=128), so **the earlier `exit -9` OOM should not recur** — but watch RAM on the first shard.
- **Success:** `dense_metrics.json` shows non-zero Recall@k and `accounting.balances: true`.
- **Failure:** `exit -9` again ⇒ lower `--batch-size`. A near-zero Recall ⇒ an index/scope bug; **stop and re-audit before training on it.**

⚠️ **Note the leakage constraint:** config sets `index_scope.train: [train]`. Train claims may only retrieve from train-split evidence. Verify `m6_04` honours this for `--split train` before trusting the output.

### C3 — Re-run X-FACT Stage 7 to settle the reranker question (~3 min, GPU)

```bash
cp -r outputs/module6/evaluation/x_fact_test \
      outputs/module6/evaluation/x_fact_test.PRE_RERANK_FIX_BACKUP
python scripts/m6_07_evaluate_verifier.py --dataset x_fact --split test
```

- **Why:** PART 7 — the untrained `reranked_top5.json` was renamed **after** this evaluation ran. `load_retrieved` will now fall back to dense candidates.
- **The `cp` first is essential:** `m6_07` has no `--output-dir` (defect D4) and will overwrite in place.
- **Success:** metrics **change** ⇒ confirms the original run used the untrained reranker; **use the new numbers and discard the old**. Metrics **identical** ⇒ the original already used dense; the timestamp inference was wrong and the published numbers stand.
- **Failure:** `SystemExit: no checkpoint in ...` ⇒ the X-FACT checkpoint has degraded since 04:27; go back to C0.
- **Cost:** trivial. **Do this one first among the GPU steps.**

### C4 — Retrain both verifiers WITH evidence (X-FACT ~15 min; FEVER ~25 min, GPU)

**Prerequisite: C2 complete, and D4/D5 fixed (C5) so `--no-resume` exists.**

```bash
mv models/module6/fever  models/module6/fever.CLAIMONLY_BASELINE
mv models/module6/x_fact models/module6/x_fact.CLAIMONLY_BASELINE

python scripts/m6_06_train_verifier.py --dataset fever --no-resume
python scripts/m6_06_train_verifier.py --dataset x_fact \
       --init-from models/module6/fever --no-resume
```

- **Why:** produces the system your paper actually describes. The `mv` preserves the current claim-only checkpoints — **they become your claim-only ablation arm, so they have real scientific value. Do not delete them.**
- **Runtime:** FEVER measured at **1,407.79 s ≈ 23.5 min** (50k × 3 epochs @ len 256). Adding evidence lengthens sequences toward the 256 cap, so expect **25–40 min**. X-FACT ≈ **10–15 min**.
- **Success:** the training manifest now reads `"evidence_source": "dense_candidates"` and `n_with_no_evidence` ≪ `n_built`. **Check this before doing anything else.** Also confirm `verifier_training/x_fact/manifest.json` now exists (it did not last time).
- **Failure:** still `"evidence_source": "none"` ⇒ C2 wrote to the wrong path; re-check `stage_dir(cfg,"retrieve",root)` resolution. **Do not proceed to Stage 7 on a claim-only model a second time.**

### C5 — Fix D4 + D5 before any ablation (code change, ~30 min — NOT part of this audit)

Minimum change set:

1. Add `--run-name` to `m6_06` and `m6_07`; suffix `ckpt_dir` and the evaluation `out` dir with it.
2. Add `--no-resume`; pass it through to `trainer.train(resume=not args.no_resume)`.
3. Make `m6_07` record the string returned by `load_retrieved()` into `verification_metrics.json` as `evidence_source` — this audit could not answer PART 7 definitively **only** because that field is absent.
4. Fix D6: `build_examples()` should set `stats["evidence_source"]` from what was *actually attached*, e.g. `"retrieved"` only if `n_with_no_evidence < n_built`, else `"none"`.

Optional but recommended: fix D3 so `--init-from` prefers `best.pt`; add a `tests/test_verification.py` case asserting head weights change across `swap_head()`.

**I did not make any of these changes.**

### C6 — Then, and only then: verifier ablations, Stage 8 re-run, final verification

Order: PART 12 A → B → C (X-FACT only; FEVER ablations are optional and cost ~25 min each) → re-run `m6_08` → `scripts/verify_module6_outputs.py`.

### What NOT to do

- ❌ Do **not** retrain FEVER before C0 — the checkpoint is probably fine.
- ❌ Do **not** run `m6_06` with `--claim-only`/`--oracle` against the current scripts (fabricates numbers, destroys checkpoints — PART 12).
- ❌ Do **not** run `m6_07` for an ablation before C5 (overwrites your only Stage 7 results).
- ❌ Do **not** delete `models/module6/*.CLAIMONLY_BASELINE` — they are your ablation arm.
- ❌ Do **not** train a cross-encoder reranker yet. It is the single most expensive component and, per your own config note, dominates the GPU budget. It is worthless until the verifier can read evidence at all.

---

## PART 18 — Final summary

**Confirmation of read-only compliance:** No script was edited. No checkpoint was deleted, overwritten or loaded. No file was renamed. No model was retrained. No configuration was changed. No Module 6 command was executed. Every artefact was opened read-only; the only computation I ran was arithmetic on numbers already stored in your JSON files.

### 1. COMPLETE TASKS

| Task | Confidence |
|---|---|
| Label-space separation (FEVER vs X-FACT) — enforced in code, config, checkpoints and outputs | ✅ Verified, and it is the strongest part of Module 6 |
| Test-leakage protocol — temperature on validation with a code-level guard; checkpoint selection on validation; no test string in `m6_06` | ✅ Verified across 11 checks |
| Stage 4 retrieval for `{fever,x_fact} × {validation,test}` — dense + BM25, with correct accounting | ✅ Verified |
| Retrieval-quality comparison (BM25 vs dense; recall-vs-k curves) — all 6 headline values | ✅ Verified against actual JSON |
| Stage 6 FEVER training *ran to completion* — 3 epochs, 14,125 steps, 1,407.79 s | ✅ Verified (methodology separately invalid) |
| Stage 6 X-FACT training *ran to completion*; `best.pt` = the true best validation checkpoint (step 138 ≠ final 207, distinct timestamps) | ✅ Verified |
| Stage 7 evaluation for both datasets — all metrics recomputed and reproduce exactly | ✅ Verified |
| Calibration machinery — ECE, reliability data, temperature scaling | ✅ Verified |

### 2. FAILED TASKS

- **Cross-encoder reranking.** Never trained. `rerank/x_fact_test/manifest.json` → `"checkpoint": null`. No reranker checkpoint exists anywhere. No training script exists. Degraded Recall@5 from 0.5347 → 0.0750 (7.1×).
- **FEVER reranking.** Never run at all — `rerank/fever_test/` does not exist. The FEVER rerank ablation returned `{"error": "candidates not available"}`.
- **X-FACT Stage 6 manifest.** Absent from Drive. `verifier_training/` contains only `fever`. `init_from`, `head_swap`, `n_train`, `n_val`, `claim_only`, `oracle` are permanently unrecorded for that run.
- **`retrieval_only_vs_retrieval_rerank` ablation.** Invalid for both datasets (error / untrained).

### 3. PENDING TASKS

| Task | Why |
|---|---|
| Train-split retrieval (`retrieve/{fever,x_fact}_train/`) | Never run — config `evaluate_splits: [validation, test]` excludes train |
| Claim-only vs claim+evidence ablation | `skipped: no runner registered`; blocked by D4+D5 |
| Oracle gold-evidence ablation | Same |
| English-only vs multilingual-evidence ablation | Same, **plus no flag exists and no mixed-language evidence pool exists** |
| Reranker training + a valid rerank ablation | No training script |
| Multi-seed runs (43, 44) | Configured, never run |
| Bootstrap CIs on macro-F1 | Not computed |
| `scripts/verify_module6_outputs.py` final pass | No report found in Drive |

### 4. METHODOLOGICAL ISSUES

| # | Issue | Severity |
|---|---|---|
| **M1** | **Verifiers trained claim-only, evaluated with evidence.** `evidence_source:"none"`, `n_with_no_evidence == n_built == 260,032`. A train/eval input-distribution mismatch that invalidates the "evidence-grounded" framing. | 🔴 **CRITICAL** |
| **M2** | **Stage 8 "ablations" are retrieval-quality analyses.** No verifier is loaded in `m6_08`; values are byte-identical to Stage 4 `dense_metrics.json`. Must be renamed in the paper. | 🔴 **CRITICAL** |
| **M3** | **X-FACT Stage 7 probably consumed the untrained reranker** — evaluated 04:27:22, file renamed 04:29:14. Resolvable in 3 minutes (C3). | 🔴 **HIGH** |
| **M4** | **FEVER test accuracy 0.3844 is below the majority baseline 0.4028** (McNemar χ² = 8.53, **p = 0.0035**). Must be reported, not buried. | 🔴 **HIGH** |
| **M5** | **X-FACT's gain over majority is not significant** (χ² = 0.308, **p = 0.579**, n = 221). | ⚠️ HIGH |
| **M6** | **X-FACT calibration got worse** (ECE 0.0531 → 0.0661). Report the regression. | ⚠️ MEDIUM |
| **M7** | **Transfer learning unverified** — no record that `--init-from` was passed (M-manifest missing). | ⚠️ HIGH |
| **M8** | **`build_examples()` hard-codes `evidence_source:"retrieved"`** regardless of reality (D6). Already produced one false record. | ⚠️ HIGH |
| **M9** | **`resume=True` hard-coded + no output isolation** (D4/D5). Any ablation run today silently fabricates a number and destroys a checkpoint, **with exit code 0**. | 🔴 **CRITICAL (blocker)** |
| **M10** | **`gradient_accumulation_steps: 2` never read** — effective batch 16, not 32. `use_mixed_precision` also unused (fp32). Do not claim either. | ⚠️ MEDIUM |
| **M11** | **Top-k sweep is vacuous** — recall is monotone in k, so k=50 wins by construction. | ⚠️ MEDIUM |
| **M12** | **Unweighted cross-entropy under 6.2 : 1 imbalance**, no class weights, no balanced sampler. `partly true/misleading` never predicted once in 366 evaluation items. | ⚠️ MEDIUM |
| **M13** | **Single seed** (42) despite `seeds: [42, 43, 44]`; no confidence intervals at n = 221. | ⚠️ MEDIUM |
| **M14** | `--init-from` loads `latest.pt`, not `best.pt` (D3). Harmless here only because FEVER's best step == final step. | ⚠️ LOW |

### 5. DATA / LEAKAGE ISSUES

**No test-set leakage was found.** All 11 checks in PART 13 pass or are vacuously clean. The protocol discipline (validation-only temperature fitting with a hard code guard, validation-only checkpoint selection, `assert_module5_untouched()` bracketing every stage, no test string in the training script) is genuinely good and should be described in the paper.

Residual items, stated honestly:

| Item | Status |
|---|---|
| Duplicate claims across train/val/test | ⚠️ **Not re-verified in this audit.** Delegated to Module 5's frozen group-safe split gate. |
| Oracle evidence in the X-FACT primary run | ⚠️ **Unverified** — training manifest missing. (`verification_metrics.json` correctly records `oracle: false` at evaluation time.) |
| Evidence-index scope for train-split retrieval (C2) | ⚠️ **Must be checked when C2 runs** — config says `index_scope.train: [train]`; verify `m6_04` honours it. |
| Test claims retrieving from the test evidence pool | ✅ **Not leakage.** Evidence texts carry no labels; nested scope is the standard design. |

### 6. EXACT NEXT COMMANDS

```bash
# C0  integrity  (~5 min, CPU)        — settle "corruption" before touching anything
#     (loop + zipfile test, full script in PART 17)

# C3  re-run X-FACT Stage 7  (~3 min, GPU)  — cheapest, highest-information step
cp -r outputs/module6/evaluation/x_fact_test \
      outputs/module6/evaluation/x_fact_test.PRE_RERANK_FIX_BACKUP
python scripts/m6_07_evaluate_verifier.py --dataset x_fact --split test

# C1  transfer check  (~10 min, CPU)  — settle whether --init-from was used

# C2  train-split retrieval  (X-FACT ~5 min; FEVER ~1.5-3 h, GPU)
python scripts/m6_04_retrieve.py --dataset x_fact --split train
python scripts/m6_04_retrieve.py --dataset fever  --split train

# C5  fix D4 + D5 + D6 in code  (~30 min)   — REQUIRED before any ablation

# C4  retrain with evidence  (X-FACT ~15 min; FEVER ~25-40 min, GPU)
mv models/module6/fever  models/module6/fever.CLAIMONLY_BASELINE
mv models/module6/x_fact models/module6/x_fact.CLAIMONLY_BASELINE
python scripts/m6_06_train_verifier.py --dataset fever --no-resume
python scripts/m6_06_train_verifier.py --dataset x_fact --init-from models/module6/fever --no-resume

# C6  ablations, Stage 8 re-run, final verification
```

**Run C0 and C3 first. They are nearly free and they change what everything else means.**

### 7. CAN MODULE 6 CURRENTLY BE CALLED COMPLETE?

# ❌ NO.

Module 6 is **not complete**, and more importantly it is **not currently scientifically valid as a fact-verification module**.

Three independent reasons, any one of which is sufficient:

1. **The verifiers were never trained on evidence** (M1). The module's stated purpose — verification against retrieved evidence — was not exercised. What exists is a claim-only 3-way classifier evaluated on out-of-distribution inputs.
2. **Four of six configured ablations were never run**, and the fifth is invalid (untrained reranker). The two that did run are retrieval analyses mislabelled as ablations (M2).
3. **The X-FACT run is not reproducible** — its training manifest does not exist, so the transfer-learning claim cannot be substantiated (M7).

**What Module 6 *can* legitimately be called today:** a completed **evidence-retrieval** module with a **verified retriever comparison**, plus a **claim-only classification baseline** with calibration analysis, for two datasets kept correctly separate. That is real work and it belongs in the paper — under that description.

### 8. WHAT YOU CAN SAFELY REPORT IN YOUR RESEARCH PAPER

#### ✅ SAFE TO REPORT AS-IS

**Retrieval (the strongest results you have):**

| Corpus | Retriever | R@1 | R@5 | R@10 | R@20 | R@50 | MRR | n_scored | excluded |
|---|---|---|---|---|---|---|---|---|---|
| FEVER test | dense (LaBSE) | 0.1824 | 0.3227 | 0.3793 | 0.4338 | 0.5066 | 0.2581 | 15,995 | 31 |
| FEVER test | BM25 | — | **0.5491** | — | — | — | 0.4242 | 15,995 | 31 |
| X-FACT test (ta) | dense (LaBSE) | 0.2064 | 0.5347 | 0.6112 | 0.6747 | 0.7365 | 0.9621 | 221 | 0 |
| X-FACT test (ta) | BM25 | — | **0.6706** | — | — | — | 0.9685 | 221 | 0 |

> **BM25 outperforms LaBSE dense retrieval on both corpora** (FEVER Δ = 0.226; X-FACT Δ = 0.136 at Recall@5). Frame as: *"the multilingual dense bi-encoder does not justify its cost at this corpus scale"* — a clean, honest negative result.

**Methodology (describe, and it strengthens the paper):** disjoint label spaces enforced in code; validation-only temperature fitting with a code-level guard against test; validation-only checkpoint selection; group-safe splits from Module 5; explicit `n_excluded_no_gold_evidence` accounting; the full documented `compute_scope` reduction table.

**Calibration — report both directions:** FEVER T = 3.9213, ECE 0.3346 → **0.0725** (4.6× improvement). X-FACT T = 0.8408, ECE 0.0531 → **0.0661** (**a 1.24× regression**). Both. Do not report only the good one.

#### ⚠️ REPORT ONLY WITH THE MANDATORY CAVEAT

FEVER test acc 0.3844 / macro-F1 0.3227 (n=16,026) and X-FACT test acc 0.5928 / macro-F1 0.3146 (n=221), plus full per-class tables and confusion matrices — **all verified exact**. Every mention must carry:

> *"These verifiers were trained on claim text only; evidence was appended at evaluation time but was not present during training. The figures are therefore a claim-only baseline under an input-distribution mismatch, not evidence-grounded verification."*

Plus, in the same table or immediately adjacent:

- FEVER: **majority-class baseline accuracy 0.4028 > model 0.3844** (McNemar χ² = 8.527, **p = 0.0035**). The model is significantly worse than a constant predictor on accuracy, while better on macro-F1 (0.3227 vs 0.1914).
- X-FACT: majority-class accuracy 0.5792 vs model 0.5928 (χ² = 0.308, **p = 0.579 — not significant**); macro-F1 0.2445 vs 0.3146.
- X-FACT `partly true/misleading`: **P = R = F1 = 0.000**, never predicted once across 366 evaluation items.
- Single seed (42), n = 221 for X-FACT, no confidence intervals.

Retrieval-quality analysis: report **only** under that name, never as an "ablation" (PART 11), and state that recall is monotone in k so the top-k curve is descriptive, not a selection criterion.

#### 🔴 DO NOT REPORT — under any framing

| Do not claim | Why |
|---|---|
| "Evidence-grounded / retrieval-augmented fact verification" | Evidence never reached training (M1) |
| "Ablation study" / "verifier ablation" / "removing X degrades F1" | No verifier was ever run under any ablated condition (M2) |
| Any reranking result, including the 0.0750 figure | The cross-encoder was untrained (`checkpoint: null`) |
| "Reranking degrades retrieval" | That is random reordering, not reranking |
| "FEVER → Tamil transfer learning" | `--init-from` unverified; manifest missing (M7) |
| "Multilingual evidence improves Tamil verification" | Never evaluated; no flag; no mixed-language pool |
| "Cross-lingual fact verification" (as a measured capability) | Two separate models, two separate label spaces, `en`-only and `ta`-only evaluations |
| "k = 50 was selected by ablation" | Recall is monotone in k — tautological (M11) |
| "Effective batch size 32" / "mixed-precision training" | Config values never read (M10) |
| "Calibration improves verification reliability" (unqualified) | True for FEVER, **false for X-FACT** (M6) |
| Any combined FEVER + X-FACT metric | Different tasks, different label spaces — the code correctly refuses this; the paper must too |

#### The honest one-paragraph framing

> We build a leakage-controlled Tamil–English claim–evidence corpus and evaluate multilingual dense retrieval (LaBSE) against a BM25 lexical baseline on FEVER (English, n = 16,026) and the Tamil subset of X-FACT (n = 221). **BM25 outperforms dense retrieval on both corpora** (Recall@5 0.549 vs 0.323 and 0.671 vs 0.535), indicating that a multilingual bi-encoder does not justify its cost at this scale. We separately train XLM-RoBERTa classifiers for FEVER entailment and X-FACT veracity, keeping the two label spaces strictly disjoint, and analyse calibration under temperature scaling fitted on validation only (FEVER ECE 0.335 → 0.073; X-FACT 0.053 → 0.066). **In the present configuration these classifiers are trained on claim text alone; connecting retrieved evidence to verifier training remains future work, and we make no claim of evidence-grounded verification.** We further report that the FEVER classifier does not exceed a majority-class baseline in accuracy (0.384 vs 0.403, p = 0.004), and that the X-FACT minority class *partly true/misleading* is never predicted.

That paragraph is publishable, defensible under review, and entirely true.

---

*End of audit. No project file was modified.*
