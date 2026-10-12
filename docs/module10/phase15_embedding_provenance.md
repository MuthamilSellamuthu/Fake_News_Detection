# Phase 1.5 — Exact Embedding Provenance Investigation

**Read-only. Nothing modified, regenerated, retrained or fixed.**

---

## A. Exact Module 3 embedding generation — traced from source

| Property | Value | Source (file:line) |
|---|---|---|
| Driver script | `scripts/extract_features.py --config configs/feature_config.yaml` | `scripts/extract_features.py:11` (docstring), `:53` (`--config`), `:60` → `run_feature_extraction()` |
| **Authoritative config** | **`configs/feature_config.yaml`** | as above |
| Implementation | `src/embeddings/feature_extraction.py::FeatureExtractor` | `:187`, `:540` |
| Model | `xlm-roberta-base` | `feature_config.yaml` → `model.name` |
| Local override | `model.local_path: null` (Hub download) | `feature_config.yaml` |
| Tokenizer | `load_tokenizer(model.name, local_path)` → XLM-R SentencePiece | `feature_extraction.py:214` |
| Input dir | `datasets/processed` | `feature_config.yaml` → `input.processed_dir` |
| Files | `train.csv`, `validation.csv`, `test.csv` | `input.*_filename` |
| **Text field** | **`text`** — a single column | `input.text_column`; read at `feature_extraction.py:321` |
| Title/body concat | **NONE.** `df[text_col].fillna("").astype(str).tolist()` | `feature_extraction.py:321` |
| id / label / language cols | `record_id`, `label`, `language` | `feature_config.yaml` → `input.*` |
| **max_length** | **512** | `tokenization.max_length` → `feature_extraction.py:203` |
| **Padding** | **`longest`** (dynamic, per batch) | `tokenization.padding` → `:204`; `tokenizer_utils.py:86` |
| Truncation | `true` | `tokenization.truncation` |
| Tokenise call | `tokenize_batch(tokenizer, texts, max_length, padding)` | `feature_extraction.py:261` |
| Forward | `FeatureExtractor._forward_batch()`; `torch.autocast(cuda, float16)` when AMP active | `:254`, `:267` |
| Mixed precision | requested `true`; **active only on CUDA** (`amp_active`) | `model.use_mixed_precision`; `:219` |
| **Pooling** | `mean_pooling(last_hidden_state, attention_mask)` | `embedding_utils.py:44`, called at `feature_extraction.py:277` |
| Mask handling | `mask = attention_mask.unsqueeze(-1).expand(...)`; `summed/counted`, `counted.clamp(min=1e-9)` | `embedding_utils.py:58-62` |
| Strategies saved | `["cls", "mean"]` | `extraction.pooling_strategies` |
| dtype | `.detach().float().cpu().numpy()` → **float32**, saved `.astype("float32")` | `feature_extraction.py:288`; `embedding_utils.py:150` |
| **🔴 NORMALISATION** | **`normalize_embeddings(arr)` — L2 to unit length — applied when `extraction.normalize_embeddings: true`, and it IS true** | `feature_config.yaml:67`; `feature_extraction.py:289`; `embedding_utils.py:65-79` |
| Batch size | 32 | `extraction.batch_size` |
| Output | `embeddings/{split}/{pooling}_embeddings.npy` + `.pt`, `record_ids.npy` (`<U64`), `labels.npy` (int64), `languages.npy` (`<U2`), **`{pooling}_metadata.json`** | `embedding_utils.py:126-170` |

### ⚠️ `configs/embeddings.yaml` is NOT the config that ran

There are two Module 3 configs, and they disagree:

| Setting | `configs/feature_config.yaml` (**used**) | `configs/embeddings.yaml` (**not used**) |
|---|---|---|
| processed dir | `datasets/processed` ✅ exists | `data/processed` ❌ |
| text column | `text` | `clean_text` |
| max length | **512** | 128 |
| splits | train / validation / test | train / **val** / test |
| pooling | `["cls","mean"]` | `"mean"` |
| normalise | **`true`** | *not present* |

`configs/embeddings.yaml` also drives a second, separate encoder class (`src/embeddings/xlmr_encoder.py::XLMREncoder`) used by `scripts/generate_embeddings.py`. **That class performs no L2 normalisation** (`_pool` returns `summed / counted` and nothing else, `xlmr_encoder.py:170-175`). It is not the path that produced the stored arrays.

---

## B. Exact Module 4 consumption

| Property | Value | Source |
|---|---|---|
| Files consumed | `embeddings/{split}/mean_embeddings.npy`, `record_ids.npy`, `labels.npy`, `languages.npy` | `embedding_dataset.py::load_split:74-88`; `module4_inference.py::load_split_arrays:46-53` |
| Row → record mapping | **positional**: row *i* of the embedding matrix ↔ `record_ids[i]`. Lengths asserted equal. | `module4_inference.py:59-62` |
| Pooling selected | `mean` | `module7_integration.yaml` → `inputs.pooling` |
| Checkpoint | `models/module4/ce_mlp_mean_uniform_seed42.pt` | `inputs.module4_checkpoint` |
| Architecture | `build_ce_mlp_baseline(768, [512,128], 2, dropout=0.1)` → `Linear(768,512) ReLU Dropout Linear(512,128) ReLU Dropout Linear(128,2)` | `projection_head.py:100-128` |
| **Pre-classifier transform** | **NONE.** `np.asarray(embeddings, dtype=np.float32)` straight into the MLP. No scaler, no re-normalisation, no centering. | `module4_inference.py:95-102` |
| Probability | `torch.softmax(logits, dim=-1)[:, 1]`, class **1 = fake** | `module4_inference.py:101` |
| Determinism | `model.eval()` disables dropout; no sampling | `module4_inference.py` docstring |

### 🔴 The decisive artifact: Module 4's own input gate

`src/contrastive/embedding_dataset.py:142-150`, inside `verify_frozen_inputs()` — a hard gate the docstring says the caller must **STOP** on:

```python
# ---- L2 normalisation (Module 3 wrote unit vectors) ----
for s, e in data.items():
    n = np.linalg.norm(e.embeddings, axis=1)
    ...
    if not np.allclose(n, 1.0, atol=1e-3):
        norm_ok = False
add("l2_normalised", norm_ok, f"{norms}", norms)
```

**Module 4 asserts, as a pre-training gate, that Module 3's stored embeddings are unit vectors.** Module 4 trained. Therefore the stored arrays are L2-normalised.

---

## C. Trace for `record_id = welfake_47809`

```
datasets/processed/test.csv, column "text"          [input.text_column = "text"]
   ↓ tokenize_batch(xlm-roberta-base SP tokenizer,
                    max_length=512, padding="longest", truncation=True)
   ↓ AutoModel(xlm-roberta-base).forward()      [autocast fp16 iff CUDA]
   ↓ mean_pooling(last_hidden_state, attention_mask)   embedding_utils.py:44
   ↓ .detach().float().cpu().numpy()                   → float32
   ↓ normalize_embeddings(arr)                         → UNIT L2 NORM   ← the missing step
   ↓ np.save("embeddings/test/mean_embeddings.npy", .astype("float32"))
        row i, where record_ids.npy[i] == "welfake_47809"
   ↓ build_ce_mlp_baseline(768,[512,128],2,0.1) + ce_mlp_mean_uniform_seed42.pt
   ↓ softmax(logits)[:, 1]
   = 0.0030954221729189157
```

`embedding_utils.py:133` uses **`"welfake_47809"`** as its own worked example of a `record_id`, confirming the identifier format is a Module 2 string, stored `<U64`.

**Section D of the diagnostic executes this exact path** — importing `FeatureExtractor`, `mean_pooling` and `normalize_embeddings` from the project rather than reimplementing them.

---

## D / E. Diagnostic + saved provenance

`notebooks/module10_phase15_embedding_provenance.py` — read-only, one Colab cell. It:

1. prints both configs side by side so the disagreement is visible;
2. lists `embeddings/test/` and dumps **`mean_metadata.json`** (written by `embedding_utils.py`, carrying `model_name`, `pooling_strategy`, `normalized`, and run provenance — this answers **E**);
3. locates `welfake_47809` via `record_ids.npy` and reports the stored vector's **L2 norm**;
4. reproduces `p_fake` from the *stored* embedding + frozen checkpoint (must equal 0.0030954221729189157);
5. regenerates the embedding from raw CSV text through `FeatureExtractor._forward_batch()` and reports **both** `_to_numpy` variants — with and without `normalize_embeddings`;
6. reports shape, max |diff|, mean |diff|, cosine similarity, and `p_fake` for each;
7. sweeps `p_fake(stored × {1, 2, 5, 10})` to demonstrate the saturation mechanism directly.

The hypothesis and its predictions are stated **in the file, above the numbers**, so the conclusion cannot be retrofitted.

---

## F. Why the earlier reproductions saturated

`build_ce_mlp_baseline` begins with `Linear(768 → 512)` and contains **no input normalisation layer**. It was trained exclusively on unit-norm vectors.

- **Unnormalised XLM-R mean-pooled vectors** have L2 norms on the order of 5–15 for `xlm-roberta-base`. Scaling every input by ~N scales the first pre-activation by ~N; through ReLU and two further Linear layers the logit gap grows with it and softmax saturates. → observed **p ≈ 1.000000**. Right space, **wrong scale**.
- **LaBSE via `SentenceTransformer.encode()`** returns **unit-norm** vectors (LaBSE's sentence-transformers pipeline terminates in a `Normalize` module). Correct scale, but a different embedding space; the MLP's learned directions are meaningless there and it saturates the other way. → observed **p ≈ 0.000000**. Right scale, **wrong space**.

Both fail, in opposite directions, for different reasons. Neither is "closer to correct" — which is exactly why the choice must not be made on error magnitude.

Consistency check on your own numbers: mean stored `p_fake` over the 20 records ≈ 0.31 (since LaBSE returned 0.0 throughout, `mean|LaBSE − stored| = mean(stored) = 0.3100`). `mean|XLM-R − stored| = 0.5911` is then consistent with XLM-R sitting near 1.0 on most but not all of the 20. Both reproductions are degenerate; the stored values are graded. That signature is scale/space corruption, not a near-miss.

---

# DIAGNOSIS

## 1. CONFIRMED FACTS

1. Module 3 ran via `scripts/extract_features.py` against **`configs/feature_config.yaml`** — not `configs/embeddings.yaml`.
2. Encoder `xlm-roberta-base`; tokenizer its SentencePiece; `max_length=512`; `padding="longest"`; truncation on.
3. Text came from a **single column `text`** of `datasets/processed/*.csv`. **No title/body concatenation.**
4. Pooling is attention-mask-weighted mean (`embedding_utils.py:44`).
5. **`extraction.normalize_embeddings: true`**, applied in `FeatureExtractor._to_numpy` (`feature_extraction.py:289`) via `normalize_embeddings()` (`embedding_utils.py:65`).
6. Stored dtype float32; both `.npy` and `.pt`; plus `record_ids/labels/languages` and `{pooling}_metadata.json`.
7. Module 4 applies **no transform** before the MLP.
8. Module 4's pre-training gate **asserts stored embeddings are unit vectors** (`embedding_dataset.py:142`), and Module 4 trained.
9. `configs/embeddings.yaml` + `XLMREncoder` form a **second, non-normalising** path that did not produce the stored arrays.
10. Module 8's `encoder: sentence-transformers/LaBSE` carries the comment *"must match Module 3's encoder path"*; Module 3's encoder is `xlm-roberta-base`. **The comment is false.**

## 2. EXACT SOURCE OF STORED EMBEDDINGS

> `scripts/extract_features.py --config configs/feature_config.yaml`
> → `FeatureExtractor._forward_batch()` — xlm-roberta-base, max_length 512, padding "longest", column `text`
> → `mean_pooling(last_hidden_state, attention_mask)`
> → `_to_numpy()` = `.detach().float().cpu().numpy()` **+ `normalize_embeddings()` → unit L2 norm**
> → `embeddings/{split}/mean_embeddings.npy`, float32

*Pending confirmation by the diagnostic (`||stored||₂ == 1.0` and path B reproducing 0.0030954221729189157).*

## 3. MISMATCH FOUND

**Two distinct mismatches.**

- **M1 — Module 8 uses the wrong encoder.** `configs/module8_explainability.yaml:34` sets LaBSE; Module 3/4 use `xlm-roberta-base`. Both are 768-d, so `m8_02::build_text_predict_fn` runs silently and returns saturated probabilities.
- **M2 — the earlier reproduction omitted L2 normalisation** (and used `max_length=128` and possibly `clean_text`). This was my diagnostic's error, not a project defect — but it is the reason XLM-R saturated to ~1.0 and it identifies normalisation as the load-bearing step.

## 4. EVIDENCE FOR THE MISMATCH

| # | Evidence |
|---|---|
| E1 | `feature_config.yaml:67` `normalize_embeddings: true` |
| E2 | `feature_extraction.py:289` `return normalize_embeddings(arr) if self.normalize else arr` |
| E3 | `embedding_dataset.py:142-150` Module 4 gate: *"Module 3 wrote unit vectors"*, `np.allclose(n, 1.0, atol=1e-3)` |
| E4 | `module4_inference.py:95-102` no transform before the MLP |
| E5 | `projection_head.py:119-128` first layer `Linear(768,512)`, no input normalisation |
| E6 | Measured: LaBSE → 0.000000 (unit norm, wrong space); XLM-R unnormalised → ~1.000000 (right space, wrong scale) |
| E7 | `feature_config.yaml` vs `embeddings.yaml` disagree on max_length, text column and processed dir |
| E8 | `xlmr_encoder.py:170-175` `_pool` returns `summed/counted` with no normalisation — a genuinely different path |

## 5. WHAT MUST BE CHANGED, IF ANY

**Nothing yet — and nothing in Modules 1–9 without your explicit instruction.**

Conditional on the diagnostic confirming §2:

| Scope | Change | Status |
|---|---|---|
| **Module 10 (new)** | `src/inference/text_encoder.py` must reproduce the §2 path exactly — `xlm-roberta-base`, `max_length=512`, `padding="longest"`, column-`text` semantics, mean pooling, **then L2 normalise** — by importing `FeatureExtractor` / `mean_pooling` / `normalize_embeddings`, not reimplementing. | required |
| **Module 8** | `configs/module8_explainability.yaml:34` `encoder:` is wrong, and `m8_02::build_text_predict_fn` would need the same corrected path. | **flagged, NOT changed** — you forbade modifying Modules 1–9 |
| **Module 3** | Nothing. It behaved as configured. | none |
| **Module 4 / 7** | Nothing. Stored predictions derive from correctly-normalised stored embeddings. | none |
| **Housekeeping** | `configs/embeddings.yaml` is a stale second config that contradicts the one actually used, and Module 8's comment asserts a match that does not hold. Both are traps for the next reader. | flagged only |

## 6. WHETHER MODULE 8 EXPLANATIONS MUST BE REGENERATED

**Yes — if the diagnostic confirms §2.** *(Not regenerated. Flagged only, per your restrictions 5 and 8.)*

`m8_02::build_text_predict_fn` encodes with LaBSE and feeds a classifier trained on XLM-R mean-pooled unit vectors. Occlusion attributions are `base − p_masked` from that `predict_fn`; if `predict_fn` is not the real classifier response, the attributions describe a model that was never evaluated. Your measured evidence — LaBSE returning **exactly 0.000000** on every one of 5 sampled articles — indicates a saturated function whose masked-token differences would be ~0 or numerical noise.

Affected: the **200 validation + 500 test** explanation records and the faithfulness metrics computed from them.

⚠️ Regenerating is inference-only (no retraining), but occlusion costs one forward pass per token: 700 records × up to 128 tokens ≈ **~90,000 encoder forward passes**. Budget accordingly; do not start it until §2 is confirmed.

**Do not report Module 8 faithfulness results in the paper until this is resolved.**

## 7. WHETHER IT IS SAFE TO PROCEED TO MODULE 10

# ❌ NOT YET — but the blocker is now identified and cheap to clear.

Module 10 must not be built until the diagnostic prints:

```
stored embedding is unit norm      : True
path A (no normalisation) matches  : False
path B (with normalisation) matches: True   cosine≈1.0
```

Then the ambiguity is gone and Phase 2 design can proceed, with `text_encoder.py` importing the project's own implementation.

If instead **neither** path matches, stop again: the remaining suspects are the text column, `max_length`, or a `datasets/processed/test.csv` revision differing from the one that produced the stored arrays. In that case Module 10's classifier and explanation outputs cannot be trusted at all, and neither can Module 8's.

---

## Run this next

```python
# Colab, read-only, ~2 min
exec(open("/content/drive/MyDrive/Fake_News_Detection/"
          "notebooks/module10_phase15_embedding_provenance.py").read())
```

Paste the full output back. I will not write a line of Module 10 until the three verdict lines are confirmed.
