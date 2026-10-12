# Module 10 — Phase 1 Inspection Report

**Status: inspection only. No file was created, modified or executed in Modules 1–9.**

Everything below was read from the actual project source, not assumed.

---

## 1. Exact artifacts discovered

### Module 4 — frozen CE-MLP classifier

| Item | Value | Source |
|---|---|---|
| Checkpoint | `models/module4/ce_mlp_mean_uniform_seed42.pt` | `configs/module7_integration.yaml` → `inputs.module4_checkpoint` |
| Constructor | `src/contrastive/projection_head.py::build_ce_mlp_baseline` | `src/integration/module4_inference.py` |
| Architecture | `input_dim=768`, `hidden_dims=[512, 128]`, `n_classes=2`, `dropout=0.1` | `module7_integration.yaml` → `inputs.module4_*` |
| Class index | **1 = fake** (`softmax(...)[:, 1]`) | `module4_inference.predict_p_fake` docstring |
| Inference fn | `src/integration/module4_inference.py::predict_p_fake` (embeddings → P(fake)) | — |
| Load tolerance | `load_state_dict(..., strict=False)`, then raises if any key missing/unexpected | same file |

### Module 3 — embeddings the classifier expects

| Item | Value | Source |
|---|---|---|
| Encoder | **`xlm-roberta-base`** | `configs/embeddings.yaml` → `model.name`; confirmed in `outputs/module3/feature_statistics.json` → `model.name` |
| Pooling | `mean` (attention-mask weighted) | `configs/embeddings.yaml` → `model.pooling_strategy`; `module7_integration.yaml` → `inputs.pooling` |
| Max seq length | 128 | `configs/embeddings.yaml` |
| Array layout | `embeddings/{split}/mean_embeddings.npy` + `record_ids.npy`, `labels.npy`, `languages.npy` | `module4_inference.load_split_arrays` |

### Module 6 — frozen verifiers

| Item | Value | Source |
|---|---|---|
| FEVER checkpoint dir | `models/module6/fever/` (`best.pt`, else `latest.pt`) | `m7_03_run_verifier.py` |
| X-FACT checkpoint dir | `models/module6/x_fact/` (`best.pt`, else `latest.pt`) | same |
| Selection rule | `cm.load(model, path=cm.best if cm.best.is_file() else None)` | same |
| Model builder | `src/verification/verifier_model.py::build_verifier` + `VerifierConfig` | same |
| Encoder | `xlm-roberta-base`, `hidden_dim=512`, `dropout=0.1`, `max_length=256` | `configs/module6_retrieval_verification.yaml` → `verifier` |
| Label spaces | FEVER `[SUPPORTS, REFUTES, NOT ENOUGH INFO]`; X-FACT `[false, true, partly true/misleading]` | `src/verification/dataset.py` |
| Input format | `tok(claim_text, " </s> ".join(evidence_texts), ...)` | `m7_03_run_verifier.py` |
| Evidence per claim | 5 (`verifier.evidence_per_claim`) | module 6 config |

### Module 7 — retrieval

| Item | Value | Source |
|---|---|---|
| Implementation | `scripts/m7_02_prepare_claims_and_retrieve.py` | — |
| Pools | `{en: fever, ta: x_fact}` | `module7_integration.yaml` → `inputs.evidence_datasets` |
| **Primary retriever** | **`bm25`** | `retrieval.primary` (comment: "measured: BM25 > dense on X-FACT R@5 0.671 vs 0.535") |
| top_k | 10 | `retrieval.top_k` |
| Index scope phase | `test` | `retrieval.phase` |
| BM25 | `src/retrieval/bm25_baseline.py::BM25Retriever`, k1/b from module 6 config | — |
| Dense | `EvidenceIndex.load(build_index/{dataset}_test)` + `build_labse_encode_fn` (**LaBSE**) | — |
| Score normalisation | `src/integration/fusion.py::rank_normalise` → per-query confidence in [0,1] | `retrieval.score_normalisation` |
| Scope guard | `build_evidence_scope(...)` from `src/retrieval/index_scope.py` | — |

### Module 7 — fusion

| Item | Value | Source |
|---|---|---|
| Implementation | `src/integration/fusion.py` | — |
| Apply fn | **`apply_fusion(p_fake_m4, p_fake_verifier, has_verifier, params)`** — docstring: *"Apply FROZEN parameters. Contains no fitting of any kind."* | — |
| Params container | `FusionParameters` dataclass, `.from_dict()` / `.to_dict()` | — |
| Verifier→P(fake) | `verifier_to_fake_prob(p_refute, p_support, p_neutral)` = `0.5 + 0.5*(p_refute - p_support)` | `fusion.py` + config comment |
| Label roles | FEVER `{fake: REFUTES, real: SUPPORTS, neutral: NOT ENOUGH INFO}`; X-FACT `{fake: "false", real: "true", neutral: "partly true/misleading"}` | `fusion.fever_label_roles` / `x_fact_label_roles` |
| Fusion maths | `fused = alpha*p_m4 + (1-alpha)*p_ver` (α=1 where `has_verifier` is False), then `_temperature_scale_binary`, then `>= threshold` | `apply_fusion` |
| Temperature form | binary logit space: `1/(1+exp(-(logit(p)/T)))`, clipped to [1e-6, 1-1e-6] | `_temperature_scale_binary` |
| Missing verifier | `missing_verifier_policy: classifier_only` (α→1, counted, never imputed as 0.5) | config |
| Frozen params file | `outputs/module7/validation/fusion_parameters.json` | `output.fusion_params_filename` + `m7_05` |
| Leakage guard | `m7_05` raises if `params.fit_split != "validation"` | `m7_05_apply_fusion.py` |

### Module 8 — explainability

| Item | Value | Source |
|---|---|---|
| Primary method | **`occlusion`** | `configs/module8_explainability.yaml` → `explanation.primary_method` |
| Implementation | `src/explainability/token_attribution.py::occlusion_attribution` | — |
| Signature | `(text, predict_fn, tokenize_fn=None, mask_token="", max_tokens=128, batch_size=16) -> TokenAttribution` | — |
| Attribution | `base − p_masked` per token; positive = removing the token *lowered* P(fake) | — |
| Tokeniser | `simple_tokenize` = whitespace (deliberately word-level so Tamil words stay intact) | — |
| Mask token | `"[MASK]"` | `explanation.mask_token` |
| Top-k | `TokenAttribution.top_tokens(k)` → `top_positive_tokens` / `top_negative_tokens` | — |
| **text → P(fake)** | **`scripts/m8_02_generate_classifier_explanations.py::build_text_predict_fn`** | — |
| Reason builder | `src/explainability/combined_explanation.py::build_reason`, `build_explanation_record`, `render_text_report` | — |
| Verifier explainer | `src/explainability/verification_explainer.py::explain_verification`, `contribution_breakdown` | — |

### Claim selection

| Item | Value | Source |
|---|---|---|
| Implementation | `src/fact_verification/claim_selection.py::select_claim_from_record` / `select_claims` | reused unchanged by `m7_02` |
| Rule | title as claim; fall back to `text` prefix if title shorter than `min_claim_chars`; truncate at `max_claim_chars`; drop below minimum | — |
| Config source | Module 5 config `claim_selection` + `normalization` blocks | `m7_02` lines 71–76 |
| **Language detection** | `str(record.get("language","")) or ("ta" if tamil_ratio(normalised) > 0.3 else "en")` — `tamil_ratio()` at `claim_selection.py:68` | — |
| Returns | `Claim` dataclass: `claim_id, text, raw_text, language, dataset, source, label, record_id, url, selection_rule, metadata` | — |

---

## 2. 🔴 BLOCKER — a real inconsistency, found before writing any code

### `configs/module8_explainability.yaml` line 34

```yaml
encoder: sentence-transformers/LaBSE      # must match Module 3's encoder path
```

**The comment asserts a match that does not hold.**

| | Encoder |
|---|---|
| Module 3 (produced the embeddings) | **`xlm-roberta-base`**, mean pooling — `configs/embeddings.yaml` and `outputs/module3/feature_statistics.json` |
| Module 4 CE-MLP | trained on those XLM-R mean-pooled vectors |
| Module 8 `build_text_predict_fn` | **`sentence-transformers/LaBSE`** |

Both encoders output 768 dimensions, so `clf(torch.from_numpy(v))` **runs without error**. But LaBSE's embedding space is not XLM-R's mean-pooled space. Feeding LaBSE vectors into an MLP trained on XLM-R vectors produces P(fake) values that do not reflect the classifier's actual behaviour.

**Why this matters for Module 10:** `build_text_predict_fn` is *precisely* the raw-text → P(fake) function Module 10 needs, for both Step 4 (classifier) and Step 8 (occlusion). If it is mis-wired, Module 10 inherits the fault in its two most visible outputs.

**Why it matters for Module 8 as it stands:** the 200 validation + 500 test occlusion explanations, and the faithfulness metrics computed from them, would be attributions of a mis-fed classifier.

### ⚠️ I am flagging this, not fixing it

You instructed me not to modify Modules 1–9. I have changed nothing. But I will not build Module 10 on top of it without telling you, and I cannot silently pick one encoder — that would be me deciding a research question.

**A decisive 2-minute check** (read-only, no training). If Module 3's stored embeddings were produced by `xlm-roberta-base`, re-encoding the same texts with LaBSE will give a very different vector, and the resulting P(fake) will disagree with the stored Module 7 predictions:

```python
# Colab, read-only
import json, numpy as np, torch, csv
from pathlib import Path
ROOT = Path("/content/drive/MyDrive/Fake_News_Detection")

pred = json.loads((ROOT/"outputs/module7/module4_predictions/test_predictions.json"
                   ).read_text())["records"][:20]
ids = [r["record_id"] for r in pred]

csv.field_size_limit(10**9)
rows = {r["record_id"]: r for r in csv.DictReader(
    (ROOT/"datasets/processed/test.csv").open(encoding="utf-8", newline=""))}
texts = [rows[i].get("clean_text") or rows[i].get("text", "") for i in ids]

import sys; sys.path.insert(0, str(ROOT))
from src.contrastive.projection_head import build_ce_mlp_baseline
clf = build_ce_mlp_baseline(768, [512, 128], 2, 0.1)
st = torch.load(ROOT/"models/module4/ce_mlp_mean_uniform_seed42.pt",
                map_location="cpu", weights_only=False)
clf.load_state_dict(st.get("state_dict", st) if isinstance(st, dict) else st); clf.eval()

def p_fake(vecs):
    with torch.inference_mode():
        return torch.softmax(clf(torch.from_numpy(np.asarray(vecs, np.float32))), -1)[:,1].numpy()

from sentence_transformers import SentenceTransformer
labse = SentenceTransformer("sentence-transformers/LaBSE"); labse.max_seq_length = 128
p_labse = p_fake(labse.encode(texts, convert_to_numpy=True))

from transformers import AutoTokenizer, AutoModel
tk = AutoTokenizer.from_pretrained("xlm-roberta-base")
xm = AutoModel.from_pretrained("xlm-roberta-base").eval()
with torch.inference_mode():
    e = tk(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    h = xm(**e).last_hidden_state
    m = e["attention_mask"].unsqueeze(-1).float()
    p_xlmr = p_fake(((h*m).sum(1)/m.sum(1).clamp(min=1e-9)).numpy())

stored = np.array([r["p_fake"] for r in pred])
print("mean |LaBSE  - stored| :", np.abs(p_labse - stored).mean())
print("mean |XLM-R  - stored| :", np.abs(p_xlmr  - stored).mean())
```

**Interpretation, fixed in advance:**

- `|XLM-R − stored|` ≈ 0 (< 0.01) and `|LaBSE − stored|` large ⇒ Module 8's encoder is **wrong**; Module 10 must use XLM-R mean pooling, and Module 8's explanations should be regenerated.
- Both ≈ 0 ⇒ the two spaces happen to agree here; the config comment is sloppy but harmless. Unlikely.
- Both large ⇒ neither reproduces the stored predictions; something else is off and we stop and investigate.

---

## 3. Second issue — the demo's honesty problem

Your own Module 7 test results:

| | Classifier-only Macro-F1 | Integrated Macro-F1 | Δ |
|---|---:|---:|---:|
| Overall | 0.971786 | 0.972681 | **+0.000895** |
| English | 0.972654 | 0.973625 | **+0.000971** |
| **Tamil** | 0.961620 | 0.961620 | **0.000000** |

With **α = 0.95**, the verifier carries 5 % of the fused probability. For **Tamil the two numbers are identical to six decimal places — fusion changed no Tamil prediction at all.**

Module 10 will display a prominent "Verification" section. If a user reads that as the driver of the verdict, the interface misleads. The design must state, from the measured numbers, that **the verdict is ~95 % classifier-driven and that verification did not change any Tamil decision on the test split.**

I will surface this in the UI rather than hide it. It is not a defect in Module 10; it is a measured property of Module 7 that the demo must not obscure.

---

## 4. Third issue — retrieval domain mismatch, already documented by your own code

`m7_02` records, per language:

```
"domain_mismatch_note": "{lang} project news claims retrieved against the {dataset}
 fact-checking corpus. Low coverage here is a corpus-mismatch finding, not a retrieval bug."
```

A user's article is retrieved against a **fixed** pool — FEVER (67,388 items) or X-FACT (6,968 items). Evidence about anything outside those pools **cannot** be retrieved; the system will still return nearest-neighbour passages. Module 10 must label these **"candidate evidence, retrieved automatically"**, never "the evidence", and must emit `NO EVIDENCE RETRIEVED` when the list is empty, per your Step 5.

---

## 5. Naming conventions observed (to be followed, not invented)

| Convention | Observed |
|---|---|
| Scripts | `scripts/m{N}_{NN}_{verb}_{noun}.py` — e.g. `m7_02_prepare_claims_and_retrieve.py` |
| Shared helpers | `scripts/m{N}_common.py` with `PROJECT_ROOT, banner, load_m{N}_config, m{N}_dir, set_all_seeds, write_json, write_manifest, sha256_file, reproducibility_block` |
| Verifier script | `scripts/verify_module{N}_outputs.py` (Modules 3, 4, 6) — **note: `verify_module6_outputs.py`, not `m6_09_...`** |
| Library code | `src/{area}/{noun}.py`, torch imported lazily inside functions |
| Config | `configs/module{N}_{name}.yaml` with `module:`, `name:`, `inputs:`, `output:`, `reproducibility:` |
| Outputs | `outputs/module{N}/{stage}/` + `manifest.json` per stage |
| Tests | `tests/test_{area}.py` |

**Proposed Module 10 layout** (matching the above, deviating from your sketch where the project's convention differs):

```
configs/module10_inference.yaml
scripts/m10_common.py                     # mirrors m7_common / m8_common
scripts/m10_01_inference.py               # CLI: one article in, JSON + text report out
scripts/verify_module10_outputs.py        # NOT m10_02_* -- matches verify_module{N}_outputs.py
scripts/m10_app.py                        # Streamlit (Phase 6)
src/inference/__init__.py
src/inference/pipeline.py                 # orchestration, frozen-artifact loading
src/inference/text_encoder.py             # raw text -> Module 3-compatible embedding
src/inference/report.py                   # slot-filling report; no free-text generation
tests/test_inference.py
outputs/module10/{inference,manifests}/
```

`src/inference/` holds **no duplicated model logic** — it imports `predict_p_fake`-equivalent code, `select_claims`, `BM25Retriever`, `EvidenceIndex`, `build_verifier`, `apply_fusion`, `occlusion_attribution`, `build_reason` from their existing homes. The only genuinely new component is `text_encoder.py`, which exists because **no Module 1–9 code encodes a single raw article into a Module 3-compatible vector** — `m7_01` reads precomputed `.npy` arrays, and `m8_02`'s `build_text_predict_fn` is the only text-level path, which is the one under question in §2.

---

## 6. What I need before Phase 2

1. **Run the §2 encoder check** and paste the two numbers. This determines whether `src/inference/text_encoder.py` uses `xlm-roberta-base` + mean pooling (Module 3's actual configuration) or LaBSE (Module 8's config).
2. Confirm you accept `verify_module10_outputs.py` rather than `m10_02_verify_inference.py`, to match `verify_module3/4/6_outputs.py`.
3. Confirm `outputs/module7/validation/fusion_parameters.json` exists and holds α=0.95, threshold=0.56, T=0.7079 — Module 10 will **load them from that file**, not hard-code them, so the frozen-parameter test compares file against config rather than against a literal.

No code will be written until (1) is answered — building the classifier and explanation paths on an unverified encoder would put the fault straight into the two most user-visible outputs.
