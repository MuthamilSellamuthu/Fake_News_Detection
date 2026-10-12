# Module 10 — Phase 2 Design

**No implementation code written. No existing file modified.**

Your flow diagram now includes **L2 normalization** between mean pooling and the CE-MLP — that matches the Phase 1.5 finding. This design is built on that flow.

---

## ⛔ Still gating Phase 3

The Phase 1.5 diagnostic output has not come back yet. Design is safe to do now; **writing `text_encoder.py` is not**, because it would bake in an unconfirmed assumption about the exact encode path.

Required before implementation:

```
stored embedding is unit norm      : True
path A (no normalisation) matches  : False
path B (with normalisation) matches: True   cosine≈1.0
```

Evidence for the assumption is already strong (config flag `true`, code applies it, Module 4's gate asserts unit vectors). The diagnostic is confirmation, not discovery. But it is 2 minutes and it removes the last doubt from the two most visible outputs.

---

## 1. Your flow → exact existing functions

| Your step | Implementation | Reused or new |
|---|---|---|
| Text normalization | `src/fact_verification/claim_selection.py::normalize_claim(text, norm_cfg)` (NFC, Tamil-safe) | **reuse** |
| Language detection | `claim_selection.py::tamil_ratio(text)`, threshold `> 0.3` → `ta` else `en` — the same rule `m7_02` used | **reuse** |
| XLM-RoBERTa + mean pooling | `src/embeddings/feature_extraction.py::FeatureExtractor._forward_batch()` → `embedding_utils.py::mean_pooling` | **reuse** |
| **L2 normalization** | `FeatureExtractor._to_numpy()` → `embedding_utils.py::normalize_embeddings` | **reuse** |
| Module 4 frozen CE-MLP | `src/contrastive/projection_head.py::build_ce_mlp_baseline` + `models/module4/ce_mlp_mean_uniform_seed42.pt`; probability via `softmax(...)[:,1]` (class 1 = fake) | **reuse** |
| Claim extraction | `claim_selection.py::select_claim_from_record(record, cfg, dataset="inference")` | **reuse** |
| BM25 evidence retrieval | `src/retrieval/bm25_baseline.py::BM25Retriever` + `src/retrieval/index_scope.py::build_evidence_scope`; pool `{en: fever, ta: x_fact}`; `top_k=10`; confidence via `src/integration/fusion.py::rank_normalise` | **reuse** |
| Frozen fact verifier | `src/verification/verifier_model.py::build_verifier` + `VerifierConfig`; `CheckpointManager` prefers `best.pt`; input `tok(claim, " </s> ".join(evidence[:5]))` | **reuse** |
| Module 7 frozen fusion | `src/integration/fusion.py::verifier_to_fake_prob` then `apply_fusion(...)` with `FusionParameters.from_dict(...)` **loaded from `outputs/module7/validation/fusion_parameters.json`** | **reuse** |
| Token-level explainability | `src/explainability/token_attribution.py::occlusion_attribution` + `TokenAttribution.top_tokens(k)` | **reuse** |
| Reason | `src/explainability/combined_explanation.py::build_reason` + `verification_explainer.py::explain_verification`, `contribution_breakdown` | **reuse** |
| **Raw text → Module-3-compatible vector** | — | **NEW, the only genuinely new component** |

`FeatureExtractor.__init__` loads model + tokenizer, so it is constructed **once** at process start and reused. Occlusion issues ~N forward passes for N tokens; reconstructing the encoder per call would make the demo unusable.

---

## 2. 🔴 Eight corrections your mockup needs

These are not cosmetic. Each one, left as drawn, would misrepresent a measured quantity.

### C1 — "Confidence: 96.8%" must not equal `P(Fake): 0.968`

Your mockup shows both as 96.8. But the verdict comes from **fusion**, not the classifier:

```
p_ver   = 0.5 + 0.5*(P_refute − P_support)
fused   = α·p_m4 + (1−α)·p_ver          α = 0.95
fused   = temperature_scale(fused, T)    T = 0.7079
verdict = fused ≥ threshold              threshold = 0.56
```

With **T = 0.7079 < 1** the logit is *divided* by T, which **sharpens** — the displayed confidence is systematically more extreme than the classifier's raw probability. Showing one number for both hides that. Display `P(fake)` classifier and `fused P(fake)` as separate rows.

### C2 — Show the threshold, because it is not 0.5

`threshold = 0.56`. A fused `p_fake` of 0.54 is **REAL** despite exceeding 0.5. Without the threshold on screen, a user reading "P(Fake) 0.54 → REAL" will think the system is broken.

### C3 — "Verifier verdict: REFUTES" is wrong for Tamil

Label spaces are disjoint and must never be mapped (enforced in `dataset.py`; `mapping_between_spaces: false`):

| Language | Pool | Verdict strings |
|---|---|---|
| English | FEVER | `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO` |
| Tamil | X-FACT | `false` / `true` / `partly true/misleading` |

The UI must render the **native** label for the pool actually used, with the pool named. Printing "REFUTES" for a Tamil article would be a fabricated verdict string.

### C4 — "Explanation faithfulness: …" must not show Module 8's stored numbers

Module 8's faithfulness was computed over a 200/500 sample — and per Phase 1.5 those runs are under question. Two honest options:

- **Compute live** for this input with `deletion_test` / `insertion_test` at `fractions=(0.1,0.2,0.3)`. Cost: ~3 extra forward passes. Label it *"measured for this input"*.
- **Omit** the field.

Never display an aggregate from a different sample beside a single user's article.

### C5 — Attribution wording must name its target

`occlusion_attribution` computes `attributions = base − p_masked` where `base` is the **classifier's** `P(fake)`. So:

- attribution **> 0** → removing the token *lowered* P(fake) → token pushed **toward fake**
- attribution **< 0** → removing it *raised* P(fake) → token pushed **toward real**

Your 🔴/🟢 split is therefore correct, but it explains the **classifier**, not the final fused verdict. When classifier and fusion disagree, "words supporting FAKE" beside a REAL verdict is incoherent unless labelled. Use: *"influenced the classifier toward higher / lower P(fake)"*.

### C6 — "Evidence retrieved: 5" — two different counts

`retrieval.top_k = 10`; the verifier consumes `evidence_per_claim = 5`. Report both: *"10 candidates retrieved; top 5 passed to the verifier."*

### C7 — The LIMITATION box needs the fusion weight

Your text covers the fixed corpus. It should also carry the measured facts, or the Verification panel will read as if it drove the verdict:

> α = 0.95 — the verdict is ~95 % classifier-driven; verification contributes ~5 %.
> On the Module 7 test split, fusion changed **no Tamil prediction at all** (Tamil macro-F1 identical to six decimals, 0.961620 both).
> Evidence is retrieved from fixed FEVER (67,388) / X-FACT (6,968) pools; an article about anything outside them cannot have its evidence retrieved.

### C8 — "🔴 FAKE NEWS" needs an abstain state

Required by your own Step 9 (`NOT ENOUGH INFO` must not become FAKE). Three display states:

| Condition | Display |
|---|---|
| `fused ≥ threshold` | 🔴 FAKE |
| `fused < threshold` | 🟢 REAL |
| `\|fused − threshold\| < 0.05` **or** no evidence retrieved | ⚠️ **UNCERTAIN** — verdict shown with an explicit low-confidence banner |

The margin band is a **display** convention, not a refitted parameter; it is declared in `configs/module10_inference.yaml` and recorded in the output JSON.

---

## 3. Proposed files

Matching the conventions found in Modules 1–9 (`scripts/m{N}_{NN}_*.py`, `scripts/m{N}_common.py`, `scripts/verify_module{N}_outputs.py`, `src/{area}/`, `configs/module{N}_*.yaml`).

```
configs/module10_inference.yaml          NEW  paths, top_k, display bands, no fitted params
scripts/m10_common.py                    NEW  mirrors m7_common/m8_common
scripts/m10_01_inference.py              NEW  CLI: --text / --file / --stdin -> JSON + report
scripts/verify_module10_outputs.py       NEW  the 11 verification checks
scripts/m10_app.py                       NEW  Streamlit (Phase 6, after CLI works)
src/inference/__init__.py                NEW
src/inference/text_encoder.py            NEW  ONLY new logic: raw text -> unit-norm vector
src/inference/pipeline.py                NEW  orchestration; loads frozen artifacts once
src/inference/report.py                  NEW  slot-filling renderer; no free-text generation
tests/test_inference.py                  NEW  the 13 required tests
outputs/module10/{inference,manifests}/  NEW  per-run JSON + manifest
```

**Nothing under `src/` for Modules 1–9 is touched.** `pipeline.py` imports; it does not reimplement.

### `text_encoder.py` — the one new component, and its contract

```python
class InferenceEncoder:
    """Raw text -> the vector Module 4 was trained on.

    Reproduces scripts/extract_features.py --config configs/feature_config.yaml
    by IMPORTING FeatureExtractor, not by reimplementing it:
        xlm-roberta-base, max_length=512, padding="longest", truncation
        -> mean_pooling(last_hidden_state, attention_mask)
        -> normalize_embeddings(...)   <-- unit L2 norm; the step Phase 1.5 found

    SELF-CHECK AT CONSTRUCTION: encodes one held-out record whose stored
    embedding and stored p_fake are known, and refuses to start unless
    cosine >= 0.999 and |p_fake - stored| < 1e-4. A silently mis-wired
    encoder is exactly the failure this module exists to avoid.
    """
```

That self-check is the design's most important feature. It converts the Phase 1.5 problem from something a future reader could reintroduce into something the process refuses to start with.

---

## 4. Output JSON schema

Every field traces to a computed value. `null` where a stage did not run — never a filled-in guess.

```json
{
  "schema_version": "m10.1",
  "input": {"text": "...", "n_chars": 812, "accepted": true, "min_chars": 40},
  "language": {"code": "ta", "method": "tamil_ratio>0.3 (claim_selection.py)",
               "tamil_ratio": 0.87},
  "claim": {"text": "...", "selection_rule": "title_as_claim",
            "claim_id": "inference:<sha1>"},
  "classifier": {"p_fake": 0.968, "p_real": 0.032, "prediction": 1,
                 "prediction_label": "FAKE", "threshold_used": 0.5,
                 "checkpoint": "models/module4/ce_mlp_mean_uniform_seed42.pt",
                 "checkpoint_sha256": "...",
                 "embedding": {"encoder": "xlm-roberta-base", "pooling": "mean",
                               "l2_normalised": true, "max_length": 512,
                               "l2_norm": 1.0000000}},
  "retrieval": {"pool": "x_fact", "retriever": "bm25", "top_k": 10,
                "n_retrieved": 10, "n_passed_to_verifier": 5,
                "candidates": [{"rank": 1, "evidence_id": "...", "text": "...",
                                "score": 12.47, "confidence": 1.0,
                                "corpus": "x_fact"}]},
  "verification": {"available": true, "pool": "x_fact",
                   "label_space": ["false", "true", "partly true/misleading"],
                   "probs": {"false": 0.71, "true": 0.11,
                             "partly true/misleading": 0.18},
                   "verdict": "false", "verdict_probability": 0.71,
                   "margin": 0.53, "n_evidence": 5,
                   "checkpoint": "models/module6/x_fact/best.pt"},
  "fusion": {"alpha": 0.95, "threshold": 0.56, "temperature": 0.7079,
             "fit_split": "validation",
             "params_source": "outputs/module7/validation/fusion_parameters.json",
             "p_fake_classifier": 0.968, "p_fake_verifier": 0.765,
             "has_verifier": true, "fused_p_fake": 0.9713, "fused_p_real": 0.0287,
             "final_prediction": 1, "final_label": "FAKE",
             "margin_to_threshold": 0.4113, "display_state": "FAKE"},
  "explainability": {"method": "occlusion", "target": "classifier P(fake)",
                     "mask_token": "[MASK]", "n_tokens": 96, "truncated": false,
                     "base_probability": 0.968,
                     "top_positive_tokens": [{"token": "...", "attribution": 0.12,
                                              "position": 4}],
                     "top_negative_tokens": [],
                     "faithfulness": {"scope": "this input only",
                                      "deletion": {...}, "insertion": {...}}},
  "reason": {"sentences": [{"template_id": "...", "text": "...",
                            "provenance": ["fusion.fused_p_fake", "fusion.threshold"]}],
             "agreement": "classifier_and_verifier_agree"},
  "caveats": ["alpha_0.95_classifier_dominant",
              "fixed_evidence_corpus_x_fact_6968_items",
              "verifier_out_of_domain_for_user_articles"],
  "provenance": {"run_id": "...", "timestamp": "...", "seed": 42,
                 "deterministic": true, "software": {...}}
}
```

### The anti-fabrication rule, enforced mechanically

Every sentence in `reason.sentences` carries a non-empty `provenance` list naming the JSON paths it was built from. `report.py` raises on an empty provenance list, so a sentence that cannot cite a computed field cannot be emitted. `tests/test_inference.py` asserts every emitted sentence's provenance paths resolve in the record — the same discipline as the `evidence_source` fix in `dataset.py`: derive from reality, never from intent.

**No language model writes any part of the explanation.** It is slot filling over templates.

---

## 5. Corrected report layout

```
╔══════════════════════════════════════════════════════════════╗
║                     NEWS VERIFICATION                         ║
╚══════════════════════════════════════════════════════════════╝
VERDICT
  🔴 FAKE            fused P(fake) 0.9713   threshold 0.56
  Language: Tamil (ta)   tamil_ratio 0.87

──────────────── CLASSIFIER (Module 4, frozen) ────────────────
  P(fake) 0.968   P(real) 0.032   -> FAKE
  encoder xlm-roberta-base | mean pooling | L2-normalised | max_len 512

──────────────── FACT VERIFICATION (Module 6, frozen) ─────────
  Evidence pool : X-FACT (6,968 items)   retriever BM25
  Retrieved     : 10 candidates, top 5 passed to the verifier
  Label space   : false / true / partly true/misleading
  Verdict       : false  (p = 0.710, margin over runner-up 0.530)

  Candidate evidence — retrieved automatically, not verified:
    1. [x_fact:a1b2…] score 12.47  conf 1.00
       "..."
    2. ...

──────────────── DECISION FUSION (Module 7, frozen) ───────────
  alpha 0.95   threshold 0.56   T 0.7079   (fitted on validation)
  classifier 0.968 x 0.95  +  verifier 0.765 x 0.05  -> 0.9578
  temperature-scaled -> 0.9713   >= 0.56  -> FAKE
  Classifier and verifier AGREE.

──────────────── WHY ──────────────────────────────────────────
  <slot-filled sentences, each with provenance>

──────────────── EXPLAINABILITY ───────────────────────────────
  Method: occlusion   target: classifier P(fake)   base 0.968
  Pushed the classifier toward FAKE:  word1 (+0.12)  word2 (+0.09)
  Pushed the classifier toward REAL:  word4 (−0.06)
  Faithfulness (this input only): deletion Δ 0.31 | insertion Δ 0.22

  Attributions show which parts of the input most influenced the
  CLASSIFIER's probability. They do not establish factual truth.

──────────────── ⚠ LIMITATIONS ────────────────────────────────
  • alpha = 0.95: the verdict is ~95% classifier-driven.
  • On the Module 7 test split, fusion changed no Tamil prediction
    (Tamil macro-F1 0.961620 both with and without fusion).
  • Evidence comes from a FIXED corpus (FEVER 67,388 / X-FACT 6,968).
    An article about events outside it cannot have evidence retrieved.
  • The verifier was trained on FEVER/X-FACT, not on user-submitted
    news. Its verdict is an auxiliary signal, not ground truth.
```

---

## 6. Where the 13 tests get their determinism

Tests 2, 3, 5, 13 need a model. Design so most tests run **without** torch or checkpoints:

- `pipeline.py` takes injectable `encode_fn`, `classify_fn`, `retrieve_fn`, `verify_fn` (defaulting to the real ones), mirroring how `VerifierTrainer` takes `step_fn`/`eval_fn` so resume logic is testable without a GPU.
- Tests 1, 4, 6, 7, 8, 9, 10, 11, 12 then run on stubs in CI.
- Tests 2, 3, 13 run against real artifacts, skipped with a clear reason when checkpoints are absent.

Test 7 (*no test-time fitting*) is the sharpest: assert that `fit_fusion`, `fit_binary_temperature` and `fit_temperature` are **never called** during inference — via `monkeypatch` raising on entry — and that loaded `FusionParameters` equal the JSON on disk.

---

## 7. Open items before Phase 3

1. **Phase 1.5 diagnostic output** — the three verdict lines. *(blocking)*
2. **Confirm `outputs/module7/validation/fusion_parameters.json` exists** and holds α=0.95, threshold=0.56, T=0.7079. Module 10 loads them from that file; hard-coding would make test 6 tautological.
3. **Faithfulness in the UI** — compute live for the input (~3 extra forward passes) or omit? *(C4)*
4. **Module 8 regeneration** — still flagged, not done. Independent of Module 10, but its explanations should not be reported in the paper until resolved.
