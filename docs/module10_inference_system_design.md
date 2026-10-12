# Module 10 — End-to-End Inference System (design)

**Purpose:** one news article in → eight outputs out, every one of them traceable to a number the pipeline actually computed or a span the retriever actually returned.

**Non-goal:** free-text explanation generation. No LLM writes prose about the prediction. That is the only reliable way to satisfy *"do not generate unsupported reasons."*

---

## Part 1 — Verification of the results you supplied

I recomputed what could be checked arithmetically. **I changed nothing.**

### 🔴 Finding: X-FACT claim-only is metrically identical to a constant "false" predictor

| Metric | Your X-FACT claim-only | Constant "false" baseline | Match |
|---|---|---|---|
| Accuracy | 0.5791855203619910 | 0.5791855203619910 | ✅ exact |
| Macro-F1 | 0.2445081184336199 | 0.2445081184336199 | ✅ exact |
| Weighted-F1 | 0.4248466854231223 | 0.4248466854231223 | ✅ exact |

All three agree to 16 decimal places. The X-FACT claim-only verifier **predicted `false` for all 221 test items** and learned nothing from claim text alone.

This is not a problem — it is the cleanest possible ablation result. It makes the evidence contribution unambiguous: **on X-FACT, 100 % of the model's skill comes from the retrieved evidence.** Report it exactly this way.

### The repair worked, and the size of the effect is the headline

| FEVER test (n = 16,026) | Accuracy | Macro-F1 |
|---|---|---|
| Majority class (always NEI) | 0.4028 | 0.1914 |
| **Pre-repair (broken, evidence mismatch)** | **0.3844** | **0.3227** |
| Claim-only (repaired) | 0.6066 | 0.6098 |
| Evidence-aware (repaired) | **0.6337** | **0.6356** |

Fixing the train/eval evidence mismatch moved FEVER **+24.93 pp accuracy and +31.30 pp macro-F1**, and lifted it from *below* the majority baseline to comfortably above it. That is the single most important number in the whole Module 6 story.

### Evidence contribution, both corpora

| Comparison | Δ Accuracy | Δ Macro-F1 |
|---|---|---|
| FEVER: evidence-aware − claim-only | **+2.71 pp** | **+2.59 pp** |
| X-FACT: evidence-aware − claim-only | **+5.88 pp** | **+17.35 pp** |

### X-FACT with 95 % Wilson intervals (n = 221)

| Variant | Accuracy | 95 % CI | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| Majority / claim-only | 0.5792 | 0.5133 – 0.6424 | 0.2445 | 0.4248 |
| Tamil-only evidence | 0.6244 | 0.5590 – 0.6856 | 0.4126 | 0.5712 |
| Multilingual evidence | **0.6380** | 0.5728 – 0.6985 | **0.4180** | **0.5783** |

### ⚠️ Multilingual vs Tamil-only evidence: report as no measurable difference

+1.36 pp accuracy is **3 test items out of 221** (one item = 0.45 pp). Macro-F1 differs by 0.005. The confidence intervals overlap almost completely.

**Do not claim multilingual evidence helps Tamil claims.** The honest sentence is: *"Restricting evidence to Tamil produced no measurable change relative to multilingual evidence (Δ accuracy = 1.4 pp, Δ macro-F1 = 0.005, n = 221); the comparison is underpowered at this test-set size."*

### ⚠️ Blocker for PART J: paired significance tests are not currently computable

McNemar's test needs **per-item** agreement counts (b and c). Two confusion matrices cannot produce them — the joint distribution is lost. `m6_07_evaluate_verifier.py` currently saves metrics and confusion matrices but **not per-example predictions**.

Without a fix you can report point estimates and Wilson CIs, but **you cannot state that evidence-aware beats claim-only significantly.** At n = 221 with a 5.9 pp gap that claim is not safe on CIs alone.

**Minimal fix** — add to the `payload` in `m6_07`:

```python
"per_example": [{"claim_id": e.claim_id, "language": e.language,
                 "gold": int(g), "pred": int(p), "n_evidence": len(e.evidence_texts)}
                for e, g, p in zip(ex_test, labels, preds)],
```

~5 lines, no retraining — Stage 7 is a ~3 min inference pass per variant. Then McNemar is a few lines of arithmetic over any two arms.

### ✅ One risk I checked and can rule out

`m7_02 test` produced 10,351 claims while `m7_01 test` produced 10,352 (one article dropped as too short). I checked `assemble()` in `m7_04_fit_fusion.py`: it joins **on `record_id` through a dict**, not by position, and missing records get `has_verifier=False` with `p_fake_verifier=0.5`. **No off-by-one misalignment.** The dropped article degrades gracefully to classifier-only.

---

## Part 2 — System architecture

```
                        ┌──────────────────────────────┐
  raw article text ───► │ 0. Normalise + language ID   │
                        │    NFC only, never NFKC/NFKD │
                        │    Tamil block U+0B80–U+0BFF │
                        └──────────────┬───────────────┘
                                       ├───────────────────────────────┐
                                       ▼                               ▼
                   ┌───────────────────────────┐        ┌──────────────────────────┐
                   │ A. CLASSIFIER PATH        │        │ B. EVIDENCE PATH         │
                   │  Module 2 preprocessing   │        │  claim extraction        │
                   │  Module 3/4 XLM-R + SupCon│        │  LaBSE encode            │
                   │  → embedding → classifier │        │  BM25 + dense retrieval  │
                   │  → P(fake) [calibrated]   │        │  → top-k evidence        │
                   └────────────┬──────────────┘        └───────────┬──────────────┘
                                │                                   ▼
                                │                       ┌──────────────────────────┐
                                │                       │ C. VERIFIER (Module 6)   │
                                │                       │  (claim, evidence) →     │
                                │                       │  SUPPORTS/REFUTES/NEI    │
                                │                       │  temperature applied     │
                                │                       └───────────┬──────────────┘
                                └───────────────┬───────────────────┘
                                                ▼
                              ┌──────────────────────────────────┐
                              │ D. FUSION (Module 7)             │
                              │  calibrated weighted fusion      │
                              │  α, threshold, T frozen from     │
                              │  VALIDATION — never test         │
                              └────────────────┬─────────────────┘
                                               ▼
                              ┌──────────────────────────────────┐
                              │ E. ATTRIBUTION (Module 8)        │
                              │  occlusion / SHAP over tokens    │
                              └────────────────┬─────────────────┘
                                               ▼
                              ┌──────────────────────────────────┐
                              │ F. EXPLANATION ASSEMBLER         │
                              │  SLOT-FILLING ONLY. No generation│
                              └──────────────────────────────────┘
```

### The anti-fabrication rule

**Every sentence in the explanation is a template whose slots are filled from a computed field, and every sentence carries a `provenance` list naming the fields it came from.** A sentence that cannot name its source cannot be emitted.

Enforce it mechanically, not by discipline:

```python
@dataclass(frozen=True)
class ExplanationSentence:
    template_id: str
    text: str
    provenance: list[str]          # e.g. ["fusion.p_fake", "fusion.threshold"]

    def __post_init__(self):
        if not self.provenance:
            raise ValueError(f"{self.template_id}: no provenance -> refusing to emit")


def render(slots: dict) -> list[ExplanationSentence]:
    """Every branch reads a real value or emits an explicit 'unavailable'."""
    out = []
    if slots["evidence"]:
        out.append(ExplanationSentence(
            "evidence_found",
            f"Retrieved {len(slots['evidence'])} candidate passages; the verifier "
            f"judged the top passage {slots['verifier']['verdict']} with probability "
            f"{slots['verifier']['p_top']:.2f}.",
            ["retrieval.candidates", "verifier.verdict", "verifier.probs"]))
    else:
        out.append(ExplanationSentence(
            "evidence_absent",
            "No evidence passage was retrieved from the fixed corpus. "
            "The verdict rests on the text classifier alone.",
            ["retrieval.candidates"]))
    return out
```

A unit test then asserts that **no output string contains a claim not derivable from `slots`** — the same discipline as the `evidence_source` fix in `dataset.py`: derive from reality, never from intent.

### Output contract

```json
{
  "verdict":       {"label": "fake", "decision_rule": "fused_p_fake >= threshold",
                    "threshold": 0.52, "fused_p_fake": 0.83},
  "confidence":    {"value": 0.83, "band": "high", "calibrated": true,
                    "temperature": 1.34, "ece_on_validation": 0.07},
  "language":      {"code": "ta", "method": "unicode_block_ratio",
                    "tamil_char_ratio": 0.87},
  "verification":  {"verdict": "REFUTES", "label_space": "fever",
                    "probs": {"SUPPORTS": 0.11, "REFUTES": 0.74, "NOT ENOUGH INFO": 0.15},
                    "temperature": 3.9213, "applied_to": "test-time logits"},
  "evidence":      [{"rank": 1, "evidence_id": "…", "text": "…", "score": 0.71,
                     "retriever": "bm25", "language": "en", "is_gold": null}],
  "explanation":   [{"template_id": "…", "text": "…", "provenance": ["…"]}],
  "attributions":  {"method": "occlusion", "target": "module4_p_fake",
                    "tokens": [{"text": "…", "span": [10, 18], "delta": 0.12}]},
  "evidence_reason": {"template_id": "verifier_refutes",
                      "text": "…", "provenance": ["verifier.verdict", "evidence[0]"]},
  "caveats":       ["retrieval_recall_at_5_is_0.67", "out_of_domain_verifier"]
}
```

`is_gold: null` is deliberate — at inference there is no gold label, and the field must not be silently filled with a guess.

---

## Part 3 — Three honest limitations the UI must surface

These are not optional disclaimers. Omitting them would make the demo misleading.

### 1. It is not a live fact-checker

Retrieval runs over a **fixed** corpus: 67,388 FEVER evidence items + 6,968 X-FACT items. An article about an event outside that corpus **cannot** have its evidence retrieved. The system will still return the nearest passages, and they may be irrelevant.

**UI requirement:** when the top retrieval score falls below a validation-set threshold, display *"No relevant evidence found in the reference corpus"* and fall back to the classifier-only verdict with reduced confidence.

### 2. Retrieval is wrong more often than it is right

Your own measurements:

| Corpus | Retriever | Recall@5 |
|---|---|---|
| X-FACT (Tamil) | **BM25** | **0.671** |
| X-FACT (Tamil) | dense (LaBSE) | 0.535 |
| FEVER (English) | **BM25** | **0.549** |
| FEVER (English) | dense (LaBSE) | 0.323 |

Even at best, **roughly a third of the time the correct evidence is not in the top 5.** Displayed passages must be labelled *"candidate evidence, retrieved automatically"* — never *"the evidence."*

### 🔧 Direct consequence: use BM25 in the demo, not dense

Your ablation says BM25 beats dense by **+13.6 pp** on X-FACT and **+22.6 pp** on FEVER. Yet the verifier is currently trained and evaluated on `dense_candidates.json`.

**This is the single most defensible route to a higher number**, because it follows from your own measurement rather than from tuning:

```bash
# retrieve already writes bm25_candidates.json for train/validation/test.
# Point load_retrieved() at it via a --retriever flag, then:
python scripts/m6_06_train_verifier.py --dataset x_fact --run-name bm25_evidence \
       --no-resume --init-from models/module6/fever --retriever bm25
python scripts/m6_07_evaluate_verifier.py --dataset x_fact --split test \
       --run-name bm25_evidence --retriever bm25
```

~15 min GPU for X-FACT. If X-FACT accuracy rises above 0.638, that is a **real, earned improvement with a documented cause**. If it does not, that is also a finding worth reporting. Requires a small `--retriever` flag in `load_retrieved()` — I have not written it yet.

### 3. The verifier is out of domain for arbitrary user input

It was trained on FEVER (English Wikipedia entailment) and X-FACT (Tamil fact-check verdicts). A user pasting a random news article is outside both distributions. The verification result is an **auxiliary signal**, not ground truth, and the fusion weight α already reflects that — it was fitted on validation.

---

## Part 4 — Comparing against the base paper

### I cannot do this yet, and I will not guess

The base paper is **not in your reference list** (`docs/research_paper_draft.md` §REFERENCES lists only foundational works — Conneau, Lundberg, Feng, Khosla, Thorne, Gupta & Srikumar, Robertson, Johnson, Guo).

It is referenced only obliquely in `docs/module1_overview.md`: it uses the **IndoHoax** dataset, a **fact-checking-platform + mainstream-newsroom** sourcing design, and a **stacking-by-averaging MLP ensemble** (`src/classifier/mlp_ensemble.py` says *"Mirrors the base paper's ensemble strategy"*).

**Please give me the citation** — authors, title, venue, year — and its reported metrics with the dataset and split each was measured on. Without that, any comparison I produced would be invented, which your own constraints forbid.

### ⚠️ The numbers you listed are the wrong ones to compare

| Your metric | Task | Classes | Comparable to a fake-news paper? |
|---|---|---|---|
| FEVER 0.6337 | textual entailment | 3 (SUPPORTS/REFUTES/NEI) | ❌ **No** |
| X-FACT 0.6380 | fact-checker veracity | 3 (false/true/partly) | ❌ **No** |
| **Module 7 fused fake/real** | **fake-news detection** | **2** | ✅ **Yes — and it is not computed yet** |

Comparing a 3-class entailment score against a binary fake/real accuracy is meaningless in **both** directions — it would understate your system to a reviewer, and overstate it if the numbers happened to fall the other way. The comparable figure comes from `m7_06_evaluate.py`, which has not run. `m7_01` and `m7_02` are done; `m7_03`–`m7_08` are pending.

### On *"my accuracy should be higher than the base paper"*

I have to be straight with you, because you set this rule yourself: *"Do not optimize for making the project look successful; optimize for making the research scientifically correct."*

I cannot promise your number will come out higher, and steering the pipeline until it does is exactly the failure mode your own constraints were written to prevent. What I can do is make the comparison **fair**, and give you the legitimate levers.

**Making it fair** — a comparison is only valid if all four hold:

1. **Same dataset.** If the base paper reports on IndoHoax and you report on WELFake + ISOT + Tamil, the numbers are not comparable at all. State the datasets side by side.
2. **Same split protocol.** Yours is group-safe with near-duplicate removal (leakage gate PASS: 153,664 FEVER groups and 1,462 X-FACT groups each confined to one split). Many published baselines use random splits, which **inflate** their accuracy through near-duplicate leakage. If so, say it — your lower number may be the more honest one.
3. **Same metric.** Report accuracy **and** macro-F1. Under imbalance, accuracy flatters the majority class.
4. **Same task difficulty.** Yours is multilingual Tamil + English with evidence retrieval. Most base papers are monolingual binary classification. Harder task, lower ceiling.

**Legitimate levers, in order of expected value:**

| Lever | Why it is legitimate | Est. gain |
|---|---|---|
| **Run Module 7 fusion** | Fusing classifier + verifier is the contribution. It is not yet measured. | Unknown — this *is* the experiment |
| **Switch verifier evidence to BM25** | Your own ablation measured BM25 ≫ dense. Following your own evidence. | X-FACT: plausibly a few pp |
| **Multiple seeds (42, 43, 44), select on validation** | Already configured; never run. Selection on validation is sound. | Reduces variance; small gain |
| **Class weighting for X-FACT** | Approved ablation arm; the rare class scores F1 = 0. | Macro-F1, not accuracy |
| **Report macro-F1 as primary** | Correct for imbalanced multi-class, and you already do. | Framing, not a gain |

**Not legitimate, and I will not do any of these:** tuning the threshold on test, selecting the best seed by test score, dropping the hard Tamil subset, reporting weighted-F1 while calling it macro-F1, or comparing your best arm against the base paper's worst.

### The honest framing if you end up lower

A well-analysed lower number on a harder task, with a leakage-controlled split and a working evidence pipeline, is defensible in a viva and publishable. A fabricated win is neither, and it is the kind of thing an examiner finds by asking one question about your split protocol.

> "On [dataset], we obtain accuracy X / macro-F1 Y, against the base paper's reported Z. The comparison is not like-for-like: we evaluate on group-safe splits with near-duplicate removal across train/validation/test (leakage gate PASS), whereas [base paper] uses a random split; and our system is multilingual with Tamil labels assigned by source-type heuristic rather than annotation. We additionally report the first evidence-grounded ablation on this task, showing that retrieved evidence contributes +17.4 pp macro-F1 on Tamil X-FACT — with a claim-only baseline that is provably identical to a constant majority predictor."

That last clause is a stronger contribution than a decimal point of accuracy.

---

## Part 5 — Build order

| # | Step | Depends on | Est. |
|---|---|---|---|
| 1 | Add `per_example` predictions to `m6_07`, re-run Stage 7 per arm | — | 5 lines + ~15 min GPU |
| 2 | McNemar + Holm correction across ablation arms | 1 | ~10 min CPU |
| 3 | Finish Module 7 (`m7_03`–`m7_08`) → **the comparable fake/real number** | — | ~1–2 h |
| 4 | Module 8 attributions on the repaired checkpoints | 3 | ~1 h |
| 5 | `src/inference/pipeline.py` — single-article path reusing frozen artifacts | 3, 4 | ~2 h |
| 6 | `src/inference/explain.py` — slot-filling assembler + provenance test | 5 | ~2 h |
| 7 | Demo UI (Streamlit/Gradio) with the three caveats surfaced | 5, 6 | ~2 h |

**Step 4 must not run before step 3.** Attribution over an evidence segment is only interpretable once the model is trained to read that segment — which is precisely what the repair fixed.

---

## What I need from you

1. **The base paper citation** and its reported metrics, with dataset and split for each.
2. Approval to add the `per_example` field to `m6_07` (5 lines) — without it, PART J significance testing is impossible.
3. Whether to add the `--retriever bm25` flag and run the BM25-evidence arm (~15 min GPU, X-FACT).
