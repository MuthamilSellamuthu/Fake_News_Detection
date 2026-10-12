# Base Paper Comparison — Ni'mah et al. (2025)

**Full citation (verified from the PDF you supplied — use this, do not paraphrase from memory):**

> I. Ni'mah, R. Wijayanti, A. Santosa, A. Jarin, T. Sampurno, M. T. Uliniansyah, M. Fang, V. Menkovski, and M. Pechenizkiy, "A simple contrastive embedding framework for low-resource fake news detection," *Neural Computing and Applications*, vol. 37, pp. 21407–21433, 2025. DOI: 10.1007/s00521-025-11467-0. Received 2 June 2024; accepted 3 July 2025; published online 5 August 2025. Open Access (© The Author(s) 2025).

---

## 1. What the base paper actually reports

**Table 5 — IndoHoax (Indonesian), claim + evidence merged as input. Base model `LazarusNLP/simcse-indobert-base`. All values are percentages.**

| Method | ACC | Macro-F1 | Micro-F1 | Real F1 | Real P | Real R | Fake F1 | Fake P | Fake R | Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Evidence-aware** | | | | | | | | | | |
| MAC (ensemble) | 72.7 | 71.1 | 72.7 | 64.3 | 72.8 | 57.6 | 77.8 | 72.6 | 83.9 | 19.2 |
| GET (ensemble) | 63.5 | 52.2 | 63.5 | 28.9 | 86.2 | 17.4 | 75.4 | 61.4 | 97.9 | 20.9 |
| GETRAL (ensemble) | 66.2 | 57.1 | 66.2 | 37.4 | 89.5 | 23.6 | 76.8 | 63.2 | 97.9 | 20.8 |
| ReRead (ensemble) | 44.5 | 34.4 | 44.5 | 8.78 | 75.0 | 4.66 | 60.1 | 43.3 | 97.9 | 26.1 |
| **OpenAI LLM** | | | | | | | | | | |
| GPT-3.5-turbo, claim-only | 67.5 | 67.5 | 67.5 | 66.9 | 58.9 | 77.2 | 68.1 | 78.2 | 60.3 | 25.8 |
| GPT-3.5-turbo, + evidence | 85.9 | 84.7 | 85.9 | 80.4 | 95.7 | 69.3 | 89.1 | 81.9 | 97.8 | N/A |
| GPT-4-turbo, claim-only | 72.7 | 70.8 | 72.7 | 63.2 | 74.5 | 54.9 | 78.3 | 71.9 | 86.0 | 24.1 |
| **GPT-4-turbo, + evidence** | **92.8** | **92.4** | 92.8 | 90.8 | 94.8 | 87.2 | 94.1 | 91.6 | 96.7 | N/A |
| **E2E BERT classifier** | | | | | | | | | | |
| BERT | 68.7 | 68.2 | 68.7 | 64.2 | 95.4 | 48.4 | 72.1 | 57.5 | 96.8 | 31.0 |
| **BERT (ensemble)** | **77.7** | **77.7** | 77.7 | 78.4 | 89.9 | 69.5 | 77.1 | 67.9 | 89.1 | 17.9 |
| **BERT embeddings** | | | | | | | | | | |
| No fine-tuning | 75.2 | 74.7 | 75.2 | 78.3 | 79.6 | 77.1 | 71.1 | 69.6 | 72.6 | 16.3 |
| MLM tuning | 74.6 | 73.6 | 74.6 | 78.6 | 76.8 | 80.6 | 68.6 | 71.1 | 66.3 | 18.2 |
| Classif. tuning | 70.4 | 70.1 | 70.4 | 67.0 | 94.7 | 51.9 | 73.1 | 59.0 | 95.9 | 29.1 |
| **Contrast-BERT (their proposal)** | | | | | | | | | | |
| **Pairwise** | **77.3** | **76.8** | 77.3 | 80.2 | 81.3 | 79.1 | 73.3 | 72.1 | 74.7 | **15.9** |
| Triplet | 74.7 | 74.5 | 74.7 | 76.9 | 81.9 | 72.5 | 72.1 | 67.1 | 77.9 | 16.9 |
| Unsupervised | 76.1 | 75.5 | 76.1 | 79.3 | 79.8 | 78.8 | 71.7 | 71.1 | 72.3 | **15.8** |

**Headline number to cite: Contrast-BERT (Pairwise) — Accuracy 77.3 %, Macro-F1 76.8 %.**

**Retrieval (Table 7, IndoHoax):** Contrast-BERT retrieves ≥ 86.4 % of relevant documents at top-5 (HR@5), MRR@5 up to 0.738. Supervised embeddings: HR@5 ≤ 0.5, MRR@5 < 0.4.

**Dataset:** IndoHoax — Indonesian, class ratio **96:4** fake:real (≈ 1:24), 10,755 train / **337 test** in DeClare format, samples without evidence discarded.

---

## 2. 🔴 Three things you must understand before comparing

### (a) The base paper does **not** win on accuracy — in its own table

GPT-4-turbo with evidence scores **92.8 %** accuracy versus Contrast-BERT's **77.3 %**. BERT (ensemble) scores **77.7 %**, also above their proposed method.

The paper is completely open about this. Its contribution is **not** highest accuracy. From the abstract:

> "we improve the **precision** score in fake news classification up to 26.64 %, while **maintaining** accuracy and recall scores of above 75 %, given extreme class imbalance ratio 1:24"

The claim is a **better precision/recall balance and calibration** (lowest Brier loss, 15.8–15.9) from a **109.5 M-parameter** model, against a ~1.8 T-parameter black box.

**This reframes your question.** "My accuracy should be higher than the base paper" is chasing a target the base paper itself does not claim to own. A reviewer who has read this paper will know that.

### (b) Their test set is 337 samples — the accuracy figure is soft

| Base-paper result | Accuracy | 95 % Wilson CI | CI width |
|---|---:|---|---:|
| Contrast-BERT (Pairwise) | 77.3 % | 72.5 – 81.5 % | 8.9 pp |
| BERT (ensemble) | 77.7 % | 73.0 – 81.8 % | 8.9 pp |
| GPT-4-turbo + evidence | 92.8 % | 89.5 – 95.1 % | 5.6 pp |

Contrast-BERT (77.3) and BERT-ensemble (77.7) have **almost entirely overlapping intervals** — that 0.4 pp gap is not a real difference. Any claim of yours that lands between roughly 72 % and 82 % is statistically indistinguishable from theirs.

### (c) The datasets are completely different — this is **not** a like-for-like comparison

| | Base paper | Yours |
|---|---|---|
| Dataset | IndoHoax | WELFake + ISOT + Tamil (BOOM, Daily Thanthi, Aanirudh) |
| Language | Indonesian | Tamil + English |
| Test size | 337 | ~10,352 (Module 7 test) |
| Class ratio | 96:4 (≈1:24) | far less extreme |
| Labels | fact-checker verdicts | English: real labels; **Tamil: source-type heuristic** |
| Split protocol | not stated | group-safe, near-duplicate removed, **leakage gate PASS** |

You cannot put 77.3 % and your number in the same table as if they measure the same thing. What you can do is put them in adjacent rows with the dataset column filled in, and say so.

---

## 3. 🔴 The numbers you gave me are still the wrong ones to compare

| Your metric | Task | Classes | Comparable to Table 5? |
|---|---|---|---|
| FEVER 0.6337 | textual entailment | 3 | ❌ No |
| X-FACT 0.6380 | fact-checker veracity | 3 | ❌ No |
| X-FACT BM25 Recall@5 = 0.671 | retrieval | — | ⚠️ See below |
| **Module 7 fused fake/real accuracy** | **binary fake-news detection** | **2** | ✅ **Yes — NOT YET COMPUTED** |

**The comparable number does not exist yet.** It comes from `m7_06_evaluate.py`. You have finished `m7_01` and `m7_02`; `m7_03`–`m7_08` are pending. Until those run, there is nothing to compare, and I will not manufacture a comparison from the 3-class numbers.

### ⚠️ Retrieval: HR@5 and Recall@5 are different metrics — do not compare them

- Base paper reports **HR@5** (Hit Rate) = 0.864–0.895: the fraction of queries with **at least one** relevant document in the top 5.
- You report **Recall@5** = 0.671 (BM25) / 0.535 (dense): the fraction of **all** gold evidence items retrieved in the top 5.

When a claim has multiple gold evidence items, **HR@5 ≥ Recall@5 by construction**. Putting 0.671 next to 0.895 and calling it a 22-point deficit would be wrong, and a reviewer would catch it.

To compare honestly you must compute HR@5 on your own data — it is a two-line change to `src/retrieval/retrieval_metrics.py` (`any(g in top_k for g in gold)` instead of the recall fraction), and it needs no retraining, only a re-score of existing candidate files. **I recommend adding it.**

---

## 4. Where you genuinely beat the base paper — and it is not accuracy

| Capability | Base paper | Yours |
|---|---|---|
| Contrastive embedding + MLP ensemble | ✅ | ✅ (Module 4, SupCon) |
| Evidence retrieval | ✅ (embedding similarity) | ✅ (dense LaBSE **+ BM25**, FAISS exact) |
| **Trained fact verifier** | ❌ **none** — retrieval only | ✅ FEVER + X-FACT, separate label spaces |
| **Classifier + verifier decision fusion** | ❌ none | ✅ Module 7, calibrated, fitted on validation |
| **Explainability (SHAP / occlusion)** | ❌ none | ✅ Module 8 |
| **Multilingual** | ❌ monolingual Indonesian | ✅ Tamil + English, XLM-R |
| **Leakage-controlled splits** | not reported | ✅ group-safe, near-duplicate removal, gate PASS |
| **Temperature scaling + ECE** | Brier loss only | ✅ ECE before/after, fitted on validation only |
| **Claim-only control for its own model** | ❌ (only for the LLM baselines) | ✅ **and it is a decisive result** |

### Your strongest single result is one the base paper cannot report

Your X-FACT claim-only arm scored **exactly** the majority-class baseline — accuracy 0.5791855203619910, macro-F1 0.2445081184336199, weighted-F1 0.4248466854231223, matching to 16 decimal places. The claim-only model predicted `false` for all 221 items.

Against that, evidence-aware scores 0.6380 / 0.4180 — **+17.35 pp macro-F1, from a provably-zero-skill baseline.**

The base paper never runs this control on its own model. It only compares claim-only vs. with-evidence for GPT-3.5 and GPT-4. **You can therefore make an attribution claim they cannot**: on Tamil, *all* verification skill comes from retrieved evidence, with a claim-only control that is provably equivalent to a constant predictor.

That is a stronger contribution than a decimal point of accuracy on a different dataset.

---

## 5. How to write the comparison honestly

### ✅ Defensible

> "Our framework extends the contrastive-embedding approach of Ni'mah et al. [X] from monolingual Indonesian to multilingual Tamil–English, and adds three components absent from that work: a trained fact verifier over retrieved evidence, calibrated decision fusion of classifier and verifier, and token-level explanations. On IndoHoax, Ni'mah et al. report 77.3 % accuracy and 76.8 % macro-F1 (n = 337, class ratio 96:4). We evaluate on a different corpus and language pair, so the figures are not directly comparable; we report ours on group-safe splits with cross-split near-duplicate removal, a control not reported in the base paper."

> "Unlike the base paper, which evaluates claim-only versus with-evidence input only for its LLM baselines, we run this control on our own verifier. On Tamil X-FACT the claim-only variant is *exactly* equivalent to a majority-class predictor (accuracy 0.579, macro-F1 0.245), while retrieved evidence raises macro-F1 to 0.418 (+17.4 pp), isolating the evidence contribution unambiguously."

### ❌ Not defensible

- Putting your 3-class FEVER/X-FACT accuracy beside their binary 77.3 % in one table.
- Comparing your Recall@5 to their HR@5.
- "We outperform the base paper" without naming the dataset, the language, the split protocol and the class ratio.
- Comparing your best arm against their worst baseline (ReRead at 44.5 %).
- Omitting that GPT-4-turbo + evidence reaches 92.8 % on their data — the paper reports it, so hiding it looks evasive.

---

## 6. Legitimate levers, ranked

| # | Lever | Justification | Est. gain |
|---|---|---|---|
| 1 | **Run Module 7 (`m7_03`–`m7_08`)** | The comparable number does not exist yet. Fusion is your contribution. | This *is* the experiment |
| 2 | **Switch verifier evidence to BM25** | Your own ablation: BM25 beats dense by +13.6 pp (X-FACT), +22.6 pp (FEVER). Following your own measurement, not tuning. | X-FACT: plausibly several pp |
| 3 | **Add `per_example` predictions to `m6_07`** | Without it McNemar is impossible and no significance claim is safe. ~5 lines, ~3 min/arm inference. | Enables PART J |
| 4 | **Add HR@5** to retrieval metrics | Makes the retrieval comparison to the base paper valid at all. ~2 lines, no retraining. | Enables a fair comparison |
| 5 | **Seeds 42/43/44, select on validation** | Already configured, never run. Validation selection is sound. | Variance reduction |
| 6 | **Class weighting for X-FACT** | Approved arm; rare class currently F1 = 0. | Macro-F1 |
| 7 | **Report Brier loss** | Base paper's calibration metric. You have ECE; Brier makes you directly comparable on calibration — where their own contribution lies. | Comparability |

**Lever 7 is underrated.** The base paper's real claim is calibration and balance under imbalance (Brier 15.8–15.9 vs E2E BERT's 31.0). You already do temperature scaling and ECE. Adding Brier gives you a *directly comparable* axis on the very dimension they compete on — and your FEVER ECE improvement (0.3346 → 0.0725) is a strong calibration result in its own right.

---

## 7. Honest bottom line

I cannot promise your accuracy will exceed 77.3 %, and engineering toward that number is what your own standing rule forbids: *"optimize for making the research scientifically correct."*

What I can say from the evidence:

1. **The target is softer than it looks.** n = 337, CI width 8.9 pp, and their own BERT-ensemble baseline beats their proposed method by 0.4 pp.
2. **It is a different dataset in a different language.** A direct win or loss carries little meaning either way.
3. **You are ahead on capability**, not on that one scalar: verifier, fusion, explainability, multilinguality, leakage control — none of which the base paper has.
4. **Your claim-only control is a result they cannot report**, and it is clean.

Compete on contribution and on calibration. If your accuracy lands near theirs on a harder multilingual task with leakage-controlled splits, that is a good outcome — say so plainly and let the ablation carry the paper.
