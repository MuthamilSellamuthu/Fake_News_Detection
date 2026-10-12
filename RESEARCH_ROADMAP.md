# Research Development Plan
### Explainable Multilingual Fake News Detection using XLM-RoBERTa, Supervised Contrastive Learning, Fact Verification and Explainable AI for Low-Resource Languages

Prepared as Research Supervisor / IEEE Reviewer / AI Research Scientist / Senior ML Engineer review, based on:
1. Base paper — Ni'mah et al., *"A simple contrastive embedding framework for low-resource fake news detection,"* Neural Computing and Applications (2025) 37:21407–21433.
2. Your proposed abstract (team: Kavin Rithik S, Muthamil S, Niranjana Y).
3. Your uploaded workflow diagram ("PROPOSED WORKFLOW").

No code is generated in this document. This is the plan only — implementation proceeds module by module, with your approval required before each one, exactly as the existing project (Modules 1–6, already built in this repository) has been developed so far.

---

## TASK 1 — Base Paper Analysis

### 1.1 Problem Statement
Fake news detection models trained end-to-end on real-world, low-resource-language data fail under **extreme class imbalance** — because such data is typically sourced from fact-checking / hoax-reporting platforms where the public mostly submits claims they *suspect are false*, the collected data skews overwhelmingly toward the fake class (in the paper's own Indonesian dataset, IndoHoax: **96% fake, 4% real**; in ablations pushed to 98:2). Standard end-to-end (E2E) classifiers trained on this distribution generalize poorly to unseen data.

### 1.2 Motivation
- High-quality fact-checked labels require human expert verification, making labeled data for non-English languages scarce and costly.
- Existing low-resource fake-news work (Arabic, Romanian, Malay, Hindi, Indonesian) had not addressed detection under *realistic, extreme* imbalance — most prior benchmarks assume moderate or balanced class ratios.
- Translation-based data augmentation (common for low-resource NLP) introduces cultural/linguistic-nuance errors and degrades model quality.
- Context-based approaches (social graphs, source credibility, engagement signals) need data that's often inaccessible, and don't transfer across domains.

### 1.3 Objectives
- Recast binary fake/real classification as a **contrastive representation-learning problem** instead of direct end-to-end supervised classification, so the model isn't forced to learn directly from a heavily skewed label distribution.
- Build **Contrast-BERT**: BERT fine-tuned with a SimCSE-style contrastive objective (unsupervised, weakly-supervised pairwise, and supervised triplet variants).
- Use the resulting embeddings as frozen features for a **stacking ensemble of MLP classifiers**, each trained on a class-balanced subset, averaged at inference.
- Establish a **checklist of research questions (RQ1–RQ5)** — a reusable evaluation protocol for low-resource contrastive fake-news detection covering contrastive objective design, data filtering, imbalance handling, pretraining choice, and evaluation methodology.

### 1.4 Methodology
1. **Data**: `{Claim (Title), Context (Article), Fact, Label}` tuples collected from four real Indonesian hoax-reporting/fact-checking sources (Mafindo, Saber-Hoax Jabar, Opendata Jabar, Klinik-Hoax Jatim) → merged into **IndoHoax** (12,532 claims, 96:4 imbalance).
2. **Preprocessing**: strip explicit class-leakage phrases (e.g., "This is Hoax," "[FALSE CLAIM]"), drop content/evidence under 20 words, split content/evidence into sentences, pair claims with sentences.
3. **Contrastive objective (Section 3.1)** — three variants, all approximating InfoNCE with in-batch negatives:
   - *Unsupervised*: positive pair = same sentence encoded twice with different dropout masks (SimCSE trick).
   - *Weakly-supervised (pairwise)*: positive pair = (claim title, a sentence from its own content/fact).
   - *Supervised (triplet)*: adds an explicit hard negative — a title randomly sampled from the *opposite* class.
4. **Filtering (Section 3.2)** — removes noisy pairs caused by topic overlap, stylistic similarity, and shared entities across classes, using:
   - *Consistency filtering*: keep a claim–sentence pair only if it appears in the top-5 (cosine similarity) retrieval results of **at least one of nine** independently pretrained embedding models.
   - *Threshold filtering*: for triplets, require similarity(claim, positive) − similarity(claim, negative) > 0.2.
5. **Class imbalance handling (Section 3.3)** — two sequential stages: (a) contrastive pretraining itself (learns from pair structure, not raw label ratio); (b) **stacking ensemble of 10 MLPs**, each trained on an independently balanced 500-fake / 500-real subset, final prediction = average of all 10 probability outputs.
6. **Pretraining choice ablation (Section 3.4)** — compares 10 published Indonesian/multilingual base embedding models across three pretraining categories (symmetric similarity, asymmetric semantic search, classification-tuned).
7. **Evaluation (Section 3.5)** — three tasks: classification (Accuracy, Macro/Micro-F1, per-class P/R/F1, Brier loss), clustering (completeness, homogeneity, ARI, NMI at k=2 and k=40 fine-grained topics), retrieval (HR@5, MRR@5, claim-as-query against content/fact corpus).

### 1.5 Algorithms Used
- **BERT** (monolingual, per language: Indonesian `simcse-indobert-base` etc.) — **not** a multilingual encoder, and **not XLM-RoBERTa**.
- **SimCSE-style contrastive learning** (Gao et al. 2021) → InfoNCE loss with in-batch negatives, three supervision variants (unsupervised / pairwise / triplet).
- **Stacking ensemble of MLPs** trained on balanced subsets, prediction-averaged at inference.
- Baselines compared against: MAC, GET, GETRAL, ReRead (evidence-aware GNN/attention methods), plain E2E BERT classifier, BERT embeddings (no fine-tune / MLM-tuned / classification-tuned), GPT-3.5-turbo and GPT-4-turbo (claim-only and claim+evidence).

### 1.6 Workflow
```
Hoax-reporting data collection (Claim, Context, Fact, Label)
        ↓
Preprocessing (leakage removal, sentence splitting, pairing)
        ↓
Consistency + Threshold Filtering (noisy pair removal)
        ↓
Contrastive fine-tuning of BERT → Contrast-BERT
        ↓
Frozen embedding extraction (inference-time encoder)
        ↓
Ensemble of 10 balanced-subset MLP classifiers → averaged prediction
        ↓
Evaluation: Classification / Clustering / Retrieval / Bias audit
```

### 1.7 Datasets Used
IndoHoax (own, Indonesian, 96:4), Tallip-ID (Indonesian, zero-shot topic-clustering eval only), Clickbait-ID (Indonesian, multiple constructed imbalance ratios 50:50 → 98:2), BanFakeNews (Bangla, ~2:98), CsFEVER / CsFEVER-NLI (Czech, translated from FEVER, re-balanced to match IndoHoax's ratio).

### 1.8 Strengths
- Contrast-BERT (Pairwise) beats E2E BERT on **balanced** Precision/Recall across both classes under extreme imbalance, and has the **lowest Brier loss (15.8)** — i.e., is less overconfident when wrong.
- Far stronger **clustering** (fine-grained topic alignment, k=40: NMI 0.461 vs. ≤0.2 for supervised embeddings) and **retrieval** (HR@5 up to 88.2% vs. ≤42% for supervised embeddings) performance.
- **Less biased** toward sensational-tone text than E2E BERT (Table 9 fairness audit: statistical parity, disparate impact, equality-of-opportunity all closer to ideal).
- Findings replicate across three genuinely different low-resource languages (Indonesian, Bangla, Czech).
- Zero-shot transfer: performs well on an unseen Indonesian-translated dataset with different topic categories.

### 1.9 Weaknesses
- **Recall trade-off**: because the contrastive model never sees explicit label supervision, it under-recalls the fake class relative to E2E classifiers in some settings (Section 5.4, imbalance-ratio ablation) — precision improves, recall can drop.
- **Augmenting real-news data from other domains doesn't help** — distribution mismatch between the platform's own "real" claims and generic reputable-news real articles; the authors note this as an unresolved stylistic problem.
- GNN-based evidence-aware baselines (GET, GETRAL) underperform specifically because IndoHoax provides only a **single fact/evidence field**, not a rich multi-evidence corpus — the graph structure these baselines rely on has nothing to exploit.
- GPT-4-turbo, the strongest baseline overall, **underperforms specifically on the Indonesian dataset** relative to other languages — an unaddressed linguistic/cultural gap.

### 1.10 Limitations (explicit + implicit)
| # | Limitation | Explicit / Implicit |
|---|---|---|
| 1 | Single-language, single-encoder scope — a *separate* BERT variant is trained per language; no shared multilingual representation space. | Implicit (architecture never varies from monolingual BERT) |
| 2 | No fact-verification pipeline — "evidence" is a static, pre-paired single fact field, never actively retrieved from an external trusted-source corpus at inference time. | Implicit |
| 3 | No explainability / interpretability component at all — no SHAP, LIME, or attention-based rationale for individual predictions. | Implicit |
| 4 | Filtering pipeline (consistency filtering) depends on **nine separately pretrained embedding models being available in the target language** — impractical to replicate for languages without that pretrained-model ecosystem (e.g., Tamil). | Implicit |
| 5 | Ensemble is simple bagging-style averaging of 10 MLPs on fixed 500:500 subsets — no adaptive weighting, no boosting, no loss reweighting alternative explored. | Implicit |
| 6 | No deployed system / web interface — purely an offline research evaluation. | Implicit |
| 7 | No uncertainty/confidence-calibration mechanism beyond reporting Brier loss post hoc — no abstention or "unverified" output category. | Implicit |
| 8 | Real-news augmentation from other domains fails; the stylistic/topic-overlap problem across classes remains unsolved. | **Explicit** ("we view a new approach with data augmentation strategy to enforce models to be invariant to topic overlap, stylistic similarity, and shared entities as a future research direction," §4.1) |

### 1.11 Future Work (stated by the authors)
The paper has no dedicated "Future Work" section; the only explicit forward-looking statement is in §4.1: a future data-augmentation strategy to make models invariant to topic overlap, stylistic similarity, and shared entities across classes. Everything else in the gap table above is *implicit* — inferable from what the paper does not attempt, not stated as a roadmap.

### 1.12 Research Gaps — Consolidated List
1. **No shared multilingual encoder** (biggest gap — separate monolingual model per language).
2. **No active fact-verification / evidence-retrieval system** (evidence is static and pre-supplied, never retrieved).
3. **No explainability** — zero interpretability of individual predictions.
4. **Filtering pipeline not portable to low-resource languages lacking many pretrained embedding variants.**
5. **Simple bagging ensemble only** — no adaptive/weighted ensembling explored.
6. **No deployment layer.**
7. **No confidence/uncertainty output** — binary prediction only, no "unverified" option.
8. **Unresolved stylistic-invariance problem** (explicitly flagged by the authors as future work).

---

## TASK 2 — Abstract Review

### 2.1 Your Abstract (as submitted)
> *"Explainable Multilingual Fake News Detection using Contrastive Learning and Fact Verification for Low-Resource Languages."* Proposes an Explainable Multilingual Fake News Detection Framework using Contrastive Learning and Fact Verification. Uses XLM-RoBERTa + contrastive learning for robust multilingual representations. A fact-verification module retrieves evidence from trusted sources and measures claim–evidence consistency. SHAP is integrated for interpretability. Evaluated via accuracy, precision, recall, F1-score.

### 2.2 Coverage Comparison

| Aspect | Base Paper | Your Abstract |
|---|---|---|
| Encoder | Monolingual BERT (per language) | **XLM-RoBERTa (shared multilingual)** ✅ new |
| Contrastive learning | Yes (SimCSE-style: unsup/pairwise/triplet) | Mentioned generically as "contrastive learning" — method unspecified |
| Class imbalance | Central, explicit contribution (extreme 96:4+) | **Not mentioned at all** ⚠️ |
| Ensemble MLP classifier | Yes (10-way balanced-subset ensemble) | Not mentioned |
| Fact verification | Static, pre-paired single fact field; never actively retrieved | **Active retrieval from "trusted sources" + consistency scoring** ✅ new |
| Explainability | None | **SHAP** ✅ new |
| Evaluation metrics | Accuracy, F1, Brier loss, clustering (ARI/NMI), retrieval (HR@k/MRR@k), fairness audit | Accuracy, Precision, Recall, F1 only — narrower |
| Target languages | Indonesian, Bangla, Czech | Unspecified ("low-resource languages") |
| Deployment | None | Implied by your workflow diagram (Application/Deployment block) but not in abstract text |

### 2.3 What Is Already Covered vs. What Is New
**Already covered by base paper:** contrastive-embedding-based (not end-to-end) fake news detection; low-resource-language framing.

**Genuinely new in your abstract:** (a) a *shared* multilingual encoder (XLM-RoBERTa) replacing per-language monolingual BERT — this directly closes Gap #1; (b) an *active* fact-verification module — closes Gap #2; (c) SHAP explainability — closes Gap #3.

### 2.4 Is It Sufficient for Publication? (Reviewer's Verdict)
**Directionally yes, but not as currently worded.** The three contributions above are real, defensible, and each individually closes a genuine, citable gap in the base paper — that is a legitimate paper. However, as an IEEE/Springer reviewer, I would flag the abstract as **not yet publication-ready** for three concrete reasons:

1. **It never mentions the base paper's central problem — extreme class imbalance.** If your framework doesn't explicitly retain and extend that handling (contrastive pretraining + balanced ensemble), a reviewer familiar with the base paper will ask "did they solve the *same* problem, or a different, easier one?" Silence here reads as either an oversight or an admission you dropped the hardest part.
2. **"Contrastive learning" and "measures consistency between claims and evidence" are too vague.** Reviewers want named methods: *which* contrastive formulation (Supervised SimCSE, specifically what defines positive/negative pairs in your setting since you don't have the base paper's claim–evidence structure), *which* similarity method for verification (cosine similarity over Sentence-BERT embeddings, per your workflow diagram).
3. **No explicit gap statement.** IEEE-style abstracts open with "prior work suffers from X; we address X by doing Y." Yours states what you built, not the gap it closes.

### 2.5 Missing Novelty / What to Add
- Explicit statement of the imbalance problem you inherit and extend.
- Explicit language pair(s) — name them (recommend English + Tamil, see Task 8).
- Explicit note that your contrastive pairing is **label-supervised** (same-label = positive, opposite-label = hard negative) rather than the base paper's claim–evidence structure — since your collected data (news articles with fake/real labels) doesn't have a paired-evidence field the way IndoHoax does. This is an honest, necessary adaptation, not a weakness, but it must be stated, or a reviewer who reads the base paper will ask why your pairing differs.
- A retrieval/clustering evaluation claim (carry forward the base paper's HR@k/MRR@k and ARI/NMI metrics — they are exactly what let the base paper demonstrate embedding quality beyond raw accuracy).
- Confidence/uncertainty framing — your own workflow diagram already lists "Partially True / Unverified" as fact-verification outputs; the abstract should claim this as a contribution (Gap #7).

### 2.6 Suggested Improved Abstract
> *"Explainable Multilingual Fake News Detection using XLM-RoBERTa, Supervised Contrastive Learning, and Evidence-Aware Fact Verification for Low-Resource Languages"*
>
> Fake news detection in low-resource languages remains difficult due to scarce labeled data and, in real-world fact-checking pipelines, extreme class imbalance between fake and real claims. Prior work (Ni'mah et al., 2025) addresses this imbalance using contrastive-embedding-based classification, but trains a separate monolingual encoder per language, provides no active evidence-retrieval mechanism, and offers no prediction-level interpretability. This work extends that framework along three axes. First, we replace the monolingual encoder with XLM-RoBERTa, a shared multilingual backbone, fine-tuned via **supervised contrastive learning** (label-supervised positive/negative pairing) to produce a single cross-lingual embedding space usable across English and Tamil without per-language retraining. Second, we introduce a **fact verification module** that retrieves top-k evidence sentences from a trusted reference corpus via Sentence-BERT embeddings and cosine similarity, and outputs a claim–evidence consistency verdict (True / False / Partially True / Unverified) alongside the classifier's prediction. Third, we integrate **SHAP** to attribute each prediction to the input tokens that most influenced it, giving human-readable rationale absent from the base framework. The resulting embeddings are evaluated not only on classification accuracy, precision, recall, and F1-score, but also on clustering quality and retrieval performance (HR@k, MRR@k), following the base paper's own evaluation protocol, to directly demonstrate improvement over the single-language baseline. The system is deployed as an interactive web application, providing an end-to-end, explainable, evidence-aware fake news detection pipeline for low-resource multilingual settings.

---

## TASK 3 — New Workflow Design

### 3.1 Critical Issue With Your Uploaded Workflow Diagram
Before presenting the new design, a specific finding as your reviewer: **your uploaded workflow diagram, if implemented exactly as drawn, does not extend the base paper — it reproduces the base paper's *weaker* baseline.** Block 5 ("MODELING — Multilingual BERT/XLM-RoBERTa Model for Fake News Detection") goes directly from tokenization to a fine-tuned classification model, with no contrastive-learning stage in between. That is precisely the **E2E BERT classifier** the base paper explicitly benchmarks *against* and shows underperforms Contrast-BERT under extreme imbalance (Table 5: E2E BERT F1-macro 68.2 vs. Contrast-BERT Pairwise 76.8). If this diagram is implemented literally, the resulting paper would not be extending Ni'mah et al. — it would be a multilingual version of the exact baseline they beat. The fix is structural, not cosmetic: insert a **Supervised Contrastive Fine-Tuning** stage and a **frozen-embedding + MLP classification** stage where the diagram currently has one direct "Modeling" block. This is already how the project's existing implementation (Modules 3–5, built earlier) is structured — the diagram needs to catch up to that, not the other way around.

### 3.2 New Workflow (Text Form)
```
1. Multilingual Dataset Collection (English + Tamil)
        ↓
2. Data Preprocessing (clean, dedup, label-encode, language-tag, merge)
        ↓
3. Train / Validation / Test Split (stratified, 70/15/15)
        ↓
4. XLM-RoBERTa Feature Extraction (frozen contextual embeddings, SentencePiece tokenizer)
        ↓
5. Supervised Contrastive Fine-Tuning (label-supervised SimCSE / InfoNCE) → Contrast-XLMR
        ↓
6. MLP Classification (ensemble or Focal-Loss single model, on Contrast-XLMR embeddings)
        ↓
7. Performance Evaluation (classification + clustering + retrieval + imbalance ablation + fairness)
        ↓
8. Fact Verification (Sentence-BERT retrieval + cosine similarity + verdict)
        ↓
9. Explainable AI (SHAP attribution)
        ↓
10. Confidence / Uncertainty Layer (calibrated probability → Unverified band)
        ↓
11. Web Application Deployment (Flask)
```

### 3.3 Module-by-Module Detail

**Module 1 — Multilingual Dataset Collection**
- *Purpose*: gather real English + Tamil fake/real news at sufficient volume.
- *Input*: none (raw web/dataset sources).
- *Output*: raw per-source files (CSV/JSONL).
- *Algorithm*: source-specific collectors — direct download for structured datasets (WELFake, ISOT), sitemap-driven scraping for Tamil news portals, listing-page scraping for Tamil fact-check sites.
- *Why this*: base paper's data comes from a closed, single-organization pipeline (Mafindo etc.) not reproducible for English/Tamil; a custom multi-source collector is required.
- *Why better than base paper*: base paper doesn't need this step (data was already collected by their institution); yours must be built and documented as an original contribution, and covers two typologically different languages/scripts instead of one.

**Module 2 — Preprocessing**
- *Purpose*: produce a clean, merged, stratified-split, label-consistent multilingual corpus.
- *Input*: raw per-source files.
- *Output*: `train.csv`, `val.csv`, `test.csv`.
- *Algorithm*: rule-based cleaning (HTML/URL/emoji/symbol removal, Tamil Unicode preservation), Unicode-ratio + statistical language detection, per-source label-convention mapping, stratified (label × language) split.
- *Why this*: base paper's preprocessing is simpler (single language, single label convention already established by the source platform); yours must reconcile *heterogeneous* label conventions across sources and preserve non-Latin script correctly.
- *Why better*: explicit multilingual-safe text cleaning is a real, citable methodological addition the base paper never had to solve.

**Module 3 — XLM-RoBERTa Feature Extraction**
- *Purpose*: baseline frozen multilingual contextual embeddings (also used for ablation: "with vs. without contrastive fine-tuning").
- *Input*: cleaned text.
- *Output*: pooled embedding vectors (HDF5).
- *Algorithm*: pretrained `xlm-roberta-base`, SentencePiece tokenization, attention-mask-weighted mean pooling.
- *Why XLM-RoBERTa*: see Task 6.
- *Why better than base paper*: replaces per-language monolingual BERT with one shared encoder — directly closes Gap #1.

**Module 4 — Supervised Contrastive Fine-Tuning**
- *Purpose*: produce Contrast-XLMR — an embedding space where same-label texts cluster and opposite-label texts separate, mirroring the base paper's core mechanism but on a shared multilingual backbone.
- *Input*: cleaned text + label.
- *Output*: fine-tuned encoder checkpoint + Contrast-XLMR embeddings.
- *Algorithm*: label-supervised SimCSE (same-label = positive, opposite-label = hard negative, in-batch negatives, false-negative-masked InfoNCE loss).
- *Why this over the base paper's triplet method*: base paper defines positive pairs via claim–evidence structure your data doesn't have; label-based pairing is the correct, honest substitute — must be stated explicitly (see Task 2.5).
- *Why better*: single contrastive space shared across English and Tamil, enabling cross-lingual transfer the base paper's per-language models cannot do by construction.

**Module 5 — MLP Classification**
- *Purpose*: final fake/real prediction from Contrast-XLMR embeddings.
- *Input*: embeddings + label.
- *Output*: trained classifier + prediction probabilities.
- *Algorithm*: MLP (either balanced-subsample bagging ensemble mirroring the base paper, or single MLP + Focal Loss as a documented alternative — both already implemented in this project).
- *Why better*: Focal Loss variant directly targets imbalance in the loss function itself, an alternative the base paper never explores (they only use ensemble-balancing, not loss reweighting) — a legitimate additional contribution if you report both and compare.

**Module 6 — Performance Evaluation**
- *Purpose*: prove the multilingual extension doesn't sacrifice what made Contrast-BERT good.
- *Input*: trained classifier + test embeddings.
- *Output*: metrics report.
- *Algorithm*: accuracy/precision/recall/F1/Brier loss/ROC-AUC, KMeans clustering + ARI/NMI/silhouette, cosine-similarity retrieval (HR@k/MRR@k), imbalance-ratio ablation, **per-language fairness audit** (English vs. Tamil — not in base paper, since base paper's fairness audit was about *tone*, not *language*, because it never had more than one language per model).
- *Why better*: adds a fairness axis (per-language performance gap) that is structurally impossible in the base paper's single-language-per-model design.

**Module 7 — Fact Verification** *(new stage vs. base paper — not yet implemented)*
- *Purpose*: give claims active, retrieved evidence rather than a static pre-paired fact field.
- *Input*: a claim + a reference evidence corpus (trusted sources).
- *Output*: top-k retrieved evidence + True/False/Partially True/Unverified verdict.
- *Algorithm*: Sentence-BERT embeddings + cosine-similarity top-k retrieval + a similarity-threshold-based verdict rule (extendable to a lightweight entailment classifier later).
- *Why Sentence-BERT + cosine, not XLM-R contrastive embeddings*: see Task 6 — Sentence-BERT is purpose-built and pretrained specifically for semantic search/retrieval, which the base paper's own consistency-filtering step (Section 3.2) already implicitly relies on for a *different* purpose (data cleaning) — you are promoting that same retrieval capability into a live, user-facing pipeline stage instead of an offline preprocessing trick.
- *Why better than base paper*: base paper's "evidence" is a static dataset field; yours is an active retrieval system against a live/expandable corpus — this alone is publishable novelty (Gap #2).

**Module 8 — Explainable AI** *(new stage vs. base paper — not yet implemented)*
- *Purpose*: attribute each prediction to the words that drove it.
- *Input*: trained MLP + a single prediction instance.
- *Output*: per-token contribution scores + visualization.
- *Algorithm*: SHAP (KernelExplainer or a linear/deep explainer matched to the MLP).
- *Why better than base paper*: zero interpretability existed before — direct closure of Gap #3.

**Module 9 — Confidence / Uncertainty Layer** *(new — realistic novelty add-on, see Task 7)*
- *Purpose*: avoid overconfident wrong answers; expose an explicit "Unverified" band instead of forcing binary output.
- *Input*: classifier probability + fact-verification verdict.
- *Output*: calibrated confidence score + abstention flag.
- *Algorithm*: probability calibration (temperature scaling or isotonic regression) + a confidence threshold below which output is "Unverified" rather than a forced Fake/Real call.
- *Why better*: base paper reports Brier loss as a *diagnostic* metric but never *acts* on it (no abstention mechanism) — yours closes Gap #7 by turning calibration into a user-facing decision.

**Module 10 — Web Application Deployment**
- *Purpose*: make the pipeline usable, not just evaluable.
- *Input*: user-submitted claim/article text.
- *Output*: prediction + evidence + SHAP explanation, rendered in a UI.
- *Algorithm*: Flask backend serving the trained pipeline; simple HTML/JS frontend.
- *Why better than base paper*: base paper has no deployment layer at all — closes Gap #6.

---

## TASK 4 — Two-Semester Division

### Semester 1 — Core Publishable Model
*Goal: a complete, defensible extension of the base paper's core method — multilingual contrastive embeddings + classifier + evaluation. This alone, done rigorously, is sufficient for a first IEEE/Springer submission (workshop or main-track, framed as the primary contribution).*

| Modules | Expected Outcome | Deliverable | Publication Contribution |
|---|---|---|---|
| 1. Dataset Collection | ≥25k rows each, English + Tamil | Raw datasets, collection scripts | Multilingual low-resource dataset construction (citable in Datasets section) |
| 2. Preprocessing | Clean, merged, stratified splits | `train/val/test.csv` | Reproducible multilingual preprocessing pipeline |
| 3. XLM-R Feature Extraction | Frozen baseline embeddings | Embedding files + baseline metrics | Ablation reference point ("without contrastive fine-tuning") |
| 4. Supervised Contrastive Fine-Tuning | Contrast-XLMR checkpoint | Fine-tuned encoder + embeddings | **Primary novel contribution**: shared multilingual contrastive space (closes Gap #1) |
| 5. MLP Classification | Trained classifier(s) | Ensemble + Focal-Loss checkpoints | Comparative classifier-design contribution |
| 6. Performance Evaluation | Full metrics incl. clustering/retrieval/fairness | Evaluation report | Direct, metric-for-metric comparison against base paper's own evaluation protocol |

**Status**: Modules 1–6 above are **already implemented** in this repository (code, configs, and CLI scripts exist for all six; Module 1 needs a longer real collection run for Tamil volume, and Modules 3–6 need a real GPU run to produce actual numbers). Semester 1, in other words, is largely *built* — what remains is executing it for real and writing it up.

### Semester 2 — Advanced Features
*Goal: the differentiating features that make this a stronger, more complete paper than a "same idea, more languages" resubmission.*

| Modules | Expected Outcome | Deliverable | Publication Contribution |
|---|---|---|---|
| 7. Fact Verification | Working retrieval + verdict module | Sentence-BERT retrieval pipeline | Closes Gap #2 — evidence-aware, not just classification |
| 8. Explainable AI | Per-prediction SHAP explanations | Explainability module + example figures | Closes Gap #3 — interpretability |
| 9. Confidence/Uncertainty | Calibrated, abstention-capable output | Calibration module | Closes Gap #7 — realistic novelty, low implementation cost |
| 10. Deployment | Working web app | Flask application | Demonstrates end-to-end system, not just offline metrics |
| — Additional novelty (Task 7) | Selected combination implemented | Ablation results | Strengthens novelty claim beyond "just added 3 modules" |

---

## TASK 5 — Complete Architecture

```
Dataset Collection (English + Tamil)
        ↓        [algorithm: source-specific collectors | reason: no existing merged multilingual corpus | improvement: enables the whole study]
Preprocessing
        ↓        [algorithm: rule-based cleaning + stratified split | reason: heterogeneous multi-source antlabel conventions | improvement: reproducible, leakage-free corpus]
Feature Extraction (XLM-RoBERTa, frozen)
        ↓        [algorithm: pretrained XLM-R + mean pooling | reason: need a multilingual-capable baseline | improvement: single encoder replaces N monolingual ones]
Representation Learning (Supervised Contrastive Fine-Tuning)
        ↓        [algorithm: label-supervised SimCSE / InfoNCE | reason: base paper's core mechanism, adapted for label-only supervision | improvement: robustness under extreme imbalance, shared cross-lingual space]
Classification (MLP)
        ↓        [algorithm: balanced-ensemble MLP or Focal-Loss MLP | reason: base paper's proven classifier stage on frozen embeddings | improvement: two documented imbalance-handling strategies compared]
Evaluation
        ↓        [algorithm: classification + clustering + retrieval + fairness metrics | reason: reproduce base paper's full evaluation protocol | improvement: adds per-language fairness axis]
Fact Verification (Sentence-BERT + Cosine Similarity)
        ↓        [algorithm: dense retrieval + similarity threshold | reason: base paper's evidence is static, not retrieved | improvement: active, evidence-aware verdict]
Explainability (SHAP)
        ↓        [algorithm: Shapley-value attribution | reason: zero interpretability in base paper | improvement: human-readable rationale per prediction]
Confidence Layer
        ↓        [algorithm: probability calibration + threshold | reason: base paper never acts on its own Brier-loss finding | improvement: abstention on low-confidence cases]
Deployment (Flask Web Application)
                 [algorithm: REST backend + simple UI | reason: base paper is offline-only | improvement: usable end-to-end system]
```

---

## TASK 6 — Algorithm Justification

### Feature Extraction: Why XLM-RoBERTa?
- **Why not BERT**: BERT (as used in the base paper) is monolingual per checkpoint — using it would require training and maintaining a separate model per language, exactly the limitation you're extending past.
- **Why not mBERT**: mBERT is multilingual but trained on comparatively small Wikipedia-only corpora per language with a shared vocabulary that underserves low-resource, non-Latin scripts like Tamil — subwords fragment excessively, degrading representation quality. XLM-RoBERTa trains on ~2.5TB of CommonCrawl text (CC-100) across 100 languages with upsampling for low-resource languages and a much larger (250k-token) SentencePiece vocabulary, giving Tamil meaningfully better tokenization and more pretraining exposure. Empirically (Conneau et al., 2020), XLM-R closes much of mBERT's low-resource-language performance gap.
- **Why XLM-R over other multilingual options** (e.g., mT5, LaBSE): XLM-R is an encoder-only model matched to the base paper's own BERT-encoder-based design, making the comparison to Contrast-BERT direct and fair — swapping in a generative (mT5) or dual-encoder-specialized (LaBSE) architecture would confound the comparison with an unrelated architectural change.

### Contrastive Learning: Why Supervised SimCSE?
- **Why not Triplet Loss (standalone)**: standard triplet loss uses one negative per anchor and no in-batch negatives, which is less sample-efficient and more sensitive to which single negative happens to be sampled. Supervised SimCSE (as adapted here) keeps an explicit hard negative *and* gets additional negatives "for free" from every other example in the batch, which is more data-efficient — an important property given how little Tamil data currently exists.
- **Why not MoCo (Momentum Contrast)**: MoCo's momentum encoder + large negative queue was designed for vision, where negative diversity at very large batch-equivalent scale matters most; it adds architectural complexity (momentum encoder, queue management) that isn't justified at this project's data scale, and the base paper's own method (SimCSE-style) is the direct lineage you're extending — switching to MoCo would break the "extends the base paper's mechanism" framing a reviewer will look for.
- **Why label-supervised pairing (not claim–evidence)**: your data lacks the base paper's paired evidence field; label-based positive/negative pairing is the correct, defensible substitute (must be stated explicitly in the paper, per Task 2.5).

### Classification: Why MLP?
- **Why not CNN**: CNNs exploit local spatial/n-gram structure, which is redundant here — the input is already a single pooled, fixed-length embedding vector (not a token sequence), so there's no spatial structure left for convolution to exploit.
- **Why not LSTM**: LSTMs are built for sequential/token-level input; again, the classifier's input here is a single fixed-size vector per example, not a sequence — an LSTM over a length-1 "sequence" degenerates to no better than an MLP, with unnecessary complexity and training cost. The base paper's own classifier stage is MLP-based specifically because the sequence modeling has already happened inside the encoder — the classifier's job is a much simpler decision-boundary problem on top of already-rich embeddings.

### Fact Verification: Why Sentence-BERT? Why Cosine Similarity?
- **Why Sentence-BERT over raw XLM-R/Contrast-XLMR embeddings for retrieval**: Sentence-BERT (and its multilingual variants) is explicitly trained with a retrieval/semantic-similarity objective (Siamese/triplet loss on sentence pairs), producing embeddings where cosine distance directly reflects semantic similarity out of the box. Contrast-XLMR is optimized to separate *fake vs. real*, not to rank arbitrary claim–evidence semantic relevance — using it for retrieval would conflate two different objectives.
- **Why cosine similarity over other similarity measures**: cosine similarity is scale-invariant (ignores embedding magnitude, focuses purely on direction/semantic alignment), is the standard, well-validated choice for sentence-embedding retrieval (used by Sentence-BERT's own authors and by the base paper's own consistency-filtering step, Section 3.2), and is computationally cheap enough for real-time retrieval in a deployed web app.

### Explainability: Why SHAP? Why not LIME?
- SHAP provides theoretically grounded, consistent attributions (Shapley values from cooperative game theory guarantee properties like local accuracy and consistency that LIME's local-surrogate-model approach does not guarantee).
- LIME's explanations can vary between runs (it fits a new local linear model via random perturbation sampling each time) and its fidelity depends heavily on the perturbation/sampling strategy chosen; SHAP's KernelExplainer is a strict generalization of LIME with better theoretical guarantees.
- SHAP has strong, well-maintained tooling (`shap` Python package, already in this project's `requirements.txt`) with built-in visualization (force plots, summary plots) suited to a deployed web app's "why did the model say this" UI requirement.

---

## TASK 7 — Additional Novelty (Realistic, Low-Cost)

Recommended combination — each item is achievable with data/tools already available to this project, none require paid APIs or unavailable datasets:

1. **Multilingual low-resource learning** *(already core to your abstract)* — the shared XLM-R contrastive space itself.
2. **Confidence scoring / uncertainty estimation** *(Module 9 above)* — cheap to add (a calibration layer on top of an already-trained classifier), and directly answers the base paper's own unresolved overconfidence finding (Brier loss).
3. **Multilingual retrieval** — extend the fact-verification module's retrieval step to explicitly support cross-lingual retrieval (e.g., a Tamil claim retrieving English evidence, or vice versa) using the shared Contrast-XLMR space — a natural, low-cost extension since the multilingual embedding space already exists once Module 4 is done.
4. **Per-language fairness/bias audit** *(already in Module 6)* — reframe this explicitly as a novelty claim: the base paper's fairness axis was *tone*; yours adds *language* as a second, structurally new fairness axis only possible because you have a shared multilingual model.

**Not recommended** given your constraints: adaptive contrastive learning (hard-negative mining schedules, curriculum learning) and evidence-aware graph reasoning (GNN-based, mirroring GET/GETRAL) — both are legitimate research directions but add substantial implementation and tuning risk for a two-semester final-year timeline, and the base paper's own results show GNN-based evidence reasoning underperforms when evidence is sparse (exactly your data situation too).

**Best combination for your scope**: items 1–4 above. Together they form a coherent narrative — *"a shared multilingual contrastive space enables not just classification, but cross-lingual retrieval, per-language fairness auditing, and calibrated confidence — none of which are possible in a single-language model"* — which is a stronger, more unified novelty argument than a disconnected list of add-ons.

---

## TASK 8 — Public Dataset Recommendations

### English
| Dataset | Why Selected | Advantages | Preprocessing Required |
|---|---|---|---|
| **WELFake** *(already collected — 63,368 rows in this project)* | Large (~72k rows), merges multiple established English fake-news sources, widely used/citable benchmark | Big enough alone to exceed a 25k-row target; well-documented label convention | Label flip (dataset ships 0=fake/1=real, opposite of this project's convention) |
| **ISOT Fake News Dataset** *(already collected)* | Classic, widely cited benchmark, file-per-class labeling is unambiguous | Easy label assignment (Fake.csv/True.csv) | **Caution**: this project found ISOT's content ~100% duplicated inside WELFake after merging — treat as a redundancy check, not a volume source |

### Tamil
| Dataset / Source | Why Selected | Advantages | Preprocessing Required |
|---|---|---|---|
| **Dina Thanthi (dailythanthi.com), sitemap-collected** *(already implemented, partially collected)* | Real, actively updated Tamil mainstream news; sitemap crawling reaches genuine volume (hundreds of articles/day) unlike front-page scraping | Large potential volume, real "mainstream news = likely real" labeling signal (mirrors base paper's own IndoHoax source-type-default labeling approach) | Source-type-default labeling (no structured verdict field); Tamil Unicode-safe cleaning |
| **BOOM Tamil (fact-check desk)** *(already implemented, small volume, currently blocked by bot detection)* | Real Tamil-language fact-checking outlet — mirrors the base paper's own IndoHoax data-generation process (fact-check platform → fake-skewed claims) | Directly analogous role to the base paper's Mafindo source | Keyword-based verdict extraction from article text (no structured field); needs a non-bot-like scraping approach to unblock |
| **Dinamalar, The Hindu Tamil** *(scaffolded, not yet populated)* | Additional mainstream Tamil sources for volume/diversity once BOOM/Dina Thanthi prove the pipeline | Source diversity reduces overfitting to one outlet's writing style | Same source-type-default labeling caveat |
| **Tamil Fake News datasets on Kaggle / academic releases** (if found with usable licensing) | Would give a structured, pre-labeled Tamil dataset analogous to WELFake/ISOT, removing the source-type-default labeling approximation | Removes the biggest methodological weak point in current Tamil labeling (self-declared vs. verified labels) | Standard dedup/clean/label-mapping only, no bespoke scraping |

**Honest caveat to report in the paper**: unlike English (WELFake/ISOT ship with verified ground-truth labels), the current Tamil pipeline assigns labels via a source-type default (fact-check-source claims default to fake, mainstream-news defaults to real) plus keyword scanning — the same approximation strategy the base paper itself uses for IndoHoax, but worth stating explicitly as a limitation, with a recommendation for a manual verification pass on a sample before final publication-grade claims.

---

## TASK 9 — Development Roadmap (Day 1 → Publication)

No source code is generated in this roadmap. Each module below is implemented **only after your explicit approval**, exactly as this project has proceeded so far.

| Phase | Timeframe (indicative) | Modules | Status |
|---|---|---|---|
| **Phase 0** | Week 1 | Literature review consolidation (this document), abstract finalization, architecture sign-off | ✅ This document |
| **Phase 1** | Weeks 2–4 | Module 1: Dataset Collection (English + Tamil, real volume) | Built; needs a full real run for Tamil volume |
| **Phase 2** | Weeks 4–5 | Module 2: Preprocessing | Built and previously run for real |
| **Phase 3** | Weeks 5–6 | Module 3: XLM-R Feature Extraction (frozen baseline) | Built; needs a real run (requires GPU/network you have, sandbox does not) |
| **Phase 4** | Weeks 6–9 | Module 4: Supervised Contrastive Fine-Tuning → Contrast-XLMR | Built; needs a real training run |
| **Phase 5** | Weeks 9–11 | Module 5: MLP Classification (both ensemble + Focal-Loss variants) | Built; needs real embeddings to train against |
| **Phase 6** | Weeks 11–13 | Module 6: Performance Evaluation (classification/clustering/retrieval/imbalance/fairness) | Built; needs a real trained classifier to evaluate |
| **— Semester 1 checkpoint —** | End of Semester 1 | Draft results section, comparison table vs. base paper's own metrics | Write-up only |
| **Phase 7** | Weeks 14–17 | Module 7: Fact Verification (Sentence-BERT retrieval + verdict) | Not yet built — awaiting approval |
| **Phase 8** | Weeks 17–19 | Module 8: Explainability (SHAP) | Not yet built — awaiting approval |
| **Phase 9** | Weeks 19–20 | Module 9: Confidence/Uncertainty calibration | Not yet built — awaiting approval |
| **Phase 10** | Weeks 20–23 | Module 10: Flask deployment | Not yet built — awaiting approval |
| **Phase 11** | Weeks 23–25 | Full ablations: with/without contrastive fine-tuning, ensemble vs. Focal-Loss, per-language fairness, multilingual retrieval novelty (Task 7) | — |
| **Phase 12** | Weeks 25–27 | Paper writing: Introduction, Related Work (base paper + gap framing from Task 1), Methodology (Task 3/5), Results (Phase 11 ablations), Discussion, Conclusion | — |
| **Phase 13** | Weeks 27–28 | Internal review, formatting to target venue (IEEE/Springer template), submission | — |

**Standing rules carried forward from this project's existing work, to continue applying:**
- No fabricated/synthetic data — every dataset row must be real and traceable to its source.
- No module is implemented without your explicit prior approval.
- Every "why this algorithm" choice must trace back either to the base paper's own method or to a stated, defensible gap it closes — not a generic best-practice default.

---

**Waiting for your review and approval of this plan before implementing any module.** Once you confirm the architecture (Tasks 3 and 5) and the two-semester split (Task 4), we proceed one module at a time, starting with whichever of Modules 7–10 (Semester 2, since 1–6 already exist) you'd like first — or a real end-to-end execution of the already-built Modules 1–6 to generate actual results for the Semester 1 checkpoint.
