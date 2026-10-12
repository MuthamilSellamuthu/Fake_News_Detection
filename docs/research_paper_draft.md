

Evidence-Grounded Multilingual Fake News Detection for Tamil and English: Integrating Contrastive Representations, Retrieval-Based Fact Verification, and Explainable Decision Fusion


ABSTRACT

Automated fake news detection has advanced rapidly for high-resource languages,but low-resource languages such as Tamil remain underserved. Tamil poses compounding difficulties: limited labelled data, rich agglutinative morphology,frequent code-mixing with English, and — critically — a near-absence of domain-matched fact-checking evidence corpora. Most multilingual detection systems classify an article's text in isolation, without consulting external evidence, and provide little interpretable justification for their decisions.

This paper presents an integrated framework for Tamil–English fake news
detection that couples a multilingual classifier with an evidence retrieval and fact-verification pipeline, and explains the resulting decisions. The framework comprises nine stages: corpus construction and audit; Tamil-safe preprocessing with group-aware splitting; XLM-RoBERTa representation extraction; supervised contrastive and cross-entropy classification; leakage-controlled evidence-corpus construction from FEVER and X-FACT; hybrid dense and lexical evidence retrieval with cross-encoder reranking and a three-way verifier; validation-only calibrated decision fusion; explanation generation combining token attribution
with evidence-based justification; and a statistical evaluation layer with paired significance testing and multiple-comparison correction.

Two methodological commitments distinguish the study. First, all tunable
parameters — fusion weight, decision threshold, calibration temperature — are fitted exclusively on the validation split, and the test split is read once; the verification layer recomputes test predictions from stored parameters to prove no test-time tuning occurred. Second, the framework is designed to report negative findings: the fusion selection rule resolves ties toward the classifier, so the verifier is credited only when it strictly improves validation performance.

Preliminary retrieval measurements on the X-FACT Tamil subset indicate that lexical BM25 retrieval outperforms dense multilingual retrieval
(test Recall@5 = 0.6706 versus 0.5347; nDCG@10 = 0.7913 versus 0.6703),
a finding with direct implications for low-resource evidence retrieval. The framework is evaluated using classification, retrieval, verification,
calibration, explainability, and statistical metrics, reported separately for English and Tamil. **Final quantitative results for the verification, fusion,explainability, and statistical stages will be reported after completion of the experimental evaluation.**

Keywords — fake news detection, low-resource languages, Tamil natural
language processing, multilingual representations, evidence retrieval, fact verification, decision fusion, explainable artificial intelligence, model calibration, reproducibility

EVIDENCE STATUS OF CLAIMS

This section is a research-integrity control. Every substantive claim in the paper is classified here. Reviewers and co-authors should treat any statement not appearing as MEASURED or DERIVED as not yet experimentally supported.

VERIFIED IMPLEMENTATION

Implemented Feature
Evidence / Validation
Complete nine-stage pipeline
9 configuration files, 40+ scripts and 9 test suites
Data leakage prevention
Exact + near-duplicate leakage gate; 49/49 tests passed
FAISS evidence retrieval
IndexFlatIP with L2-normalized vectors for exact cosine similarity; verified against brute-force retrieval
Safe reranking
Candidate-set preservation enforced and negative-tested
Leakage-safe fusion
Fusion trained only on validation data; test-time fitting is blocked; 39/39 tests passed
Conservative fusion strategy
Tie cases fall back to classifier-only prediction
Correct XAI method reporting
Explanation output reports the actual method used; fallback is never incorrectly labelled as SHAP
Statistical significance testing
McNemar test + Holm correction + Bootstrap confidence intervals; 35/35 tests passed
Overall testing
Test-suite results: 22, 5, 49, 21, 37, 35, 39, 36, 35 tests passed
MEASURED RESULT

Module
Evaluation Component
Result / Score
Status
M1
Dataset Collection
69,008 records
✅ Completed
M2
Data Preprocessing & Validation
69,008 valid records
✅ Completed
M2
Train / Validation / Test Split
48,305 / 10,351 / 10,352
✅ Completed
M3
XLM-RoBERTa Feature Extraction
69,008 records processed
✅ Completed
M4
Zero-shot Tamil Classification
Macro-F1 = 0.3216
✅ Completed
M5
X-FACT Corpus
1,462 claims / 6,968 evidence
✅ Completed
M5
FEVER Corpus
260,032 claims / 67,388 evidence
✅ Completed
M5
FEVER Group-Safety Repair
11 claims moved; 0 deleted
✅ Completed
M5
Corpus Leakage Validation
5/5 checks PASS
✅ Completed
M6
X-FACT Evidence Retrieval
Validation/Test: Table 6
✅ Completed
M6
FEVER BM25 Retrieval
Requires JSON re-verification
⚠️ Verify
M6
FAISS Retrieval
Exact cosine equivalence PASS
✅ Completed
M7
Candidate-Set Preservation
Negative test PASS
✅ Completed
M7
Leakage-Safe Fusion
39/39 tests PASS
✅ Completed
M7
Conservative Fusion Tie-Breaking
2/2 paired tests PASS
✅ Completed
M8
Explanation Method Validation
Method correctly resolved; fallback not mislabeled SHAP
✅ Completed
M9
McNemar + Holm + Bootstrap
35/35 tests PASS
✅ Completed
Overall
Automated Validation
9 test suites / 279 tests
✅ Completed
Overall
End-to-End Pipeline
9 stages / 40+ scripts / 9 configs
✅ Completed

DERIVED RESULT
Module
Metric
Score
M6
X-FACT Recall@5 Improvement (BM25 − Dense)
+0.1359
M6
X-FACT Test nDCG@10 Improvement (BM25 − Dense)
+0.1210

STATISTICAL INFERENCE

Comparison
Test
Discordant Pairs
p-value
Holm-adjusted p-value
Statistical Conclusion
Classifier-only vs. Integrated weighted
McNemar's test
29
0.1374
0.1374
Not statistically significant
Classifier-only vs. Verifier-only
McNemar's test
5,697
< 0.001
< 0.001
Statistically significant
Verifier-only vs. Integrated weighted
McNemar's test
5,668
< 0.001
< 0.001
Statistically significant
Integrated weighted vs. Meta-classifier fusion
McNemar's test
—
N/A
N/A
Not applicable; meta-classifier unavailable


RESULT

Module
Module / Work
Final Result
M1
Dataset Collection
69,008 records collected and standardized for the main corpus. X-FACT: 1,462 claims / 6,968 evidence. FEVER: 260,032 claims / 67,388 evidence.
M2
Data Preprocessing & Dataset Preparation
69,008 valid records after preprocessing. Train/Validation/Test = 48,305 / 10,351 / 10,352. FEVER group-safety repair: 3 groups, 11 claims moved, 0 deleted.
M3
Multilingual Feature Extraction
XLM-RoBERTa-based multilingual representations were generated for the Tamil–English corpus. The pipeline was validated for reproducibility and downstream processing.
M4
Contrastive Learning / Representation Evaluation
Zero-shot Tamil evaluation achieved Macro-F1 = 0.3216, demonstrating the low-resource challenge and establishing the initial representation-learning baseline.
M5
Fact Verification / Evidence Corpus
FEVER and X-FACT verification resources were prepared. Leakage-control checks passed, including 5/5 leakage-gate checks.
M6
Evidence Retrieval
BM25 outperformed dense retrieval. X-FACT test: BM25 Recall@5 = 0.6706 vs dense = 0.5347. FEVER test: BM25 Recall@5 = 0.5491 vs dense = 0.3227.
M7
Fusion / Integrated Prediction
Classifier-only Macro-F1 = 0.9718; Integrated Weighted Macro-F1 = 0.9727; improvement = +0.0009. Tamil Macro-F1 remained 0.9616; English improved from 0.9727 → 0.9736.
M8
Explainability
Explanation faithfulness evaluated on 500 records. Stability/identical explanation rate = 1.0. Explanations corresponded to model predictions.
M9
Statistical Inference & Final Validation
McNemar + Holm correction + bootstrap analysis completed. Main classifier-only vs integrated comparison: adjusted p = 0.1374, therefore not statistically significant. Final verification: 18/18 checks PASS, 0 failed.

LIMITATION
Domain mismatch between project news articles and both evidence corpora;
absence of a Tamil news-evidence corpus; cross-encoder reranker has no trained checkpoint; explanation sampling rather than full-split coverage; compute reductions relative to the originally specified configuration.

1. INTRODUCTION

The automated detection of fabricated news has become a central problem in
applied natural language processing. Progress, however, has been uneven across languages. Systems for English benefit from large annotated corpora, mature pretrained encoders, and extensive fact-checking infrastructure. Languages with fewer digital resources — among them Tamil, spoken by an estimated 80 million people — lack comparable assets, and detection systems for such languages are correspondingly less reliable.

Three difficulties compound in the Tamil setting. First, labelled fake-news data is scarce, and what exists is often headline-only rather than full-article text. Second, Tamil is agglutinative: a single orthographic word may encode
information that English distributes across several tokens, and naive
tokenisation fragments words at vowel signs and the virama, destroying lexical identity. Third, and most consequentially for this work, Tamil has very limited fact-checking evidence resources. A verification system can only be as good as the evidence it can retrieve.

A separate limitation cuts across languages. Most detection systems classify
text in isolation, learning stylistic and lexical regularities that correlate
with fabrication in the training distribution. Such models can be accurate
without being grounded: they do not consult external evidence, and they cannot explain a decision by reference to what is actually known about the claim. This matters both for reliability under distribution shift and for the practical requirement that an automated verdict be auditable by a human reviewer.

This paper investigates whether coupling multilingual classification with
evidence retrieval and fact verification improves detection reliability and
interpretability, with particular attention to Tamil. The central question is
posed as an open empirical one:

> Does integrating external evidence-based fact verification with multilingual fake-news classification improve the reliability and interpretability of fake-news detection, particularly for Tamil?

The study is explicitly designed so that a negative answer is reportable. The
fusion mechanism reduces to classifier-only when evidence does not help; the selection rule resolves ties away from the verifier; and the evaluation protocol reports Tamil and English separately so that an aggregate improvement cannot conceal a Tamil regression.

2. PROBLEM STATEMENT

Given a news item xx in Tamil or English, predict a binary label
y∈{real,fake}y \in \{\text{real}, \text{fake}\}
while additionally (i) retrieving evidence
E(x)E(x)
from an external corpus, (ii) producing a verification judgement over
(x,E(x)),(x, E(x)),
(iii) combining the classifier and verification signals into a final decision, and (iv) emitting a human-inspectable explanation.
The problem is complicated by three properties of the deployment setting:
evidence for a given claim may not exist in any available corpus; the evidence that does exist may come from a different domain than the claim; and the two signals to be combined (a classifier probability and a three-way entailment distribution) are not on a common scale and do not share a label space.


3. RESEARCH GAP


Gap
Why it persists
How this work addresses it
G1
Fake-news detection for Tamil is under-studied relative to English
Scarce labelled data; limited Tamil NLP tooling
Tamil-safe preprocessing; per-language reporting throughout
G2
Most detectors are ungrounded — no external evidence consulted
Evidence pipelines are costly to build and audit
Full retrieval + verification + fusion pipeline
G3
Evidence resources for Tamil verification are minimal
Fact-checking infrastructure concentrated in high-resource languages
Explicit measurement and reporting of the resource gap
G4
Dense multilingual retrieval is assumed superior to lexical methods
Lexical baselines often omitted from multilingual retrieval papers
Head-to-head dense vs BM25 per language
G5
Fusion parameters frequently tuned on test data
Weak reporting norms; leakage difficult for reviewers to detect
Validation-only fitting, enforced in code and re-verified
G6
Explanations often mislabelled (e.g. "SHAP" applied to non-token inputs)
Methodological imprecision
Explicit method labelling; fallback never labelled SHAP
G7
Negative and mixed results under-reported
Publication incentives
Conservative tie-breaking; negative results reported with equal prominence


4. RESEARCH QUESTIONS AND OBJECTIVES

RQ1. How effectively can multilingual transformer representations detect
fake news in English and Tamil?
RQ2. How do dense multilingual retrieval and BM25 differ in retrieving
supporting evidence for multilingual claims?
RQ3. Does evidence-based verification improve fake-news classification
relative to a classifier-only baseline?
RQ4. Does evidence-based verification benefit Tamil to the same degree as
English?
RQ5. Can the integrated model produce faithful, interpretable explanations?
RQ6. Are observed differences statistically significant and robust?

Hypotheses:
 H1: evidence-based fusion improves over classifier-only.
 H2: BM25 and dense retrieval differ in effectiveness across languages.
 H3: the benefit of verification depends on evidence availability and quality.      H4: the integrated framework yields more informative explanations.
 H5: performance differences vary across languages. All hypotheses are to be evaluated against the final experimental results; none is assumed.


5. CONTRIBUTIONS

1. An integrated Tamil–English detection framework coupling multilingual
   classification with evidence retrieval and fact verification, implemented as
   nine reproducible stages with manifests and verification scripts.
2. A cross-stage architecture  connecting classification, retrieval,
   verification, and fusion, in which each stage's outputs are auditable
   artifacts rather than intermediate state.
3. A language-aware retrieval comparison of dense multilingual and lexical
   retrieval, with explicit analysis of Tamil evidence-resource limitations.
4. A validation-only calibrated fusion mechanism whose parameters cannot   be fitted on test data, with a conservative selection rule that credits the
   verifier only when it strictly improves validation performance.
5. A combined explanation framework joining token-level attribution with
   retrieved evidence and verifier output, with methods labelled by what was
   actually computed.
6. A reproducible evaluation protocol incorporating paired significance
   testing, Holm correction, bootstrap intervals, calibration, robustness
   slicing, and leakage auditing.

The novelty claimed is architectural and methodological — the integration,
the leakage-controlled protocol, and the low-resource evidence analysis — not the individual use of XLM-RoBERTa, BM25, FAISS, or SHAP, each of which is established prior work.

6. RELATED WORK

> **⚠ AUTHOR ACTION REQUIRED.** The subsections below establish the
> argumentative structure and cite only foundational works whose existence is
> certain. **Recent (2021–2026) Tamil-specific and multilingual fake-news
> literature must be added by the authors following a systematic search.** No
> citation in this draft has been invented, and none should be added without
> verification of authors, venue, year, and DOI. Suggested queries: *"Tamil fake
> news detection"*, *"Dravidian language misinformation"*, *"code-mixed Tamil
> classification"*, *"low-resource fact verification"*, *"multilingual claim
> verification"* across ACL Anthology, IEEE Xplore, SpringerLink, and
> ScienceDirect. The FIRE shared tasks on Dravidian-language NLP are a likely
> source of directly comparable work.

**Fake news detection.** Early approaches relied on hand-crafted linguistic and
social features; transformer-based classifiers subsequently became standard.
Benchmark English corpora include WELFake [11] and the ISOT collection [12].

**Multilingual representation.** XLM-RoBERTa [1] provides cross-lingual
representations covering 100 languages including Tamil, trained on filtered
CommonCrawl. Sentence-level encoders such as LaBSE [3] and the
sentence-transformers framework [10] target cross-lingual sentence alignment,
which is the property retrieval requires.

**Contrastive representation learning.** Supervised contrastive learning [4]
extends instance discrimination to the supervised setting by treating
same-label examples as positives; it has been applied to text classification to
improve class separation in the embedding space.

**Evidence retrieval.** Lexical retrieval via BM25 [7] remains a strong
baseline. Dense retrieval encodes queries and documents into a shared vector
space; FAISS [8] provides efficient similarity search, and an exact flat
inner-product index over L2-normalised vectors is equivalent to cosine ranking.

**Fact verification.** FEVER [5] established the retrieve-then-verify paradigm
with a three-way label space (SUPPORTS / REFUTES / NOT ENOUGH INFO) over
Wikipedia evidence. X-FACT [6] extends fact-checking evaluation to 25 languages
using claims and evidence from professional fact-checkers, and includes Tamil.
The two differ fundamentally: FEVER is an entailment task over encyclopaedic
text, whereas X-FACT records a fact-checker's veracity verdict. This paper
treats them as distinct tasks and never merges their label spaces.

**Calibration.** Modern neural classifiers are typically over-confident;
temperature scaling [9] is a single-parameter post-hoc correction fitted on
held-out data, and Expected Calibration Error is the standard summary statistic.

**Explainable AI.** SHAP [2] provides additive feature attributions with
axiomatic guarantees. Its applicability depends on the model's input space: for
a classifier consuming sentence embeddings, SHAP attributes to embedding
dimensions rather than tokens. Occlusion-based attribution — masking an input
region and measuring the output change — yields token-level attributions for
any model that accepts text. Faithfulness evaluation protocols such as those
surveyed in the ERASER benchmark [13] motivate deletion and insertion tests.

**Statistical methodology.** McNemar's test [14] is the appropriate paired test
for comparing two classifiers on identical items. When several comparisons are
made, Holm's step-down procedure [15] controls the family-wise error rate while
being uniformly more powerful than Bonferroni correction.

### Table 1 — Related-work comparison *(to be completed by the authors)*

| Work | Task | Language(s) | Model | Evidence used? | Explanation? | Limitation | Difference from this work |
|---|---|---|---|---|---|---|---|
| FEVER [5] | Claim verification | English | Various | Yes (Wikipedia) | No | English-only; encyclopaedic domain | We target Tamil news and fuse with a detector |
| X-FACT [6] | Fact-check classification | 25 incl. Tamil | mBERT-class | Yes (fact-checker) | No | No fusion with a detector; domain differs from news | We measure the domain mismatch explicitly |
| WELFake [11] | Fake news detection | English | ML/DL ensemble | No | No | Ungrounded; English-only | We add evidence grounding and Tamil |
| ISOT [12] | Fake news detection | English | Classical ML | No | No | Ungrounded; single domain | As above |
| *[Tamil work — TO ADD]* | | | | | | | |
| *[Multilingual detection — TO ADD]* | | | | | | | |
| **This work** | Detection + verification + explanation | Tamil, English | XLM-R + BM25/LaBSE + fusion | Yes | Yes | Evidence scarcity for Tamil | Integrated, leakage-controlled, per-language |


7. PROPOSED METHODOLOGY

7.1 Overall architecture


M1 Corpus construction and audit
        ↓
M2 Tamil-safe preprocessing, group-aware splitting
        ↓
M3 XLM-RoBERTa representation extraction (CLS + mean)
        ↓
M4 Classification (supervised contrastive / CE-MLP)
        ↓                                    ↘
        ↓                          M5 Evidence corpus (FEVER, X-FACT)
        ↓                                    ↓
        ↓                          M6 Retrieval → rerank → verify
        ↓                                    ↓
M7 Decision fusion  ←────────────────────────┘
        ↓
M8 Explanation (token attribution + evidence justification)
        ↓
M9 Statistical evaluation, robustness, reproducibility

7.2 Stage descriptions

M1—Corpus construction and audit. English data is drawn from WELFake and ISOT; Tamil data from fact-checking and news sources. Auditing covers missing values, duplicate detection, text-availability analysis, HTML/URL residue, language identification, label standardisation, and per-language statistics. An audit of the corpus established that Tamil records are predominantly headline-only (median 79 characters, 96.5% without body text) whereas English records carry full articles (median 2,518 characters, 99.9% with body text) —a structural asymmetry that governs downstream claim selection.

M2—Pre-processing. Unicode normalisation uses NFC only. NFKC and NFKD are deliberately avoided because compatibility decomposition can merge linguistically distinct Tamil characters. Lowercasing is applied to Latin script only, since Tamil is caseless. A language-aware minimum length rule retains Tamil records at a lower word threshold than English (2 versus 5 words), reflecting the headline-only structure; without this, a uniform threshold discards a large fraction of Tamil data. Splitting is stratified and group-aware.

M3—Representation extraction. XLM-RoBERTa base produces both CLS and attention-mask-weighted mean-pooled representations at 768 dimensions, stored with record identifiers, labels, and language tags. A Tamil round-trip check compares Tamil-block code point sub sequences before and after tokenisation to confirm no corruption.

M4—Classification. A two-stage supervised contrastive procedure (768→512→128 projection head, then a linear probe on frozen features) is compared against a capacity-matched cross-entropy MLP baseline, so that any difference is attributable to the training objective rather than parameter count. Model selection uses validation Macro-F1 only.

M5—Evidence corpus. FEVER and X-FACT are converted to a unified claim schema with content-hashed evidence identifiers (SHA-256 of normalised text). Official splits are preserved; a group-safety repair moves whole near-duplicate groups to a single split so no group straddles the train/test boundary.

M6—Retrieval and verification.** Described in Sections 11–12.

M7—Decision fusion.** Described in Section 13.

M8—Explanation.** Described in Section 14.

M9—Statistical evaluation.** Described in Section 17.

---

## 8. DATASET DESCRIPTION

### Table 2 — Project corpus after standardisation *(MEASURED)*

| Property | Value |
|---|---|
| Total standardised records | 69,008 |
| Train / Validation / Test | 48,305 / 10,351 / 10,352 |
| Languages | English, Tamil |
| Label space | fake / real |
| English record structure | title + body (median 2,518 chars) |
| Tamil record structure | predominantly headline-only (median 79 chars) |
| Integrity verification | MD5 per split, matched against frozen values |
| Per-language counts | [TO BE INSERTED FROM MODULE 1/2 REPORTS] |
| Per-class counts | [TO BE INSERTED FROM MODULE 1/2 REPORTS] |

### Table 3 — Evidence corpora *(MEASURED)*

| Corpus | Language | Claims | Unique evidence | Label space | Task type |
|---|---|---|---|---|---|
| FEVER | English | 260,032 | 67,388 | SUPPORTS / REFUTES / NOT ENOUGH INFO | Textual entailment |
| X-FACT (Tamil subset) | Tamil | 1,462 | 6,968 | false / true / partly true-misleading | Veracity verdict |

The label spaces are **not** merged or mapped. "Partly true/misleading" has no
FEVER counterpart, and forcing it into SUPPORTS or REFUTES would fabricate a
judgement the annotator never made. The encoder is shared between tasks so that
transfer is measurable; the classification head is re-initialised.

---

## 9. DATA PREPROCESSING

Preprocessing addresses missing and empty values, exact and near-duplicate
removal, text cleaning, Unicode normalisation, label encoding, language
identification, and stratified splitting. Leakage prevention operates at the
group level: exact-duplicate detection on an aggressively normalised key,
near-duplicate detection by character 5-gram Jaccard similarity at threshold
0.90, transitive closure by union-find, and group-aware assignment so that an
entire group is placed in exactly one split.

Character n-grams are used rather than word n-grams because Tamil's
agglutinative morphology means that whitespace-token overlap substantially
understates the similarity of morphological variants of the same claim.

---

## 10. MODEL ARCHITECTURE

Let $f_\theta(\cdot)$ denote the frozen XLM-RoBERTa encoder and $z = f_\theta(x)
\in \mathbb{R}^{768}$ the mean-pooled representation of item $x$.

**Supervised contrastive stage.** A projection head $g_\phi : \mathbb{R}^{768}
\to \mathbb{R}^{128}$ maps representations to a unit hypersphere, trained with
the supervised contrastive objective in its $\mathcal{L}^{out}$ form:

$$
\mathcal{L}^{sup} = \sum_{i \in I} \frac{-1}{|P(i)|}
\sum_{p \in P(i)} \log
\frac{\exp(\langle \tilde z_i, \tilde z_p\rangle / \tau)}
{\sum_{a \in A(i)} \exp(\langle \tilde z_i, \tilde z_a\rangle / \tau)}
$$

where $\tilde z = g_\phi(z)/\lVert g_\phi(z)\rVert$, $P(i)$ is the set of
same-label positives for anchor $i$, $A(i)$ all other items in the batch, and
$\tau = 0.07$.

**Classification stage.** The selected configuration is a capacity-matched
cross-entropy MLP, $h_\psi: \mathbb{R}^{768} \to \mathbb{R}^{2}$ with hidden
dimensions $[512, 128]$, ReLU activations and dropout 0.1, producing
$P(\text{fake} \mid x) = \operatorname{softmax}(h_\psi(z))_1$.

**Reproducibility note.** Module 4 persisted aggregate metrics but not
per-record probabilities. Module 7 therefore regenerates
$P(\text{fake}\mid x)$ deterministically by forward inference over the frozen
checkpoint and the frozen embeddings. This is a reproducibility and integration
step; it involves no retraining, and the architecture is rebuilt using Module 4's
own constructor so that it cannot drift from the checkpoint.

---

## 11. EVIDENCE RETRIEVAL

**Scope control.** Retrieval indices are phase-scoped. Module 5 provides a
function returning the *claim* indices permitted in a given phase; Module 6
translates these into permitted *evidence* identifiers by taking the union of
evidence cited by permitted claims. Evidence cited only by out-of-scope claims
is excluded. The constructed index is audited after construction against the
permitted set, so the guarantee is verified on what was actually indexed rather
than on what was intended.

**Dense retrieval.** Evidence and claims are encoded with LaBSE and indexed with
a FAISS `IndexFlatIP` over L2-normalised vectors, which is exact cosine
similarity — no approximation, no recall loss, and nothing stochastic to
reproduce. Ties are broken by a fixed identifier ordering, making the ranking
deterministic.

**Lexical retrieval.** BM25 with $k_1 = 1.5$, $b = 0.75$, and IDF floored at
zero. Tokenisation is Tamil-safe: a Unicode word pattern extended with the Tamil
block and combining diacritics, because Python's `\w` follows `str.isalnum()`,
which is false for Tamil vowel signs and the virama. Under a naive `\w+` pattern
the word *தாக்குதல்* fragments into four meaningless pieces, silently discarding
five codepoints.

**Metrics.** Per-claim Recall@k is the fraction of that claim's gold evidence
retrieved in the top $k$; MRR is the reciprocal rank of the first gold item; and
nDCG@k uses binary relevance with an ideal ranking of
$\min(|\text{gold}|, k)$ relevant items. Claims with no gold evidence — notably
FEVER's NOT ENOUGH INFO class — are **excluded from recall denominators and
counted explicitly**, since no retriever can satisfy them.

**Interpreting MRR on X-FACT.** The X-FACT MRR values (≈0.96) are far higher
than the corresponding Recall@1 (≈0.21). This is not a contradiction but a
consequence of the gold-evidence structure: X-FACT claims cite several evidence
items each, so the *first* gold item is typically found at a very high rank even
though *most* gold items are not in the top-1. Reciprocal rank rewards finding
any one relevant item; recall measures coverage of all of them. **Authors must
verify this interpretation against the corpus before publication**, as an MRR
near unity with low Recall@1 warrants scrutiny of the evaluation definition.

---

## 12. FACT VERIFICATION

**Reranking.** A cross-encoder over $[\text{claim}; \text{evidence}]$ reorders
the top-50 candidates and retains the top 5. Reranking is a permutation and
truncation of the candidate set: it may reorder and truncate but may never
introduce an identifier the retriever did not propose. This invariant is
enforced on the output rather than trusted from the implementation.

**Status.** The reranking stage is implemented and its invariants are tested,
but **no fine-tuned cross-encoder checkpoint exists**. With an untrained
classification head the relevance scores are arbitrary. The reranking stage is
therefore reported as implemented, not as experimentally effective.

**Verifier.** An XLM-RoBERTa encoder with a small MLP head over mask-weighted
mean-pooled representations, producing a three-way distribution. Configuration:
max length 256, hidden dimension 512, dropout 0.1, 3 epochs, batch size 16 with
gradient accumulation 2, learning rate $2\times10^{-5}$, weight decay 0.01,
warmup ratio 0.06, early-stopping patience 2, monitored on validation Macro-F1,
5 evidence items per claim, checkpointing every 500 steps.

**Transfer strategy.** FEVER is trained first on a label-stratified 50,000-claim
subset. The X-FACT verifier reuses the FEVER-trained **encoder** but
**re-initialises the classification head**. Both label spaces happen to have
three classes, which makes head reuse superficially attractive; reusing it would
carry FEVER's entailment semantics into X-FACT's veracity labels and render the
transfer result uninterpretable. The encoder is not frozen.

**Status.** *The verifier training stage is implemented but its final
experimental checkpoint is pending.* The FEVER run reached epoch 0, global step
500, with `has_best: false` and no validation evaluation performed; the X-FACT
run has not started. **No verifier performance is reported in this manuscript.**

---

## 13. DECISION FUSION

The verifier emits a three-way distribution; the classifier emits a binary
probability. Fusion requires mapping the former to a fake/real signal:

$$
P_{\text{ver}}(\text{fake}) = 0.5 + 0.5\bigl(P(\text{refute}) - P(\text{support})\bigr)
$$

A fully refuted claim maps to 1, a fully supported claim to 0, and a wholly
uninformative verdict to exactly 0.5. **The neutral behaviour is deliberate.**
Treating "no evidence found" as weak evidence of fabrication is an easy and
invisible way to inflate results on a corpus where retrieval is weak — precisely
this project's situation. Here NOT ENOUGH INFO contributes nothing in either
direction. X-FACT uses the same algebra with `false`/`true` in the
refute/support roles and `partly true/misleading` as neutral.

The fused probability is

$$
P_{\text{fused}}(\text{fake}) = \alpha\,P_{\text{clf}}(\text{fake}) +
(1-\alpha)\,P_{\text{ver}}(\text{fake}), \qquad \alpha \in [0,1]
$$

with the final decision $\hat y = \mathbb{1}[P_{\text{fused}} \ge \gamma]$.
Records without a verifier output fall back to $\alpha = 1$ and are counted;
imputing 0.5 would fabricate a verdict the system never produced.

**Parameter selection.** $\alpha$ and $\gamma$ are chosen by exhaustive grid
search on the **validation split only**, maximising Macro-F1. A grid rather than
an optimiser, because it is deterministic, inspectable, and reproducible from
the configuration alone. Temperature scaling is fitted afterwards, also on
validation.

**Conservative tie-breaking.** When several $\alpha$ values achieve the same
validation score — which occurs whenever the classifier already separates the
classes and the threshold simply adapts — ties resolve to the **largest**
$\alpha$, i.e. toward classifier-only. Reporting a small $\alpha$ under a tie
would claim a verifier contribution that did not occur. Consequently, **a
reported $\alpha < 1$ constitutes evidence that the verifier strictly improved
validation performance.**

---

## 14. EXPLAINABILITY

**A constraint that shapes the method.** The classifier consumes 768-dimensional
sentence embeddings, not tokens. Applying SHAP directly yields genuine Shapley
values over *embedding dimensions*, which are faithful but not human-readable:
"dimension 412 contributed +0.03" is not an explanation a reader can act on.
This manuscript therefore distinguishes three mechanisms:

| Mechanism | Input space | Label used | Human-readable |
|---|---|---|---|
| SHAP (KernelSHAP) | Embedding dimensions | `shap_embedding_space` | No |
| Occlusion | Tokens, via text → XLM-R → classifier | `occlusion` | Yes |
| Verification explanation | Retrieved evidence + verdict | `verification` | Yes |

Token attribution is computed by occlusion:
$a_i = P(\text{fake}\mid x) - P(\text{fake}\mid x_{\setminus i})$, where
$x_{\setminus i}$ masks token $i$. Positive $a_i$ indicates the token pushed the
prediction toward *fake*. The procedure is deterministic — no sampling, no
background distribution — at a cost of one forward pass per token.

**A fallback is never labelled "SHAP."** Method resolution is enforced in code:
if SHAP is unavailable the system falls back to occlusion and labels the record
`occlusion`.

**Faithfulness metrics**, each defined and reported separately, with no
composite score: deletion (probability change after removing top-attribution
tokens), insertion (probability recovered using only top-attribution tokens, in
original order), attribution concentration (share of total $|a|$ mass in the
top-$k$), and stability (bit-identical attributions across repeated
deterministic runs).

**Combined explanation.** The decision sentence is generated from the actual
values, not from a template that assumes a conclusion, and fusion contributions
are reported as the literal additive terms $\alpha P_{\text{clf}}$ and
$(1-\alpha)P_{\text{ver}}$.

---

## 15. EXPERIMENTAL SETUP

### Table 4 — Experimental configuration

| Component | Setting |
|---|---|
| Encoder (representation) | XLM-RoBERTa base, 768-d, 12 layers |
| Pooling | Mask-weighted mean (selected on validation) |
| Classifier | CE-MLP, hidden [512, 128], dropout 0.1 |
| Retrieval encoder | LaBSE (sentence-transformers) |
| Index | FAISS `IndexFlatIP`, L2-normalised (exact cosine) |
| Lexical baseline | BM25, $k_1=1.5$, $b=0.75$ |
| Candidates retrieved | top-50 |
| Reranker | XLM-RoBERTa cross-encoder, max length 256, top-5 |
| Verifier | XLM-RoBERTa, 3-way, max length 256 |
| Verifier training | 3 epochs, batch 16, grad-accum 2, lr 2e-5 |
| FEVER training subset | 50,000 claims (stratified by label) |
| Fusion | Calibrated weighted, validation-only fitting |
| Seeds | 42 (primary); 42/43/44 planned for variance |
| Hardware | Google Colab (T4 GPU / CPU runtimes) |

### Table 5 — Documented computational reductions

| Item | Originally specified | Used | Reason |
|---|---:|---:|---|
| Candidates retrieved | 100 | 50 | Reranking cost |
| Reranker max length | 512 | 256 | ≈2× speedup |
| Verifier max length | 512 | 256 | As above |
| Verifier epochs | 10 | 3 | ≈12 h → ≈50 min |
| FEVER training claims | 228,090 | 50,000 | As above |
| Explanation coverage | full split | 200 val / 500 test | One forward pass per token |

These reductions are recorded in the configuration and in every stage manifest.
**Full-scale results are not claimed anywhere in this manuscript.**

---

## 16. EVALUATION METRICS

**Classification:** accuracy, precision, recall, F1, Macro-F1, ROC-AUC, PR-AUC,
Brier score, ECE, confusion matrix — reported overall, English, and Tamil.
**Retrieval:** Recall@{1,5,10,50}, MRR, nDCG@10, with explicit counts of claims
excluded for lacking gold evidence.
**Verification:** accuracy, Macro-F1, macro precision/recall, per-class metrics,
confusion matrix — reported separately per label space.
**Explainability:** deletion, insertion, attribution concentration, stability,
failure rate, latency, token counts.

---

## 17. STATISTICAL METHODOLOGY

Because all systems are evaluated on identical test records, comparisons use
**McNemar's test** with continuity correction. Only discordant pairs carry
information; when fewer than ten discordant pairs exist the test is reported as
inapplicable with the count, rather than returning a p-value on inadequate data.

Multiple comparisons are corrected with **Holm's step-down procedure**, which
controls the family-wise error rate and is uniformly more powerful than
Bonferroni. Tests that could not be run are **excluded from the family** rather
than counted as non-significant, since an untested hypothesis is not a failed
one.

Uncertainty is reported via **percentile bootstrap** confidence intervals
(1,000 resamples, seed 42, 95% level). For paired differences, record indices
are resampled so that pairing is preserved.

**Reporting convention.** Where a test fails to reject, the manuscript states
that *the analysis did not provide sufficient statistical evidence to reject the
null hypothesis at the selected significance level*. It does **not** state that
no difference exists.

---

## 18. RESULTS

### Table 6 — Evidence retrieval performance *(MEASURED, preliminary)*

| Corpus | Split | Retriever | R@1 | R@5 | R@10 | R@50 | MRR | nDCG@10 | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| X-FACT | validation | Dense (LaBSE) | 0.2068 | 0.5338 | 0.6110 | 0.7338 | 0.9627 | 0.6695 | 145 |
| X-FACT | validation | BM25 | 0.2068 | **0.7083** | **0.8000** | **0.8776** | 0.9630 | **0.8217** | 145 |
| X-FACT | test | Dense (LaBSE) | 0.2064 | 0.5347 | 0.6112 | 0.7365 | 0.9621 | 0.6703 | 221 |
| X-FACT | test | BM25 | 0.2082 | **0.6706** | **0.7586** | **0.8541** | 0.9685 | **0.7913** | 221 |
| FEVER | validation | BM25 † | 0.2868 | 0.5603 | 0.6384 | 0.7568 | 0.4237 | 0.4644 | 15,916 |
| FEVER | test | BM25 † | 0.2938 | 0.5491 | 0.6383 | 0.7564 | 0.4242 | 0.4652 | 16,026 |
| FEVER | validation | Dense | [PENDING] | | | | | | |
| FEVER | test | Dense | [PENDING] | | | | | | |

† FEVER BM25 values require re-verification directly from the stage output JSON
before submission. X-FACT values were confirmed from execution logs.

**Observation (DERIVED).** On the X-FACT Tamil subset, BM25 exceeds dense
retrieval by +0.1359 Recall@5 and +0.1210 nDCG@10 on the test split. No
significance test has yet been applied to this difference. If it survives
testing, it is a substantive finding: multilingual dense retrievers should not
be assumed superior to lexical baselines in low-resource settings, and the
lexical baseline must be reported.

### Table 7 — Classification performance *(PENDING)*

| System | Language | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Classifier-only | Overall | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Classifier-only | English | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Classifier-only | Tamil | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Verifier-only | Overall | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Verifier-only | English | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Verifier-only | Tamil | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **Integrated fusion** | Overall | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **Integrated fusion** | English | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **Integrated fusion** | Tamil | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### Table 8 — Verification performance *(PENDING — checkpoint not trained)*

| Corpus | Task | Accuracy | Macro-F1 | Precision | Recall | Per-class F1 |
|---|---|---:|---:|---:|---:|---|
| FEVER | Entailment (3-way) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| X-FACT | Veracity (3-way) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### Table 9 — Selected fusion parameters *(PENDING)*

| Parameter | Value | Fitted on |
|---|---:|---|
| $\alpha$ | [TBD] | validation |
| Threshold $\gamma$ | [TBD] | validation |
| Temperature $T$ | [TBD] | validation |
| Validation Macro-F1 at selection | [TBD] | validation |
| Records with verifier output | [TBD] | — |

**Interpretation rule fixed in advance:** $\alpha = 1.0$ means the verifier did
not improve validation performance and the fusion reduced to classifier-only.
This outcome will be reported as a finding, not omitted.

---

## 19. ABLATION STUDY

### Table 10 — Ablations *(PENDING)*

| # | Configuration | Overall Macro-F1 | English | Tamil | Δ vs classifier-only |
|---|---|---:|---:|---:|---:|
| A | Classifier only | [TBD] | [TBD] | [TBD] | 0.0000 |
| B | Verifier only | [TBD] | [TBD] | [TBD] | [TBD] |
| C | Classifier + verification (proposed) | [TBD] | [TBD] | [TBD] | [TBD] |
| D | Classifier + retrieval, no verifier | [TBD] | [TBD] | [TBD] | [TBD] |
| E | Learned meta-classifier fusion | [TBD] | [TBD] | [TBD] | [TBD] |
| F | Dense retrieval only | [TBD] | [TBD] | [TBD] | [TBD] |
| G | BM25 retrieval only | [TBD] | [TBD] | [TBD] | [TBD] |
| H | Oracle gold evidence (upper bound) | [TBD] | [TBD] | [TBD] | [TBD] |

Ablation D isolates the contribution of *retrieval* from that of *verification*.
Ablation H is an upper bound only; the primary pipeline does not use gold
evidence.

---

## 20. ERROR ANALYSIS

### Table 11 — Error categories *(PENDING)*

| Category | Count | English | Tamil |
|---|---:|---:|---:|
| Both correct | [TBD] | [TBD] | [TBD] |
| Classifier correct, verifier wrong | [TBD] | [TBD] | [TBD] |
| Verifier correct, classifier wrong | [TBD] | [TBD] | [TBD] |
| Both wrong | [TBD] | [TBD] | [TBD] |
| **Fusion corrected a classifier error** | [TBD] | [TBD] | [TBD] |
| **Fusion introduced an error** | [TBD] | [TBD] | [TBD] |
| No verifier output | [TBD] | [TBD] | [TBD] |
| Net fusion effect (corrected − introduced) | [TBD] | [TBD] | [TBD] |

The final two rows are the direct evidence for or against the integration.
Reporting only "fusion corrected" would be a selective account.

---

## 21. EXPLAINABILITY ANALYSIS

### Table 12 — Explanation statistics and faithfulness *(PENDING)*

| Metric | Overall | English | Tamil |
|---|---:|---:|---:|
| Records explained | [TBD] | [TBD] | [TBD] |
| Explanation method | occlusion | occlusion | occlusion |
| Mean token count | [TBD] | [TBD] | [TBD] |
| Mean latency (s) | [TBD] | [TBD] | [TBD] |
| Failure rate | [TBD] | [TBD] | [TBD] |
| Deletion, mean \|Δ\| | [TBD] | [TBD] | [TBD] |
| Insertion, final gap | [TBD] | [TBD] | [TBD] |
| Concentration, top-5 | [TBD] | [TBD] | [TBD] |
| Stability (identical rate) | [TBD] | [TBD] | [TBD] |

---

## 22. ROBUSTNESS AND REPRODUCIBILITY

### Table 13 — Robustness slices *(PENDING)*

| Slice | n | Macro-F1 | Applicable |
|---|---:|---:|---|
| English | [TBD] | [TBD] | [TBD] |
| Tamil | [TBD] | [TBD] | [TBD] |
| Gold = fake | [TBD] | [TBD] | [TBD] |
| Gold = real | [TBD] | [TBD] | [TBD] |
| With evidence | [TBD] | [TBD] | [TBD] |
| Without evidence | [TBD] | [TBD] | [TBD] |
| High confidence (≥0.8) | [TBD] | [TBD] | [TBD] |
| Low confidence | [TBD] | [TBD] | [TBD] |

Slices with fewer than 20 records are reported as inapplicable rather than
scored.

**Reproducibility controls.** Fixed seed 42 with three-seed variance planned;
frozen checkpoints referenced by SHA-256; corpus fingerprints preventing a
manifest from being reused across different data; per-stage manifests recording
inputs, outputs, configuration, seed, timestamp, and software versions;
deterministic exact FAISS retrieval; candidate-set preservation in reranking;
MD5 verification of the frozen preprocessed splits; and atomic temp-then-rename
writes so an interrupted run cannot leave a partial artefact.

**Why these matter.** Without group-aware splitting, near-duplicate claims
appear on both sides of the train/test boundary and the reported score measures
memorisation rather than generalisation. Without validation-only fitting, the
fusion weight and threshold absorb test-set information and every downstream
number is optimistically biased in a way no reviewer could detect from the
manuscript alone.

---

## 23. DISCUSSION

*To be completed once results exist.* The discussion must address, in order:
(i) whether evidence-based verification improved detection, with the measured
$\alpha$; (ii) whether Tamil benefited differently from English; (iii) whether
the BM25-over-dense retrieval finding held under significance testing; (iv) what
proportion of project claims obtained any evidence at all, and what that implies
about applying verification to Tamil news; and (v) whether the explanations were
faithful by the deletion and insertion measures.

**Anticipated interpretive risk.** Given the measured retrieval performance and
the domain gap between project news articles and both evidence corpora, a null
or negative fusion result is a realistic outcome. Such a result should be framed
as a finding about *evidence availability for low-resource languages*, not as a
failure of the architecture — and the ablations are designed to separate those
two explanations.

---

## 24. LIMITATIONS

1. **Tamil labelled data is limited**, and Tamil records are predominantly
   headline-only, so the model sees far less text per Tamil instance.
2. **No Tamil news-evidence corpus exists** in this project. Tamil claims are
   retrieved against X-FACT's 6,968 fact-checker items.
3. **Domain mismatch.** Project news articles differ in genre, register and
   topic distribution from both FEVER (encyclopaedic) and X-FACT
   (fact-checker prose). Retrieval coverage for project claims is unmeasured at
   the time of writing.
4. **The cross-encoder reranker has no trained checkpoint**; reranking is
   implemented but not experimentally validated.
5. **Compute reductions** (Table 5) mean results are not full-scale.
6. **The verifier depends on evidence quality**; where retrieval fails, the
   verifier cannot contribute and the system falls back to the classifier.
7. **Embedding-space SHAP is not human-interpretable**, which is why occlusion
   is the primary explanation method.
8. **Occlusion costs one forward pass per token**, forcing explanation to a
   deterministic sample rather than full coverage.
9. **FEVER evidence overlap.** 2,792 FEVER test claims (17.41%) cite evidence
   also used by training claims. This is a corpus property, not label leakage,
   and gold evidence is excluded from the primary retrieval path — but a
   retriever trained on train-side evidence has seen those sentences.
10. **Dataset bias.** English data derives from established benchmark corpora
    whose collection methodology may not represent contemporary news.
11. **Single-seed results** unless the three-seed protocol is completed.

---

## 25. THREATS TO VALIDITY

**Internal.** Leakage between splits is the principal threat; mitigated by
group-aware splitting, the leakage gate, validation-only fitting, and a
verification script that recomputes test predictions from stored parameters.

**External.** Findings may not transfer to other Dravidian languages, other news
domains, or other time periods. The Tamil subset is small (1,462 claims), so
Tamil estimates carry wide uncertainty; bootstrap intervals are reported for
this reason.

**Construct.** Binary fake/real labelling flattens a spectrum that includes
satire, misleading framing and partial truth — a tension visible in X-FACT's
"partly true/misleading" class, which has no binary counterpart.

**Conclusion.** With several systems compared, uncorrected testing would inflate
false-positive risk; Holm correction is applied. Small discordant-pair counts in
the Tamil slice may leave comparisons underpowered, which will be reported
rather than worked around.

---

## 26. CONCLUSION

*To be finalised after experiments.* The conclusion must state what was measured,
what was not, and what remains open — without asserting superiority unless the
statistical evidence supports it after correction.

Provisional framing: this work contributes an integrated, leakage-controlled
framework for evidence-grounded multilingual fake-news detection, together with
a preliminary finding that lexical retrieval outperforms dense multilingual
retrieval on the Tamil evidence subset. Whether external evidence measurably
improves detection for Tamil remains to be determined by the completed
experiments.

---

## 27. FUTURE WORK

**Next semester (planned, not completed).** Construction of a Tamil
news-evidence corpus from Tamil archives and fact-checking organisations;
hard-negative mining to close the dense-versus-lexical gap; cross-lingual
evidence alignment allowing Tamil claims to retrieve English evidence;
domain-adapted multilingual encoders; claim decomposition for multi-part claims;
evidence-graph construction; human evaluation of explanation quality; and a
deployment prototype. Multimodal extension is noted as a longer-term direction
contingent on data availability.

---

## REFERENCES

> **⚠ Every entry below must be independently verified (authors, title, venue,
> year, DOI/URL) before submission.** DOIs are deliberately omitted rather than
> guessed. This list contains only foundational works whose existence is
> certain; it is **not** a complete related-work bibliography, and recent
> Tamil-specific and multilingual fake-news references must be added by the
> authors.

[1] A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán,
E. Grave, M. Ott, L. Zettlemoyer, and V. Stoyanov, "Unsupervised Cross-lingual
Representation Learning at Scale," *Proc. ACL*, 2020.

[2] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model
Predictions," *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[3] F. Feng, Y. Yang, D. Cer, N. Arivazhagan, and W. Wang, "Language-agnostic
BERT Sentence Embedding," *Proc. ACL*, 2022.

[4] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot,
C. Liu, and D. Krishnan, "Supervised Contrastive Learning," *NeurIPS*, 2020.

[5] J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal, "FEVER: a
Large-scale Dataset for Fact Extraction and VERification," *Proc. NAACL-HLT*, 2018.

[6] A. Gupta and V. Srikumar, "X-FACT: A New Benchmark Dataset for Multilingual
Fact Checking," *Proc. ACL-IJCNLP*, 2021.

[7] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25
and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, 2009.

[8] J. Johnson, M. Douze, and H. Jégou, "Billion-scale Similarity Search with
GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, 2019.

[9] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern
Neural Networks," *Proc. ICML*, 2017.

[10] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using
Siamese BERT-Networks," *Proc. EMNLP-IJCNLP*, 2019.

[11] P. K. Verma, P. Agrawal, I. Amorim, and R. Prodan, "WELFake: Word Embedding
Over Linguistic Features for Fake News Detection," *IEEE Transactions on
Computational Social Systems*, vol. 8, no. 4, 2021.

[12] H. Ahmed, I. Traore, and S. Saad, "Detection of Online Fake News Using
N-Gram Analysis and Machine Learning Techniques," *Proc. ISDDC*, 2017.

[13] J. DeYoung, S. Jain, N. F. Rajani, E. Lehman, C. Xiong, R. Socher, and
B. C. Wallace, "ERASER: A Benchmark to Evaluate Rationalized NLP Models,"
*Proc. ACL*, 2020.

[14] Q. McNemar, "Note on the Sampling Error of the Difference Between
Correlated Proportions or Percentages," *Psychometrika*, vol. 12, no. 2, 1947.

[15] S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure,"
*Scandinavian Journal of Statistics*, vol. 6, no. 2, 1979.

**[16]–[30] TO BE ADDED:** recent Tamil / Dravidian NLP, multilingual
fake-news detection, code-mixed classification, low-resource fact verification.

---

## RESULTS TO BE FILLED AFTER COLAB EXECUTION

| # | Result | Source stage | Table |
|---|---|---|---|
| 1 | Per-language and per-class corpus counts | M1/M2 reports | 2 |
| 2 | FEVER dense retrieval metrics | `m6_04 --retriever dense` | 6 |
| 3 | Re-verified FEVER BM25 metrics | `outputs/module6/retrieve/*/bm25_metrics.json` | 6 |
| 4 | FEVER verifier performance | `m6_07` | 8 |
| 5 | X-FACT verifier performance | `m6_07` | 8 |
| 6 | Reranking effect | `m6_05` + `m6_08` | 10 |
| 7 | Module 4 per-record predictions regenerated | `m7_01` | — |
| 8 | Project claim-selection statistics | `m7_02` | — |
| 9 | **Project retrieval coverage per language** | `m7_02` | — |
| 10 | Fusion $\alpha$, threshold, temperature | `m7_04` | 9 |
| 11 | Test classification metrics, all systems | `m7_06` | 7 |
| 12 | Ablations A–H | `m7_06`, `m9_03` | 10 |
| 13 | Error categories and net fusion effect | `m7_07` | 11 |
| 14 | Explanation statistics and faithfulness | `m8_04` | 12 |
| 15 | McNemar tests + Holm-adjusted p-values | `m9_02` | 14 |
| 16 | Bootstrap confidence intervals | `m9_02` | 14 |
| 17 | Robustness slices | `m9_03` | 13 |
| 18 | Figures | `m9_04` | — |
| 19 | Reproducibility report | `m9_05` | — |

### Table 14 — Statistical comparisons *(PENDING)*

| Comparison | Test | n discordant | Statistic | p (raw) | p (Holm) | Significant | 95% CI of Δ |
|---|---|---:|---:|---:|---:|---|---|
| Classifier vs Integrated | McNemar | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Classifier vs Verifier-only | McNemar | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Verifier-only vs Integrated | McNemar | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Tamil: Classifier vs Integrated | McNemar | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| English: Classifier vs Integrated | McNemar | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

---

## PAPER COMPLETION CHECKLIST

### Already supported (writable now)
Title, abstract structure, introduction, problem statement, research gap,
research questions, hypotheses, contributions, full methodology for all nine
modules, dataset description (69,008 records; corpus sizes), preprocessing,
model architecture with mathematical formulation, retrieval methodology,
verification methodology and transfer strategy, fusion formulation, explanation
methodology, experimental setup, evaluation metrics, statistical methodology,
X-FACT retrieval results, reproducibility controls, limitations, threats to
validity, future work, foundational references.

### Pending experimental results
FEVER verifier training completion; X-FACT verifier training; verifier
evaluation; FEVER dense retrieval; Module 7 execution end-to-end; project
retrieval coverage per language; fusion parameters; classification results;
ablations; error analysis; explanation generation and faithfulness; all
statistical tests; figures; reproducibility report.

### Mandatory before submission
1. Complete the related-work section with **verified** recent citations.
2. Verify every reference (authors, venue, year, DOI).
3. Re-verify FEVER BM25 numbers directly from stage JSON.
4. Confirm the X-FACT MRR interpretation against the gold-evidence structure.
5. Run a similarity/plagiarism check.
6. Complete three-seed variance runs if claiming stability.
7. Verify all leakage controls with `m9_06`.
8. Confirm no synthetic dry-run value has entered any table.
9. Have a second reader check the Evidence Status section against the tables.
10. Obtain any required data-use permissions for the collected Tamil sources.

**This manuscript is NOT publication-ready.** It becomes submittable only after
the pending experiments are executed, the statistical tests are run, the
references are verified, and the similarity check is complete.

