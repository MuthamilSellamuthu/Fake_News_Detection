# RESEARCH PAPER AUDIT — Evidence-Grounded Multilingual Fake News Detection
**Audited document:** `docs/research_paper_draft.md`
**Audit date:** current session
**Auditor scope:** scientific correctness, logical coherence, implementation support, reproducibility, unsupported claims, reference integrity.

---

## ⚠ CRITICAL FINDING #1 — THE X-FACT METRIC ANOMALY IS RESOLVED, AND IT WEAKENS A HEADLINE CLAIM

You instructed that the high-MRR / low-Recall@1 combination must not be
interpreted without inspecting the metric implementation and the gold-evidence
definition. Both have now been inspected. **The metrics are implemented
correctly. The anomaly is fully explained by the corpus structure — and the
explanation materially changes how the BM25-versus-dense result must be
reported.**

### The implementation is correct

`reciprocal_rank()` returns `1/rank` of the *first* gold item — standard MRR.
`recall_at_k()` returns `hits_in_top_k / |gold|` — per-claim recall, correctly
dividing by the number of gold items. Neither contains a defect.

### The arithmetic explains the numbers exactly

| Quantity | X-FACT | FEVER |
|---|---:|---:|
| Evidence mentions per claim | **4.951** | 1.156 |
| Mentions per unique evidence item | **1.039** | 4.462 |
| Predicted Recall@1 *if the first gold item were always at rank 1* | **0.2020** | 0.8648 |
| **Measured Recall@1** (val / test) | **0.2068 / 0.2064** | 0.2868 / 0.2938 |
| **Measured MRR** (val / test) | **0.9627 / 0.9621** | 0.4237 / 0.4242 |

For X-FACT the predicted value 0.2020 and the measured value 0.2068 agree to
within 0.005. This is not a coincidence: it demonstrates that for essentially
**every** X-FACT claim, the first gold evidence item is retrieved at rank 1.
Recall@1 is low only because per-claim recall divides by ≈4.95 gold items.

### The consequential part: X-FACT evidence is claim-specific

The ratio of **1.039 mentions per unique evidence item** means each X-FACT
evidence item is cited by essentially exactly one claim. The evidence is not a
shared corpus; it is a per-claim attachment. FEVER, by contrast, has 4.46
mentions per evidence item — evidence there is genuinely shared across claims,
and its MRR (0.42) reflects a substantially harder retrieval problem.

### What this means for the paper

The X-FACT retrieval task, as constructed, is **close to degenerate**: the
system retrieves a claim's own attached evidence from a pool in which that
evidence appears nowhere else, and which shares heavy lexical overlap with the
claim text. This is precisely the condition under which a lexical matcher
excels.

**Therefore the claim "BM25 outperforms dense retrieval for Tamil" must be
narrowed.** The measured result is real, but it supports only:

> On the X-FACT Tamil subset, where each claim is paired with near-exclusive
> evidence exhibiting high lexical overlap, BM25 retrieves more of the gold
> evidence set than LaBSE dense retrieval (test Recall@5 0.6706 vs 0.5347).
> Because evidence in this corpus is almost entirely claim-specific
> (1.04 citations per item), this setting favours lexical matching and does not
> by itself establish that BM25 is superior for Tamil evidence retrieval in
> general.

The FEVER comparison — where evidence is genuinely shared — is the scientifically
informative one, and **FEVER dense retrieval has not yet been run**. Until it is,
the dense-versus-lexical question is not properly answered.

**Required paper actions:** (1) narrow the claim as above; (2) add the
mentions-per-evidence statistic to the dataset table as a corpus property;
(3) move the degeneracy analysis into Threats to Validity; (4) run FEVER dense
retrieval and treat *that* as the primary dense-vs-lexical comparison;
(5) do not report the X-FACT MRR without the accompanying explanation, or a
reviewer will reasonably suspect an evaluation error.

---

## ⚠ CRITICAL FINDING #2 — A SECOND-ORDER CONSEQUENCE FOR MODULE 7

If X-FACT evidence is claim-specific, then retrieving evidence for **project
news articles** against the X-FACT pool is a fundamentally different and much
harder task than the X-FACT-internal retrieval measured above. A project claim
has no attached evidence in that pool; the retriever must find a topically
related fact-check.

Consequently the X-FACT retrieval numbers in Table 6 **must not be presented as
predictive of Module 7 retrieval quality**. The draft does not currently make
this distinction explicit. The `m7_02` coverage measurement is the only figure
that speaks to the real task, and it has not been produced.

---

## PART A — SECTION-BY-SECTION AUDIT (35 sections)

Legend: ✅ sound · 🟡 needs revision · 🔴 blocked on results · ⚠ contains a risk

| # | Section | Status | Finding / required action |
|---|---|---|---|
| 1 | Title | ✅ | Accurate; no superiority claim. Retain. |
| 2 | Abstract | 🟡 | Must add the corpus-degeneracy caveat beside the BM25 result. Currently states the comparison without the qualification. |
| 3 | Keywords | ✅ | Appropriate coverage. |
| 4 | Introduction | ✅ | Motivation is well grounded; the three Tamil difficulties are supported by the measured corpus audit (79 vs 2,518 median chars). |
| 5 | Motivation | ✅ | Embedded in §1. Consider promoting to its own subsection for IEEE format. |
| 6 | Problem statement | ✅ | Formal, correctly scoped. |
| 7 | Research gaps | ✅ | G1–G7 each map to an implemented control. |
| 8 | Research questions | ✅ | RQ1–RQ6 are answerable by the implemented pipeline. |
| 9 | Objectives | 🟡 | Not a separate section in the draft. Add explicit objectives mapped 1:1 to RQs. |
| 10 | Contributions | 🟡 | Six contributions listed; see Novelty audit — two must be downgraded. |
| 11 | Related work | 🔴 | **Largest gap.** Foundational refs present; no verified recent Tamil/Dravidian literature. |
| 12 | Dataset description | 🟡 | Add mentions-per-evidence and per-language/per-class counts (both `[TBD]`). |
| 13 | Dataset construction | ✅ | Audit operations documented and traceable. |
| 14 | Preprocessing | ✅ | NFC-only rationale and language-aware thresholds are correct and defensible. |
| 15 | Multilingual representation | ✅ | Correctly avoids claiming XLM-R optimality. |
| 16 | Fake-news classifier | ✅ | Capacity-matched baseline argument is sound. |
| 17 | Evidence corpus | 🟡 | Must add the claim-specific-evidence property for X-FACT. |
| 18 | BM25 retrieval | ✅ | Tamil tokenisation defect and fix documented — a genuine methodological contribution. |
| 19 | Dense retrieval | ✅ | Correct. |
| 20 | FAISS indexing | ✅ | `IndexFlatIP` over L2-normalised vectors = exact cosine. Stated as implementation, not performance. |
| 21 | Cross-encoder reranking | ✅ | Correctly reported as implemented but **not experimentally validated** (no trained checkpoint). |
| 22 | Fact verification | ✅ | Transfer strategy and head re-initialisation rationale are correct. Status honestly stated. |
| 23 | Decision fusion | ✅ | Mathematics correct; NEI-neutrality rationale is a genuine methodological strength. |
| 24 | Explainability | ✅ | SHAP/occlusion distinction is precise and correctly separated. |
| 25 | Statistical evaluation | ✅ | McNemar + Holm + bootstrap correctly specified; non-rejection wording correct. |
| 26 | Ablation studies | 🔴 | Template only. Ablations F/G/H not yet implemented as runnable experiments — mark as planned. |
| 27 | Leakage prevention | ✅ | Strongest section. Group-aware splitting, validation-only fitting, recomputation check. |
| 28 | Reproducibility | ✅ | Seeds, hashes, manifests, atomic writes all documented and implemented. |
| 29 | Experimental setup | ✅ | Tables 4–5 complete; compute reductions honestly disclosed. |
| 30 | Results | 🔴 | Only Table 6 populated. All else `[TBD]`. Table 6 requires the new caveat. |
| 31 | Discussion | 🔴 | Correctly deferred. Interpretive-risk paragraph is good practice. |
| 32 | Limitations | 🟡 | Add: X-FACT evidence claim-specificity; X-FACT retrieval not predictive of project retrieval. |
| 33 | Future work | ✅ | Clearly marked as planned. |
| 34 | Conclusion | 🔴 | Correctly deferred. |
| 35 | References | 🟡 | 15 verifiable foundational entries, no DOIs. Recent literature missing. |

---

## PART B — TABLE TEMPLATES (14)

Every cell carries an evidence class: **[M]** measured · **[D]** derived ·
**[S]** statistical inference · **[L]** limitation · **[TBD]** pending.

### Table 1 — Dataset statistics
| Property | Value | Class |
|---|---:|---|
| Total standardised records | 69,008 | [M] |
| Train | 48,305 | [M] |
| Validation | 10,351 | [M] |
| Test | 10,352 | [M] |
| Integrity | MD5 verified per split | [M] |
| Duplicates removed | [TBD] | [TBD] |
| Records dropped (below min length) | [TBD] | [TBD] |

### Table 2 — Language distribution
| Language | Train | Validation | Test | Total | Fake | Real | Class |
|---|---:|---:|---:|---:|---:|---:|---|
| English | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Tamil | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Median chars (Tamil) | 79 | — | — | — | — | — | [M] |
| Median chars (English) | 2,518 | — | — | — | — | — | [M] |

### Table 3 — Module 4 classification performance
| Config | Scope | Acc | P | R | Macro-F1 | ROC-AUC | Brier | Class |
|---|---|---:|---:|---:|---:|---:|---:|---|
| CE-MLP (selected) | Overall | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| CE-MLP | English | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| CE-MLP | Tamil | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| SupCon + probe | Overall | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Zero-shot Tamil | Tamil | — | — | — | **0.3216** | — | — | **[M]** |

### Table 4 — FEVER retrieval performance
| Split | Retriever | R@1 | R@5 | R@10 | R@50 | MRR | nDCG@10 | n | Class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Validation | BM25 | 0.2868 | 0.5603 | 0.6384 | 0.7568 | 0.4237 | 0.4644 | 15,916 | [M]† |
| Test | BM25 | 0.2938 | 0.5491 | 0.6383 | 0.7564 | 0.4242 | 0.4652 | 16,026 | [M]† |
| Validation | Dense | [TBD] | | | | | | | [TBD] |
| Test | Dense | [TBD] | | | | | | | [TBD] |

† Requires re-verification directly from `outputs/module6/retrieve/*/bm25_metrics.json`.

### Table 5 — X-FACT retrieval performance
| Split | Retriever | R@1 | R@5 | R@10 | R@50 | MRR | nDCG@10 | n | Class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Validation | Dense | 0.2068 | 0.5338 | 0.6110 | 0.7338 | 0.9627 | 0.6695 | 145 | [M] |
| Validation | BM25 | 0.2068 | 0.7083 | 0.8000 | 0.8776 | 0.9630 | 0.8217 | 145 | [M] |
| Test | Dense | 0.2064 | 0.5347 | 0.6112 | 0.7365 | 0.9621 | 0.6703 | 221 | [M] |
| Test | BM25 | 0.2082 | 0.6706 | 0.7586 | 0.8541 | 0.9685 | 0.7913 | 221 | [M] |

**Mandatory footnote.** X-FACT evidence is near-exclusive to each claim
(1.04 citations per evidence item; 4.95 evidence items per claim). The high MRR
reflects that the first gold item is retrieved at rank 1 for nearly every claim,
while Recall@1 is bounded near 1/4.95 = 0.202 by per-claim normalisation
(measured 0.2064–0.2068). **[D]**

### Table 6 — Dense vs BM25 comparison
| Corpus | Metric | Dense | BM25 | Δ (BM25 − Dense) | Significant? | Class |
|---|---|---:|---:|---:|---|---|
| X-FACT (test) | Recall@5 | 0.5347 | 0.6706 | **+0.1359** | [TBD] | [D] |
| X-FACT (test) | Recall@10 | 0.6112 | 0.7586 | +0.1474 | [TBD] | [D] |
| X-FACT (test) | nDCG@10 | 0.6703 | 0.7913 | +0.1210 | [TBD] | [D] |
| FEVER (test) | Recall@5 | [TBD] | 0.5491 | [TBD] | [TBD] | [TBD] |
| FEVER (test) | nDCG@10 | [TBD] | 0.4652 | [TBD] | [TBD] | [TBD] |

**Interpretation constraint [L]:** the X-FACT rows are measured on a corpus
whose evidence is claim-specific, a condition favouring lexical matching. The
FEVER rows are the generalisable comparison and are incomplete.

### Table 7 — Fact-verification performance
| Corpus | Task | Labels | Acc | Macro-F1 | P | R | Per-class F1 | Class |
|---|---|---|---:|---:|---:|---:|---|---|
| FEVER | Entailment | S/R/NEI | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| X-FACT | Veracity | false/true/partly | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**[L]** Label spaces are distinct tasks and are never merged; results are not comparable across rows.

### Table 8 — Classifier-only vs verifier-only vs fused
| System | Acc | P | R | Macro-F1 | ROC-AUC | Brier | ECE | Class |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A Classifier-only | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| B Verifier-only | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| C Fused (proposed) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### Table 9 — English vs Tamil
| System | Language | Macro-F1 | Δ vs classifier-only | n | Class |
|---|---|---:|---:|---:|---|
| Classifier-only | English | [TBD] | 0.0000 | [TBD] | [TBD] |
| Classifier-only | Tamil | [TBD] | 0.0000 | [TBD] | [TBD] |
| Fused | English | [TBD] | [TBD] | [TBD] | [TBD] |
| Fused | Tamil | [TBD] | [TBD] | [TBD] | [TBD] |

**Reporting rule:** a negative Tamil delta is a finding and is reported with equal prominence.

### Table 10 — Ablation study
| # | Configuration | Macro-F1 | English | Tamil | Δ vs A | Implemented? | Class |
|---|---|---:|---:|---:|---:|---|---|
| A | Classifier only | [TBD] | [TBD] | [TBD] | 0.0000 | ✅ | [TBD] |
| B | Verifier only | [TBD] | [TBD] | [TBD] | [TBD] | ✅ | [TBD] |
| C | Fusion (proposed) | [TBD] | [TBD] | [TBD] | [TBD] | ✅ | [TBD] |
| D | Classifier + retrieval, no verifier | [TBD] | [TBD] | [TBD] | [TBD] | ✅ | [TBD] |
| E | Learned meta-classifier | [TBD] | [TBD] | [TBD] | [TBD] | ✅ | [TBD] |
| F | Dense retrieval only | [TBD] | [TBD] | [TBD] | [TBD] | 🟡 planned | [TBD] |
| G | BM25 only | [TBD] | [TBD] | [TBD] | [TBD] | 🟡 planned | [TBD] |
| H | Oracle gold evidence | [TBD] | [TBD] | [TBD] | [TBD] | ✅ | [TBD] |
| I | No contrastive representation | [TBD] | [TBD] | [TBD] | [TBD] | 🔴 not implemented | [TBD] |

**Do not report row I unless the experiment is actually run.**

### Table 11 — Explainability evaluation
| Metric | Overall | English | Tamil | Definition | Class |
|---|---:|---:|---:|---|---|
| Records explained | [TBD] | [TBD] | [TBD] | sampled, not full split | [TBD] |
| Method | occlusion | occlusion | occlusion | token-level | [M] |
| Deletion mean \|Δ\| | [TBD] | [TBD] | [TBD] | P change after removing top tokens | [TBD] |
| Insertion final gap | [TBD] | [TBD] | [TBD] | P from top tokens only, vs full | [TBD] |
| Concentration top-5 | [TBD] | [TBD] | [TBD] | share of \|attribution\| mass | [TBD] |
| Stability identical rate | [TBD] | [TBD] | [TBD] | bit-identical across runs | [TBD] |
| Failure rate | [TBD] | [TBD] | [TBD] | explanations that errored | [TBD] |

**[L]** No composite explanation-quality score is defined.

### Table 12 — Statistical significance
| Comparison | Test | n discordant | Stat | p raw | p Holm | Reject H₀? | Class |
|---|---|---:|---:|---:|---:|---|---|
| A vs C | McNemar | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [S] |
| A vs B | McNemar | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [S] |
| B vs C | McNemar | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [S] |
| Tamil A vs C | McNemar | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [S] |
| English A vs C | McNemar | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [S] |

**Wording rule:** non-rejection is reported as *"the analysis did not provide
sufficient statistical evidence to reject the null hypothesis at α = 0.05"*.

### Table 13 — Confidence intervals
| System | Metric | Point | 95% CI lower | 95% CI upper | Resamples | Seed | Class |
|---|---|---:|---:|---:|---:|---:|---|
| A | Macro-F1 | [TBD] | [TBD] | [TBD] | 1000 | 42 | [S] |
| C | Macro-F1 | [TBD] | [TBD] | [TBD] | 1000 | 42 | [S] |
| C − A | Δ Macro-F1 | [TBD] | [TBD] | [TBD] | 1000 | 42 | [S] |

### Table 14 — Error analysis categories
| Category | Count | English | Tamil | Class |
|---|---:|---:|---:|---|
| Both correct | [TBD] | [TBD] | [TBD] | [TBD] |
| Classifier correct, verifier wrong | [TBD] | [TBD] | [TBD] | [TBD] |
| Verifier correct, classifier wrong | [TBD] | [TBD] | [TBD] | [TBD] |
| Both wrong | [TBD] | [TBD] | [TBD] | [TBD] |
| **Fusion corrected classifier error** | [TBD] | [TBD] | [TBD] | [TBD] |
| **Fusion introduced error** | [TBD] | [TBD] | [TBD] | [TBD] |
| No verifier output | [TBD] | [TBD] | [TBD] | [TBD] |
| Net fusion effect | [TBD] | [TBD] | [TBD] | [D] |

---

## PART C — NOVELTY AUDIT

| # | Claimed contribution | Class | Strength | Justification |
|---|---|---|---|---|
| A | **Methodological** — validation-only calibrated fusion with conservative tie-breaking that credits the verifier only on strict improvement | Methodological | **STRONG** | Not standard practice; enforced in code and negative-tested; directly prevents overstating a component's contribution |
| B | **System-level** — nine-stage auditable pipeline with per-stage manifests, hashes and verification scripts | System | **MODERATE** | Engineering rigour above the norm, but pipeline integration alone is not a research contribution |
| C | **Multilingual / Tamil** — Tamil-safe preprocessing (NFC-only, language-aware length thresholds) and the demonstrated `\w+` tokenisation defect that fragments Tamil at vowel signs and virama | Multilingual | **STRONG** | Concrete, reproducible, measurable; the tokenisation finding is independently useful to Tamil NLP |
| D | **Evidence-grounding** — measurement of evidence availability for Tamil news, including the claim-specificity property of X-FACT evidence | Evidence | **STRONG** *(conditional on `m7_02` coverage)* | This is the paper's most publishable finding, positive or negative |
| E | **Explainability** — precise separation of embedding-space SHAP from token-level occlusion, with enforced method labelling | Explainability | **MODERATE** | Methodologically correct and addresses a real mislabelling problem, but neither method is new |
| F | **Evaluation / reproducibility** — leakage-controlled protocol with group-aware splitting, Holm correction, recomputation-based leakage verification | Evaluation | **MODERATE–STRONG** | Rigorous; strength depends on whether reviewers value protocol contributions |

**Downgrade required in the draft.** Contribution 1 ("an integrated framework")
and Contribution 2 ("a cross-stage architecture") as currently worded are
**WEAK** on their own — combining established components is not novelty. Rewrite
them to foreground the *measurement* they enable (evidence availability for
Tamil) rather than the combination itself.

**Claims that must NOT be made:** that XLM-R, BM25, FAISS, SHAP or LaBSE are
novel; that the system is state-of-the-art; that fusion improves detection;
that BM25 is generally superior for Tamil retrieval.

---

## PART D — REFERENCE AUDIT

### Correctly represented (verify bibliographic details before submission)
| Ref | Work | Represented correctly? |
|---|---|---|
| [1] XLM-R — Conneau et al., ACL 2020 | ✅ cited as multilingual representation, not as optimal |
| [2] SHAP — Lundberg & Lee, NeurIPS 2017 | ✅ correctly scoped to embedding-space attribution |
| [3] LaBSE — Feng et al., ACL 2022 | ✅ cited for cross-lingual sentence alignment |
| [4] SupCon — Khosla et al., NeurIPS 2020 | ✅ objective correctly described (L^out form, τ=0.07) |
| [5] FEVER — Thorne et al., NAACL 2018 | ✅ 3-way label space correctly stated |
| [6] X-FACT — Gupta & Srikumar, ACL-IJCNLP 2021 | ✅ correctly distinguished from FEVER as veracity not entailment |
| [7] BM25 — Robertson & Zaragoza, FnTIR 2009 | ✅ parameters stated |
| [8] FAISS — Johnson et al., IEEE TBD 2019 | ✅ exactness of FlatIP correctly claimed |
| [9] Calibration — Guo et al., ICML 2017 | ✅ temperature scaling correctly described |
| [10] SBERT — Reimers & Gurevych, EMNLP 2019 | ✅ framework citation |
| [11] WELFake — Verma et al., IEEE TCSS 2021 | ✅ dataset citation |
| [12] ISOT — Ahmed et al., ISDDC 2017 | ✅ dataset citation |
| [13] ERASER — DeYoung et al., ACL 2020 | ✅ motivates deletion/insertion |
| [14] McNemar, Psychometrika 1947 | ✅ paired test correctly justified |
| [15] Holm, Scand. J. Statistics 1979 | ✅ step-down procedure correctly described |

### REFERENCES REQUIRING VERIFICATION
All 15 above require confirmation of exact author list, title, venue, volume,
pages, year and DOI. **DOIs were deliberately omitted rather than guessed.**

### MISSING REFERENCES — search queries, not fabricated citations
No Tamil/Dravidian citation is supplied because none could be verified. Run
these searches in ACL Anthology, IEEE Xplore, SpringerLink, ScienceDirect and
Google Scholar (2021–2026):

1. `"Tamil" AND ("fake news" OR misinformation OR "false information") AND detection`
2. `"Dravidian" AND ("fake news" OR misinformation) AND (LSTM OR BERT OR transformer)`
3. `"code-mixed Tamil" AND (classification OR "offensive language" OR sentiment)`
4. `"low-resource" AND "fact verification" AND (multilingual OR cross-lingual)`
5. `"cross-lingual" AND "claim verification" AND retrieval`
6. `FIRE shared task AND Dravidian AND (fake news OR misinformation)`
7. `"multilingual fake news detection" AND (XLM-R OR mBERT) AND benchmark`
8. `"evidence retrieval" AND "fact checking" AND "low-resource language"`
9. `"Tamil NLP" AND (corpus OR dataset OR benchmark) AND 2023..2026`
10. `"explainable" AND "fake news detection" AND multilingual`

**Minimum for submission:** 10–15 verified recent references, of which at least
4–5 must be Tamil- or Dravidian-specific, and at least 3 must be directly
comparable prior systems for the related-work table.

---

## PART E — FINAL REPORTS

### A. PAPER READINESS REPORT
| Component | State |
|---|---|
| Structure (35 sections) | ✅ complete |
| Methodology | ✅ fully written and implementation-backed |
| Mathematical formulation | ✅ SupCon, fusion, verifier mapping, occlusion |
| Measured results | 🟡 retrieval only |
| Experimental results | 🔴 verification, fusion, explainability, statistics all pending |
| Related work | 🔴 recent literature missing |
| References | 🟡 15 foundational, unverified details, no DOIs |
| Integrity controls | ✅ Evidence Status section present and enforced |
| **Overall** | **NOT SUBMITTABLE** — methodology-complete, results-incomplete |

### B. MISSING EXPERIMENTS
1. FEVER verifier training completion (interrupted at epoch 0, step 500)
2. X-FACT verifier training (not started)
3. Verifier evaluation, both corpora
4. **FEVER dense retrieval** — required for a valid dense-vs-lexical claim
5. Cross-encoder reranker training (else drop reranking from claims)
6. Module 7 end-to-end, including **project retrieval coverage per language**
7. Module 8 explanation generation and faithfulness
8. Module 9 statistics, tables, figures
9. Three-seed variance runs (42/43/44) if stability is claimed

### C. MISSING REFERENCES
Recent Tamil/Dravidian NLP; multilingual fake-news detection systems;
low-resource fact verification; code-mixed processing; explainable multilingual
detection. See Part D search queries.

### D. CLAIMS THAT MUST NOT YET BE MADE
- That evidence-based verification improves detection (any language)
- That the system outperforms any prior work
- That BM25 is generally superior to dense retrieval for Tamil
- That explanations are faithful
- That any difference is statistically significant
- That reranking is effective
- That results are state-of-the-art
- That the architecture is novel merely by combining known components

### E. RESULTS THAT CAN ALREADY BE REPORTED
- Corpus: 69,008 records; 48,305 / 10,351 / 10,352; MD5-verified
- Evidence corpora: FEVER 260,032 claims / 67,388 evidence; X-FACT 1,462 / 6,968
- Group-safety repair: 3 groups, 11 claims moved, 0 deleted; leakage gate 5/5
- X-FACT retrieval, dense and BM25, validation and test **with the degeneracy caveat**
- FEVER BM25 retrieval **pending re-verification from JSON**
- Corpus structure: 4.95 evidence/claim and 1.04 citations/evidence (X-FACT); 1.16 and 4.46 (FEVER)
- Module 4 zero-shot Tamil Macro-F1 = 0.3216
- Tamil tokenisation defect and its correction
- Implementation and test status (all suites, counts stated as software verification, not experiments)

### F. RESULTS REQUIRING MODULE 7
Per-record P(fake); project claim selection statistics; **project retrieval
coverage per language**; fusion α, threshold, temperature; Tables 8, 9, 10, 14;
error categories and net fusion effect.

### G. RESULTS REQUIRING MODULE 8
Table 11 in full; explanation examples for the paper; per-language explanation
statistics; deletion/insertion/concentration/stability.

### H. RESULTS REQUIRING MODULE 9
Tables 12 and 13; Holm-adjusted p-values; bootstrap intervals; robustness
slices; figures; reproducibility report.

### I. FINAL SUBMISSION CHECKLIST
1. ☐ Complete all experiments in B
2. ☐ Re-verify FEVER BM25 from stage JSON
3. ☐ Add the X-FACT degeneracy caveat to abstract, Table 5, and Threats to Validity
4. ☐ Narrow the BM25-vs-dense claim as specified in Critical Finding #1
5. ☐ Run FEVER dense retrieval and make it the primary dense-vs-lexical comparison
6. ☐ Add 10–15 verified recent references, ≥4 Tamil-specific
7. ☐ Verify all 15 foundational references and add DOIs
8. ☐ Complete the related-work comparison table
9. ☐ Rewrite Contributions 1–2 to foreground measurement over combination
10. ☐ Add an explicit Objectives section mapped to RQs
11. ☐ Confirm no synthetic dry-run value appears in any table
12. ☐ Run `m9_06` and confirm all leakage checks pass
13. ☐ Similarity/plagiarism check
14. ☐ Second reader verifies Evidence Status against every table
15. ☐ Confirm data-use permissions for collected Tamil sources
16. ☐ Author list, affiliations, funding, ethics statement
17. ☐ Format to the target venue's template
