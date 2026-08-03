# Explainable Multilingual Fake News Detection

Final Year B.E. (Computer Science) project — target venue: IEEE / Springer.

**Title:** Explainable Multilingual Fake News Detection using XLM-RoBERTa, Supervised SimCSE, Fact Verification and Explainable AI for Low-Resource Languages

**Status:** Design phase — architecture, folder structure, and dependencies defined. No model or application code has been implemented yet.

---

## 1. Base Paper

Ni'mah, I., Wijayanti, R., Santosa, A., Jarin, A., Sampurno, T., Uliniansyah, M.T., Fang, M., Menkovski, V., & Pechenizkiy, M. (2025). *A simple contrastive embedding framework for low-resource fake news detection.* Neural Computing and Applications, 37, 21407–21433. https://doi.org/10.1007/s00521-025-11467-0

**What the base paper does:**
The authors treat fake news detection as a *representation learning* problem rather than an end-to-end classification problem. Their pipeline (Contrast-BERT):

1. Collects real-world, **extremely imbalanced** claims from Indonesian hoax-reporting platforms (IndoHoax, ratio ≈ 96:4 fake:real), plus Bangla (BanFakeNews) and Czech (CsFEVER) data for cross-language validation.
2. Builds positive/negative sentence pairs from `{Claim, Context/Article, Fact, Label}` and filters noisy pairs (topic overlap, stylistic similarity, shared entities) using **consistency filtering** and **threshold filtering**.
3. Fine-tunes a monolingual BERT encoder (IndoBERT) with a **SimCSE-style contrastive objective** (unsupervised, weakly-supervised pairwise, or supervised triplet) — never training an end-to-end classifier directly on the imbalanced labels.
4. Uses the resulting embeddings as frozen features for a **stacking ensemble of MLP classifiers**, each trained on a class-balanced subset, with predictions averaged at inference.
5. Evaluates on three axes: classification (P/R/F1/Brier loss), clustering (topic alignment), and retrieval (evidence retrieval quality), plus a fairness audit on sensational-vs-neutral tone.

**Key finding:** Contrast-BERT beats an end-to-end fine-tuned BERT classifier on precision and calibration under extreme imbalance, and generalizes zero-shot to unseen topics/domains — because contrastive pretraining does not directly overfit to the skewed label distribution.

---

## 2. Research Gap

The base paper is a strong contrastive-learning baseline, but leaves several gaps that this project targets:

| # | Gap in base paper | What this project adds |
|---|---|---|
| 1 | **Single-language, single-encoder scope.** Contrast-BERT is retrained separately per language (IndoBERT for Indonesian, separate runs for Bangla/Czech) — there is no shared multilingual representation space. | Use **XLM-RoBERTa** as a single shared multilingual encoder, enabling one model — and one cross-lingual embedding space — to serve multiple low-resource languages, including zero-/few-shot transfer to languages unseen during fine-tuning. |
| 2 | **Contrastive objective is monolingual SimCSE.** Positive/negative pairs are drawn from within one language's claim–evidence structure. | Adopt **supervised SimCSE on XLM-RoBERTa**, explicitly using label-supervised (claim, evidence, hard-negative) triplets so contrastive training benefits from available ground truth while retaining robustness to class imbalance. |
| 3 | **No integrated fact-verification stage.** The paper only *evaluates* embedding quality on a retrieval benchmark (HR@5, MRR@5) — retrieval is a diagnostic metric, not a deployed verification step that produces a verdict. | Add a dedicated **fact verification module** (Sentence-BERT + cosine similarity) that retrieves the most relevant evidence for a claim *and* uses the similarity score as a verifiable, inspectable support/refute signal at inference time. |
| 4 | **No interpretability layer.** The paper's "fairness" analysis (Table 9) audits bias in aggregate but gives no per-prediction explanation of *why* a specific claim was flagged. | Integrate **SHAP** to explain individual predictions (which tokens/features pushed the MLP toward "fake"), addressing the black-box criticism common to BERT-family classifiers and giving end users (e.g., fact-checkers) actionable rationale. |
| 5 | **Research artifact, not a deployable system.** Code/data are released for reproducibility, but there is no user-facing interface. | Ship a **Flask web application** wrapping the full pipeline (claim in → embedding → classification → fact verification → SHAP explanation → verdict out) so the system is usable outside a notebook. |
| 6 | **Evaluated only on Indonesian/Bangla/Czech in isolation.** No demonstration of a single model's cross-lingual low-resource generalization. | Explicitly evaluate **zero-shot/cross-lingual transfer** (train-on-one-language, test-on-another) as a first-class experiment, matching the "for Low-Resource Languages" claim in the project title rather than three separate monolingual pipelines. |

**In one sentence:** the base paper proves contrastive embeddings beat end-to-end classifiers under extreme class imbalance for *one language at a time*; this project extends that idea to a *shared multilingual encoder*, closes the loop with an *explainable, verifiable, deployable* system.

---

## 3. Proposed Workflow

```
Public Dataset Collection
        ↓
Data Preprocessing
        ↓
XLM-RoBERTa Feature Extraction
        ↓
Supervised SimCSE Contrastive Learning
        ↓
MLP Classifier
        ↓
Performance Evaluation
        ↓
Fact Verification (Sentence-BERT + Cosine Similarity)
        ↓
Explainable AI (SHAP)
        ↓
Flask Web Application
```

**Stage-by-stage:**

1. **Public Dataset Collection** — source low-resource-language fake news / fact-checking datasets (e.g., Indonesian, Bangla, Czech, or others), each ideally providing `{claim, context/article, fact, label}`.
2. **Data Preprocessing** — clean text (strip HTML, explicit class-identifier leakage like "[FALSE CLAIM]"), language-detect/normalize, split claims and evidence into sentences, construct positive/negative pairs, and apply consistency + threshold filtering to remove noisy pairs (topic overlap, stylistic similarity, shared entities).
3. **XLM-RoBERTa Feature Extraction** — encode claims and evidence with a pretrained multilingual XLM-RoBERTa checkpoint to obtain contextual sentence embeddings shared across languages.
4. **Supervised SimCSE Contrastive Learning** — fine-tune XLM-RoBERTa with the SimCSE InfoNCE objective using labeled positive pairs and hard negatives (claim vs. matching evidence = positive; claim vs. opposite-class evidence = hard negative), producing "Contrast-XLMR" embeddings robust to class imbalance.
5. **MLP Classifier** — train a stacking ensemble of MLP classifiers on balanced subsets of the resulting embeddings (mirroring the base paper's ensemble strategy), averaging predictions at inference.
6. **Performance Evaluation** — measure accuracy, precision, recall, F1, Brier loss, clustering quality (topic alignment), and retrieval quality (HR@k, MRR@k); include imbalance-ratio ablations and a fairness/bias audit.
7. **Fact Verification (Sentence-BERT + Cosine Similarity)** — given a claim, retrieve top-k most similar evidence sentences from a reference corpus using Sentence-BERT embeddings and cosine similarity, surfacing supporting/refuting evidence alongside the classifier's verdict.
8. **Explainable AI (SHAP)** — apply SHAP to the MLP classifier to attribute the prediction to input features/tokens, giving a human-readable rationale per prediction.
9. **Flask Web Application** — expose the end-to-end pipeline (submit claim → classification + retrieved evidence + SHAP explanation) through a web UI/API for interactive use.

---

## 4. Folder Structure

```
fake-news-detection-xlmr/
├── README.md
├── requirements.txt
├── configs/                    # YAML configs: hyperparameters, paths, thresholds
├── data/
│   ├── raw/                    # untouched source datasets
│   ├── interim/                # cleaned but not yet paired
│   ├── processed/               # contrastive pairs, train/val/test splits
│   └── external/                # third-party reference corpora
├── notebooks/                   # EDA and reporting notebooks only
├── src/
│   ├── data/                    # loading, cleaning, pair construction, filtering
│   ├── embeddings/              # XLM-RoBERTa encoder wrapper
│   ├── contrastive/              # supervised SimCSE model, loss, training loop
│   ├── classifier/               # MLP + stacking ensemble
│   ├── fact_verification/        # Sentence-BERT retriever, cosine similarity verification
│   ├── explainability/           # SHAP explainer
│   ├── evaluation/               # metrics, error analysis, fairness audit
│   └── utils/                    # logging, config loading, seeding
├── app/                          # Flask web application
│   ├── templates/
│   └── static/
├── models/                       # trained checkpoints (git-ignored)
│   ├── contrastive/
│   ├── classifiers/
│   └── checkpoints/
├── reports/
│   ├── figures/
│   └── results/
├── tests/                        # unit tests mirroring src/
└── scripts/                      # orchestration entrypoints (train, evaluate, run app)
```

Each directory contains its own `README.md` explaining its contents in more detail. This structure keeps data, modeling code, trained artifacts, and the serving layer independently swappable — e.g., the encoder or classifier can be upgraded without touching `app/` or `fact_verification/`.

---

## 5. Tech Stack

| Component | Choice |
|---|---|
| Multilingual encoder | `XLM-RoBERTa` (HuggingFace `transformers`) |
| Contrastive learning | Supervised SimCSE (InfoNCE loss) |
| Classifier | MLP ensemble (`scikit-learn` / `torch`) |
| Fact verification | `sentence-transformers` + cosine similarity (`faiss-cpu` for retrieval) |
| Explainability | `shap` |
| Serving | `Flask` |
| Experiment tracking | `mlflow` (optional) |

See `requirements.txt` for pinned versions.

---

## 6. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

(No entrypoint scripts exist yet — this section will be expanded once training/evaluation/app scripts are implemented.)

---

## 7. Roadmap

- [ ] Finalize dataset selection and licensing check
- [ ] Implement `src/data` preprocessing + filtering
- [ ] Implement `src/embeddings` + `src/contrastive` (Contrast-XLMR training)
- [ ] Implement `src/classifier` ensemble
- [ ] Implement `src/evaluation` (classification/clustering/retrieval/fairness)
- [ ] Implement `src/fact_verification`
- [ ] Implement `src/explainability`
- [ ] Implement `app/` (Flask)
- [ ] Write up results for IEEE/Springer submission

---

## 8. Citation

If referencing the base framework this project builds on:

```bibtex
@article{nimah2025contrastive,
  title   = {A simple contrastive embedding framework for low-resource fake news detection},
  author  = {Ni'mah, Iftitahu and Wijayanti, Rini and Santosa, Agung and Jarin, Asril and Sampurno, Tri and Uliniansyah, Mohammad Teduh and Fang, Meng and Menkovski, Vlado and Pechenizkiy, Mykola},
  journal = {Neural Computing and Applications},
  volume  = {37},
  pages   = {21407--21433},
  year    = {2025},
  doi     = {10.1007/s00521-025-11467-0}
}
```

## 9. License

TBD by author/institution.
