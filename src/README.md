# src/

Core library code, organized to mirror the pipeline stages:

- `data/` — loading, cleaning, sentence-pair construction, consistency/threshold filtering.
- `embeddings/` — XLM-RoBERTa encoder wrapper (tokenization, pooling, dropout-noise augmentation for SimCSE views).
- `contrastive/` — Supervised SimCSE objective (InfoNCE loss), training loop for Contrast-XLMR.
- `classifier/` — MLP classifier(s) and stacking-based ensemble over balanced subsets.
- `fact_verification/` — Sentence-BERT encoder + cosine-similarity claim-evidence matching.
- `explainability/` — SHAP explainer wrapping the classifier for token/feature attribution.
- `evaluation/` — classification, clustering, retrieval metrics; error analysis and bias/fairness checks.
- `utils/` — logging, config loading, seeding, shared helpers.

Each module is import-only (no side effects at import time); orchestration lives in `scripts/`.
