"""Module 2 — Data Preprocessing.

Cleans and verifies the dataset Module 1 produced: HTML/URL removal,
Tamil-safe Unicode normalization, whitespace/text normalization,
language and label verification, document-length analysis, invalid/
short-document removal, post-cleaning duplicate verification, final
dataset quality checks, and a reproducible train/val/test split
stratified jointly by language and label. See
`src/preprocessing/preprocessing_pipeline.py` for the orchestrator and
`docs/module1_overview.md` for the Module 1 / Module 2 scope boundary.
Contrastive pair construction and tokenization are Module 3's job, not
this one's.
"""