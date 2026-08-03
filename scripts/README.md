# scripts/

Orchestration entrypoints (to be added once implementation starts):

- `download_datasets.py` — fetch/prepare public datasets into `data/raw/`.
- `build_pairs.py` — run preprocessing + filtering to produce `data/processed/`.
- `train_contrastive.py` — train Contrast-XLMR (SimCSE objective).
- `train_classifier.py` — train MLP ensemble on top of frozen embeddings.
- `evaluate.py` — run classification/clustering/retrieval/fairness evaluation.
- `run_app.sh` — launch the Flask app.
