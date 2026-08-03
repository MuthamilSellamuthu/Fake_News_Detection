# models/

Trained artifacts (git-ignored by default; store via release assets, DVC, or cloud storage):

- `contrastive/` — fine-tuned Contrast-XLMR checkpoints (per training objective: unsupervised / pairwise / triplet).
- `classifiers/` — trained MLP weights (one per ensemble subset) plus the stacking/averaging config.
- `checkpoints/` — intermediate training checkpoints for resuming runs.
