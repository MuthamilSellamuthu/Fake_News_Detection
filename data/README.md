# data/

- `raw/` — untouched source datasets as downloaded (IndoHoax/Mafindo-style hoax reports, BanFakeNews, CsFEVER, clickbait sets, or your chosen public dataset).
- `interim/` — intermediate cleaned data (deduplicated, HTML-stripped, class-identifier-stripped) before pair construction.
- `processed/` — final model-ready artifacts: contrastive pairs/triplets, train/val/test splits, filtered pairs (post consistency + threshold filtering).
- `external/` — third-party reference resources (e.g. fact-check corpora, pretrained embedding caches) not produced by this pipeline.

Large files here should stay out of git (see `.gitignore`); track provenance instead (source URL, date pulled, license).
