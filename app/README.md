# app/

Flask web application exposing the trained pipeline:

- `app.py` — application factory / entrypoint.
- `routes.py` — endpoints (submit claim, view verdict, view SHAP explanation, view retrieved evidence).
- `inference_pipeline.py` — glue code loading trained artifacts (XLM-RoBERTa+SimCSE encoder, MLP ensemble, SBERT retriever, SHAP explainer) and running end-to-end inference on a new claim.
- `templates/`, `static/` — frontend assets.
