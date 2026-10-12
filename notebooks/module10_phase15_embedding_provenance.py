#!/usr/bin/env python
"""Module 10 Phase 1.5 — encoder provenance diagnostic. STRICTLY READ-ONLY.

Writes nothing. Trains nothing. Modifies no config, checkpoint or output.
Regenerates no Module 3 embedding and no Module 8 explanation.

Reproduces the ONE path the project's own source says produced the stored
embeddings, by IMPORTING that implementation rather than reimplementing it:

    scripts/extract_features.py --config configs/feature_config.yaml
      -> src.embeddings.feature_extraction.FeatureExtractor._forward_batch()
      -> src.embeddings.embedding_utils.mean_pooling()
      -> FeatureExtractor._to_numpy() -> embedding_utils.normalize_embeddings()

Hypothesis, stated before the numbers so it cannot be retrofitted:
    stored embeddings are L2-NORMALISED xlm-roberta-base mean-pooled vectors
    over datasets/processed/*.csv column "text" at max_length=512.
Predicts: ||stored|| == 1.0; path B matches; path A does not.

Tolerances are FIXED here and must not be loosened after seeing results:
    cosine >= 0.999   and   |P(fake) - stored| < 1e-4
"""

import csv
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/content/drive/MyDrive/Fake_News_Detection")
RECORD_ID = "welfake_47809"
SPLIT = "test"
COS_TOL = 0.999
P_TOL = 1e-4

sys.path.insert(0, str(ROOT))
csv.field_size_limit(10 ** 9)

import torch                                                          # noqa: E402
from src.contrastive.projection_head import build_ce_mlp_baseline     # noqa: E402
from src.embeddings import embedding_utils                            # noqa: E402
from src.embeddings import feature_extraction as fx_mod               # noqa: E402
from src.embeddings.feature_extraction import FeatureExtractor        # noqa: E402
from src.utils.config_loader import load_yaml                         # noqa: E402

LINE = "=" * 70
print(LINE)
print("MODULE 10 - PHASE 1.5 ENCODER PROVENANCE DIAGNOSTIC")
print(LINE)

fcfg = load_yaml(ROOT / "configs/feature_config.yaml")
m7in = load_yaml(ROOT / "configs/module7_integration.yaml")["inputs"]

# ---- 1. stored Module 3 embedding -------------------------------------
emb_dir = ROOT / fcfg["output"]["embeddings_dir"] / SPLIT
rec_ids = np.load(emb_dir / "record_ids.npy", allow_pickle=False).astype(str)
hits = np.where(rec_ids == RECORD_ID)[0]
if len(hits) != 1:
    sys.exit(f"record_id {RECORD_ID!r} found {len(hits)} times - stop and report.")
row = int(hits[0])
stored_emb = np.load(emb_dir / "mean_embeddings.npy", allow_pickle=False)[row].astype(np.float64)
stored_norm = float(np.linalg.norm(stored_emb))

# ---- 2. stored Module 7 classifier P(fake) (read from file, not hard-coded)
pred_doc = json.loads((ROOT / "outputs/module7/module4_predictions"
                       / f"{SPLIT}_predictions.json").read_text(encoding="utf-8"))
stored_p = next((float(r["p_fake"]) for r in pred_doc["records"]
                 if str(r["record_id"]) == RECORD_ID), None)
if stored_p is None:
    sys.exit(f"{RECORD_ID} has no stored p_fake in Module 7 predictions - stop.")

# ---- 3. original text --------------------------------------------------
in_cfg = fcfg["input"]
csv_p = ROOT / in_cfg["processed_dir"] / in_cfg["test_filename"]
raw_text = None
with csv_p.open(encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        if r.get(in_cfg["id_column"]) == RECORD_ID:
            raw_text = r.get(in_cfg["text_column"]) or ""
            break
if raw_text is None:
    sys.exit(f"{RECORD_ID} not found in {csv_p} - stop and report.")

# ---- 4/5. reproduce, with and without normalisation --------------------
fx = FeatureExtractor(fcfg)
pooled, _ = fx._forward_batch([raw_text])
mean_t = pooled["mean"]
repro_A = mean_t.detach().float().cpu().numpy()[0].astype(np.float64)   # no normalisation
repro_B = fx._to_numpy(mean_t)[0].astype(np.float64)                    # project's own path

# ---- 7. frozen Module 4 CE-MLP ----------------------------------------
ckpt = ROOT / m7in["module4_models_dir"] / m7in["module4_checkpoint"]
clf = build_ce_mlp_baseline(input_dim=int(m7in["module4_input_dim"]),
                            hidden_dims=list(m7in["module4_hidden_dims"]),
                            n_classes=int(m7in["module4_n_classes"]),
                            dropout=float(m7in["module4_dropout"]))
st = torch.load(ckpt, map_location="cpu", weights_only=False)
clf.load_state_dict(st.get("state_dict", st) if isinstance(st, dict) else st, strict=False)
clf.eval()


def p_fake_of(vec):
    with torch.inference_mode():
        x = torch.from_numpy(np.asarray(vec, dtype=np.float32).reshape(1, -1))
        return float(torch.softmax(clf(x), dim=-1)[0, 1])


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---- 6/8/9. report -----------------------------------------------------
print(f"stored record ID              : {RECORD_ID}")
print(f"stored embedding shape        : {stored_emb.shape}")
print(f"stored embedding L2 norm      : {stored_norm:.10f}")
print(f"stored classifier P(fake)     : {stored_p:.16f}")
print(f"text length (chars)           : {len(raw_text)}")

results = {}
for tag, vec, desc in (("A", repro_A, "mean pooling WITHOUT normalization"),
                       ("B", repro_B, "mean pooling WITH L2 normalization")):
    cos = cosine(vec, stored_emb)
    p = p_fake_of(vec)
    dp = abs(p - stored_p)
    ok = (cos >= COS_TOL) and (dp < P_TOL)
    results[tag] = ok
    print(f"\n[{tag}] {desc}")
    print(f"    L2 norm                   : {np.linalg.norm(vec):.10f}")
    print(f"    cosine similarity         : {cos:.10f}")
    print(f"    max |diff| vs stored      : {np.abs(vec - stored_emb).max():.10f}")
    print(f"    mean |diff| vs stored     : {np.abs(vec - stored_emb).mean():.10f}")
    print(f"    reproduced P(fake)        : {p:.16f}")
    print(f"    |P(fake) - stored|        : {dp:.3e}")
    print(f"    matches                   : {ok}")

print()
print(f"path A (no normalisation) matches  : {results['A']}")
print(f"path B (with normalisation) matches: {results['B']}")

# ---- 11. source provenance --------------------------------------------
print(f"\n{LINE}\nSOURCE OF EACH COMPONENT\n{LINE}")


def where(obj):
    try:
        f = Path(inspect.getsourcefile(obj))
        return f"{f.relative_to(ROOT)}:{inspect.getsourcelines(obj)[1]}"
    except Exception:
        return "<unresolved>"


print(f"config          : configs/feature_config.yaml")
print(f"  text_column   : {in_cfg['text_column']!r}")
print(f"  max_length    : {fcfg['tokenization']['max_length']}")
print(f"  padding       : {fcfg['tokenization']['padding']!r}")
print(f"  normalize     : {fcfg['extraction']['normalize_embeddings']}")
print(f"tokenizer       : {type(fx.tokenizer).__name__} <- {fcfg['model']['name']}")
print(f"encoder model   : {type(fx.model).__name__} <- {fcfg['model']['name']}")
print(f"  loaded by     : {where(fx_mod._load_model)}")
print(f"forward+pool    : {where(FeatureExtractor._forward_batch)}")
print(f"mean pooling    : {where(embedding_utils.mean_pooling)}")
print(f"normalization   : {where(embedding_utils.normalize_embeddings)}")
print(f"  applied in    : {where(FeatureExtractor._to_numpy)}")
print(f"classifier      : {where(build_ce_mlp_baseline)}")
print(f"  checkpoint    : {ckpt.relative_to(ROOT)}")
print(f"device          : {fx.device}   amp_active={fx.amp_active}")
print(f"tolerances      : cosine >= {COS_TOL}, |dP(fake)| < {P_TOL}")

meta_p = emb_dir / "mean_metadata.json"
if meta_p.is_file():
    print(f"\nstored provenance metadata ({meta_p.relative_to(ROOT)}):")
    print(json.dumps(json.loads(meta_p.read_text(encoding="utf-8")),
                     indent=2, ensure_ascii=False)[:900])
else:
    print(f"\n{meta_p.relative_to(ROOT)} not present.")

# ---- verdict -----------------------------------------------------------
print(f"\n{LINE}\nVERDICT\n{LINE}")
if results["B"] and not results["A"]:
    print("CONFIRMED: stored embeddings are L2-normalised xlm-roberta-base")
    print("mean-pooled vectors. Module 10 must use this exact path.")
elif results["A"] and not results["B"]:
    print("Stored embeddings are NOT normalised. Hypothesis refuted - report back.")
elif results["A"] and results["B"]:
    print("Both matched - unexpected. Report back before proceeding.")
else:
    print("NEITHER matched. Do NOT proceed to Phase 3. Remaining suspects:")
    print("  text column, max_length, or a datasets/processed revision that")
    print("  differs from the one that produced the stored arrays.")
print(f"\n{LINE}\nDIAGNOSTIC COMPLETE - NOTHING WAS WRITTEN OR MODIFIED.\n{LINE}")
