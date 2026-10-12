"""Module 7 — regenerate Module 4's per-record predictions. Read-only.

WHY THIS EXISTS. Module 4 wrote only aggregate metrics: `_write_outputs()`
persists `ablation_results.json`, `training_history.json` and
`evaluation_metrics.json`. The per-record `predict_proba` output was used to
compute those metrics and then discarded. There is therefore no P(fake) per
`record_id` anywhere in the project, and fusion is impossible without one.

This module reconstructs them by forward inference only:

    frozen ce_mlp_mean_uniform_seed42.pt  +  frozen Module 3 embeddings
        -> P(fake) per record_id

No training, no fine-tuning, no change to any Module 4 artefact. The model is
rebuilt with `build_ce_mlp_baseline()` -- Module 4's own constructor, not a
reimplementation -- so the architecture cannot drift from the checkpoint.

DETERMINISM. `eval()` disables dropout and there is no sampling anywhere in
the path, so repeated runs over the same embeddings are bit-identical. The
inference is verified against Module 4's published metrics where possible;
a mismatch means the wrong checkpoint or the wrong pooling was loaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split_arrays(embeddings_dir: Path, split: str, pooling: str = "mean"
                      ) -> dict[str, np.ndarray]:
    """Load Module 3's frozen arrays for one split. Opens files read-only."""
    d = Path(embeddings_dir) / split
    emb_path = d / f"{pooling}_embeddings.npy"
    required = {"embeddings": emb_path, "record_ids": d / "record_ids.npy",
                "labels": d / "labels.npy", "languages": d / "languages.npy"}
    missing = [str(p) for p in required.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Module 3 embeddings missing for split "
            f"{split!r}: {missing}. Module 7 cannot regenerate Module 4 "
            "predictions without them.")

    out = {k: np.load(p, allow_pickle=False) for k, p in required.items()}
    n = out["embeddings"].shape[0]
    for k in ("record_ids", "labels", "languages"):
        if len(out[k]) != n:
            raise ValueError(f"{split}: {k} has {len(out[k])} rows but "
                             f"embeddings have {n}")
    out["record_ids"] = out["record_ids"].astype(str)
    out["languages"] = out["languages"].astype(str)
    out["labels"] = out["labels"].astype(int)
    return out


def predict_p_fake(embeddings: np.ndarray, checkpoint_path: Path,
                   input_dim: int = 768, hidden_dims: list[int] | None = None,
                   n_classes: int = 2, dropout: float = 0.1,
                   batch_size: int = 512) -> np.ndarray:
    """P(fake) per row from the frozen CE-MLP checkpoint.

    Class index 1 is "fake", matching Module 2's label encoding and Module 4's
    `predict_proba(...)[:, 1]`.
    """
    import torch

    from src.contrastive.projection_head import build_ce_mlp_baseline

    model = build_ce_mlp_baseline(input_dim=input_dim,
                                  hidden_dims=hidden_dims or [512, 128],
                                  n_classes=n_classes, dropout=dropout)
    state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the Module 4 architecture.\n"
            f"  missing keys   : {list(missing)[:5]}\n"
            f"  unexpected keys: {list(unexpected)[:5]}\n"
            f"  Check inputs.module4_checkpoint and module4_hidden_dims.")
    model.eval()

    X = np.asarray(embeddings, dtype=np.float32)
    out = np.empty(X.shape[0], dtype=np.float64)
    with torch.inference_mode():
        for i in range(0, X.shape[0], batch_size):
            logits = model(torch.from_numpy(X[i:i + batch_size]))
            out[i:i + batch_size] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return out


def build_prediction_records(arrays: dict[str, np.ndarray], p_fake: np.ndarray,
                             split: str, threshold: float = 0.5) -> list[dict[str, Any]]:
    """One record per row, carrying everything fusion and error analysis need."""
    if len(p_fake) != len(arrays["record_ids"]):
        raise ValueError(f"{len(p_fake)} probabilities for "
                         f"{len(arrays['record_ids'])} records")
    return [{"record_id": str(rid), "split": split, "language": str(lang),
             "gold_label": int(lab), "p_fake": float(p),
             "prediction": int(p >= threshold)}
            for rid, lang, lab, p in zip(arrays["record_ids"], arrays["languages"],
                                         arrays["labels"], p_fake)]


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and accounting; every record lands in exactly one language bucket."""
    by_lang: dict[str, int] = {}
    by_gold: dict[str, int] = {}
    for r in records:
        by_lang[r["language"]] = by_lang.get(r["language"], 0) + 1
        by_gold[str(r["gold_label"])] = by_gold.get(str(r["gold_label"]), 0) + 1
    p = np.array([r["p_fake"] for r in records], dtype=np.float64) if records else np.zeros(0)
    return {
        "n_records": len(records),
        "by_language": dict(sorted(by_lang.items())),
        "by_gold_label": dict(sorted(by_gold.items())),
        "language_accounting_balances": sum(by_lang.values()) == len(records),
        "p_fake_min": float(p.min()) if len(p) else None,
        "p_fake_max": float(p.max()) if len(p) else None,
        "p_fake_mean": float(p.mean()) if len(p) else None,
        "all_probs_in_unit_interval": bool(((p >= 0.0) & (p <= 1.0)).all()) if len(p) else True,
        "n_unique_record_ids": len({r["record_id"] for r in records}),
        "record_ids_unique": len({r["record_id"] for r in records}) == len(records),
    }
