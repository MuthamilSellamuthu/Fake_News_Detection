"""Module 10 — raw text to a Module-3-compatible embedding.

THE ONLY NEW MODEL-FACING COMPONENT IN MODULE 10, and it exists solely because
Modules 1-9 provide no single-raw-text -> embedding API: `m7_01` reads
precomputed `.npy` arrays, and `m8_02`'s text path uses a different encoder.

It does not reimplement the encode path. It constructs Module 3's own
`FeatureExtractor` from Module 3's own `configs/feature_config.yaml` and calls
its `_forward_batch()` and `_to_numpy()`. Tokenizer, model, pooling and
normalisation are therefore Module 3's, by construction rather than by
resemblance.

    xlm-roberta-base | max_length 512 | padding "longest" | truncation
      -> embedding_utils.mean_pooling(last_hidden_state, attention_mask)
      -> embedding_utils.normalize_embeddings(...)   -> unit L2 norm
      -> float32 (768,)

WHY THE SELF-CHECK EXISTS
------------------------
Module 8 encodes with `sentence-transformers/LaBSE` while the classifier was
trained on xlm-roberta-base mean-pooled unit vectors. Both are 768-dimensional,
so the mismatch raises no error -- it silently produces saturated
probabilities. Measured on the same records:

    LaBSE (unit norm, wrong space)          -> P(fake) = 0.000000
    XLM-R mean-pool, unnormalised           -> P(fake) = 1.000000
    XLM-R mean-pool + L2 normalisation      -> P(fake) = 0.0030934  (stored: 0.0030954)

Right space with the wrong scale and right scale with the wrong space both
saturate. A dimension check cannot catch either. So this encoder reproduces a
known record at construction and REFUSES TO START if it cannot.

NOTE ON COSINE: cosine similarity is scale-invariant. In the Phase 1.5
diagnostic BOTH the normalised and unnormalised paths scored 0.9999999994
against the stored vector. Cosine detects a wrong-space encoder; only the
P(fake) comparison detects a wrong-scale one. Both are required.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


class EncoderProvenanceError(RuntimeError):
    """The encoder cannot reproduce a known stored embedding or probability.

    Raised at construction. Inference must stop rather than silently run on an
    incompatible encoder -- that is the exact failure this class guards.
    """


class InferenceEncoder:
    """Raw text -> the unit-norm 768-d vector Module 4 was trained on."""

    def __init__(self, root: Path, m10_cfg: dict[str, Any],
                 classify_fn: Any = None, self_check: bool = True) -> None:
        from scripts.m6_common import load_config as _  # noqa: F401  (path guard)
        from src.utils.config_loader import load_yaml

        self.root = Path(root)
        self.cfg = m10_cfg
        self.feature_cfg = load_yaml(self.root / m10_cfg["inputs"]["feature_config"])

        model_cfg = self.feature_cfg["model"]
        tok_cfg = self.feature_cfg["tokenization"]
        self.encoder_name: str = model_cfg["name"]
        self.max_length: int = int(tok_cfg["max_length"])
        self.padding: str = str(tok_cfg["padding"])
        self.pooling: str = "mean"
        self.normalized: bool = bool(self.feature_cfg["extraction"]["normalize_embeddings"])

        if self.encoder_name != "xlm-roberta-base":
            raise EncoderProvenanceError(
                f"Module 3 was run with 'xlm-roberta-base'; feature_config now "
                f"says {self.encoder_name!r}. Refusing to start.")
        if not self.normalized:
            raise EncoderProvenanceError(
                "feature_config sets normalize_embeddings=false, but Module 4's "
                "input gate asserts unit vectors (src/contrastive/embedding_dataset.py). "
                "Refusing to start.")

        from src.embeddings.feature_extraction import FeatureExtractor
        self._fx = FeatureExtractor(self.feature_cfg)      # loads model + tokenizer once

        self.self_check_result: dict[str, Any] | None = None
        if self_check:
            if classify_fn is None:
                raise EncoderProvenanceError(
                    "self_check=True requires classify_fn to compare P(fake).")
            self.self_check_result = self._run_self_check(classify_fn)

    # ------------------------------------------------------------------ API
    def encode(self, text: str) -> np.ndarray:
        """One text -> float32 (768,), unit L2 norm."""
        return self.encode_many([text])[0]

    def encode_many(self, texts: list[str]) -> np.ndarray:
        """Batch encode. Shape (n, 768), float32, each row unit norm."""
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        pooled, _ = self._fx._forward_batch(list(texts))
        return self._fx._to_numpy(pooled[self.pooling]).astype(np.float32)

    def provenance(self) -> dict[str, Any]:
        return {"encoder": self.encoder_name, "pooling": self.pooling,
                "normalized": self.normalized, "max_length": self.max_length,
                "padding": self.padding, "embedding_dim": 768,
                "source_config": self.cfg["inputs"]["feature_config"],
                "implementation": ("src.embeddings.feature_extraction.FeatureExtractor"
                                   " (_forward_batch + _to_numpy)"),
                "self_check": self.self_check_result}

    # ---------------------------------------------------------- self-check
    def _run_self_check(self, classify_fn: Any) -> dict[str, Any]:
        c = self.cfg["encoder_self_check"]
        rid, split = str(c["record_id"]), str(c["split"])
        min_cos, max_dp = float(c["min_cosine"]), float(c["max_p_fake_delta"])

        stored_emb, stored_p = self._load_known_record(rid, split)
        text = self._load_record_text(rid, split)
        repro = self.encode(text).astype(np.float64)
        stored_emb = stored_emb.astype(np.float64)

        cos = float(repro @ stored_emb /
                    (np.linalg.norm(repro) * np.linalg.norm(stored_emb) + 1e-12))
        p_repro = float(classify_fn(repro.astype(np.float32)))
        dp = abs(p_repro - stored_p)
        ok = (cos >= min_cos) and (dp < max_dp)

        result = {"record_id": rid, "split": split, "cosine": cos,
                  "stored_p_fake": stored_p, "reproduced_p_fake": p_repro,
                  "abs_p_fake_delta": dp, "min_cosine": min_cos,
                  "max_p_fake_delta": max_dp,
                  "reproduced_l2_norm": float(np.linalg.norm(repro)),
                  "passed": ok}
        if not ok:
            raise EncoderProvenanceError(
                "ENCODER PROVENANCE SELF-CHECK FAILED — refusing to run inference.\n"
                f"  record            : {rid} ({split})\n"
                f"  cosine            : {cos:.10f}   (need >= {min_cos})\n"
                f"  stored P(fake)    : {stored_p:.16f}\n"
                f"  reproduced P(fake): {p_repro:.16f}\n"
                f"  |difference|      : {dp:.3e}      (need < {max_dp})\n"
                f"  encoder           : {self.encoder_name}, pooling={self.pooling}, "
                f"normalized={self.normalized}, max_length={self.max_length}\n"
                "  A saturated reproduced probability (0.000000 or 1.000000) means the\n"
                "  encoder is in the wrong SPACE or at the wrong SCALE. Do not loosen\n"
                "  these tolerances; fix the encode path.")
        return result

    def _load_known_record(self, rid: str, split: str) -> tuple[np.ndarray, float]:
        import json

        emb_dir = self.root / self.cfg["inputs"]["embeddings_dir"] / split
        ids_p, emb_p = emb_dir / "record_ids.npy", emb_dir / f"{self.pooling}_embeddings.npy"
        for p in (ids_p, emb_p):
            if not p.is_file():
                raise EncoderProvenanceError(
                    f"Cannot run the encoder self-check: {p} is missing. Module 10 "
                    "will not start without verifying encoder provenance.")
        ids = np.load(ids_p, allow_pickle=False).astype(str)
        hit = np.where(ids == rid)[0]
        if len(hit) != 1:
            raise EncoderProvenanceError(
                f"self-check record {rid!r} appears {len(hit)} times in {ids_p}")
        emb = np.load(emb_p, allow_pickle=False)[int(hit[0])]

        m7 = self.root / "outputs/module7/module4_predictions" / f"{split}_predictions.json"
        if not m7.is_file():
            raise EncoderProvenanceError(
                f"Cannot run the encoder self-check: {m7} is missing.")
        doc = json.loads(m7.read_text(encoding="utf-8"))
        p = next((float(r["p_fake"]) for r in doc["records"]
                  if str(r["record_id"]) == rid), None)
        if p is None:
            raise EncoderProvenanceError(f"{rid} has no stored p_fake in {m7}")
        return emb, p

    def _load_record_text(self, rid: str, split: str) -> str:
        csv.field_size_limit(10 ** 9)
        in_cfg = self.feature_cfg["input"]
        fname = {"train": in_cfg["train_filename"], "validation": in_cfg["val_filename"],
                 "test": in_cfg["test_filename"]}[split]
        path = self.root / in_cfg["processed_dir"] / fname
        if not path.is_file():
            raise EncoderProvenanceError(f"self-check CSV missing: {path}")
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get(in_cfg["id_column"]) == rid:
                    return row.get(in_cfg["text_column"]) or ""
        raise EncoderProvenanceError(f"{rid} not found in {path}")
