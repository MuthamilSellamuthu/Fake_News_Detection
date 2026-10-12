"""Module 6 — FAISS flat inner-product index over normalised embeddings.

IndexFlatIP on L2-normalised vectors IS exact cosine similarity, and "flat"
means exhaustive: no approximation, no recall loss, no training step, and
therefore nothing stochastic to reproduce. At this corpus size (67,388 FEVER
and 6,968 X-FACT evidence items) an approximate index would buy little and
cost exactness, so the flat index is the right trade and keeps the retrieval
numbers auditable.

Determinism: vectors are indexed in a fixed, caller-supplied id order, and
ties in the score are broken by that order. `search()` is therefore
reproducible across runs and machines.

A pure-NumPy fallback is used when faiss is unavailable, so tests and CPU
stages run anywhere. `tests/test_retrieval.py` asserts the two agree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover - availability differs by environment
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore
    _HAS_FAISS = False


def faiss_available() -> bool:
    return _HAS_FAISS


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Zero rows stay zero rather than becoming NaN."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class EvidenceIndex:
    """Exact inner-product index with an explicit id ordering."""

    def __init__(self, evidence_ids: Sequence[str], vectors: np.ndarray,
                 normalize: bool = True, use_faiss: bool | None = None,
                 vectors_prenormalized: bool = False) -> None:
        """`normalize` governs BOTH stored vectors and incoming queries.

        `vectors_prenormalized` says the vectors are already unit-length (as
        when reloading from disk) so they are not normalised twice -- but
        queries still are. Conflating these two is a subtle bug: an index
        reloaded with `normalize=False` would stop normalising queries and
        silently return inner products instead of cosines.
        """
        if len(evidence_ids) != vectors.shape[0]:
            raise ValueError(f"{len(evidence_ids)} ids but {vectors.shape[0]} vectors")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("duplicate evidence ids: index order would be ambiguous")
        self.evidence_ids = list(evidence_ids)
        self.dim = int(vectors.shape[1])
        self.normalized = normalize
        vecs = (np.asarray(vectors, dtype=np.float32)
                if (vectors_prenormalized or not normalize) else l2_normalize(vectors))
        self.vectors = np.ascontiguousarray(vecs, dtype=np.float32)
        self.use_faiss = _HAS_FAISS if use_faiss is None else (use_faiss and _HAS_FAISS)
        self._index = None
        if self.use_faiss:
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(self.vectors)

    def __len__(self) -> int:
        return len(self.evidence_ids)

    def search(self, queries: np.ndarray, top_k: int,
               batch_size: int = 128) -> tuple[list[list[str]], np.ndarray]:
        """Top-k evidence ids and scores per query row.

        `top_k` is clamped to the index size so a small scope cannot raise.

        QUERIES ARE PROCESSED IN BATCHES. The NumPy fallback materialises a
        (n_queries x n_evidence) similarity matrix; at FEVER scale that is
        15,916 x 67,388 = 1.07e9 float32 = 4.3 GB, and the subsequent argsort
        allocates a further 8.6 GB of int64. Together they exceed a standard
        Colab runtime and the process is SIGKILLed (exit -9) with no traceback.
        Batching bounds the peak at roughly
        `batch_size * n_evidence * 12 bytes` -- about 100 MB at the default.

        Batching does not change the result: each query's ranking depends only
        on its own row. Full stable argsort is retained within each batch
        rather than argpartition, so tie-breaking remains exactly "highest
        score first, ties by ascending index" and the brute-force equivalence
        test still holds.
        """
        q = l2_normalize(queries) if self.normalized else np.asarray(queries, dtype=np.float32)
        q = np.ascontiguousarray(q, dtype=np.float32)
        n_q = q.shape[0]
        k = min(int(top_k), len(self.evidence_ids))
        if k == 0 or n_q == 0:
            return [[] for _ in range(n_q)], np.zeros((n_q, 0), dtype=np.float32)

        if self.use_faiss and self._index is not None:
            scores, idx = self._index.search(q, k)
        else:
            bs = max(1, int(batch_size))
            idx = np.empty((n_q, k), dtype=np.int64)
            scores = np.empty((n_q, k), dtype=np.float32)
            for start in range(0, n_q, bs):
                stop = min(start + bs, n_q)
                sims = q[start:stop] @ self.vectors.T
                # stable argsort on -sims keeps equal scores in index order
                order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
                idx[start:stop] = order
                scores[start:stop] = np.take_along_axis(sims, order, axis=1)
                del sims, order

        ids = [[self.evidence_ids[j] for j in row if j >= 0] for row in idx]
        return ids, np.asarray(scores, dtype=np.float32)

    def save(self, out_dir: Path) -> dict[str, str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        vec_tmp = out_dir / "vectors.tmp.npy"
        np.save(vec_tmp, self.vectors)
        vec_tmp.replace(out_dir / "vectors.npy")
        meta_tmp = out_dir / "index_meta.tmp.json"
        meta_tmp.write_text(json.dumps({
            "evidence_ids": self.evidence_ids, "dim": self.dim,
            "normalized": self.normalized, "n": len(self.evidence_ids),
            "backend": "faiss" if self.use_faiss else "numpy",
        }, ensure_ascii=False), encoding="utf-8")
        meta_tmp.replace(out_dir / "index_meta.json")
        return {"vectors": str(out_dir / "vectors.npy"),
                "meta": str(out_dir / "index_meta.json")}

    @classmethod
    def load(cls, out_dir: Path, use_faiss: bool | None = None) -> "EvidenceIndex":
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "index_meta.json").read_text(encoding="utf-8"))
        vectors = np.load(out_dir / "vectors.npy")
        return cls(meta["evidence_ids"], vectors,
                   normalize=bool(meta.get("normalized", True)),
                   vectors_prenormalized=True,   # stored vectors already unit-length
                   use_faiss=use_faiss)


def brute_force_search(queries: np.ndarray, vectors: np.ndarray,
                       evidence_ids: Sequence[str], top_k: int
                       ) -> tuple[list[list[str]], np.ndarray]:
    """Reference implementation used only by tests to prove the index exact."""
    q, v = l2_normalize(queries), l2_normalize(vectors)
    sims = q @ v.T
    k = min(top_k, len(evidence_ids))
    idx = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    scores = np.take_along_axis(sims, idx, axis=1)
    return [[evidence_ids[j] for j in row] for row in idx], scores
