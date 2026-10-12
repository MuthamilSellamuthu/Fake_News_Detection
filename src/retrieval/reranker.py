"""Module 6 — XLM-R cross-encoder reranking.

A bi-encoder scores claim and evidence independently, so it never sees how
they interact. A cross-encoder reads `[claim; evidence]` jointly and can
catch negation, entity swaps and hedging -- exactly the distinctions that
matter for fact verification. It is far too slow to score the whole pool,
hence the two-stage design: retrieve 50 cheaply, rerank those 50 carefully.

Hard invariant: reranking is a PERMUTATION of the candidate set. It may
reorder and it may truncate to top-n, but it must never introduce an
evidence id the retriever did not propose. A violation would silently turn
retrieval failures into retrieval successes and inflate every downstream
number. `rerank_candidates()` enforces it in code, and
`tests/test_retrieval.py` negative-tests the guard.

`score_fn` is injected so the whole path is testable without downloading
XLM-R.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


class CandidateSetViolation(AssertionError):
    """Raised when reranking returns an id that was not a candidate."""


def assert_subset_of_candidates(reranked: Sequence[str], candidates: Sequence[str],
                                claim_id: str = "") -> None:
    """Reranking must be a permutation-and-truncation of the candidate set.

    Enforced on the OUTPUT rather than trusted from the implementation, so a
    custom scorer or a subclass cannot quietly widen the pool. Inventing
    evidence would turn retrieval misses into apparent hits and inflate every
    downstream metric.
    """
    extra = set(reranked) - set(candidates)
    if extra:
        raise CandidateSetViolation(
            f"reranker introduced {len(extra)} evidence id(s) absent from the candidate "
            f"set{f' for claim {claim_id}' if claim_id else ''}: {sorted(extra)[:5]}")
    if len(set(reranked)) != len(reranked):
        raise CandidateSetViolation(
            f"reranker returned duplicate evidence ids"
            f"{f' for claim {claim_id}' if claim_id else ''}")


def rerank_candidates(candidates: Sequence[str], scores: Sequence[float],
                      top_n: int) -> tuple[list[str], list[float]]:
    """Sort candidates by score (desc) and keep `top_n`.

    Ties break by original candidate order, so the output is deterministic
    for equal scores -- important because a cross-encoder can easily emit
    identical logits for near-identical evidence sentences.
    """
    if len(candidates) != len(scores):
        raise ValueError(f"{len(candidates)} candidates but {len(scores)} scores")
    order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
    keep = order[: min(int(top_n), len(order))]
    out_ids = [candidates[i] for i in keep]
    out_scores = [float(scores[i]) for i in keep]

    extra = set(out_ids) - set(candidates)
    if extra:
        raise CandidateSetViolation(
            f"reranker produced {len(extra)} id(s) absent from the candidate set: "
            f"{sorted(extra)[:5]}")
    if len(set(out_ids)) != len(out_ids):
        raise CandidateSetViolation("reranker returned duplicate evidence ids")
    return out_ids, out_scores


class CrossEncoderReranker:
    """Resumable reranking over per-claim candidate lists."""

    def __init__(self, score_fn: Callable[[Sequence[tuple[str, str]]], np.ndarray],
                 top_n: int = 5, batch_size: int = 32) -> None:
        self.score_fn = score_fn
        self.top_n = int(top_n)
        self.batch_size = int(batch_size)

    def rerank_one(self, claim_text: str, candidate_ids: Sequence[str],
                   evidence_texts: dict[str, str]) -> tuple[list[str], list[float]]:
        if not candidate_ids:
            return [], []
        pairs = [(claim_text, evidence_texts.get(e, "")) for e in candidate_ids]
        scores: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            scores.extend(np.asarray(self.score_fn(pairs[i:i + self.batch_size]),
                                     dtype=np.float32).ravel().tolist())
        if len(scores) != len(candidate_ids):
            raise RuntimeError(f"score_fn returned {len(scores)} scores for "
                               f"{len(candidate_ids)} candidates")
        return rerank_candidates(candidate_ids, scores, self.top_n)

    def rerank_all(self, claims: dict[str, str], candidates: dict[str, Sequence[str]],
                   evidence_texts: dict[str, str], out_dir: Path,
                   shard_size: int = 1000,
                   log: Callable[[str], None] = print) -> dict[str, Any]:
        """Rerank every claim, sharding + checkpointing as we go."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"
        man: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                man = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                man = {}

        claim_ids = sorted(candidates)          # fixed order -> resumable shards
        n_shards = (len(claim_ids) + shard_size - 1) // shard_size if claim_ids else 0
        done = set(man.get("completed_shards", []))
        for i in range(n_shards):
            if (out_dir / f"rerank_{i:05d}.json").is_file():
                done.add(i)

        man.update({"n_claims": len(claim_ids), "shard_size": shard_size,
                    "n_shards": n_shards, "top_n": self.top_n,
                    "completed_shards": sorted(done), "status": "in_progress"})
        manifest_path.write_text(json.dumps(man, indent=2), encoding="utf-8")

        t0 = time.perf_counter()
        for i in range(n_shards):
            if i in done:
                continue
            lo, hi = i * shard_size, min((i + 1) * shard_size, len(claim_ids))
            shard: dict[str, Any] = {}
            for cid in claim_ids[lo:hi]:
                ids, scores = self.rerank_one(claims.get(cid, ""), candidates[cid],
                                              evidence_texts)
                # Enforce the invariant on whatever rerank_one returned. A
                # subclass or a custom scorer could break it; the pipeline must
                # not trust the reranker's own bookkeeping.
                assert_subset_of_candidates(ids, candidates[cid], cid)
                shard[cid] = {"evidence_ids": ids, "scores": scores}
            tmp = out_dir / f"rerank_{i:05d}.tmp.json"
            tmp.write_text(json.dumps(shard, ensure_ascii=False), encoding="utf-8")
            tmp.replace(out_dir / f"rerank_{i:05d}.json")
            done.add(i)
            man["completed_shards"] = sorted(done)
            manifest_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
            log(f"  rerank shard {i + 1}/{n_shards} [{lo}:{hi}]")

        man["status"] = "complete"
        man["total_seconds"] = round(time.perf_counter() - t0, 2)
        manifest_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
        return man


def load_reranked(out_dir: Path) -> dict[str, dict[str, Any]]:
    out_dir = Path(out_dir)
    merged: dict[str, dict[str, Any]] = {}
    for p in sorted(out_dir.glob("rerank_*.json")):
        if p.name.endswith(".tmp.json"):
            continue
        merged.update(json.loads(p.read_text(encoding="utf-8")))
    return merged


def build_cross_encoder_score_fn(model_name: str = "xlm-roberta-base",
                                 max_length: int = 256, device: str | None = None
                                 ) -> Callable[[Sequence[tuple[str, str]]], np.ndarray]:
    """Lazily build a relevance scorer from an XLM-R sequence classifier.

    Imported inside the function so the module stays importable without
    transformers and without downloading weights. An untrained head gives
    arbitrary scores -- this is wired for a fine-tuned checkpoint, and the
    Colab notebook says so at the point of use.
    """
    import torch  # local imports on purpose
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    @torch.no_grad()
    def _score(pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        enc = tok([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                  truncation=True, max_length=max_length, return_tensors="pt").to(dev)
        return model(**enc).logits.squeeze(-1).float().cpu().numpy()

    return _score
