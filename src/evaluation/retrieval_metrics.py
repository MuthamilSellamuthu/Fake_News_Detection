"""Retrieval quality: HR@k and MRR@k — Module 6.

This project's data doesn't have an explicit claim/evidence retrieval
structure (same gap noted in Module 4's `pair_sampler.py`), so
retrieval quality is operationalized as a same-class nearest-neighbor
task instead: for every test-split embedding (the "query"), rank the
train-split embeddings (the "reference pool") by cosine similarity and
ask how quickly a same-label neighbor turns up. A high-quality
embedding space should surface same-class neighbors within the very
top ranks; a poor one won't do meaningfully better than chance.

Using the *train* split as the reference pool (rather than the test
split against itself) is deliberate: it avoids the degenerate case
where a point's nearest neighbor is trivially itself, and it mirrors
how retrieval would actually be used downstream (a new claim retrieved
against a fixed, already-indexed reference corpus).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _cosine_similarity_matrix(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Row-normalize both matrices, then a single matrix multiply gives
    every pairwise cosine similarity at once: shape `(n_query, n_reference)`.
    """
    query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    reference_norm = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-12)
    return query_norm @ reference_norm.T


def compute_retrieval_metrics(
    query_X: np.ndarray,
    query_y: np.ndarray,
    reference_X: np.ndarray,
    reference_y: np.ndarray,
    k_values: list[int],
) -> dict[str, Any]:
    """Compute Hit Rate@k and Mean Reciprocal Rank@k for each `k` in
    `k_values`.

    For each query row: rank every reference row by cosine similarity
    (descending), find the rank of the *first* same-label reference row
    (1-indexed), and derive both metrics from that one number —
    `HR@k` = fraction of queries whose first same-label match ranked
    at or above `k`; `MRR@k` = average of `1/rank` for queries whose
    match ranked at or above `k` (0 for queries whose first match, if
    any, fell outside the top `k`).
    """
    max_k = max(k_values)
    similarity = _cosine_similarity_matrix(query_X, reference_X)

    # Top max_k reference indices per query, by descending similarity.
    top_k_indices = np.argpartition(-similarity, kth=min(max_k, similarity.shape[1] - 1), axis=1)[:, :max_k]
    # argpartition doesn't guarantee order *within* the top-k slice; sort just that slice.
    row_indices = np.arange(similarity.shape[0])[:, None]
    order_within_topk = np.argsort(-similarity[row_indices, top_k_indices], axis=1)
    top_k_indices = top_k_indices[row_indices, order_within_topk]

    top_k_labels = reference_y[top_k_indices]  # (n_query, max_k)
    same_label = top_k_labels == query_y[:, None]

    n_query = query_X.shape[0]
    first_match_rank = np.full(n_query, np.inf)
    for i in range(n_query):
        matches = np.nonzero(same_label[i])[0]
        if len(matches) > 0:
            first_match_rank[i] = matches[0] + 1  # convert 0-indexed position to 1-indexed rank

    results: dict[str, float] = {}
    for k in k_values:
        hit = first_match_rank <= k
        results[f"HR@{k}"] = float(hit.mean())
        reciprocal_rank = np.where(hit, 1.0 / first_match_rank, 0.0)
        results[f"MRR@{k}"] = float(reciprocal_rank.mean())

    return results
