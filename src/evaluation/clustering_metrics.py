"""Clustering quality / topic alignment — Module 6.

Runs unsupervised KMeans on the embeddings and asks: does an
unsupervised grouping of the embedding space line up with the true
fake/real label, even though the clustering algorithm never sees that
label? High alignment is evidence the embedding space itself separates
fake/real content along its dominant axes of variation — a stronger
claim than "a supervised classifier can find a decision boundary here",
since clustering has no label signal to exploit at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def compute_clustering_metrics(
    X: np.ndarray, y_true: np.ndarray, n_clusters: int = 2, random_state: int = 42
) -> dict[str, Any]:
    """Cluster `X` into `n_clusters` groups with KMeans, then score the
    result two ways:

    - Adjusted Rand Index / Normalized Mutual Information: how well the
      *unsupervised* cluster assignment matches the *true* fake/real
      label (both are label-alignment metrics; ARI corrects for chance
      agreement, NMI is information-theoretic — reporting both is
      standard practice since they can occasionally disagree at the
      margins).
    - Silhouette score: how well-separated the clusters are from each
      other in the embedding space, independent of whether they
      happen to align with the true label at all.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    metrics: dict[str, Any] = {
        "adjusted_rand_index": adjusted_rand_score(y_true, cluster_labels),
        "normalized_mutual_info": normalized_mutual_info_score(y_true, cluster_labels),
    }

    try:
        metrics["silhouette_score"] = silhouette_score(X, cluster_labels)
    except ValueError:
        # Degenerate case: KMeans collapsed to fewer than 2 distinct
        # clusters actually being populated (can happen on tiny or
        # highly-clustered inputs) — silhouette is undefined there.
        metrics["silhouette_score"] = None

    return metrics
