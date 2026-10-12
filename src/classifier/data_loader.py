"""Load Module 3/4 embeddings for classifier training — Module 5.

A thin, classifier-specific wrapper around `EmbeddingStore.load` so
`src/classifier` depends on a small, stable interface rather than
reaching into `src/embeddings` internals directly from every caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.embedding_store import EmbeddingStore


def load_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read one split's `.h5` embedding file.

    Returns `(X, y, record_ids, metadata)`:
      X          — shape (n_rows, embedding_dim), float32
      y          — shape (n_rows,), int (1=fake, 0=real)
      record_ids — shape (n_rows,), joins back to data/processed/<split>.csv
      metadata   — the encoder config the embeddings were produced with
                    (model name, checkpoint path, etc.), so a training
                    run can log exactly which encoder its embeddings
                    came from.
    """
    embeddings, record_ids, labels, metadata = EmbeddingStore.load(path)
    return embeddings, labels, record_ids, metadata
