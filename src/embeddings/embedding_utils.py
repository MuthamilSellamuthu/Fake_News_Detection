"""Module 3 — pooling, normalization, and on-disk embedding storage.

Everything in this file operates on model *outputs* (a transformer's
`last_hidden_state`) and downstream artifacts (the pooled vectors), and
never touches the tokenizer or the model's forward pass itself — see
`tokenizer_utils.py` for input-side concerns and `feature_extraction.py`
for the orchestrator that ties both together.

Deliberately written against plain NumPy for storage and metadata
handling, with torch only imported inside the pooling functions (which
must operate on the actual `last_hidden_state`/`attention_mask` tensors
a transformer forward pass produces). This keeps the storage/metadata
logic testable without a torch installation, which matters in
environments like this project's development sandbox where torch could
not be installed (no GPU, network access to PyPI's own wheel host
blocked, and the resulting CUDA-toolkit dependency chain is far larger
than the sandbox's available disk — see
docs/module3_completion_report.md §0 for the full explanation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def extract_cls_embedding(last_hidden_state: Any) -> Any:
    """Return the leading `<s>` token's final hidden state for every
    sequence in the batch.

    `last_hidden_state` shape: (batch, seq_len, hidden_size). Returned
    shape: (batch, hidden_size). No masking needed here — position 0 is
    always a real (non-padding) token by construction, since the
    tokenizer places `<s>` before any padding for every sequence
    regardless of that sequence's actual length.
    """
    return last_hidden_state[:, 0, :]


def mean_pooling(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Attention-mask-weighted mean over the sequence dimension.

    `last_hidden_state` shape: (batch, seq_len, hidden_size).
    `attention_mask` shape: (batch, seq_len), 1 for real tokens, 0 for
    padding. Returned shape: (batch, hidden_size).

    Padding tokens must be excluded from the average — a plain
    `last_hidden_state.mean(dim=1)` would let batches with more padding
    (i.e. batches containing one long sequence alongside several short
    ones) dilute the short sequences' embeddings with the long
    sequence's padding-position hidden states, which carry no real
    content but are not guaranteed to be zero.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counted = mask.sum(dim=1)
    counted = counted.clamp(min=1e-9)  # guards a hypothetical all-padding row
    return summed / counted


def normalize_embeddings(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row to unit length.

    Required by Module 4's supervised contrastive loss, which is
    defined on the unit hypersphere — cosine similarity between two
    unit vectors is just their dot product, which is what contrastive
    losses (InfoNCE-style) are typically implemented against directly
    for efficiency. `eps` guards the theoretical zero-vector case
    (shouldn't occur for a transformer's hidden states in practice, but
    dividing by exactly zero would silently produce NaNs that could
    propagate unnoticed into Module 4).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=eps, a_max=None)
    return embeddings / norms


@dataclass
class EmbeddingMetadata:
    """Everything needed to trace a saved embedding file back to the
    exact run that produced it, and to join it back to the Module 2
    dataset it came from.
    """

    model_name: str
    pooling_strategy: str
    embedding_dim: int
    max_length: int
    normalized: bool
    split: str
    n_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "pooling_strategy": self.pooling_strategy,
            "embedding_dim": self.embedding_dim,
            "max_length": self.max_length,
            "normalized": self.normalized,
            "split": self.split,
            "n_rows": self.n_rows,
        }


def save_embeddings(
    embeddings: np.ndarray,
    record_ids: np.ndarray,
    labels: np.ndarray,
    languages: np.ndarray,
    output_dir: str | Path,
    split: str,
    pooling_strategy: str,
    metadata: EmbeddingMetadata,
    save_numpy: bool = True,
    save_torch: bool = True,
) -> dict[str, str]:
    """Write one split's pooled embeddings to disk, alongside
    `record_id`, `label`, and `language` arrays so every embedding row
    can be joined straight back to the Module 2 CSV it came from without
    re-deriving anything.

    File layout under `output_dir/<split>/`:
        {pooling_strategy}_embeddings.npy   float32 (n_rows, hidden_size)
        {pooling_strategy}_embeddings.pt    torch.FloatTensor, same data
        record_ids.npy                      <U64  (n_rows,)  [written once per split]
        labels.npy                          int64 (n_rows,)  [written once per split]
        languages.npy                       <U2   (n_rows,)  [written once per split]
        {pooling_strategy}_metadata.json    run/config provenance

    `record_id` is a Module 2 *string* identifier (e.g.
    `"welfake_47809"`, `"aanirudh_headlines_1203"` — source name + a
    numeric suffix, not a bare integer), so it is stored as fixed-width
    Unicode, matching `language`, not as int64.

    Both NumPy and PyTorch formats are written by default (per the
    module spec) because downstream code differs: Module 4/5's PyTorch
    `Dataset`/`DataLoader` classes want tensors; ad-hoc analysis/plots
    (and this report's own verification step) are simplest against
    plain `.npy` arrays with pandas/NumPy, without requiring torch to be
    installed just to inspect the data.
    """
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    if save_numpy:
        np.save(split_dir / f"{pooling_strategy}_embeddings.npy", embeddings.astype("float32"))
        # record_ids/labels/languages are identical across pooling
        # strategies for the same split, but writing them alongside
        # each strategy call is idempotent (same values) and keeps this
        # function self-contained rather than requiring callers to
        # sequence a separate "write metadata once" step correctly.
        np.save(split_dir / "record_ids.npy", np.asarray(record_ids, dtype="<U64"))
        np.save(split_dir / "labels.npy", np.asarray(labels, dtype="int64"))
        np.save(split_dir / "languages.npy", np.asarray(languages, dtype="<U2"))

    written: dict[str, str] = {}
    if save_numpy:
        written["numpy_embeddings"] = str(split_dir / f"{pooling_strategy}_embeddings.npy")
        written["numpy_record_ids"] = str(split_dir / "record_ids.npy")
        written["numpy_labels"] = str(split_dir / "labels.npy")
        written["numpy_languages"] = str(split_dir / "languages.npy")

    if save_torch:
        import torch

        torch.save(torch.from_numpy(embeddings.astype("float32")), split_dir / f"{pooling_strategy}_embeddings.pt")
        # torch has no native string tensor type; record_ids and
        # languages are saved via torch.save on plain Python lists,
        # which pickle cleanly and round-trip with torch.load like any
        # other torch.save'd object. labels are numeric (0/1), so they
        # stay a real int64 tensor.
        torch.save(list(record_ids), split_dir / "record_ids.pt")
        torch.save(torch.from_numpy(np.asarray(labels, dtype="int64")), split_dir / "labels.pt")
        torch.save(list(languages), split_dir / "languages.pt")

        written["torch_embeddings"] = str(split_dir / f"{pooling_strategy}_embeddings.pt")
        written["torch_record_ids"] = str(split_dir / "record_ids.pt")
        written["torch_labels"] = str(split_dir / "labels.pt")
        written["torch_languages"] = str(split_dir / "languages.pt")

    metadata_path = split_dir / f"{pooling_strategy}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2)
    written["metadata"] = str(metadata_path)

    return written


def load_embeddings(
    output_dir: str | Path,
    split: str,
    pooling_strategy: str,
    framework: str = "numpy",
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read one split's saved embeddings back into memory.

    Returns `(embeddings, record_ids, labels, languages, metadata)`.
    `framework` selects both which on-disk format is read for every
    array in the tuple ("numpy" -> the `.npy` files, "torch" -> the
    `.pt` files) and the return type of `embeddings` specifically
    (`np.ndarray` vs. `torch.Tensor`); `record_ids`/`labels`/`languages`
    are always returned as NumPy arrays / a Python list of ints /
    NumPy array respectively regardless of `framework`, since Module
    4/5's own `Dataset` classes are expected to wrap these themselves
    at the point they build batches.
    """
    split_dir = Path(output_dir) / split
    metadata_path = split_dir / f"{pooling_strategy}_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if framework == "torch":
        import torch

        embeddings = torch.load(split_dir / f"{pooling_strategy}_embeddings.pt")
        record_ids = np.asarray(torch.load(split_dir / "record_ids.pt"))
        labels = torch.load(split_dir / "labels.pt").numpy()
        languages = np.asarray(torch.load(split_dir / "languages.pt"))
    else:
        embeddings = np.load(split_dir / f"{pooling_strategy}_embeddings.npy")
        record_ids = np.load(split_dir / "record_ids.npy")
        labels = np.load(split_dir / "labels.npy")
        languages = np.load(split_dir / "languages.npy")

    return embeddings, record_ids, labels, languages, metadata
