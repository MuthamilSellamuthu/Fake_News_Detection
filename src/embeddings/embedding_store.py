"""Efficient on-disk storage for embeddings — Module 3.

Why HDF5 instead of, say, one big `.npy` or `.csv` per split: embeddings
are large, numeric, and fixed-shape (N rows x hidden_size columns),
which is exactly the case HDF5 is built for. Concretely, this gives us
three things a flat `.npy`/`.csv` file doesn't:

1. Incremental, resizable writes. `AutoModel` forward passes happen in
   batches; with HDF5's `maxshape=(None, dim)` datasets we can append
   each batch to disk as it's produced instead of accumulating every
   embedding for a 60,000+ row split in RAM before writing anything.
2. Partial reads. Module 4/5 can read a slice (e.g. rows 10,000:20,000,
   or just the `label` column) without loading the entire file — useful
   once Tamil volume grows and English is already 60k+ rows.
3. Compression. `gzip` compression on the embeddings dataset
   meaningfully shrinks the on-disk footprint for float32 vectors,
   which tend to compress reasonably well, at a small CPU cost on
   read/write.

File layout written by this module (one .h5 file per split, e.g.
`data/embeddings/train.h5`):

    /embeddings   float32, shape (N, hidden_size)   the pooled vectors
    /record_id    int64,   shape (N,)               joins back to data/processed/<split>.csv
    /label        int64,   shape (N,)               1=fake, 0=real (Module 2 convention)

Root-level HDF5 attributes (`f.attrs[...]`) record the encoder config
that produced the file (model name, pooling strategy, max sequence
length, embedding dim) so a downstream script can sanity-check it's
reading embeddings from the encoder it expects, without re-parsing
`configs/embeddings.yaml`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


class EmbeddingStore:
    """Write-then-read handle for one split's embedding file.

    Usage pattern (see `scripts/generate_embeddings.py`):
        store = EmbeddingStore.create(path, embedding_dim=768, metadata={...})
        for batch in batches:
            store.append(embeddings_batch, record_ids_batch, labels_batch)
        store.close()

        # later, in Module 4/5:
        embeddings, record_ids, labels, meta = EmbeddingStore.load(path)
    """

    def __init__(self, file: h5py.File, embedding_dim: int) -> None:
        # Not meant to be called directly — use `create()` or `open_for_append()`,
        # which set up (or reopen) the resizable datasets correctly first.
        self._file = file
        self._embedding_dim = embedding_dim
        self._count = self._file["embeddings"].shape[0]

    @classmethod
    def create(
        cls,
        path: str | Path,
        embedding_dim: int,
        metadata: dict[str, Any],
        compression: str = "gzip",
        compression_level: int = 4,
    ) -> "EmbeddingStore":
        """Create a brand-new .h5 file with empty, resizable datasets.

        `maxshape=(None, embedding_dim)` on `embeddings` (and
        `maxshape=(None,)` on the 1-D id/label datasets) is the part
        that makes this resizable: HDF5 allocates space lazily and lets
        `append()` grow each dataset's first dimension indefinitely,
        rather than requiring the final row count up front.

        `chunks=True` lets h5py pick a sensible chunk shape for
        incremental writes/compression; without chunking, a resizable
        *and* compressed dataset isn't possible in HDF5.

        `metadata` (encoder config: model name, pooling strategy, etc.)
        is stored as root-level attributes so it travels with the file
        and can be checked by whatever reads it later.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file = h5py.File(path, "w")

        file.create_dataset(
            "embeddings",
            shape=(0, embedding_dim),
            maxshape=(None, embedding_dim),
            dtype="float32",
            chunks=True,
            compression=compression,
            compression_opts=compression_level,
        )
        file.create_dataset(
            "record_id", shape=(0,), maxshape=(None,), dtype="int64", chunks=True
        )
        file.create_dataset(
            "label", shape=(0,), maxshape=(None,), dtype="int64", chunks=True
        )

        for key, value in metadata.items():
            file.attrs[key] = value

        return cls(file, embedding_dim)

    def append(
        self,
        embeddings: np.ndarray,
        record_ids: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Grow every dataset by `len(embeddings)` rows and write the new batch.

        This is what makes the "batch at a time, never hold the whole
        split in memory" property work: each call resizes the on-disk
        arrays to `old_count + new_rows` and writes only the new slice,
        so peak memory is one batch's worth of embeddings, not the
        whole split's.
        """
        n_new = embeddings.shape[0]
        if n_new == 0:
            return
        if embeddings.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self._embedding_dim}, got shape {embeddings.shape}."
            )

        new_count = self._count + n_new
        self._file["embeddings"].resize(new_count, axis=0)
        self._file["record_id"].resize(new_count, axis=0)
        self._file["label"].resize(new_count, axis=0)

        self._file["embeddings"][self._count : new_count] = embeddings.astype("float32")
        self._file["record_id"][self._count : new_count] = np.asarray(record_ids, dtype="int64")
        self._file["label"][self._count : new_count] = np.asarray(labels, dtype="int64")

        self._count = new_count

    def close(self) -> None:
        """Flush and release the underlying file handle.

        Not using a context manager (`with EmbeddingStore(...) as store`)
        for the whole write loop because `create()` returns an already-open
        store meant to be appended to incrementally across many calls;
        the caller closes it once, after the last `append()`.
        """
        self._file.close()

    @staticmethod
    def load(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Read an entire embeddings file back into memory.

        Returns `(embeddings, record_ids, labels, metadata)`. Loading
        everything at once is appropriate for Module 4/5 training,
        which needs the full split in memory anyway (e.g. for building
        contrastive pairs) — for larger-than-RAM scenarios, open the
        file directly with `h5py.File(path, "r")` and slice the
        datasets instead of calling this method, since HDF5 datasets
        support NumPy-style slicing without loading the whole array.
        """
        with h5py.File(path, "r") as file:
            embeddings = file["embeddings"][:]
            record_ids = file["record_id"][:]
            labels = file["label"][:]
            metadata = dict(file.attrs)
        return embeddings, record_ids, labels, metadata
