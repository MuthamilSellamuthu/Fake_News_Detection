"""Module 4 — language-aware batch sampling.

This is the module's principal methodological contribution, so the
reasoning is recorded here rather than only in the paper.

The problem
-----------
The corpus is 12:1 English:Tamil (63,600 / 5,408 after Module 2). Under
uniform sampling a batch of 256 contains roughly 20 Tamil rows spread
across two labels, so a Tamil anchor frequently has only a handful of
same-label positives -- and sometimes none of the same language.
Supervised contrastive loss builds its geometry from in-batch positive
pairs, so the low-resource language ends up shaping the representation
far less than its research importance warrants. For a paper whose
contribution is *low-resource* multilingual detection, that is the
wrong failure mode to leave in place.

The intervention
----------------
Force a minimum Tamil fraction per batch (default 25%, configurable).
Every batch then reliably contains:

  - same-label / same-language positives for Tamil anchors, and
  - same-label / cross-language positives,

which is exactly the signal that pulls the two languages into a shared
label-consistent geometry.

Honesty about the cost
----------------------
Oversampling Tamil with replacement means Tamil rows repeat within an
epoch while English rows are subsampled. That is a real distributional
intervention, not a free lunch: it can overfit the smaller Tamil pool.
This is why `uniform` sampling is implemented here too and run as a
mandatory ablation -- the claim that language-aware sampling helps is
measured, not assumed.

No row is ever moved between splits and the frozen embeddings are never
modified; this only changes the ORDER and MULTIPLICITY in which train
rows are visited.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


class LanguageAwareBatchSampler:
    """Yields batches of indices with a guaranteed minimum Tamil share.

    Parameters
    ----------
    languages:
        Per-row language tags for the training split.
    batch_size:
        Rows per batch.
    tamil_fraction:
        Target share of each batch drawn from Tamil rows. 0.25 means
        64 Tamil / 192 English in a 256-batch.
    replacement:
        Whether the minority language may repeat within an epoch. True
        is required to sustain the target fraction once the Tamil pool
        is exhausted.
    seed:
        Reproducibility. Recorded in the run's config snapshot.
    drop_last:
        Drop a trailing partial batch. SupCon needs several positives
        per anchor, so ragged final batches are dropped by default.
    """

    def __init__(
        self,
        languages: np.ndarray,
        batch_size: int = 256,
        tamil_fraction: float = 0.25,
        replacement: bool = True,
        seed: int = 42,
        minority_language: str = "ta",
        drop_last: bool = True,
    ) -> None:
        if not 0.0 < tamil_fraction < 1.0:
            raise ValueError(f"tamil_fraction must be in (0, 1), got {tamil_fraction}")
        if batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {batch_size}")

        self.languages = np.asarray(languages).astype(str)
        self.batch_size = int(batch_size)
        self.tamil_fraction = float(tamil_fraction)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.minority_language = minority_language
        self.drop_last = bool(drop_last)

        self.minority_indices = np.where(self.languages == minority_language)[0]
        self.majority_indices = np.where(self.languages != minority_language)[0]

        if len(self.minority_indices) == 0:
            raise ValueError(
                f"No rows with language '{minority_language}' found. Language-aware "
                "sampling cannot guarantee a minority fraction from an empty pool."
            )

        self.n_minority_per_batch = max(1, int(round(self.batch_size * self.tamil_fraction)))
        self.n_majority_per_batch = self.batch_size - self.n_minority_per_batch

        # An epoch is defined by exhausting the MAJORITY pool once, so
        # "epoch" keeps its usual meaning of one pass over the bulk of
        # the data rather than being inflated by oversampling.
        self.n_batches = len(self.majority_indices) // self.n_majority_per_batch
        if not self.drop_last and len(self.majority_indices) % self.n_majority_per_batch:
            self.n_batches += 1

        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle deterministically per epoch (seed + epoch)."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self.seed + self._epoch)

        majority = rng.permutation(self.majority_indices)
        minority_pool = rng.permutation(self.minority_indices)
        minority_cursor = 0

        for b in range(self.n_batches):
            start = b * self.n_majority_per_batch
            maj = majority[start : start + self.n_majority_per_batch]

            need = self.n_minority_per_batch
            if self.replacement:
                # Walk the shuffled minority pool, reshuffling when it
                # runs out. Cheaper and lower-variance than independent
                # draws with replacement, and it guarantees every Tamil
                # row is seen before any repeats.
                chunks = []
                while need > 0:
                    if minority_cursor >= len(minority_pool):
                        minority_pool = rng.permutation(self.minority_indices)
                        minority_cursor = 0
                    take = min(need, len(minority_pool) - minority_cursor)
                    chunks.append(minority_pool[minority_cursor : minority_cursor + take])
                    minority_cursor += take
                    need -= take
                minr = np.concatenate(chunks)
            else:
                if minority_cursor >= len(minority_pool):
                    break
                take = min(need, len(minority_pool) - minority_cursor)
                minr = minority_pool[minority_cursor : minority_cursor + take]
                minority_cursor += take

            batch = np.concatenate([maj, minr])
            if self.drop_last and len(batch) < self.batch_size:
                continue
            # Shuffle so language does not correlate with position.
            yield rng.permutation(batch)

    def describe(self) -> dict:
        return {
            "strategy": "language_aware",
            "batch_size": self.batch_size,
            "tamil_fraction_target": self.tamil_fraction,
            "minority_per_batch": self.n_minority_per_batch,
            "majority_per_batch": self.n_majority_per_batch,
            "actual_fraction": round(self.n_minority_per_batch / self.batch_size, 4),
            "n_minority_rows": int(len(self.minority_indices)),
            "n_majority_rows": int(len(self.majority_indices)),
            "n_batches_per_epoch": self.n_batches,
            "replacement": self.replacement,
            "seed": self.seed,
        }


class UniformBatchSampler:
    """Standard shuffled batching — the ablation control.

    Deliberately plain: this is what the language-aware sampler is
    measured against, so it must be an ordinary random permutation with
    no hidden balancing.
    """

    def __init__(
        self,
        n_rows: int,
        batch_size: int = 256,
        seed: int = 42,
        drop_last: bool = True,
        languages: np.ndarray | None = None,
    ) -> None:
        self.n_rows = int(n_rows)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.languages = None if languages is None else np.asarray(languages).astype(str)
        self.n_batches = self.n_rows // self.batch_size
        if not self.drop_last and self.n_rows % self.batch_size:
            self.n_batches += 1
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self.seed + self._epoch)
        order = rng.permutation(self.n_rows)
        for b in range(self.n_batches):
            batch = order[b * self.batch_size : (b + 1) * self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch

    def describe(self) -> dict:
        info = {
            "strategy": "uniform",
            "batch_size": self.batch_size,
            "n_rows": self.n_rows,
            "n_batches_per_epoch": self.n_batches,
            "seed": self.seed,
        }
        if self.languages is not None:
            n_ta = int((self.languages == "ta").sum())
            info["expected_tamil_per_batch"] = round(self.batch_size * n_ta / self.n_rows, 2)
            info["expected_tamil_fraction"] = round(n_ta / self.n_rows, 4)
        return info


def build_sampler(
    strategy: str,
    languages: np.ndarray,
    batch_size: int,
    tamil_fraction: float,
    seed: int,
    replacement: bool = True,
):
    """Factory used by the trainer so the ablation switches on one string."""
    if strategy == "language_aware":
        return LanguageAwareBatchSampler(
            languages=languages, batch_size=batch_size, tamil_fraction=tamil_fraction,
            replacement=replacement, seed=seed,
        )
    if strategy == "uniform":
        return UniformBatchSampler(
            n_rows=len(languages), batch_size=batch_size, seed=seed, languages=languages,
        )
    raise ValueError(f"Unknown sampling strategy: {strategy!r} (expected 'language_aware' or 'uniform')")
