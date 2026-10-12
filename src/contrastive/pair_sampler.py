"""Builds (anchor, positive, hard-negative) index triplets for supervised
SimCSE training — Module 4.

Pure NumPy, no PyTorch dependency, so this piece is independently
testable and reusable regardless of what deep learning framework the
rest of the module uses.

Pairing rule (label-supervised, language-agnostic — see
configs/contrastive.yaml for why): for anchor row `i` with label
`labels[i]`,
  - positive_idx[i] is a *different* row that shares `labels[i]`.
  - negative_idx[i] is a row with a *different* label than `labels[i]`.

Both pools are drawn from the *entire* split (English and Tamil rows
mixed together), which is what makes the resulting contrastive space
shared across languages rather than per-language.
"""

from __future__ import annotations

import numpy as np


def build_supervised_triplets(labels: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return `(positive_idx, negative_idx)`, each shape `(len(labels),)`.

    Row `i` of the training split should be trained as:
        anchor   = row i
        positive = row `positive_idx[i]`  (different row, same label)
        negative = row `negative_idx[i]`  (different label)

    Positive assignment uses a shuffle-and-rotate trick per label group
    instead of rejection sampling: shuffle every same-label group's
    indices, then pair each shuffled index with the *next* one in the
    shuffled order (wrapping around). Because every element is paired
    with a different position in the same permutation, no row is ever
    paired with itself — without needing a resample-until-different
    loop. The one exception is a label group of size 1, where there is
    no valid "different row, same label" partner; that row is paired
    with itself as a documented fallback (equivalent to unsupervised
    SimCSE's dropout-noise self-pair for that single row).

    Negative assignment samples uniformly at random (with replacement)
    from every row *not* sharing the anchor's label — cheap, and
    sufficient given InfoNCE also treats every other example's positive
    and negative in the same training batch as additional implicit
    negatives (see `src/contrastive/losses.py`), so this explicit
    negative doesn't need to be adversarially "hard" on its own.

    `seed` should differ across epochs (e.g. pass the epoch number) so
    each epoch samples a fresh set of pairs rather than repeating the
    same ones — this is cheap re-randomization "for free" data
    augmentation across epochs.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError(
            "build_supervised_triplets needs at least 2 distinct labels present "
            f"to form hard negatives; got only {unique_labels.tolist()}."
        )

    label_to_indices = {lbl: np.where(labels == lbl)[0] for lbl in unique_labels}

    positive_idx = np.empty(n, dtype=np.int64)
    negative_idx = np.empty(n, dtype=np.int64)

    for lbl, indices in label_to_indices.items():
        m = len(indices)
        shuffled = rng.permutation(indices)

        if m > 1:
            rotated = np.roll(shuffled, -1)
        else:
            rotated = shuffled  # only one row with this label; self-pair fallback (see docstring)
        positive_idx[shuffled] = rotated

        other_label_indices = np.concatenate(
            [idx for other_lbl, idx in label_to_indices.items() if other_lbl != lbl]
        )
        negative_idx[indices] = rng.choice(other_label_indices, size=m, replace=True)

    return positive_idx, negative_idx
