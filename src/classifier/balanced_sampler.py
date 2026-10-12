"""Class-balanced random subsampling — Module 5.

Used to give each MLP in the ensemble its own, differently-balanced
view of the training data: every class is downsampled (without
replacement) to the size of the smallest class, with a fresh random
seed producing a different subsample each call. This plays the same
role bootstrap resampling plays in a random forest (decorrelating
ensemble members so their errors don't all coincide), while also
directly fixing class imbalance for each member's training run rather
than leaving it to the model to compensate for on its own.
"""

from __future__ import annotations

import numpy as np


def balanced_bootstrap_sample(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a class-balanced random subsample of `(X, y)`.

    Every distinct label in `y` is downsampled (without replacement,
    within that class) to `min_count` — the size of the smallest class
    present — so the returned arrays have exactly equal counts of each
    class. Rows are shuffled after concatenation so a downstream model
    doesn't see all of one class before the other.

    Raises no error for an already-balanced input (min_count would
    just equal every class's size, and the function reduces to a
    shuffle). Requires at least 2 rows total.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"Need at least 2 classes to balance; got only {classes.tolist()}.")

    min_count = int(counts.min())
    rng = np.random.default_rng(seed)

    selected_indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        selected_indices.append(chosen)

    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)

    return X[selected_indices], y[selected_indices]
