"""Imbalance-ratio ablation — Module 6.

Re-evaluates an already-trained classifier's metrics after
synthetically adjusting the test set's fake:real ratio via random
*subsampling* — never duplication or synthesis, per this project's
standing rule against fabricating data, which applies to evaluation
data just as much as training data. This isolates how much reported
performance is an artifact of the test set's own class balance, held
separate from anything about the model itself.
"""

from __future__ import annotations

import numpy as np


def subsample_to_ratio(
    X: np.ndarray, y: np.ndarray, fake_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a random subsample of `(X, y)` whose fake-class (label 1)
    fraction is `fake_ratio`, achieved purely by discarding excess
    majority-class rows.

    The total row count necessarily shrinks as `fake_ratio` moves away
    from the data's natural balance — e.g. asking for 90% fake from a
    roughly 50/50 test set means most real rows get dropped so the
    remaining fake:real ratio hits 90:10, not that fake rows get
    invented to reach a bigger total.

    Finds the largest `(n_fake_keep, n_real_keep)` pair that (a) hits
    the target ratio exactly (up to rounding) and (b) doesn't exceed
    the rows actually available in either class.
    """
    if not 0.0 < fake_ratio < 1.0:
        raise ValueError(f"fake_ratio must be strictly between 0 and 1; got {fake_ratio}.")

    rng = np.random.default_rng(seed)
    fake_indices = np.where(y == 1)[0]
    real_indices = np.where(y == 0)[0]
    n_fake, n_real = len(fake_indices), len(real_indices)

    max_total_from_fake = n_fake / fake_ratio
    max_total_from_real = n_real / (1 - fake_ratio)
    total = min(max_total_from_fake, max_total_from_real)

    n_fake_keep = min(int(round(total * fake_ratio)), n_fake)
    n_real_keep = min(int(round(total * (1 - fake_ratio))), n_real)

    fake_selected = rng.choice(fake_indices, size=n_fake_keep, replace=False)
    real_selected = rng.choice(real_indices, size=n_real_keep, replace=False)

    selected = np.concatenate([fake_selected, real_selected])
    rng.shuffle(selected)
    return X[selected], y[selected]
