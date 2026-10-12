"""Stacking-by-averaging MLP ensemble classifier — Module 5.

Mirrors the base paper's ensemble strategy: train several MLP
classifiers, each on an independently-resampled class-balanced subset
of the (typically imbalanced) training embeddings, then average their
predicted probabilities at inference. Spreading "which examples a
balancing draw kept vs. discarded" across many independent random
draws — rather than committing to one fixed undersample — tends to
produce a more stable, less majority-biased final prediction than any
single member alone.

This module does not itself decide what to feed the ensemble — it
trains/predicts on whatever embeddings `scripts/train_classifier.py`
loads (Module 3's frozen embeddings or Module 4's fine-tuned
Contrast-XLMR embeddings), keeping this class agnostic to which
encoder produced its input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.classifier.balanced_sampler import balanced_bootstrap_sample


class MLPEnsembleClassifier:
    """A bagging-style ensemble of `sklearn.neural_network.MLPClassifier`s.

    Not a scikit-learn `BaseEstimator` subclass on purpose — it doesn't
    need `get_params`/`set_params`/`GridSearchCV` compatibility, and
    keeping it a plain class keeps `fit`/`predict`/`save`/`load` easy
    to read top to bottom.
    """

    def __init__(
        self,
        n_estimators: int = 7,
        hidden_layer_sizes: Sequence[int] = (256, 128),
        activation: str = "relu",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        early_stopping: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.estimators_: list[MLPClassifier] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPEnsembleClassifier":
        """Train `self.n_estimators` MLPs, each on its own class-balanced
        subsample (seed = `random_state + i`, so every member sees
        different data *and* has a different weight initialization).
        """
        self.estimators_ = []
        for i in range(self.n_estimators):
            seed = self.random_state + i
            X_balanced, y_balanced = balanced_bootstrap_sample(X, y, seed=seed)

            mlp = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=self.early_stopping,
                random_state=seed,
            )
            mlp.fit(X_balanced, y_balanced)
            self.estimators_.append(mlp)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average every estimator's predicted class probabilities —
        the "predictions averaged at inference" step. Shape `(n_samples,
        2)`, columns ordered `[P(real=0), P(fake=1)]` to match
        scikit-learn's convention (matches `self.estimators_[i].classes_`
        order, which is sorted, i.e. `[0, 1]`).
        """
        if not self.estimators_:
            raise RuntimeError("MLPEnsembleClassifier.fit() must be called before predict_proba().")
        stacked = np.stack([estimator.predict_proba(X) for estimator in self.estimators_], axis=0)
        return stacked.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Threshold the averaged P(fake) at 0.5 for a hard 0/1 prediction."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "MLPEnsembleClassifier":
        return joblib.load(path)
