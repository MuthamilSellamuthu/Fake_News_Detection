"""Focal Loss — Module 5 (PyTorch + Focal Loss variant).

Down-weights the loss contribution of examples the model already
classifies confidently (typically the majority class in an imbalanced
problem), so gradient signal concentrates on hard / minority examples
without needing to resample or discard any training data — an
alternative way of addressing the exact same underlying problem
(class imbalance) that `mlp_ensemble.py`'s balanced-subsampling
approach handles by resampling instead.

Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017).
Adapted here from its original per-pixel object-detection setting to
per-example binary fake/real classification.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_alpha_from_labels(labels: np.ndarray, num_classes: int) -> list[float]:
    """Compute per-class Focal Loss weights inversely proportional to
    each class's frequency in `labels`, normalized so weights average
    to 1 across classes (a class at exactly the average frequency
    gets weight ~1; rarer classes get weight > 1; more common classes
    get weight < 1). This is what `configs/mlp_classifier.yaml`'s
    `loss.alpha: "auto"` resolves to at training time, computed fresh
    from whatever training split is actually loaded rather than
    hardcoded.
    """
    counts = np.array([(labels == c).sum() for c in range(num_classes)], dtype=np.float64)
    counts = np.clip(counts, 1, None)  # avoid divide-by-zero if a class is absent from this split
    inverse = 1.0 / counts
    normalized = inverse * (num_classes / inverse.sum())
    return normalized.tolist()


class FocalLoss(nn.Module):
    """Multi-class Focal Loss over raw logits (softmax applied internally).

    Written for the general `num_classes`-way case (not hardcoded to
    binary) even though this project only currently has 2 classes, so
    the same loss would still work unmodified if a "mixed/unverified"
    third label were ever introduced.
    """

    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits: shape (batch_size, num_classes), raw (pre-softmax) scores.
        targets: shape (batch_size,), integer class indices.

        Mechanics
        ---------
        1. `log_softmax` turns logits into log-probabilities.
        2. Gather each row's log-probability (and, by exponentiating,
           plain probability) for its *true* class only — call this
           `p_t`.
        3. Focal weighting: multiply the standard cross-entropy term
           `-log(p_t)` by `(1 - p_t) ** gamma`. When the model is
           already confidently correct (`p_t` near 1), this factor
           shrinks toward 0, suppressing further gradient from examples
           that don't need it. When the model is wrong or unsure
           (`p_t` near 0), the factor stays near 1, so those examples
           keep contributing at close to full strength — standard
           cross-entropy is the `gamma = 0` special case of this.
        4. Optional `alpha` per-class reweighting on top (see
           `compute_alpha_from_labels`), for when the focusing term
           alone shouldn't be relied on to fully compensate for a
           skewed class distribution.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - target_probs).pow(self.gamma)
        loss = -focal_term * target_log_probs

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
