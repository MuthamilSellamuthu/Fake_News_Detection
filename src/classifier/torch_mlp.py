"""Single MLP classifier (PyTorch) — Module 5 (Focal Loss variant).

Distinct from `mlp_ensemble.py` (the scikit-learn bagging ensemble):
this is one plain feedforward network trained with a custom loss
(Focal Loss), AdamW, and a linear-warmup schedule — scikit-learn's
`MLPClassifier` only supports its own built-in loss/optimizer choices
and can't be trained with any of those, which is why this variant
needs its own small `nn.Module` rather than reusing sklearn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Feedforward network: input embedding -> hidden layers (Linear ->
    ReLU -> Dropout, repeated) -> `num_classes` output logits.

    Outputs raw logits (no final softmax/sigmoid) — `FocalLoss` expects
    raw logits and applies `log_softmax` internally, and evaluation
    code applies `argmax` directly, so no output activation belongs
    here.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.dropout = dropout
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def config(self) -> dict[str, Any]:
        """Architecture config needed to reconstruct this exact network
        before loading a saved `state_dict` — a bare `state_dict` alone
        doesn't record `input_dim`/`hidden_layer_sizes`/`num_classes`,
        so `save_checkpoint`/`load_checkpoint` carry this alongside the
        weights.
        """
        return {
            "input_dim": self.input_dim,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "dropout": self.dropout,
            "num_classes": self.num_classes,
        }


def save_checkpoint(model: MLPClassifier, path: str | Path, extra: dict[str, Any] | None = None) -> None:
    """Save architecture config + weights (+ optional metadata, e.g.
    epoch number / val loss at save time) in one file, so
    `load_checkpoint` can rebuild the exact same network before
    loading its weights back in — a plain `state_dict` alone isn't
    enough to do that.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model_config": model.config(), "state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> MLPClassifier:
    """Reconstruct an `MLPClassifier` from a file written by `save_checkpoint`."""
    payload = torch.load(path, map_location=device)
    model = MLPClassifier(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model
