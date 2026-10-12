"""Module 4 — projection head and linear classifier.

Architecture (from the config, not hardcoded):

    ProjectionHead:  768 -> 512 -> ReLU -> Dropout -> 128 -> L2-normalise
    LinearClassifier: 128 -> 2

The L2 normalisation is part of the module rather than applied by the
training loop, because supervised contrastive loss is defined on the
unit hypersphere: an un-normalised projection silently changes the
objective (dot products stop being cosine similarities) while still
producing a decreasing loss curve. Putting it inside `forward` makes it
impossible to forget.

Two-stage design: stage 1 trains the projection head with SupCon;
stage 2 freezes it and fits a linear classifier on the 128-d output.
The linear probe is deliberately the weakest reasonable classifier, so
any gain over the cross-entropy baseline is attributable to the learned
representation rather than to classifier capacity.

torch is imported lazily so this file can be imported for inspection in
an environment without torch.
"""

from __future__ import annotations

from typing import Any


def build_projection_head(
    input_dim: int = 768,
    hidden_dim: int = 512,
    output_dim: int = 128,
    dropout: float = 0.1,
    normalize: bool = True,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ProjectionHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.normalize = normalize
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            z = self.net(x)
            if self.normalize:
                # eps guards the degenerate all-zero row; without it a
                # single dead sample produces NaNs that propagate through
                # the whole batch's loss.
                z = F.normalize(z, p=2, dim=1, eps=1e-8)
            return z

        def describe(self) -> dict[str, Any]:
            return {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "dropout": dropout,
                "l2_normalised_output": self.normalize,
                "n_parameters": sum(p.numel() for p in self.parameters()),
            }

    return ProjectionHead()


def build_linear_classifier(input_dim: int = 128, n_classes: int = 2):
    """Stage 2 probe. Single linear layer, no hidden units, by design."""
    import torch
    import torch.nn as nn

    class LinearClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(input_dim, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.fc(x)

        def describe(self) -> dict[str, Any]:
            return {
                "input_dim": input_dim,
                "n_classes": n_classes,
                "n_parameters": sum(p.numel() for p in self.parameters()),
            }

    return LinearClassifier()


def build_ce_mlp_baseline(
    input_dim: int = 768,
    hidden_dims: list[int] | None = None,
    n_classes: int = 2,
    dropout: float = 0.1,
):
    """Cross-entropy MLP baseline — the honest comparison.

    Capacity is matched to ProjectionHead + LinearClassifier
    (768->512->128->2) so a difference in results reflects the
    *objective* (SupCon two-stage vs. end-to-end cross-entropy) rather
    than parameter count. Without this control, a SupCon improvement
    could simply be extra layers.
    """
    import torch
    import torch.nn as nn

    dims = hidden_dims or [512, 128]

    class CrossEntropyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = input_dim
            for h in dims:
                layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

        def describe(self) -> dict[str, Any]:
            return {
                "input_dim": input_dim,
                "hidden_dims": dims,
                "n_classes": n_classes,
                "n_parameters": sum(p.numel() for p in self.parameters()),
            }

    return CrossEntropyMLP()
