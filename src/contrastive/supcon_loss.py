"""Module 4 — Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

Formulation implemented (the paper's L_out, equation 2):

    L = sum_i  -1/|P(i)|  sum_{p in P(i)}  log
            exp(z_i . z_p / tau) / sum_{a in A(i)} exp(z_i . z_a / tau)

where for anchor i:
    A(i) = all indices except i
    P(i) = indices in A(i) sharing i's label

Why L_out rather than L_in: Khosla et al. section 3 shows the
outside-the-log form is the better-behaved gradient, and it is the
variant their reported results use. Getting this wrong is a silent
quality bug -- both forms run and produce plausible curves -- so the
choice is pinned here and covered by a test against a hand-computed
reference.

Anchors with no positives in the batch (their label appears once)
contribute nothing and are excluded from the mean. Under the
language-aware sampler this is rare by construction, but a batch of
all-one-label rows would otherwise yield NaN.

A NumPy reference implementation lives in `supcon_loss_reference()` so
the torch version can be verified numerically in an environment without
torch installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def supcon_loss_reference(
    projections: np.ndarray,
    labels: np.ndarray,
    temperature: float = 0.07,
) -> float:
    """NumPy reference for the supervised contrastive loss.

    Exists to validate the torch implementation. Slower and less
    memory-efficient; correctness is the only goal.

    `projections` must already be L2-normalised, so the dot product is
    cosine similarity.
    """
    z = np.asarray(projections, dtype=np.float64)
    y = np.asarray(labels).reshape(-1)
    n = z.shape[0]
    if n < 2:
        return 0.0

    sim = (z @ z.T) / temperature
    # Numerical stability: subtract the per-row max over the valid
    # (non-self) entries before exponentiating.
    eye = np.eye(n, dtype=bool)
    sim_masked = np.where(eye, -np.inf, sim)
    row_max = np.max(sim_masked, axis=1, keepdims=True)
    exp_sim = np.exp(sim_masked - row_max)
    exp_sim[eye] = 0.0

    denom = exp_sim.sum(axis=1, keepdims=True)          # sum over A(i)
    log_prob = (sim_masked - row_max) - np.log(denom)   # log softmax over A(i)
    # The diagonal is -inf by construction (self-similarity is excluded
    # from A(i)). Multiplying -inf by a False mask entry yields NaN
    # rather than 0, so the excluded entries are zeroed explicitly
    # before the masked sum. This is a real bug the hand-computed test
    # caught: without it every loss value is NaN.
    log_prob[eye] = 0.0

    positive_mask = (y[:, None] == y[None, :]) & ~eye
    n_positives = positive_mask.sum(axis=1)

    valid = n_positives > 0
    if not valid.any():
        return 0.0

    per_anchor = np.zeros(n, dtype=np.float64)
    per_anchor[valid] = -(
        (log_prob * positive_mask)[valid].sum(axis=1) / n_positives[valid]
    )
    return float(per_anchor[valid].mean())


def build_supcon_loss(temperature: float = 0.07):
    """Return a torch `SupConLoss` module.

    Constructed lazily so this file can be imported (and its NumPy
    reference used) in environments without torch.
    """
    import torch
    import torch.nn as nn

    class SupConLoss(nn.Module):
        """Supervised contrastive loss, L_out form."""

        def __init__(self, temperature: float = 0.07) -> None:
            super().__init__()
            self.temperature = float(temperature)

        def forward(self, projections: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
            device = projections.device
            n = projections.shape[0]
            if n < 2:
                return torch.zeros((), device=device, requires_grad=True)

            labels = labels.reshape(-1)
            sim = projections @ projections.T / self.temperature

            eye = torch.eye(n, dtype=torch.bool, device=device)
            neg_inf = torch.finfo(sim.dtype).min
            sim_masked = sim.masked_fill(eye, neg_inf)

            row_max = sim_masked.max(dim=1, keepdim=True).values.detach()
            shifted = sim_masked - row_max
            exp_sim = torch.exp(shifted).masked_fill(eye, 0.0)

            denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
            log_prob = shifted - torch.log(denom)
            # Zero the diagonal explicitly. `neg_inf` here is
            # torch.finfo.min (finite), so this would not currently
            # produce NaN the way a true -inf does -- but relying on
            # that is fragile, and the NumPy reference hit exactly this
            # bug. Making the exclusion explicit keeps the two
            # implementations provably equivalent.
            log_prob = log_prob.masked_fill(eye, 0.0)

            positive_mask = (labels[:, None] == labels[None, :]) & ~eye
            n_positives = positive_mask.sum(dim=1)
            valid = n_positives > 0
            if not valid.any():
                return torch.zeros((), device=device, requires_grad=True)

            pos_log_prob = (log_prob * positive_mask).sum(dim=1)
            per_anchor = -(pos_log_prob[valid] / n_positives[valid])
            return per_anchor.mean()

    return SupConLoss(temperature=temperature)


def batch_positive_statistics(labels: np.ndarray, languages: np.ndarray) -> dict[str, Any]:
    """Diagnostic: how many positives does the average anchor actually get?

    Recorded per epoch because it is the direct, measurable justification
    for language-aware sampling. If Tamil anchors have near-zero
    same-language positives under uniform sampling and a healthy count
    under language-aware sampling, that is the mechanism working -- and
    it belongs in the paper as evidence rather than assertion.
    """
    y = np.asarray(labels).reshape(-1)
    lang = np.asarray(languages).astype(str).reshape(-1)
    n = len(y)
    eye = np.eye(n, dtype=bool)

    same_label = (y[:, None] == y[None, :]) & ~eye
    same_lang = (lang[:, None] == lang[None, :])

    per_anchor_pos = same_label.sum(axis=1)
    per_anchor_same_lang_pos = (same_label & same_lang).sum(axis=1)
    per_anchor_cross_lang_pos = (same_label & ~same_lang).sum(axis=1)

    out: dict[str, Any] = {
        "batch_size": int(n),
        "mean_positives_per_anchor": round(float(per_anchor_pos.mean()), 3),
        "anchors_with_zero_positives": int((per_anchor_pos == 0).sum()),
    }
    for language in ("en", "ta"):
        mask = lang == language
        if mask.any():
            out[f"{language}_rows"] = int(mask.sum())
            out[f"{language}_mean_positives"] = round(float(per_anchor_pos[mask].mean()), 3)
            out[f"{language}_mean_same_language_positives"] = round(float(per_anchor_same_lang_pos[mask].mean()), 3)
            out[f"{language}_mean_cross_language_positives"] = round(float(per_anchor_cross_lang_pos[mask].mean()), 3)
        else:
            out[f"{language}_rows"] = 0
    return out
