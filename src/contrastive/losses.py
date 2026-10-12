"""Supervised SimCSE / InfoNCE contrastive loss — Module 4.

Given one batch of (anchor, positive, hard-negative) embedding triples,
pulls each anchor toward its own positive and away from every other
embedding in the batch — both its own hard negative *and* every other
row's positive/negative, which act as free "in-batch negatives". This
in-batch-negative trick (rather than scoring each triple in isolation)
is what makes SimCSE-style training effective: with batch size B, each
anchor is contrasted against 2B candidates instead of just 2.

False-negative masking, and why it's necessary here: SimCSE's in-batch
trick was designed for NLI data, where "some other anchor's positive"
is very unlikely to coincidentally belong to the current anchor's own
class. That assumption breaks with only two classes (fake/real):
roughly half of "some other anchor's positive" will, by chance, share
the current anchor's label. Treating those as negatives would actively
push same-class texts apart — the opposite of what a fake/real
contrastive space should do. `supervised_simcse_loss` masks these out
using the batch's labels before computing the loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_simcse_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute the batch's (false-negative-masked) InfoNCE loss.

    Parameters
    ----------
    anchor, positive, negative:
        Each shape `(batch_size, hidden_dim)`. Row `i` of `anchor` is
        trained against row `i` of `positive` (its true match) and,
        as free in-batch negatives, every row of `negative` plus every
        *other* row of `positive` whose label differs from anchor i's.
    labels:
        Shape `(batch_size,)`, the class label (0/1) of each anchor
        (equivalently, of each row i's own text — anchor and its
        positive always share this label by construction). Used only
        to build the false-negative mask described below; not used as
        a supervised classification target here.
    temperature:
        Softmax temperature. Embeddings are cosine-similarity scored
        (via L2 normalization + dot product) and divided by this value
        before the softmax/cross-entropy — a smaller temperature makes
        the loss punish near-miss positives more sharply, which is the
        standard SimCSE default (0.05).

    Mechanics
    ---------
    1. L2-normalize all three sets of embeddings, so their dot product
       is exactly cosine similarity (scale-invariant).
    2. Stack `positive` and `negative` into one `(2*batch_size,
       hidden_dim)` candidate pool and score every anchor against every
       candidate: `sim[i, j] = cosine(anchor[i], candidates[j]) /
       temperature`, shape `(batch_size, 2*batch_size)`.
    3. Build a boolean mask over that same shape marking which entries
       are valid loss terms for each anchor row:
         - every `negative`-block entry is valid (pair_sampler already
           guarantees these are opposite-label rows);
         - `positive`-block entry `j` is valid for anchor `i` only if
           `j == i` (the true positive, always kept) or `labels[j] !=
           labels[i]` (a genuinely different-class "other anchor's
           positive"). Same-class, non-diagonal positive-block entries
           are masked to `-inf` so they contribute exactly zero to the
           softmax denominator instead of being wrongly treated as
           negatives.
    4. The correct answer for anchor `i` is always candidate index `i`
       — a `batch_size`-way classification problem, solved with
       standard cross-entropy over the masked similarity matrix.

    Returns a scalar loss tensor, ready for `.backward()`.
    """
    batch_size = anchor.size(0)
    device = anchor.device

    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)

    candidates = torch.cat([positive, negative], dim=0)  # (2B, dim)
    similarity = torch.matmul(anchor, candidates.T) / temperature  # (B, 2B)

    labels = labels.to(device)
    same_label = labels.unsqueeze(1) == labels.unsqueeze(0)  # (B, B): anchor i vs. positive-block row j
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=device)
    # Positive-block entry (i, j) is valid if it's the true positive (j==i)
    # or a genuine opposite-label row; invalid (excluded) only when it's a
    # same-label *false* negative at a non-diagonal position.
    positive_block_valid = diagonal | (~same_label)
    negative_block_valid = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device)
    valid_mask = torch.cat([positive_block_valid, negative_block_valid], dim=1)  # (B, 2B)

    similarity = similarity.masked_fill(~valid_mask, float("-inf"))
    targets = torch.arange(batch_size, device=device)

    return F.cross_entropy(similarity, targets)
