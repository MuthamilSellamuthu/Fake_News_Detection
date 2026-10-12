"""PyTorch Dataset + collate function for supervised SimCSE training — Module 4.

Kept intentionally simple: the dataset itself just returns raw string
triples (tokenization happens in the collate function, batched, so
padding is computed once per batch rather than once per row).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SupervisedSimCSEDataset(Dataset):
    """Wraps one epoch's worth of (anchor, positive, negative) triplet
    indices — produced by `src/contrastive/pair_sampler.build_supervised_triplets`
    — and exposes them as raw text triples for a `DataLoader`.

    Re-instantiate this once per epoch with fresh `positive_idx`/
    `negative_idx` (a new call to `build_supervised_triplets` with that
    epoch's number as the seed) so pairing is re-randomized every
    epoch rather than frozen for the whole training run.
    """

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        positive_idx: np.ndarray,
        negative_idx: np.ndarray,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.positive_idx = positive_idx
        self.negative_idx = negative_idx

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, str, str, int]:
        """Return `(anchor_text, positive_text, negative_text, anchor_label)`.
        The label travels alongside the triple because
        `supervised_simcse_loss` needs each anchor's label to mask out
        same-class false negatives (see `src/contrastive/losses.py`).
        """
        return (
            self.texts[idx],
            self.texts[self.positive_idx[idx]],
            self.texts[self.negative_idx[idx]],
            int(self.labels[idx]),
        )


def make_collate_fn(encoder, max_length: int):
    """Build a `collate_fn` closed over a specific encoder/tokenizer.

    Rather than tokenizing anchor, positive, and negative texts in
    three separate calls (three separate padding decisions, three
    separate forward passes), this concatenates all `3*batch_size`
    strings into a single tokenizer call — one consistent padding
    length for the whole batch — and lets the trainer run one forward
    pass over the concatenated tensor, then split the pooled output
    back into three `batch_size`-sized chunks. This is the standard
    efficient implementation pattern for SimCSE-style training.
    """

    def collate_fn(batch: list[tuple[str, str, str, int]]) -> dict[str, torch.Tensor]:
        anchors, positives, negatives, labels = zip(*batch)
        combined_texts = list(anchors) + list(positives) + list(negatives)

        encoded = encoder.tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        encoded["batch_size"] = len(anchors)
        return encoded

    return collate_fn
