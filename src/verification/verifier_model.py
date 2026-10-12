"""Module 6 — XLM-R verifier with a replaceable classification head.

Architecture: shared XLM-R encoder + a small MLP head.

The head is deliberately separable from the encoder. FEVER pretrains the
encoder on 3-way entailment; X-FACT then REPLACES the head with a freshly
initialised 3-way veracity head and fine-tunes. Both spaces happen to have
three classes, which makes it tempting to keep the FEVER head and "just fine-
tune" -- that would smuggle FEVER's entailment semantics into X-FACT's
veracity labels and make the transfer result uninterpretable. `swap_head()`
therefore always re-initialises, and records that it did so.

Keeping the encoder shared is the whole point: it is what makes
FEVER -> Tamil transfer measurable, and connects to Module 4's finding that
naive English-to-Tamil transfer is weak (zero-shot Tamil Macro-F1 0.3216).

torch is imported lazily so this module can be imported, inspected and
partially tested on a machine with no torch and no downloaded weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VerifierConfig:
    encoder_name: str = "xlm-roberta-base"
    num_labels: int = 3
    hidden_dim: int = 512
    dropout: float = 0.1
    max_length: int = 256
    label_space: str = "fever"

    def to_dict(self) -> dict[str, Any]:
        return {"encoder_name": self.encoder_name, "num_labels": self.num_labels,
                "hidden_dim": self.hidden_dim, "dropout": self.dropout,
                "max_length": self.max_length, "label_space": self.label_space}


def build_verifier(cfg: VerifierConfig):
    """Construct the model. Imports torch/transformers only when called."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class Verifier(nn.Module):
        def __init__(self, config: VerifierConfig) -> None:
            super().__init__()
            self.config = config
            self.encoder = AutoModel.from_pretrained(config.encoder_name)
            hidden = self.encoder.config.hidden_size
            self.head = self._make_head(hidden, config)
            self.label_space = config.label_space

        @staticmethod
        def _make_head(hidden: int, config: VerifierConfig) -> "nn.Module":
            return nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(hidden, config.hidden_dim),
                nn.Tanh(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.num_labels),
            )

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Mask-weighted mean pooling. Module 4's ablation selected mean over
            # CLS for this encoder, so the same choice is kept for consistency.
            mask = attention_mask.unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            pooled = summed / mask.sum(dim=1).clamp(min=1e-9)
            return self.head(pooled)

        def swap_head(self, num_labels: int, label_space: str,
                      hidden_dim: int | None = None) -> dict[str, Any]:
            """Replace the head for a new label space. ALWAYS re-initialises.

            Returns a record of the swap so the report can state plainly that
            no FEVER head weights survived into the X-FACT model.
            """
            old = {"label_space": self.label_space, "num_labels": self.config.num_labels}
            self.config.num_labels = int(num_labels)
            self.config.label_space = label_space
            if hidden_dim is not None:
                self.config.hidden_dim = int(hidden_dim)
            hidden = self.encoder.config.hidden_size
            self.head = self._make_head(hidden, self.config)
            return {"head_reinitialised": True, "encoder_reused": True,
                    "from": old, "to": {"label_space": label_space, "num_labels": num_labels},
                    "note": ("Encoder weights are carried over (that is the transfer being "
                             "measured); head weights are discarded so FEVER entailment "
                             "semantics cannot leak into X-FACT veracity labels.")}

    return Verifier(cfg)


def count_parameters(model) -> dict[str, int]:
    enc = sum(p.numel() for p in model.encoder.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    return {"encoder": enc, "head": head, "total": enc + head}
