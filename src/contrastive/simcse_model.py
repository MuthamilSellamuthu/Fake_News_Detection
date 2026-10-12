"""Trainable XLM-RoBERTa wrapper for supervised SimCSE fine-tuning — Module 4.

This is deliberately a separate, small class from
`src/embeddings/xlmr_encoder.py`'s `XLMREncoder` (Module 3) rather than
a shared/reused one: Module 3's contract is "always frozen, always
eval mode, never accumulates gradients" — reusing that class here and
flipping it into train mode for fine-tuning would risk that contract
silently breaking if the two modules are ever imported into the same
process. Keeping them separate means each file's behavior is exactly
what its own code says, nothing more.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ContrastiveXLMREncoder(nn.Module):
    """XLM-RoBERTa wrapped as a standard `torch.nn.Module` so it can be
    fine-tuned with an ordinary PyTorch optimizer/training loop.

    Structurally almost identical to Module 3's mean-pooling (same
    formula, same reasoning — see that file's `_pool()` docstring for
    why mean pooling is used), but here the backbone is left in
    (default) train mode and its parameters are meant to accumulate
    gradients, since the entire point of Module 4 is to update them.
    """

    def __init__(self, model_name: str = "xlm-roberta-base", max_sequence_length: int = 128) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_sequence_length = max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

    def get_embedding_dim(self) -> int:
        """Hidden size of the backbone's output vectors (read from its
        own config, so this stays correct for -base or -large)."""
        return self.backbone.config.hidden_size

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """SentencePiece-tokenize a list of strings (same mechanics as
        Module 3's `XLMREncoder.tokenize()` — see that docstring)."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )

    @staticmethod
    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Attention-mask-weighted average of every real (non-padding)
        token's final hidden state. Identical math to Module 3's
        `_pool(pooling_strategy="mean")`; kept as a `staticmethod` here
        (rather than importing Module 3's private method) so this file
        has no dependency on Module 3's internals.
        """
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counted = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counted

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Standard `nn.Module` forward: tokenized tensors in, one
        pooled embedding per row out. Gradients flow through every step
        of this (unlike Module 3, there is no `torch.no_grad()` here) —
        that's what lets the optimizer update `self.backbone`'s weights
        based on the contrastive loss computed from this output.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pool(outputs.last_hidden_state, attention_mask)

    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Convenience wrapper: tokenize a batch of raw strings and run
        `forward()` in one call, moving the tokenized tensors to
        `device` first. Used by the trainer for the validation-loss
        pass; the main training loop tokenizes anchor+positive+negative
        together instead (see `src/contrastive/trainer.py`) for a
        single, more efficient forward pass per step.
        """
        encoded = self.tokenize(texts)
        encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
        return self.forward(encoded["input_ids"], encoded["attention_mask"])

    def load_backbone(self, checkpoint_dir: str | Path, device: torch.device) -> None:
        """Replace `self.backbone`'s weights with a previously-saved
        checkpoint (as written by `save()`), moved onto `device`. Used
        by the trainer to reload the best-val-loss checkpoint after the
        epoch loop ends, so the embeddings generated for "save learned
        embeddings" come from the best model seen during training, not
        whatever state the last epoch happened to leave `self.backbone` in.
        """
        self.backbone = AutoModel.from_pretrained(checkpoint_dir)
        self.backbone.to(device)

    def save(self, output_dir: str | Path) -> None:
        """Persist the fine-tuned backbone + tokenizer in standard
        HuggingFace format (`config.json`, `model.safetensors`,
        tokenizer files), so it can be reloaded later exactly like any
        other pretrained checkpoint via
        `AutoModel.from_pretrained(output_dir)` — including by Module 3's
        `XLMREncoder` if you want frozen embeddings from the *fine-tuned*
        Contrast-XLMR instead of the original pretrained checkpoint, or
        by Module 5's classifier training.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
