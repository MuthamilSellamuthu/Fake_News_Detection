"""XLM-RoBERTa encoder wrapper — Module 3.

Purpose of this file: wrap HuggingFace's pretrained XLM-RoBERTa in a
small class that turns raw text (English or Tamil, mixed freely) into
one fixed-size contextual embedding vector per text. Nothing in this
file trains or updates any weight — `self.model.eval()` plus
`torch.no_grad()` make that explicit. This is deliberate: Module 3's
job is frozen feature extraction only. Fine-tuning the encoder
(Supervised SimCSE contrastive training) is Module 4's job, and
classification is Module 5's — both consume the vectors this module
produces, they don't happen here.

Why XLM-RoBERTa specifically (short version; full comparison against
mBERT is in the project README / chat explanation, not repeated here to
avoid the two going out of sync): XLM-R is pretrained on ~2.5TB of
CommonCrawl text (CC-100) across 100 languages with a single
SentencePiece vocabulary of 250,002 tokens, and Tamil is one of the
covered languages with a meaningfully-sized share of that corpus. That
gives Tamil sub-word tokenization that doesn't fragment into
near-useless single characters, which is what tends to happen when a
low-resource, non-Latin-script language is squeezed into a Wikipedia-
sized shared vocabulary (mBERT's situation).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PoolingStrategy = Literal["mean", "cls"]


class XLMREncoder:
    """Loads a pretrained XLM-RoBERTa checkpoint once, then repeatedly
    turns batches of text into pooled contextual embeddings.

    Kept as a class (rather than free functions) because the tokenizer
    and model are expensive to load and should only be loaded once per
    process, then reused across every batch/split.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_sequence_length: int = 128,
        pooling_strategy: PoolingStrategy = "mean",
        device: str = "auto",
    ) -> None:
        """Load the pretrained tokenizer and model.

        Parameters
        ----------
        model_name:
            Any XLM-RoBERTa checkpoint on the HuggingFace Hub, e.g.
            "xlm-roberta-base" or "xlm-roberta-large". Passed straight
            through to `AutoTokenizer`/`AutoModel`, so this class isn't
            hardcoded to one checkpoint size.
        max_sequence_length:
            Tokenized sequences longer than this are truncated; shorter
            ones are padded up to it (or to the longest sequence in the
            batch, whichever is smaller — see `tokenize()`).
        pooling_strategy:
            "mean" (attention-mask-weighted average of every token's
            final hidden state) or "cls" (just the leading <s> token's
            hidden state). See `_pool()` for the reasoning.
        device:
            "auto" picks "cuda" if a GPU is available, else "cpu".
            Otherwise passed straight through (e.g. "cpu", "cuda:0").

        What actually happens on this line:
        `AutoTokenizer.from_pretrained(model_name)` downloads (or reads
        from the local HuggingFace cache) XLM-R's SentencePiece model
        file (`sentencepiece.bpe.model`) and wraps it in a fast
        Rust-backed tokenizer. This is the "Use SentencePiece tokenizer"
        requirement — XLM-R does not use WordPiece (BERT/mBERT's
        scheme); it uses a unigram-language-model SentencePiece
        vocabulary trained jointly across all 100 pretraining
        languages, which is what lets one shared vocabulary tokenize
        Tamil script sensibly without a separate Tamil-specific model.
        """
        self.model_name = model_name
        self.max_sequence_length = max_sequence_length
        self.pooling_strategy = pooling_strategy

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        # Disables dropout and any other train-time-only layers. This
        # model is never trained in this module, so it must always run
        # in eval mode — leaving it in train mode would make repeated
        # calls on the *same* text return slightly different embeddings
        # (dropout is stochastic), which would silently corrupt any
        # downstream similarity/clustering step that assumes a
        # deterministic encoder.
        self.model.eval()

    def get_embedding_dim(self) -> int:
        """Return the size of one embedding vector (768 for -base, 1024
        for -large). Reads it from the loaded model's own config rather
        than hardcoding a number, so this class stays correct if
        `model_name` is swapped for a different-sized checkpoint.
        """
        return self.model.config.hidden_size

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Convert raw strings into the integer tensors XLM-RoBERTa expects.

        Internally calls the SentencePiece-backed tokenizer loaded in
        `__init__`. `padding=True` pads every sequence in this batch up
        to the longest one *in the batch* (not always
        `max_sequence_length`, which saves compute on batches of short
        texts). `truncation=True` cuts anything longer than
        `max_sequence_length` from the end. `return_tensors="pt"`
        returns PyTorch tensors directly, ready for the model.

        Returns a dict with (at least) "input_ids" — the SentencePiece
        token IDs — and "attention_mask" — 1 for real tokens, 0 for
        padding, which `_pool()` needs to ignore padding correctly.
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Collapse per-token embeddings (one vector per token per text)
        into one fixed-size vector per text.

        Two strategies, chosen via `self.pooling_strategy`:

        "mean" (default, recommended for a *frozen* encoder): average
        every real token's final hidden state, using `attention_mask`
        so padding tokens (which have no real content) contribute
        nothing to the average. This is the pooling scheme
        Sentence-BERT popularized specifically because a plain
        pretrained transformer's leading token wasn't trained to
        summarize the whole sequence — averaging spreads that
        responsibility across every token instead, which tends to
        produce more usable general-purpose sentence vectors without
        any additional training.

        "cls": take only the first token's ("<s>", RoBERTa/XLM-R's
        equivalent of BERT's [CLS]) final hidden state. Cheaper (no
        averaging) but, for a model that has never seen a
        sentence-level training objective (that only starts in
        Module 4's SimCSE fine-tuning), this token's embedding isn't
        specifically optimized to represent the whole input — it's
        offered mainly for comparison/ablation, not as the default.

        `last_hidden_state` shape: (batch_size, sequence_length,
        hidden_size). `attention_mask` shape: (batch_size,
        sequence_length). Returned shape: (batch_size, hidden_size).
        """
        if self.pooling_strategy == "cls":
            return last_hidden_state[:, 0, :]

        # "mean": expand the mask to match hidden_size so it can be
        # multiplied elementwise against every dimension of every
        # token's vector, zeroing out padding positions before summing.
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        # Count of real (non-padding) tokens per text, clamped so a
        # hypothetical all-padding row can't divide by zero.
        counted = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counted

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Run one batch of texts through tokenizer -> model -> pooling.

        `torch.no_grad()` disables gradient tracking: this is inference
        only, so there's no backward pass to prepare for, and skipping
        it roughly halves memory use and speeds up the forward pass.

        `self.model(**encoded)` runs the actual transformer forward
        pass — every token attends to every other token through 12 (or
        24, for -large) self-attention layers, which is what makes the
        output "contextual": the same word gets a different embedding
        depending on the sentence it appears in, unlike a static
        word-embedding table (word2vec/GloVe).

        Returns a NumPy array of shape (len(texts), hidden_size), moved
        off the GPU (if used) and detached from the autograd graph
        before conversion, since NumPy can't hold GPU tensors or
        gradient metadata.
        """
        encoded = self.tokenize(texts)
        encoded = {key: tensor.to(self.device) for key, tensor in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        pooled = self._pool(outputs.last_hidden_state, encoded["attention_mask"])
        return pooled.cpu().numpy()

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode an arbitrarily long list of texts by repeatedly calling
        `encode_batch()` on slices of size `batch_size` and stacking the
        results.

        This is the method calling code (e.g. `scripts/generate_embeddings.py`)
        should use — it exists so callers never need to think about
        batching themselves, and so the batch size used for compute
        efficiency is decoupled from "one split at a time" processing at
        the call site.

        `show_progress` wraps the batch loop in `tqdm` so a multi-hour
        run (tens of thousands of rows through a transformer on CPU is
        not fast) shows a live rate/ETA instead of appearing to hang.
        """
        from tqdm import tqdm

        all_embeddings: list[np.ndarray] = []
        batch_ranges = range(0, len(texts), batch_size)
        iterator = tqdm(batch_ranges, desc="Encoding", unit="batch") if show_progress else batch_ranges

        for start in iterator:
            batch = texts[start : start + batch_size]
            all_embeddings.append(self.encode_batch(batch))

        if not all_embeddings:
            return np.empty((0, self.get_embedding_dim()), dtype=np.float32)

        return np.concatenate(all_embeddings, axis=0)
