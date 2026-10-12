"""Module 3 — tokenizer loading and multilingual/Tamil-safe tokenization.

Everything in this file is about turning raw strings (English or Tamil,
never mixed within a single Module 2 row, but mixed freely across rows
in the same batch) into the integer tensors XLM-RoBERTa expects, and
about verifying that this process is actually Tamil-safe rather than
assuming it.

Kept separate from `embedding_utils.py` (which only deals with model
*outputs* — pooling, normalization, storage) and `feature_extraction.py`
(which orchestrates the end-to-end run) so each file has one job.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

logger_name = "features.tokenizer_utils"


def load_tokenizer(model_name: str, local_path: str | None = None) -> Any:
    """Load XLM-RoBERTa's SentencePiece-backed fast tokenizer.

    `local_path` (if given) is loaded instead of `model_name`, for
    environments where Hugging Face Hub access is unavailable (e.g. this
    project's development sandbox — see docs/module3_completion_report.md
    §0 for why) but the model files have been downloaded elsewhere and
    copied into the project. Import is deferred to inside the function so
    that importing this module does not itself require `transformers` to
    be installed (useful for unit-testing the pure-Python helpers below
    with a substitute tokenizer, and for keeping `feature_extraction.py`'s
    error message clean when the dependency truly is missing).

    Raises a clear, actionable `RuntimeError` (not a raw exception trace)
    if the download fails — the most common failure mode in a
    network-restricted environment is a proxy/allowlist block, and the
    default `transformers` error for that is a generic connection
    exception that doesn't say what to do about it.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The 'transformers' package is not installed. Install it with "
            "`pip install transformers sentencepiece` (or run this pipeline "
            "in the provided Colab notebook, which installs it automatically)."
        ) from exc

    source = local_path if local_path else model_name
    try:
        return AutoTokenizer.from_pretrained(source)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised with context
        raise RuntimeError(
            f"Could not load tokenizer '{source}'. If this is a network error, "
            "this environment likely cannot reach the Hugging Face Hub — run "
            "this pipeline in the provided Colab notebook (notebooks/module3_colab.ipynb) "
            "instead, or set model.local_path in configs/feature_config.yaml to a "
            "local directory containing the tokenizer files."
        ) from exc


@dataclass
class TokenizedBatch:
    """Container for one batch's tokenizer output, kept framework-light
    (plain nested lists/ints) so this dataclass has no hard dependency
    on torch — `feature_extraction.py` converts to tensors at the point
    it actually needs them.
    """

    input_ids: Any  # torch.Tensor, shape (batch, seq_len)
    attention_mask: Any  # torch.Tensor, shape (batch, seq_len)
    seq_len: int
    n_truncated: int = 0
    truncated_mask: list[bool] = field(default_factory=list)
    # Per-document token count BEFORE truncation/padding — the raw
    # length the tokenizer would have produced given unlimited
    # max_length. Carried through (rather than just a truncated/not
    # boolean) so the extraction run can report real maximum and
    # average sequence lengths without a second tokenization pass over
    # the whole corpus.
    unpadded_lengths: list[int] = field(default_factory=list)


def tokenize_batch(
    tokenizer: Any,
    texts: list[str],
    max_length: int = 512,
    padding: str = "longest",
) -> TokenizedBatch:
    """Tokenize one batch with dynamic padding, truncation, and attention
    masks.

    `padding="longest"` (the config default) pads every sequence in
    *this batch* up to the longest sequence *in this batch*, not always
    to `max_length` — this is "dynamic padding": a batch of five Tamil
    headlines (a few tokens each) doesn't get padded out to 512 just
    because some other batch elsewhere in the dataset needed that many.
    This meaningfully reduces wasted compute given how much shorter
    Tamil documents are than English ones in this dataset (mean ~16
    words vs. ~554 — see docs/module3_completion_report.md §6).

    `truncation=True` with `max_length` caps any sequence longer than
    the limit — required because XLM-R's absolute position embeddings
    only go up to 512 positions; a longer input would error, not just
    lose information silently.

    Returns a `TokenizedBatch` also reporting how many of this batch's
    texts were long enough to be truncated, so the caller can aggregate
    this into `feature_statistics.json` (truncation is a real,
    disclosed information-loss tradeoff, not something to bury).
    """
    encoded = tokenizer(
        texts,
        padding=padding,
        truncation=truncation_flag(max_length),
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Determine which original texts would have exceeded max_length
    # before truncation/padding, by re-tokenizing without truncation and
    # comparing lengths. Only done for reporting, not for the tensors
    # actually fed to the model (those come from the padded/truncated
    # `encoded` above).
    n_truncated = 0
    truncated_mask: list[bool] = []
    lengths: list[int] = []
    unpadded = tokenizer(texts, truncation=False, padding=False)["input_ids"]
    for ids in unpadded:
        length = len(ids)
        lengths.append(length)
        is_truncated = length > max_length
        truncated_mask.append(is_truncated)
        if is_truncated:
            n_truncated += 1

    return TokenizedBatch(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        seq_len=encoded["input_ids"].shape[1],
        n_truncated=n_truncated,
        truncated_mask=truncated_mask,
        unpadded_lengths=lengths,
    )


def truncation_flag(max_length: int) -> bool:
    """Trivial helper so `tokenize_batch`'s call reads clearly; always
    True in this pipeline (Module 3 always enforces max_length), kept as
    a function rather than a bare `True` literal only so a future
    "no truncation, error instead" mode has one obvious place to add it.
    """
    return True


#: Tamil Unicode block (U+0B80–U+0BFF). Covers Tamil letters, the
#: combining vowel signs (matras), virama/pulli, and Tamil numerals —
#: i.e. every codepoint whose loss or alteration would constitute real
#: Tamil corruption.
_TAMIL_BLOCK_START, _TAMIL_BLOCK_END = "஀", "௿"


def tamil_codepoints(text: str) -> str:
    """Extract only the Tamil-script codepoints from `text`, in order.

    Combining marks in the Tamil block (vowel signs, virama) are
    included because they carry meaning — dropping a virama changes the
    word. Everything else (Latin letters, digits, punctuation,
    whitespace) is excluded, because those are exactly the characters a
    tokenizer is *permitted* to normalize.
    """
    return "".join(ch for ch in text if _TAMIL_BLOCK_START <= ch <= _TAMIL_BLOCK_END)


def verify_tamil_roundtrip(tokenizer: Any, sample_texts: list[str]) -> dict[str, Any]:
    """Verify that tokenization preserves Tamil script exactly.

    Why this is not a naive string equality check
    ---------------------------------------------
    XLM-RoBERTa's SentencePiece model applies its own text normalizer
    (an NMT/NFKC-style pass) *before* segmentation. That normalizer
    legitimately rewrites certain non-Tamil characters — most commonly
    folding curly quotation marks (U+201C/U+201D “ ”) to straight ASCII
    quotes, and normalizing assorted dashes and spaces. This is
    documented, intentional tokenizer behavior applied identically to
    every language, and it destroys no Tamil information.

    An earlier version of this function compared
    `NFC(decode(encode(x))) == NFC(x)` and therefore reported those
    benign rewrites as Tamil "corruption" — a false positive that
    failed verification on a run whose Tamil data was completely
    intact.

    The check now separates two genuinely different questions:

      1. **Was any Tamil character lost or altered?** Compared by
         extracting the Tamil-block codepoint subsequence (letters,
         vowel signs, virama) from input and output and requiring exact
         equality. This is the integrity property that actually
         matters, and it is *stricter* than before for Tamil: a dropped
         combining vowel sign still fails, and cannot be masked by
         surrounding punctuation changes.
      2. **Did the tokenizer normalize anything else?** Counted and
         reported as `n_benign_normalization`, never as a failure.

    Additionally, a U+FFFD replacement character appearing in the
    decoded output that was not in the input is always a hard failure —
    that is unambiguous data loss regardless of script.

    Returns a dict summary (never raises), so `feature_extraction.py`
    can record it directly into `feature_statistics.json`.
    """
    n_checked = 0
    n_tamil_corruption = 0
    n_benign_normalization = 0
    n_replacement_char = 0
    corruption_samples: list[dict[str, str]] = []
    benign_samples: list[dict[str, str]] = []

    for text in sample_texts:
        if not text:
            continue
        n_checked += 1
        # NFC-normalize the input the same way Module 2 already did, so
        # this compares against the canonical form the tokenizer
        # actually receives in production.
        original = unicodedata.normalize("NFC", text)

        encoded_ids = tokenizer.encode(original, add_special_tokens=False)
        decoded = unicodedata.normalize("NFC", tokenizer.decode(encoded_ids, skip_special_tokens=True))

        original_tamil = tamil_codepoints(original)
        decoded_tamil = tamil_codepoints(decoded)

        introduced_replacement = "�" in decoded and "�" not in original

        if original_tamil != decoded_tamil or introduced_replacement:
            n_tamil_corruption += 1
            if introduced_replacement:
                n_replacement_char += 1
            if len(corruption_samples) < 10:
                corruption_samples.append({
                    "original": original[:120],
                    "decoded": decoded[:120],
                    "original_tamil_len": str(len(original_tamil)),
                    "decoded_tamil_len": str(len(decoded_tamil)),
                    "introduced_replacement_char": str(introduced_replacement),
                })
        elif decoded.strip() != original.strip():
            # Tamil survived intact; something non-Tamil was normalized.
            n_benign_normalization += 1
            if len(benign_samples) < 5:
                benign_samples.append({"original": original[:120], "decoded": decoded[:120]})

    return {
        "n_checked": n_checked,
        "n_tamil_corruption": n_tamil_corruption,
        "n_replacement_char_introduced": n_replacement_char,
        "n_benign_normalization": n_benign_normalization,
        "passed": n_tamil_corruption == 0,
        "corruption_samples": corruption_samples,
        "benign_normalization_samples": benign_samples,
        "note": (
            "PASS requires zero Tamil-script corruption. Non-Tamil differences "
            "(e.g. SentencePiece folding curly quotes to straight quotes) are "
            "counted as benign normalization and do not fail the check."
        ),
        # Back-compat: older report writers read `n_mismatch`.
        "n_mismatch": n_tamil_corruption,
    }


def get_special_tokens_info(tokenizer: Any) -> dict[str, Any]:
    """Small diagnostic dict recorded in feature_statistics.json so the
    exact special-token scheme used for a given run is traceable
    (matters if `model.name` is ever swapped to a different checkpoint
    with different special tokens).
    """
    return {
        "cls_token": tokenizer.cls_token,
        "sep_token": tokenizer.sep_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
        "mask_token": tokenizer.mask_token,
        "vocab_size": tokenizer.vocab_size,
    }
