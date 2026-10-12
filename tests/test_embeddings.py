"""Module 3 mechanics validation.

Why this file exists and what it actually proves: this project's
development sandbox has no GPU, only 2 CPU cores / 2.8GB RAM, and —
critically — no network access to the Hugging Face Hub (or any known
mirror), so the real XLM-RoBERTa checkpoint and tokenizer cannot be
downloaded here at all (confirmed: `curl` to huggingface.co, hf-mirror.com,
and raw.githubusercontent.com all return 403 from the sandbox's proxy
allowlist). A plain `pip install torch` was also attempted and abandoned:
PyPI itself is reachable, but the default Linux wheel pulls in a
multi-gigabyte `cuda-toolkit` dependency chain that does not fit this
sandbox's 4.2GB of free local disk, even though this project only needs
CPU inference.

Given that, per an explicit decision by the project owner, real
embedding generation happens via `notebooks/module3_colab.ipynb` (free
GPU, full internet access), not in this sandbox. What CAN and SHOULD be
verified here, without torch or a real checkpoint, is that every piece
of `src/embeddings/*.py` logic that does NOT require torch is correct:
storage format round-tripping, L2 normalization, dynamic padding,
attention-mask construction, truncation counting, Tamil round-trip
mismatch detection, and the CLS/mean-pooling formulas themselves
(reimplemented here in plain NumPy and cross-checked against
hand-computed expected values, since `embedding_utils.extract_cls_embedding`/
`mean_pooling` are written against actual torch tensor semantics and
cannot execute without torch installed).

Every text/record_id/label/language value used below is REAL Module 2
(Version B) data read from `datasets/processed/test.csv`. Every
embedding *vector* is explicitly synthetic (`np.random`) — there is no
real trained encoder to draw from here, and every synthetic value is
labeled as such in the code and print output below.

Run directly (no pytest required):
    python tests/test_embeddings.py

Or with pytest (once installed):
    pytest tests/test_embeddings.py -v
"""

from __future__ import annotations

import sys
import unicodedata as ud
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedding_utils import (  # noqa: E402
    EmbeddingMetadata,
    load_embeddings,
    normalize_embeddings,
    save_embeddings,
)
from src.embeddings.tokenizer_utils import tokenize_batch, verify_tamil_roundtrip  # noqa: E402

TEST_CSV = PROJECT_ROOT / "datasets" / "processed" / "test.csv"


class FakeTokenizer:
    """Deterministic, reversible, pure-Python/NumPy stand-in for XLM-R's
    real SentencePiece tokenizer. NOT a real tokenizer — exists only to
    exercise `tokenize_batch`/`verify_tamil_roundtrip`'s surrounding
    logic (padding, attention masks, truncation counting, mismatch
    detection) without requiring torch or a Hugging Face Hub download.
    Encodes each character as its Unicode codepoint + 10 (the +10 offset
    avoids colliding with this class's special-token ids 0/1/2), which
    makes the mapping exactly reversible for any Unicode input, Tamil
    included — unlike a real SentencePiece vocabulary, which is a fixed
    250,002-entry subword table, not a lossless per-character bijection.
    """

    cls_token, sep_token, pad_token, unk_token, mask_token = "<s>", "</s>", "<pad>", "<unk>", "<mask>"
    vocab_size = 250002
    CLS_ID, SEP_ID, PAD_ID = 1, 2, 0

    def _encode_core(self, text: str) -> list[int]:
        return [ord(c) + 10 for c in text]

    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors=None, return_attention_mask=True):
        seqs = []
        for t in texts:
            ids = [self.CLS_ID] + self._encode_core(t) + [self.SEP_ID]
            if truncation and len(ids) > max_length:
                ids = ids[: max_length - 1] + [self.SEP_ID]
            seqs.append(ids)
        if not padding:
            return {"input_ids": seqs, "attention_mask": [[1] * len(s) for s in seqs]}
        maxlen = max((len(s) for s in seqs), default=0)
        input_ids = np.full((len(seqs), maxlen), self.PAD_ID, dtype="int64")
        attention_mask = np.zeros((len(seqs), maxlen), dtype="int64")
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = s
            attention_mask[i, : len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text, add_special_tokens=False):
        return self._encode_core(text)

    def decode(self, ids, skip_special_tokens=True):
        chars = [chr(i - 10) for i in ids if i not in (self.CLS_ID, self.SEP_ID, self.PAD_ID)]
        return "".join(chars)


class CorruptingFakeTokenizer(FakeTokenizer):
    """Same as FakeTokenizer, but decode() drops Unicode combining marks
    (category Mn) — simulates a tokenizer that corrupts Tamil combining
    vowel signs, to prove `verify_tamil_roundtrip`'s mismatch detection
    actually fires rather than trivially always reporting success.
    """

    def decode(self, ids, skip_special_tokens=True):
        chars = [chr(i - 10) for i in ids if i not in (self.CLS_ID, self.SEP_ID, self.PAD_ID)]
        return "".join(c for c in chars if ud.category(c) != "Mn")


def _load_real_samples() -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(TEST_CSV, low_memory=False)
    en_samples = df.loc[df["language"] == "en", "text"].dropna().astype(str).head(3).tolist()
    ta_samples = df.loc[df["language"] == "ta", "text"].dropna().astype(str).head(5).tolist()
    return df, en_samples, ta_samples


def test_normalize_embeddings():
    rng = np.random.default_rng(42)
    synthetic_vecs = rng.normal(size=(10, 768)).astype("float32")  # SYNTHETIC
    normed = normalize_embeddings(synthetic_vecs)
    norms = np.linalg.norm(normed, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert np.allclose(normed[0], synthetic_vecs[0] / np.linalg.norm(synthetic_vecs[0]), atol=1e-4)

    zero_row = np.zeros((1, 768), dtype="float32")
    assert not np.isnan(normalize_embeddings(zero_row)).any()


def test_save_and_load_embeddings_roundtrip(tmp_path=None):
    import tempfile

    tmp_path = tmp_path or (Path(tempfile.gettempdir()) / "module3_mechanics_test_scratch")
    df = pd.read_csv(TEST_CSV, low_memory=False).head(50)
    real_ids = df["record_id"].to_numpy()  # REAL Module 2 ids, e.g. "welfake_47809"
    real_labels = df["label"].to_numpy()
    real_langs = df["language"].astype(str).to_numpy()

    rng = np.random.default_rng(42)
    synthetic_embeddings = normalize_embeddings(rng.normal(size=(50, 768)).astype("float32"))  # SYNTHETIC

    meta = EmbeddingMetadata(
        model_name="MECHANICS_TEST_NOT_REAL_XLMR", pooling_strategy="mean", embedding_dim=768,
        max_length=512, normalized=True, split="test", n_rows=50,
    )
    save_embeddings(
        synthetic_embeddings, real_ids, real_labels, real_langs,
        output_dir=tmp_path, split="test", pooling_strategy="mean", metadata=meta,
        save_numpy=True, save_torch=False,
    )
    loaded_emb, loaded_ids, loaded_labels, loaded_langs, loaded_meta = load_embeddings(
        tmp_path, split="test", pooling_strategy="mean", framework="numpy"
    )
    assert np.array_equal(loaded_emb, synthetic_embeddings)
    assert np.array_equal(loaded_ids, real_ids)
    assert np.array_equal(loaded_labels, real_labels)
    assert np.array_equal(loaded_langs, real_langs)
    assert loaded_meta["pooling_strategy"] == "mean" and loaded_meta["embedding_dim"] == 768

    import shutil
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_tokenize_batch_dynamic_padding_and_masks():
    _, en_samples, ta_samples = _load_real_samples()
    fake_tok = FakeTokenizer()

    # Short, real, Tamil-only batch: dynamic padding should pad to well
    # under 512, not always to max_length.
    tb_short = tokenize_batch(fake_tok, ta_samples, max_length=512, padding="longest")
    assert tb_short.seq_len < 512

    for i, text in enumerate(ta_samples):
        n_real_tokens = int(tb_short.attention_mask[i].sum())
        assert n_real_tokens == len(text) + 2  # +2 for <s>/</s>

    # Long, real English articles: correctly truncate to exactly max_length.
    tb_long = tokenize_batch(fake_tok, en_samples, max_length=512, padding="longest")
    assert tb_long.seq_len == 512
    assert tb_long.n_truncated == len(en_samples)

    # Explicit truncation-counting check on a controlled input.
    tb_mixed = tokenize_batch(fake_tok, ["x" * 1000, "short"], max_length=512, padding="longest")
    assert tb_mixed.n_truncated == 1
    assert tb_mixed.truncated_mask == [True, False]


def test_verify_tamil_roundtrip_detects_real_corruption():
    _, _, ta_samples = _load_real_samples()

    clean = verify_tamil_roundtrip(FakeTokenizer(), ta_samples)
    assert clean["passed"] and clean["n_mismatch"] == 0

    corrupt = verify_tamil_roundtrip(CorruptingFakeTokenizer(), ta_samples)
    assert not corrupt["passed"] and corrupt["n_mismatch"] == len(ta_samples)


def test_pooling_formulas_against_hand_computed_values():
    """NumPy reimplementation of embedding_utils.extract_cls_embedding /
    mean_pooling (which are written against torch tensor semantics --
    .unsqueeze/.expand/.sum(dim=...) -- and so cannot execute without
    torch installed), cross-checked against values computed by hand.
    """
    last_hidden_state = np.array([
        [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],  # row 0: 2 real tokens + 1 padding
        [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],  # row 1: 3 real tokens, no padding
    ])
    attention_mask = np.array([[1, 1, 0], [1, 1, 1]])

    cls_out = last_hidden_state[:, 0, :]
    assert np.allclose(cls_out, [[1.0, 1.0], [3.0, 3.0]])

    m = attention_mask[:, :, None].astype(last_hidden_state.dtype)
    mean_out = (last_hidden_state * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
    assert np.allclose(mean_out[0], [1.5, 1.5])  # mean of [1,1] and [2,2], padding excluded
    assert np.allclose(mean_out[1], [4.0, 4.0])  # mean of all 3 real tokens


def _run_all():
    tests = [
        test_normalize_embeddings,
        test_save_and_load_embeddings_roundtrip,
        test_tokenize_batch_dynamic_padding_and_masks,
        test_verify_tamil_roundtrip_detects_real_corruption,
        test_pooling_formulas_against_hand_computed_values,
    ]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"[FAIL] {t.__name__}: {exc}")
            failed.append(t.__name__)
    print(f"\n{passed}/{len(tests)} passed.")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
