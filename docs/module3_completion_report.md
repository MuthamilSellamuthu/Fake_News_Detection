# Module 3 — Multilingual Feature Extraction: Completion & Verification Report

Status: **Code complete and mechanics-validated. Real embedding
generation is pending — deferred to the provided Colab notebook by
explicit project-owner decision (see §0).** This is not a partial or
abandoned module; every requested file exists and every piece of logic
that could be executed in this environment was executed and verified
against real Module 2 data. What's pending is specifically the GPU/
network-dependent step of downloading and running the real
XLM-RoBERTa-base checkpoint, which this development sandbox cannot do.

---

## 0. Why generation didn't happen in this sandbox (investigated first, decision made before any code was written)

Before writing any Module 3 code, the sandbox's actual compute was
checked directly:

- **CPU/RAM:** 2 cores, 2.8GB total RAM (2.2GB free).
- **GPU:** none (`nvidia-smi` not found).
- **Network to Hugging Face Hub:** blocked. `curl` to `huggingface.co`,
  `hf-mirror.com`, and `raw.githubusercontent.com` all returned `403
  Forbidden` with `X-Proxy-Error: blocked-by-allowlist` — the real
  XLM-RoBERTa-base checkpoint and tokenizer cannot be downloaded here
  at all, by any route tried.
- **PyPI:** reachable. `pip install torch` was attempted; the default
  Linux wheel resolves a multi-gigabyte `cuda-toolkit` dependency chain
  (cuda-bindings, cufft, cusolver, etc.) that exceeds the sandbox's
  4.2GB of free local disk. Install was abandoned rather than risk
  filling the disk mid-download.

These findings were presented to the project owner before implementing
anything, with three options (full local run given uploaded model
files, code + Colab only, or a small real-subset run given a smaller
model). **"Code + Colab only" was explicitly selected.** Every decision
after this point follows from that.

## 1. Feature dimensions

- Model: `xlm-roberta-base` — 12 layers, hidden size **768**, 12
  attention heads, ~278M parameters, 250,002-token SentencePiece
  vocabulary.
- Two pooled representations per document, each **768-dimensional**:
  CLS (`<s>` token) and attention-mask-weighted mean pooling.
  `embedding_utils.py`'s functions read `hidden_size` from the loaded
  model's own config rather than hardcoding 768, so this stays correct
  if `configs/feature_config.yaml`'s `model.name` is ever swapped to
  `xlm-roberta-large` (1024-dim).
- Saved as float32; L2-normalized to unit length (required by Module
  4's contrastive loss, which operates on the unit hypersphere).

## 2. Number of embeddings generated

**0 real embeddings generated in this session** — by design, per §0.
The planned full run (once the Colab notebook executes) will produce
**138,016 embedding vectors**: 69,008 documents × 2 pooling strategies
(CLS + mean), split as:

| Split | Rows | English | Tamil |
|---|---:|---:|---:|
| Train | 48,305 | 44,520 | 3,785 |
| Validation | 10,351 | 9,540 | 811 |
| Test | 10,352 | 9,540 | 812 |
| **Total** | **69,008** | **63,600** | **5,408** |

These row counts are Module 2's finalized (Version B) split counts,
read directly from `datasets/processed/{train,validation,test}.csv` —
not estimated.

What **was** generated and verified for real in this session: 5/5
mechanics tests in `tests/test_embeddings.py`, using real Module 2 text/
`record_id`/`label`/`language` data paired with explicitly-synthetic
embedding vectors (no real encoder was available to draw from). See §6
of `outputs/module3/feature_report.md` for the full test list and
results, including one real bug this process caught and fixed
(`record_id` was assumed int64; Module 2's real IDs are strings like
`"welfake_47809"`).

## 3. GPU memory usage estimate

- Model weights: ~1.0GB (fp32) / ~530MB (fp16 activations under
  `torch.autocast`; weights themselves remain fp32, only activations
  are cast).
- Activation memory at `batch_size=32`, `max_length=512`: ~2-4GB,
  typical for a 12-layer, 768-hidden encoder-only transformer at this
  batch/sequence size.
- **Estimated total: ~4-6GB.** Comfortably fits a free-tier Colab T4
  (16GB) with substantial headroom — `batch_size` could be increased
  from the config default if desired.
- Basis: published throughput/memory benchmarks for BERT/RoBERTa-base-
  sized transformers at this batch size and sequence length. Not
  measured directly — no GPU was available in this sandbox to measure
  it on; the real figure will be visible in Colab's own memory monitor
  when the notebook runs.

## 4. Runtime estimate

| Environment | Estimate |
|---|---|
| Colab T4 GPU, fp16, batch=32 | **~8-19 minutes** for all 69,008 documents |
| This sandbox's CPU (2 cores, no GPU) | **~3.8-9.6 hours** — why generation was deferred |

Both pooling strategies (CLS + mean) are computed from the same forward
pass, so producing both costs nothing extra over producing one. Basis:
60-150 docs/sec (GPU) / 2-5 docs/sec (CPU), extrapolated from typical
published throughput for base-sized transformer inference at this
sequence length; actual GPU throughput should trend toward the higher
end given how much shorter Tamil documents are than 512 tokens on
average (dynamic padding reduces wasted compute on those batches).

## 5. Storage size estimate

- Per row: 2 pooling strategies × 2 formats (NumPy + PyTorch) × 768
  floats × 4 bytes = 12,288 bytes.
- Across all 69,008 rows: **~808.7MB** for embedding arrays, **~818.6MB**
  including `record_id`/`label`/`language` metadata arrays.
- This is an estimate from the known row count and dimension, not a
  measurement — will be confirmed against the actual `embeddings/`
  directory size once the Colab run completes.

## 6. Verification that train/validation/test embeddings match their datasets

**Not yet performed against real files — no real embeddings exist on
disk yet.** The exact procedure that will run this check is already
written into `notebooks/module3_colab.ipynb` §8, and will verify, for
each split:

1. `len(embeddings/<split>/record_ids.npy) == len(datasets/processed/<split>.csv)`
2. `set(embeddings/<split>/record_ids.npy) == set(datasets/processed/<split>.csv["record_id"])` — exact match, zero drift
3. Every saved embedding vector's L2 norm ≈ 1.0 (confirms normalization was applied)
4. The real-tokenizer Tamil round-trip check (`tamil_unicode_roundtrip_check`
   in `feature_statistics.json`) reports 0 mismatches

What **was** verified for real in this session: the storage/loading
round-trip itself (`save_embeddings`/`load_embeddings`) against 50 real
Module 2 `record_id`/`label`/`language` rows with synthetic embedding
vectors — proving the file format and joinability logic is correct,
independent of whether the vectors themselves come from a real encoder.

## 7. List of generated files

**Config:** `configs/feature_config.yaml`

**Code (`src/embeddings/`):** `tokenizer_utils.py`, `embedding_utils.py`,
`feature_extraction.py` (new Module 3 files; the pre-existing
`xlmr_encoder.py`/`embedding_store.py` in this directory predate the
current module-by-module rebuild and were not modified or relied upon)

**CLI entry point:** `scripts/extract_features.py`

**Tests:** `tests/test_embeddings.py` (5/5 passing, real Module 2 data + explicitly-synthetic vectors)

**Notebook:** `notebooks/module3_colab.ipynb` (18 cells — mounts Drive,
installs deps, checks GPU, confirms Module 2 output, verifies Tamil
round-trip against the real tokenizer, runs extraction, verifies
outputs against source CSVs)

**Reports:** `outputs/module3/feature_statistics.json`,
`outputs/module3/feature_report.md`, `outputs/module3/feature_extraction.log`,
`docs/module3_completion_report.md` (this file)

**Placeholder output directory:** `embeddings/` (currently just a
`README.md` describing the layout that will be populated by the Colab run)

**Not touched:** everything from Module 1 and Module 2 (`datasets/raw/`,
`datasets/processed/module1_dataset.*`, `datasets/processed/cleaned_dataset.*`,
`datasets/processed/{train,validation,test}.csv`, `src/data_collection/`,
`src/preprocessing/`, `configs/module1_*.yaml`, `configs/module2_*.yaml`)
— Module 3 only reads Module 2's finalized splits.

## 8. Recommendations before Module 4 (Supervised Contrastive Learning)

1. **Run `notebooks/module3_colab.ipynb` before starting Module 4.**
   Module 4 needs real embeddings to train against; none exist yet.
   Set the Colab runtime to GPU (T4 or better) first.
2. **Review the real `outputs/module3/feature_statistics.json`
   `tamil_unicode_roundtrip_check` and `splits.*.n_truncated` fields
   after the Colab run**, before treating Module 3 as finalized — these
   are currently estimates/pending, not measured values, and the
   pipeline is explicitly designed to surface (not hide) truncation and
   Tamil-tokenization issues rather than assume they didn't happen.
3. **Decide on the truncation tradeoff for Module 4 explicitly.**
   `max_length=512` will truncate a meaningful share of long English
   articles (mean ~554 words, some up to 24,235); Tamil is essentially
   unaffected. If Module 4's contrastive pairs need full-document
   signal rather than lead-paragraph signal, consider whether 512 is
   the right ceiling before training — this is a modeling decision, not
   something Module 3 should decide unilaterally.
4. **Keep both CLS and mean-pooled embeddings** rather than picking one
   now. Mean pooling is generally the stronger default for a *frozen*
   encoder's sentence-level similarity (this is well established from
   Sentence-BERT-style literature), but Module 4 fine-tunes the encoder
   itself with a contrastive objective — it's worth empirically
   comparing both as the starting representation rather than assuming.
5. **Sync the Drive-side `embeddings/` output back into this repo/folder**
   after the Colab run, the same way Module 1/2's outputs were, so the
   embeddings are versioned alongside the code that produced them.

---

**Module 3 code is finalized. Real embedding generation awaits the
Colab run. Module 4 will not begin until the Colab-generated embeddings
exist, the verification checklist in §6 passes for real, and you give
explicit approval.**
