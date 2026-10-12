"""Module 3 orchestrator — multilingual feature extraction with XLM-RoBERTa.

Reads Module 2's finalized (Version B) `train.csv` / `validation.csv` /
`test.csv` (never modifies them, never re-runs any Module 2 logic) and
writes:

  - `embeddings/<split>/{cls,mean}_embeddings.{npy,pt}` — pooled,
    L2-normalized embeddings, plus `record_ids`/`labels`/`languages`
    arrays alongside them so every row is traceable back to its
    Module 2 CSV row.
  - `outputs/module3/feature_statistics.json` — the full execution
    record (see below).
  - `outputs/module3/feature_report.md` — human-readable summary.

Every metric a publication methods section needs is captured *during*
the run rather than estimated afterwards, because several of them
(peak GPU memory, per-batch timings, checkpoint-recovery status) are
only observable while execution is happening:

   environment: GPU model, CUDA version, PyTorch version, total GPU
       memory, peak GPU memory, mixed-precision status
   throughput: batch size, per-batch runtime distribution, total
       runtime, documents/second
   checkpointing: whether checkpoints were written, and whether this
       run resumed from one
   tokenization: max sequence length seen (pre-truncation), average
       sequence length (pre-truncation), truncated document count and
       percentage
   embeddings: CLS dimension, mean-pooling dimension, count generated,
       on-disk storage size, success/failure counts

Design note on why the model/tokenizer are constructor-injectable
(`FeatureExtractor(config, tokenizer=..., model=...)`): production use
(and the Colab notebook) always loads the real XLM-RoBERTa checkpoint by
name. But this project's development sandbox has no GPU, only 2 CPU
cores, and no network access to the Hugging Face Hub, so the real
checkpoint cannot be downloaded there (see
docs/module3_completion_report.md §0). Making the tokenizer/model
injectable lets the *exact* production code path be exercised
end-to-end against a small local substitute, proving the pipeline
mechanics are correct independent of checkpoint availability. It is not
a shortcut around the real implementation — `run_feature_extraction()`
always loads the real model when no override is given.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.embeddings.embedding_utils import (
    EmbeddingMetadata,
    extract_cls_embedding,
    mean_pooling,
    normalize_embeddings,
    save_embeddings,
)
from src.embeddings.runtime_stats import (
    BatchTimer,
    collect_device_info,
    human_bytes,
    human_duration,
    peak_gpu_memory,
    reset_peak_memory,
)
from src.embeddings.tokenizer_utils import (
    get_special_tokens_info,
    load_tokenizer,
    tokenize_batch,
    verify_tamil_roundtrip,
)
from src.utils.config_loader import load_yaml, project_root
from src.utils.logger import get_logger

logger = get_logger("features.extraction")

PoolingStrategy = Literal["cls", "mean"]


def _select_device(requested: str) -> Any:
    """Resolve "auto" to "cuda" if available, else "cpu"."""
    import torch

    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(model_name: str, local_path: str | None, device: Any) -> Any:
    """Load the pretrained XLM-RoBERTa encoder in frozen eval mode.

    Frozen (`model.eval()` + `torch.no_grad()` at every call site)
    because Module 3's job is feature extraction only — no weight in
    this encoder is updated here. Fine-tuning it with a supervised
    contrastive objective is Module 4's job.
    """
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "The 'transformers' package is not installed. Install it with "
            "`pip install transformers sentencepiece torch` (or run this "
            "pipeline in the provided Colab notebook)."
        ) from exc

    source = local_path if local_path else model_name
    try:
        model = AutoModel.from_pretrained(source)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not load model '{source}'. If this is a network error, this "
            "environment likely cannot reach the Hugging Face Hub — run this "
            "pipeline in the provided Colab notebook (notebooks/module3_colab.ipynb) "
            "instead, or set model.local_path in configs/feature_config.yaml to a "
            "local directory containing the model files."
        ) from exc

    model.to(device)
    model.eval()
    return model


@dataclass
class BatchError:
    batch_index: int
    record_ids: list[Any]
    error: str

    def to_dict(self) -> dict[str, Any]:
        return {"batch_index": self.batch_index, "record_ids": self.record_ids, "error": self.error}


@dataclass
class SequenceLengthStats:
    """Token-length distribution measured BEFORE truncation."""

    lengths: list[int] = field(default_factory=list)

    def add(self, values: list[int]) -> None:
        self.lengths.extend(values)

    def summary(self, max_length: int) -> dict[str, Any]:
        if not self.lengths:
            return {"n": 0}
        arr = np.asarray(self.lengths, dtype="int64")
        n_truncated = int((arr > max_length).sum())
        return {
            "n": int(arr.size),
            "max_token_length": int(arr.max()),
            "min_token_length": int(arr.min()),
            "avg_token_length": round(float(arr.mean()), 2),
            "median_token_length": float(np.median(arr)),
            "p95_token_length": float(np.percentile(arr, 95)),
            "p99_token_length": float(np.percentile(arr, 99)),
            "max_sequence_length_config": max_length,
            "n_truncated": n_truncated,
            "pct_truncated": round(100.0 * n_truncated / arr.size, 3),
        }


@dataclass
class SplitExtractionResult:
    embeddings: dict[PoolingStrategy, np.ndarray]
    record_ids: np.ndarray
    labels: np.ndarray
    languages: np.ndarray
    n_rows: int
    embedding_dim: int
    seq_len_stats: SequenceLengthStats
    timer: BatchTimer
    n_success: int
    n_failure: int
    checkpoint_created: bool
    checkpoint_resumed: bool
    resumed_from_batch: int
    n_checkpoints_written: int
    errors: list[BatchError] = field(default_factory=list)


class FeatureExtractor:
    """Loads the encoder once, then repeatedly turns batches of Module 2
    text into pooled, normalized embeddings for an entire split, while
    recording full execution telemetry.
    """

    def __init__(
        self,
        config: dict[str, Any],
        tokenizer: Any = None,
        model: Any = None,
        device: Any = None,
    ) -> None:
        self.config = config
        model_cfg = config["model"]
        tok_cfg = config["tokenization"]
        self.max_length: int = tok_cfg["max_length"]
        self.padding: str = tok_cfg["padding"]
        self.pooling_strategies: list[PoolingStrategy] = config["extraction"]["pooling_strategies"]
        self.batch_size: int = config["extraction"]["batch_size"]
        self.normalize: bool = config["extraction"]["normalize_embeddings"]
        self.checkpoint_every: int = config["extraction"]["checkpoint_every_n_batches"]
        self.progress_every: int = config["logging"]["progress_every_n_batches"]
        self.requested_amp: bool = model_cfg["use_mixed_precision"]

        self.device = device if device is not None else _select_device(model_cfg["device"])
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(model_cfg["name"], model_cfg["local_path"])
        self.model = model if model is not None else _load_model(model_cfg["name"], model_cfg["local_path"], self.device)

        # Mixed precision is only meaningful on CUDA. Recording both the
        # requested setting and whether it was actually active prevents a
        # report from claiming "mixed precision: true" on a CPU run where
        # it silently did nothing.
        self.amp_active: bool = bool(self.requested_amp and getattr(self.device, "type", None) == "cuda")

        self.embedding_dim = self._infer_embedding_dim()
        self.device_info = collect_device_info(self.device)

        logger.info(
            "FeatureExtractor ready: device=%s, embedding_dim=%d, pooling=%s, batch_size=%d, "
            "max_length=%d, mixed_precision_requested=%s, mixed_precision_active=%s",
            self.device, self.embedding_dim, self.pooling_strategies, self.batch_size,
            self.max_length, self.requested_amp, self.amp_active,
        )
        if self.device_info.get("gpu_available"):
            logger.info(
                "GPU: %s | CUDA %s | torch %s | total memory %s",
                self.device_info["gpu_name"], self.device_info["cuda_version"],
                self.device_info["torch_version"], self.device_info["gpu_total_memory_human"],
            )
        else:
            logger.warning(
                "No GPU detected — running on %s. This will be substantially slower; "
                "mixed precision is inactive.", self.device,
            )

    def _infer_embedding_dim(self) -> int:
        """Read hidden_size from the loaded model's own config rather
        than hardcoding 768, so this stays correct if `model.name` is
        swapped for -large (1024) or another checkpoint.
        """
        if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
            return int(self.model.config.hidden_size)
        if hasattr(self.model, "hidden_size"):
            return int(self.model.hidden_size)
        raise RuntimeError("Could not infer embedding dimension from the loaded model.")

    def _forward_batch(self, texts: list[str]) -> tuple[dict[PoolingStrategy, Any], list[int]]:
        """Tokenize + forward pass + pool one batch. Returns pooled
        tensors per requested strategy and the per-document
        pre-truncation token lengths.
        """
        import torch

        tokenized = tokenize_batch(self.tokenizer, texts, max_length=self.max_length, padding=self.padding)
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        with torch.no_grad():
            if self.amp_active:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        pooled: dict[PoolingStrategy, Any] = {}
        if "cls" in self.pooling_strategies:
            pooled["cls"] = extract_cls_embedding(last_hidden_state)
        if "mean" in self.pooling_strategies:
            pooled["mean"] = mean_pooling(last_hidden_state, attention_mask)

        return pooled, tokenized.unpadded_lengths

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """Detach, move to CPU, cast to float32, and optionally
        L2-normalize. float32 cast matters under mixed precision: the
        autocast forward pass yields float16 tensors, and saving those
        directly would halve the stored precision of the embeddings
        Module 4 trains on.
        """
        arr = tensor.detach().float().cpu().numpy()
        return normalize_embeddings(arr) if self.normalize else arr

    def extract_split(
        self,
        df: pd.DataFrame,
        split_name: str,
        text_col: str,
        id_col: str,
        label_col: str,
        language_col: str,
        checkpoint_dir: str | Path | None = None,
    ) -> SplitExtractionResult:
        """Run the full extraction loop for one split, with
        checkpointing, per-batch timing, and per-batch error isolation.

        Checkpointing: every `checkpoint_every_n_batches` batches, the
        accumulated pooled arrays plus the next batch index and the
        sequence-length record are written to
        `checkpoint_dir/<split>_checkpoint.npz`. If that file exists when
        this method starts, extraction resumes from the recorded batch
        index rather than restarting — important for long runs on
        interruptible infrastructure (a Colab session that disconnects).
        Completed splits are never recomputed: the checkpoint is deleted
        only after the split finishes, and `run_feature_extraction`
        skips any split whose final embedding files already exist.

        Error handling: a failing batch (OOM, malformed row) is retried
        as two half-size sub-batches, recursively, until either every
        text succeeds or a single bad row is isolated, skipped, and
        recorded in `errors` — extraction continues rather than aborting
        the split over one document.
        """
        texts = df[text_col].fillna("").astype(str).tolist()
        record_ids = df[id_col].to_numpy()
        labels = df[label_col].to_numpy()
        languages = df[language_col].astype(str).to_numpy()
        n_rows = len(df)
        n_batches = (n_rows + self.batch_size - 1) // self.batch_size

        accumulated: dict[PoolingStrategy, list[np.ndarray]] = {s: [] for s in self.pooling_strategies}
        seq_stats = SequenceLengthStats()
        timer = BatchTimer()
        errors: list[BatchError] = []
        start_batch = 0
        n_success = 0
        n_failure = 0
        checkpoint_created = False
        checkpoint_resumed = False
        n_checkpoints_written = 0

        checkpoint_path = None
        if checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir) / f"{split_name}_checkpoint.npz"
            start_batch, accumulated, seq_stats, n_success = self._maybe_resume(checkpoint_path)
            if start_batch > 0:
                checkpoint_resumed = True
                logger.info(
                    "[%s] Resumed from checkpoint at batch %d/%d (%d document(s) already embedded).",
                    split_name, start_batch, n_batches, n_success,
                )

        timer.start_split()
        for batch_idx in range(start_batch, n_batches):
            batch_start_idx = batch_idx * self.batch_size
            batch_end_idx = min(batch_start_idx + self.batch_size, n_rows)
            batch_texts = texts[batch_start_idx:batch_end_idx]
            batch_ids = record_ids[batch_start_idx:batch_end_idx].tolist()

            t0 = time.perf_counter()
            try:
                pooled, lengths = self._forward_batch(batch_texts)
                seq_stats.add(lengths)
                for strategy, tensor in pooled.items():
                    accumulated[strategy].append(self._to_numpy(tensor))
                n_success += len(batch_texts)
            except Exception as exc:  # noqa: BLE001 - isolated per batch, re-raised per row below
                logger.warning(
                    "[%s] batch %d/%d failed (%s); retrying as sub-batches to isolate the failure.",
                    split_name, batch_idx + 1, n_batches, exc,
                )
                sub = self._retry_as_subbatches(batch_texts, batch_ids)
                for strategy in self.pooling_strategies:
                    ok_arrays = [a[strategy] for a in sub["ok_pooled"] if strategy in a]
                    if ok_arrays:
                        accumulated[strategy].append(np.concatenate(ok_arrays, axis=0))
                seq_stats.add(sub["lengths"])
                n_success += sub["n_success"]
                n_failure += len(sub["errors"])
                errors.extend(sub["errors"])
            timer.record_batch(time.perf_counter() - t0, len(batch_texts))

            if (batch_idx + 1) % self.progress_every == 0 or (batch_idx + 1) == n_batches:
                elapsed = sum(timer.batch_seconds)
                rate = timer.n_documents / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[%s] batch %d/%d (%.1f%%) | %.1f docs/sec | elapsed %s",
                    split_name, batch_idx + 1, n_batches, 100.0 * (batch_idx + 1) / n_batches,
                    rate, human_duration(elapsed),
                )

            if checkpoint_path is not None and (batch_idx + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(checkpoint_path, batch_idx + 1, accumulated, seq_stats, n_success)
                checkpoint_created = True
                n_checkpoints_written += 1

        timer.end_split()

        final_embeddings = {
            strategy: (np.concatenate(arrays, axis=0) if arrays else np.empty((0, self.embedding_dim), dtype="float32"))
            for strategy, arrays in accumulated.items()
        }

        # Rows skipped due to unrecoverable per-row errors must be
        # dropped from the metadata arrays too, otherwise embeddings and
        # record_ids/labels/languages would silently misalign — which
        # would corrupt every downstream module without an obvious error.
        if errors:
            failed_ids = {str(rid) for e in errors for rid in e.record_ids}
            keep = np.array([str(r) not in failed_ids for r in record_ids])
            record_ids, labels, languages = record_ids[keep], labels[keep], languages[keep]
            logger.warning(
                "[%s] %d row(s) failed and were excluded from the saved arrays to keep "
                "embeddings and metadata aligned.", split_name, len(failed_ids),
            )

        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint_path.unlink()  # split completed; checkpoint no longer needed
            logger.info("[%s] Extraction complete — checkpoint removed.", split_name)

        return SplitExtractionResult(
            embeddings=final_embeddings,
            record_ids=record_ids,
            labels=labels,
            languages=languages,
            n_rows=n_rows,
            embedding_dim=self.embedding_dim,
            seq_len_stats=seq_stats,
            timer=timer,
            n_success=n_success,
            n_failure=n_failure,
            checkpoint_created=checkpoint_created,
            checkpoint_resumed=checkpoint_resumed,
            resumed_from_batch=start_batch,
            n_checkpoints_written=n_checkpoints_written,
            errors=errors,
        )

    def _retry_as_subbatches(self, texts: list[str], record_ids: list[Any]) -> dict[str, Any]:
        ok_pooled: list[dict[PoolingStrategy, np.ndarray]] = []
        errors: list[BatchError] = []
        lengths: list[int] = []
        n_success = 0

        def _process(sub_texts: list[str], sub_ids: list[Any]) -> None:
            nonlocal n_success
            if not sub_texts:
                return
            try:
                pooled, sub_lengths = self._forward_batch(sub_texts)
                lengths.extend(sub_lengths)
                ok_pooled.append({s: self._to_numpy(t) for s, t in pooled.items()})
                n_success += len(sub_texts)
            except Exception as exc:  # noqa: BLE001
                if len(sub_texts) == 1:
                    logger.error("Row record_id=%s failed and was skipped: %s\n%s", sub_ids[0], exc, traceback.format_exc())
                    errors.append(BatchError(batch_index=-1, record_ids=sub_ids, error=str(exc)))
                else:
                    mid = len(sub_texts) // 2
                    _process(sub_texts[:mid], sub_ids[:mid])
                    _process(sub_texts[mid:], sub_ids[mid:])

        _process(texts, record_ids)
        return {"ok_pooled": ok_pooled, "errors": errors, "lengths": lengths, "n_success": n_success}

    def _save_checkpoint(
        self,
        path: Path,
        next_batch_index: int,
        accumulated: dict[PoolingStrategy, list[np.ndarray]],
        seq_stats: SequenceLengthStats,
        n_success: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "next_batch_index": np.array(next_batch_index),
            "n_success": np.array(n_success),
            "seq_lengths": np.asarray(seq_stats.lengths, dtype="int64"),
        }
        for strategy, arrays in accumulated.items():
            if arrays:
                payload[f"{strategy}_embeddings"] = np.concatenate(arrays, axis=0)
        # Write to a temp path then rename: an interrupted write leaves
        # the previous good checkpoint intact rather than a truncated
        # file that would fail to load on the next resume attempt.
        # NOTE: np.savez_compressed appends ".npz" unless the path
        # already ends with it, so the temp name must keep that suffix
        # (".tmp.npz", not ".npz.tmp") or the file numpy actually writes
        # won't be the one we then try to rename.
        tmp_path = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_path, **payload)
        tmp_path.replace(path)
        logger.info("Checkpoint saved at batch %d -> %s", next_batch_index, path)

    def _maybe_resume(self, path: Path) -> tuple[int, dict[PoolingStrategy, list[np.ndarray]], SequenceLengthStats, int]:
        accumulated: dict[PoolingStrategy, list[np.ndarray]] = {s: [] for s in self.pooling_strategies}
        seq_stats = SequenceLengthStats()
        if not path.exists():
            return 0, accumulated, seq_stats, 0

        try:
            data = np.load(path)
            next_batch_index = int(data["next_batch_index"])
            n_success = int(data["n_success"]) if "n_success" in data else 0
            if "seq_lengths" in data:
                seq_stats.add(data["seq_lengths"].tolist())
            for strategy in self.pooling_strategies:
                key = f"{strategy}_embeddings"
                if key in data:
                    accumulated[strategy] = [data[key]]
            return next_batch_index, accumulated, seq_stats, n_success
        except Exception as exc:  # noqa: BLE001
            # A corrupt checkpoint should not be fatal — restarting the
            # split from scratch is always correct, just slower.
            logger.warning("Checkpoint %s could not be read (%s); restarting this split from batch 0.", path, exc)
            return 0, {s: [] for s in self.pooling_strategies}, SequenceLengthStats(), 0


def _split_outputs_complete(embeddings_dir: Path, split: str, strategies: list[str], expected_rows: int) -> bool:
    """True if this split's embeddings already exist on disk with the
    expected row count — used to skip completed splits entirely so a
    re-run never recomputes finished work.
    """
    split_dir = embeddings_dir / split
    if not split_dir.exists():
        return False
    for strategy in strategies:
        path = split_dir / f"{strategy}_embeddings.npy"
        if not path.exists():
            return False
        try:
            if np.load(path, mmap_mode="r").shape[0] != expected_rows:
                return False
        except Exception:  # noqa: BLE001
            return False
    return all((split_dir / f"{name}.npy").exists() for name in ["record_ids", "labels", "languages"])


def _directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def run_feature_extraction(
    config_path: str | Path | None = None,
    tokenizer: Any = None,
    model: Any = None,
    splits: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """End-to-end Module 3 run: load config, load encoder, extract
    embeddings for every split, save to disk, write statistics + report.

    Splits whose outputs already exist and have the right row count are
    skipped (never recomputed) unless `force=True`.
    """
    run_started = time.perf_counter()
    root = project_root()
    resolved_config_path = Path(config_path) if config_path else (root / "configs" / "feature_config.yaml")
    cfg = load_yaml(resolved_config_path)

    in_cfg = cfg["input"]
    out_cfg = cfg["output"]
    all_splits = {
        "train": in_cfg["train_filename"],
        "validation": in_cfg["val_filename"],
        "test": in_cfg["test_filename"],
    }
    splits_to_run = splits if splits is not None else list(all_splits.keys())

    processed_dir = root / in_cfg["processed_dir"]
    embeddings_dir = root / out_cfg["embeddings_dir"]
    reports_dir = root / out_cfg["reports_dir"]
    checkpoint_dir = reports_dir / "checkpoints"

    reset_peak_memory()
    extractor = FeatureExtractor(cfg, tokenizer=tokenizer, model=model)

    report: dict[str, Any] = {
        "module": "Module 3 - Multilingual Feature Extraction",
        "status": "running",
        "generated_at": None,
        "model": {
            "name": cfg["model"]["name"] if model is None else "injected/override",
            "local_path": cfg["model"]["local_path"],
            "embedding_dim": extractor.embedding_dim,
            "cls_embedding_dim": extractor.embedding_dim,
            "mean_embedding_dim": extractor.embedding_dim,
        },
        "execution_environment": extractor.device_info,
        "extraction_config": {
            "batch_size": extractor.batch_size,
            "max_sequence_length": extractor.max_length,
            "padding": extractor.padding,
            "pooling_strategies": extractor.pooling_strategies,
            "normalize_embeddings": extractor.normalize,
            "mixed_precision_requested": extractor.requested_amp,
            "mixed_precision_active": extractor.amp_active,
            "checkpoint_every_n_batches": extractor.checkpoint_every,
        },
        "special_tokens": get_special_tokens_info(extractor.tokenizer) if hasattr(extractor.tokenizer, "cls_token") else {},
        "splits": {},
    }

    tamil_check_done = False
    for split_name in splits_to_run:
        csv_path = processed_dir / all_splits[split_name]
        logger.info("Loading %s split from %s", split_name, csv_path)
        df = pd.read_csv(csv_path, low_memory=False)

        if not tamil_check_done:
            tamil_samples = (
                df.loc[df[in_cfg["language_column"]] == "ta", in_cfg["text_column"]]
                .dropna().astype(str).head(200).tolist()
            )
            if tamil_samples:
                report["tamil_unicode_roundtrip_check"] = verify_tamil_roundtrip(extractor.tokenizer, tamil_samples)
                logger.info(
                    "Tamil Unicode round-trip check: %d checked, %d mismatch(es), passed=%s",
                    report["tamil_unicode_roundtrip_check"]["n_checked"],
                    report["tamil_unicode_roundtrip_check"]["n_mismatch"],
                    report["tamil_unicode_roundtrip_check"]["passed"],
                )
                tamil_check_done = True

        if not force and _split_outputs_complete(embeddings_dir, split_name, extractor.pooling_strategies, len(df)):
            logger.info("[%s] Already complete on disk (%d rows) — skipping, not recomputing.", split_name, len(df))
            report["splits"][split_name] = {"skipped_already_complete": True, "n_rows": len(df)}
            continue

        result = extractor.extract_split(
            df,
            split_name=split_name,
            text_col=in_cfg["text_column"],
            id_col=in_cfg["id_column"],
            label_col=in_cfg["label_column"],
            language_col=in_cfg["language_column"],
            checkpoint_dir=checkpoint_dir,
        )

        written_files: dict[str, dict[str, str]] = {}
        for strategy, embeddings in result.embeddings.items():
            meta = EmbeddingMetadata(
                model_name=report["model"]["name"],
                pooling_strategy=strategy,
                embedding_dim=result.embedding_dim,
                max_length=extractor.max_length,
                normalized=extractor.normalize,
                split=split_name,
                n_rows=int(embeddings.shape[0]),
            )
            written_files[strategy] = save_embeddings(
                embeddings=embeddings,
                record_ids=result.record_ids,
                labels=result.labels,
                languages=result.languages,
                output_dir=embeddings_dir,
                split=split_name,
                pooling_strategy=strategy,
                metadata=meta,
                save_numpy=out_cfg["save_numpy"],
                save_torch=out_cfg["save_torch"],
            )

        split_dir_size = _directory_size(embeddings_dir / split_name)
        lang_series, label_series = pd.Series(result.languages), pd.Series(result.labels)

        report["splits"][split_name] = {
            "n_source_rows": result.n_rows,
            "n_embeddings_generated": int(next(iter(result.embeddings.values())).shape[0]) if result.embeddings else 0,
            "n_success": result.n_success,
            "n_failure": result.n_failure,
            "embedding_dims": {s: list(e.shape) for s, e in result.embeddings.items()},
            "cls_embedding_dim": result.embedding_dim if "cls" in result.embeddings else None,
            "mean_embedding_dim": result.embedding_dim if "mean" in result.embeddings else None,
            "by_language": {str(k): int(v) for k, v in lang_series.value_counts().items()},
            "by_label": {str(k): int(v) for k, v in label_series.value_counts().items()},
            "sequence_length": result.seq_len_stats.summary(extractor.max_length),
            "runtime": result.timer.summary(),
            "checkpointing": {
                "checkpoint_created": result.checkpoint_created,
                "n_checkpoints_written": result.n_checkpoints_written,
                "resumed_from_checkpoint": result.checkpoint_resumed,
                "resumed_from_batch": result.resumed_from_batch,
            },
            "storage": {"bytes": split_dir_size, "human": human_bytes(split_dir_size)},
            "n_errors": len(result.errors),
            "errors": [e.to_dict() for e in result.errors],
            "output_files": written_files,
        }

        seq = report["splits"][split_name]["sequence_length"]
        rt = report["splits"][split_name]["runtime"]
        logger.info(
            "[%s] complete: %d embedded (%d failed) | avg %s tokens, max %s, %s truncated (%s%%) | "
            "%s at %s docs/sec | %s on disk",
            split_name, result.n_success, result.n_failure,
            seq.get("avg_token_length"), seq.get("max_token_length"),
            seq.get("n_truncated"), seq.get("pct_truncated"),
            rt.get("total_human"), rt.get("throughput_docs_per_second"),
            human_bytes(split_dir_size),
        )

    # ---- Aggregate totals across every split actually processed ----
    processed = {k: v for k, v in report["splits"].items() if not v.get("skipped_already_complete")}
    all_lengths_summary = _aggregate_sequence_lengths(processed)
    total_docs = sum(v.get("n_embeddings_generated", 0) for v in processed.values())
    total_runtime = time.perf_counter() - run_started

    report["totals"] = {
        "total_documents_embedded": total_docs,
        "total_embedding_vectors": total_docs * len(extractor.pooling_strategies),
        "total_success": sum(v.get("n_success", 0) for v in processed.values()),
        "total_failure": sum(v.get("n_failure", 0) for v in processed.values()),
        "splits_skipped_already_complete": [k for k, v in report["splits"].items() if v.get("skipped_already_complete")],
        "sequence_length_overall": all_lengths_summary,
        "embeddings_dir_storage": {
            "bytes": _directory_size(embeddings_dir),
            "human": human_bytes(_directory_size(embeddings_dir)),
        },
        "total_runtime_seconds": round(total_runtime, 2),
        "total_runtime_human": human_duration(total_runtime),
        "overall_throughput_docs_per_second": round(total_docs / total_runtime, 2) if total_runtime > 0 else None,
    }
    report["peak_gpu_memory"] = peak_gpu_memory()
    report["status"] = "complete" if report["totals"]["total_failure"] == 0 else "complete_with_errors"
    report["generated_at"] = pd.Timestamp.utcnow().isoformat()

    reports_dir.mkdir(parents=True, exist_ok=True)
    stats_path = reports_dir / out_cfg["statistics_filename"]
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    write_feature_report(report, reports_dir / out_cfg["report_filename"])

    logger.info(
        "Module 3 feature extraction complete: %d documents, %d vectors, %s total, peak GPU memory %s.",
        total_docs, report["totals"]["total_embedding_vectors"],
        report["totals"]["total_runtime_human"],
        report["peak_gpu_memory"].get("peak_allocated_human") or "n/a (CPU)",
    )
    return report


def _aggregate_sequence_lengths(processed_splits: dict[str, Any]) -> dict[str, Any]:
    """Combine per-split sequence-length summaries into corpus-level
    figures. Averages are weighted by document count — a plain mean of
    the three split means would over-weight the two small splits.
    """
    total_n = sum(v.get("sequence_length", {}).get("n", 0) for v in processed_splits.values())
    if total_n == 0:
        return {"n": 0}

    weighted_avg = sum(
        v["sequence_length"]["avg_token_length"] * v["sequence_length"]["n"]
        for v in processed_splits.values() if v.get("sequence_length", {}).get("n")
    ) / total_n
    n_truncated = sum(v.get("sequence_length", {}).get("n_truncated", 0) for v in processed_splits.values())
    max_len = max(
        (v["sequence_length"]["max_token_length"] for v in processed_splits.values() if v.get("sequence_length", {}).get("n")),
        default=0,
    )
    return {
        "n": total_n,
        "avg_token_length": round(weighted_avg, 2),
        "max_token_length": max_len,
        "n_truncated": n_truncated,
        "pct_truncated": round(100.0 * n_truncated / total_n, 3),
    }


def write_feature_report(report: dict[str, Any], path: Path) -> None:
    """Render `feature_report.md` directly from the same dict written to
    `feature_statistics.json`, so the two files cannot disagree.
    """
    env = report.get("execution_environment", {})
    totals = report.get("totals", {})
    peak = report.get("peak_gpu_memory", {})
    cfg = report.get("extraction_config", {})

    lines: list[str] = [
        "# Module 3 — Feature Extraction Report",
        "",
        f"**Status:** {report.get('status')}  ",
        f"**Generated:** {report.get('generated_at')}  ",
        f"**Model:** `{report.get('model', {}).get('name')}`  ",
        f"**Embedding dimension:** {report.get('model', {}).get('embedding_dim')} "
        f"(CLS: {report.get('model', {}).get('cls_embedding_dim')}, "
        f"mean-pooled: {report.get('model', {}).get('mean_embedding_dim')})",
        "",
        "Every value in this report is measured from the actual execution — no estimates.",
        "",
        "## 1. Execution environment",
        "",
        "| | |",
        "|---|---|",
        f"| GPU | {env.get('gpu_name') or 'none (CPU run)'} |",
        f"| GPU count | {env.get('gpu_count', 0)} |",
        f"| Total GPU memory | {env.get('gpu_total_memory_human') or 'n/a'} |",
        f"| CUDA version | {env.get('cuda_version') or 'n/a'} |",
        f"| cuDNN version | {env.get('cudnn_version') or 'n/a'} |",
        f"| PyTorch version | {env.get('torch_version') or 'n/a'} |",
        f"| Device used | {env.get('device_used')} |",
        f"| Python | {env.get('python_version')} |",
        f"| Peak GPU memory (allocated) | {peak.get('peak_allocated_human') or 'n/a (CPU run)'} |",
        f"| Peak GPU memory (reserved) | {peak.get('peak_reserved_human') or 'n/a (CPU run)'} |",
        "",
        "## 2. Extraction configuration",
        "",
        "| | |",
        "|---|---|",
        f"| Batch size | {cfg.get('batch_size')} |",
        f"| Max sequence length | {cfg.get('max_sequence_length')} |",
        f"| Padding | {cfg.get('padding')} (dynamic) |",
        f"| Pooling strategies | {', '.join(cfg.get('pooling_strategies', []))} |",
        f"| L2-normalized | {cfg.get('normalize_embeddings')} |",
        f"| Mixed precision requested | {cfg.get('mixed_precision_requested')} |",
        f"| Mixed precision active | {cfg.get('mixed_precision_active')} |",
        f"| Checkpoint interval (batches) | {cfg.get('checkpoint_every_n_batches')} |",
        "",
        "## 3. Per-split results",
        "",
        "| Split | Rows | Embedded | Failed | English | Tamil | Avg tokens | Max tokens | Truncated | % trunc. | Runtime | Docs/sec | Storage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for split, s in report.get("splits", {}).items():
        if s.get("skipped_already_complete"):
            lines.append(f"| {split} | {s.get('n_rows')} | *skipped — already complete* | | | | | | | | | | |")
            continue
        seq, rt, st = s.get("sequence_length", {}), s.get("runtime", {}), s.get("storage", {})
        lines.append(
            f"| {split} | {s.get('n_source_rows', 0):,} | {s.get('n_embeddings_generated', 0):,} | "
            f"{s.get('n_failure', 0)} | {s.get('by_language', {}).get('en', 0):,} | "
            f"{s.get('by_language', {}).get('ta', 0):,} | {seq.get('avg_token_length', 'n/a')} | "
            f"{seq.get('max_token_length', 'n/a')} | {seq.get('n_truncated', 'n/a')} | "
            f"{seq.get('pct_truncated', 'n/a')} | {rt.get('total_human', 'n/a')} | "
            f"{rt.get('throughput_docs_per_second', 'n/a')} | {st.get('human', 'n/a')} |"
        )

    seq_overall = totals.get("sequence_length_overall", {})
    lines += [
        "",
        "## 4. Totals",
        "",
        "| | |",
        "|---|---:|",
        f"| Documents embedded | {totals.get('total_documents_embedded', 0):,} |",
        f"| Embedding vectors | {totals.get('total_embedding_vectors', 0):,} |",
        f"| Successes | {totals.get('total_success', 0):,} |",
        f"| Failures | {totals.get('total_failure', 0):,} |",
        f"| Average token length (pre-truncation) | {seq_overall.get('avg_token_length', 'n/a')} |",
        f"| Maximum token length (pre-truncation) | {seq_overall.get('max_token_length', 'n/a')} |",
        f"| Documents truncated | {seq_overall.get('n_truncated', 'n/a')} ({seq_overall.get('pct_truncated', 'n/a')}%) |",
        f"| Total runtime | {totals.get('total_runtime_human', 'n/a')} |",
        f"| Overall throughput | {totals.get('overall_throughput_docs_per_second', 'n/a')} docs/sec |",
        f"| Total storage | {totals.get('embeddings_dir_storage', {}).get('human', 'n/a')} |",
        "",
    ]

    tamil = report.get("tamil_unicode_roundtrip_check")
    if tamil:
        lines += [
            "## 5. Tamil Unicode integrity",
            "",
            f"- Sample Tamil documents checked: {tamil.get('n_checked')}",
            f"- Round-trip mismatches: {tamil.get('n_mismatch')}",
            f"- **Passed: {tamil.get('passed')}**",
            "",
            "Verified by encoding then decoding real Tamil text with the actual tokenizer and",
            "comparing against the NFC-normalized original — confirming combining vowel signs",
            "and virama survive tokenization intact.",
            "",
        ]

    ckpt_lines = [
        f"- **{split}**: {s.get('checkpointing', {}).get('n_checkpoints_written', 0)} checkpoint(s) written; "
        f"resumed from checkpoint: {s.get('checkpointing', {}).get('resumed_from_checkpoint')}"
        + (f" (from batch {s['checkpointing']['resumed_from_batch']})" if s.get("checkpointing", {}).get("resumed_from_checkpoint") else "")
        for split, s in report.get("splits", {}).items() if not s.get("skipped_already_complete")
    ]
    if ckpt_lines:
        lines += ["## 6. Checkpointing", ""] + ckpt_lines + [""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Backwards-compatible alias — the previous private name, kept so any
# existing import keeps working.
_write_feature_report = write_feature_report
