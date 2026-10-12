#!/usr/bin/env python
"""Module 3 orchestrator: turn Module 2's processed text into XLM-RoBERTa
contextual embeddings and store them efficiently.

This script only does frozen feature extraction — it loads a pretrained
XLM-RoBERTa checkpoint, tokenizes with its SentencePiece tokenizer,
runs a forward pass, pools the result into one vector per row, and
writes those vectors to `data/embeddings/<split>.h5`. It does not train,
fine-tune, or classify anything.

Usage
-----
  # All three splits, default model/config:
  python scripts/generate_embeddings.py

  # Just train, smaller batch size, xlm-roberta-large:
  python scripts/generate_embeddings.py --splits train --batch-size 16 --model-name xlm-roberta-large

  # Quick smoke test on a handful of rows before committing to a full run:
  python scripts/generate_embeddings.py --splits val --limit 50

Realistic time expectation: encoding tens of thousands of rows through
a 12-layer transformer is slow on CPU (expect low tens of rows/second;
a GPU is 10-50x faster). Use `--limit` to test on a small slice first.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Make `src` importable when run as `python scripts/generate_embeddings.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.embeddings.embedding_store import EmbeddingStore  # noqa: E402
from src.embeddings.xlmr_encoder import XLMREncoder  # noqa: E402
from src.utils.config_loader import load_yaml, project_root  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.generate_embeddings")


def load_split(processed_dir: Path, split: str, text_col: str, id_col: str, label_col: str, limit: int | None) -> pd.DataFrame:
    """Read one split's processed CSV and return only the columns this
    module needs (text to encode, id to join back on, label to carry
    alongside the embedding for convenience in Module 4/5).

    `limit`, when set, takes the first N rows only — meant for a quick
    smoke test of the encoder/storage path before running a full split
    (which, on CPU, can take a long time — see module docstring).
    """
    path = processed_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/preprocess_datasets.py (Module 2) first."
        )
    df = pd.read_csv(path)
    missing_cols = [c for c in (text_col, id_col, label_col) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} is missing expected column(s): {missing_cols}")
    if limit is not None:
        df = df.head(limit)
    # Rows with empty/NaN text can't be tokenized meaningfully; Module 2
    # already drops these during its own missing-value step, but guard
    # here too in case this script is ever pointed at a differently-built CSV.
    df = df[df[text_col].notna() & (df[text_col].str.strip() != "")]
    return df.reset_index(drop=True)


def process_split(
    split: str,
    encoder: XLMREncoder,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Encode one split end-to-end: load rows -> batch-encode -> stream to HDF5.

    Returns a small summary dict (row count, output path, elapsed time)
    for the final printed report — mirrors the pattern used by
    `scripts/download_datasets.py` and `scripts/preprocess_datasets.py`
    so all three CLIs report results the same way.
    """
    input_cfg = config["input"]
    output_cfg = config["output"]

    processed_dir = project_root() / input_cfg["processed_dir"]
    df = load_split(
        processed_dir,
        split,
        text_col=input_cfg["text_column"],
        id_col=input_cfg["id_column"],
        label_col=input_cfg["label_column"],
        limit=config.get("_limit"),
    )
    logger.info("Loaded %d row(s) for split '%s'.", len(df), split)

    embeddings_dir = project_root() / output_cfg["embeddings_dir"]
    output_path = embeddings_dir / f"{split}.h5"

    store = EmbeddingStore.create(
        output_path,
        embedding_dim=encoder.get_embedding_dim(),
        metadata={
            "model_name": encoder.model_name,
            "pooling_strategy": encoder.pooling_strategy,
            "max_sequence_length": encoder.max_sequence_length,
            "embedding_dim": encoder.get_embedding_dim(),
            "split": split,
            "row_count": len(df),
        },
        compression=output_cfg["compression"],
        compression_level=output_cfg["compression_level"],
    )

    batch_size = config["batch_size"]
    texts = df[input_cfg["text_column"]].tolist()
    record_ids = df[input_cfg["id_column"]].to_numpy()
    labels = df[input_cfg["label_column"]].to_numpy()

    start = time.time()
    # Encode + write in the same batch size, so each forward pass's
    # output is flushed to disk immediately rather than accumulated —
    # this is what keeps peak memory bounded regardless of split size.
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = encoder.encode_batch(batch_texts)
        store.append(
            batch_embeddings,
            record_ids[i : i + batch_size],
            labels[i : i + batch_size],
        )
    elapsed = time.time() - start
    store.close()

    logger.info(
        "Split '%s': %d rows encoded in %.1fs (%.1f rows/sec) -> %s",
        split, len(df), elapsed, len(df) / elapsed if elapsed > 0 else 0.0, output_path,
    )

    return {
        "split": split,
        "rows_encoded": len(df),
        "elapsed_seconds": round(elapsed, 1),
        "output_file": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 3: XLM-RoBERTa embedding extraction.")
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Which split(s) to encode. Default: all splits listed in configs/embeddings.yaml.",
    )
    parser.add_argument("--model-name", default=None, help="Override configs/embeddings.yaml model.name.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configs/embeddings.yaml batch_size.")
    parser.add_argument("--max-length", type=int, default=None, help="Override model.max_sequence_length.")
    parser.add_argument("--pooling", choices=["mean", "cls"], default=None, help="Override model.pooling_strategy.")
    parser.add_argument("--device", default=None, help="Override model.device (e.g. cpu, cuda, cuda:0).")
    parser.add_argument("--limit", type=int, default=None, help="Only encode the first N rows of each split (smoke test).")
    args = parser.parse_args()

    config_path = project_root() / "configs" / "embeddings.yaml"
    config = load_yaml(config_path)

    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.max_length:
        config["model"]["max_sequence_length"] = args.max_length
    if args.pooling:
        config["model"]["pooling_strategy"] = args.pooling
    if args.device:
        config["model"]["device"] = args.device
    if args.limit:
        config["_limit"] = args.limit

    splits = args.splits or config["input"]["splits"]

    logger.info("Loading encoder: %s", config["model"]["name"])
    encoder = XLMREncoder(
        model_name=config["model"]["name"],
        max_sequence_length=config["model"]["max_sequence_length"],
        pooling_strategy=config["model"]["pooling_strategy"],
        device=config["model"]["device"],
    )
    logger.info(
        "Encoder ready: hidden_size=%d, pooling=%s, device=%s",
        encoder.get_embedding_dim(), encoder.pooling_strategy, encoder.device,
    )

    results = []
    for split in splits:
        logger.info("=" * 60)
        logger.info("Encoding split: %s", split)
        results.append(process_split(split, encoder, config))

    print("\n" + "=" * 72)
    print("MODULE 3 EMBEDDING GENERATION SUMMARY")
    print("=" * 72)
    for result in results:
        print(f"  [{result['split']:^8}] {result['rows_encoded']:>7} rows in {result['elapsed_seconds']:>7.1f}s -> {result['output_file']}")
    print("=" * 72)
    print("Nothing has been trained or classified — this is Module 3 output only.")


if __name__ == "__main__":
    main()
