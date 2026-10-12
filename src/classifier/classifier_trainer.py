"""Training / validation / testing loop for the single-MLP + Focal Loss
classifier — Module 5 (Focal Loss variant).

Distinct from `src/contrastive/trainer.py` (Module 4, which fine-tunes
the XLM-R encoder itself): this only trains a small classification
head on top of already-generated, frozen embeddings (Module 3's or
Module 4's), so an epoch here is fast — no transformer forward/backward
pass is involved, only a few linear layers over pre-computed vectors.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

from src.classifier.data_loader import load_embeddings
from src.classifier.focal_loss import FocalLoss, compute_alpha_from_labels
from src.classifier.torch_mlp import MLPClassifier, load_checkpoint, save_checkpoint
from src.utils.logger import get_logger

logger = get_logger("classifier.trainer")


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap pre-computed embeddings + labels in a plain `TensorDataset`.
    No custom `Dataset` class is needed here (unlike Module 4's text
    triplets) — the whole split already fits in memory as two arrays,
    since it's just `(n_rows, embedding_dim)` floats and `(n_rows,)`
    labels, not raw text needing per-batch tokenization.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: MLPClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion: FocalLoss,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
) -> dict[str, float]:
    """Run one full pass over `loader`. Training if `optimizer` is
    given, otherwise a no-grad evaluation pass (used for both the
    validation loop and the testing loop, so they can't silently
    compute metrics differently from each other).
    """
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * len(y_batch)
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.append(y_batch.detach().cpu().numpy())

    all_preds_arr = np.concatenate(all_preds)
    all_labels_arr = np.concatenate(all_labels)

    return {
        "loss": total_loss / max(len(all_labels_arr), 1),
        "accuracy": accuracy_score(all_labels_arr, all_preds_arr),
        "precision": precision_score(all_labels_arr, all_preds_arr, zero_division=0),
        "recall": recall_score(all_labels_arr, all_preds_arr, zero_division=0),
        "f1": f1_score(all_labels_arr, all_preds_arr, zero_division=0),
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    """Full Module 5 (torch + Focal Loss) pipeline.

    1. Load train/val/test embeddings (Module 3 or Module 4 output).
    2. Build the MLP, Focal Loss (with auto-computed or configured
       per-class alpha), AdamW optimizer, and linear-warmup scheduler.
    3. Training loop: each epoch, train one pass then run the
       validation loop; track the best (lowest) val loss, checkpoint
       whenever it improves, and stop early if it hasn't improved for
       `training.early_stopping_patience` epochs.
    4. Testing loop: reload the *best* checkpoint (not necessarily the
       last epoch trained, especially once early stopping has
       triggered) and evaluate it once on the held-out test split.

    Returns a summary dict (per-epoch train/val metrics, best val loss,
    final test metrics, checkpoint paths) for the CLI script to print.
    """
    from src.utils.config_loader import project_root

    input_cfg = config["input"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    optim_cfg = config["optimizer"]
    sched_cfg = config["scheduler"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    torch.manual_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings_dir = project_root() / input_cfg["embeddings_dir"]
    X_train, y_train, _, meta = load_embeddings(embeddings_dir / f"{input_cfg['train_split']}.h5")
    X_val, y_val, _, _ = load_embeddings(embeddings_dir / f"{input_cfg['val_split']}.h5")
    X_test, y_test, _, _ = load_embeddings(embeddings_dir / f"{input_cfg['test_split']}.h5")
    logger.info(
        "Loaded embeddings from %s (encoder=%s): train=%d val=%d test=%d",
        embeddings_dir, meta.get("model_name", "unknown"), len(y_train), len(y_val), len(y_test),
    )

    train_loader = _make_loader(X_train, y_train, train_cfg["batch_size"], shuffle=True)
    val_loader = _make_loader(X_val, y_val, train_cfg["batch_size"], shuffle=False)
    test_loader = _make_loader(X_test, y_test, train_cfg["batch_size"], shuffle=False)

    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_layer_sizes=model_cfg["hidden_layer_sizes"],
        dropout=model_cfg["dropout"],
        num_classes=model_cfg["num_classes"],
    ).to(device)

    alpha = loss_cfg["alpha"]
    if alpha == "auto":
        alpha = compute_alpha_from_labels(y_train, model_cfg["num_classes"])
        logger.info("Computed Focal Loss alpha (auto, inverse class frequency): %s", alpha)
    criterion = FocalLoss(gamma=loss_cfg["gamma"], alpha=alpha)

    if optim_cfg["name"].lower() != "adamw":
        raise ValueError(f"Only the 'adamw' optimizer is currently wired up; got {optim_cfg['name']!r}.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optim_cfg["learning_rate"], weight_decay=optim_cfg["weight_decay"]
    )

    if sched_cfg["name"].lower() != "linear_warmup":
        raise ValueError(f"Only the 'linear_warmup' scheduler is currently wired up; got {sched_cfg['name']!r}.")
    total_steps = len(train_loader) * train_cfg["max_epochs"]
    warmup_steps = int(total_steps * sched_cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    checkpoint_dir = project_root() / output_cfg["checkpoint_dir"]
    best_checkpoint_path = checkpoint_dir / output_cfg["best_checkpoint_name"]
    last_checkpoint_path = checkpoint_dir / output_cfg["last_checkpoint_name"]

    best_val_loss = float("inf")
    patience_counter = 0
    epoch_summaries: list[dict[str, Any]] = []

    for epoch in range(train_cfg["max_epochs"]):
        start = time.time()
        train_metrics = _run_epoch(model, train_loader, device, criterion, optimizer, scheduler)
        val_metrics = _run_epoch(model, val_loader, device, criterion, optimizer=None, scheduler=None)
        elapsed = time.time() - start

        logger.info(
            "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_f1=%.4f (%.1fs)",
            epoch + 1, train_cfg["max_epochs"], train_metrics["loss"], val_metrics["loss"], val_metrics["f1"], elapsed,
        )
        epoch_summaries.append(
            {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics, "elapsed_seconds": round(elapsed, 1)}
        )

        improved = val_metrics["loss"] < best_val_loss - train_cfg["min_delta"]
        if improved:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            save_checkpoint(model, best_checkpoint_path, extra={"epoch": epoch + 1, "val_loss": best_val_loss})
            logger.info("New best val_loss=%.4f -> saved to %s", best_val_loss, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stopping_patience"]:
                logger.info(
                    "Early stopping at epoch %d (no val_loss improvement for %d epoch(s)).",
                    epoch + 1, patience_counter,
                )
                break

    save_checkpoint(
        model, last_checkpoint_path,
        extra={"epoch": epoch_summaries[-1]["epoch"], "val_loss": epoch_summaries[-1]["val"]["loss"]},
    )

    # Testing loop: always evaluate the BEST checkpoint (lowest val
    # loss seen during training), not whichever weights training
    # happened to end on — important once early stopping can trigger
    # before the final epoch.
    best_model = load_checkpoint(best_checkpoint_path, device=device)
    test_metrics = _run_epoch(best_model, test_loader, device, criterion, optimizer=None, scheduler=None)
    logger.info("Test metrics (best checkpoint): %s", test_metrics)

    return {
        "epochs": epoch_summaries,
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
    }
