"""Module 6 — resumable verifier training.

Built for free Colab, which disconnects. A checkpoint therefore stores
everything needed to continue *identically*, not just the weights:

    model, optimizer, scheduler, epoch, global_step, RNG state
    (python / numpy / torch / cuda), and the config that produced it.

Without the RNG state a resumed run would see a different shuffle and take a
different trajectory, so "resume" would silently mean "restart from these
weights" -- reproducible only by accident. `tests/test_verification.py`
asserts that an interrupted-then-resumed run matches an uninterrupted one.

Checkpoints are written temp-then-renamed; a disconnect mid-write leaves the
previous good checkpoint intact rather than a truncated file.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")
    best_step: int = 0
    epochs_without_improvement: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG that can affect the trajectory."""
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:  # noqa: BLE001
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        s = state["python"]
        random.setstate((s[0], tuple(s[1]), s[2]) if isinstance(s, (list, tuple)) else s)
    try:
        import numpy as np
        if "numpy" in state:
            np.random.set_state(state["numpy"])
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception:  # noqa: BLE001
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass


class CheckpointManager:
    """Atomic save/load of full training state."""

    def __init__(self, ckpt_dir: Path) -> None:
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.latest = self.dir / "latest.pt"
        self.best = self.dir / "best.pt"
        self.meta_path = self.dir / "checkpoint_meta.json"

    def save(self, model, optimizer, scheduler, state: TrainState,
             config: dict[str, Any], is_best: bool = False) -> dict[str, Any]:
        import torch

        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "train_state": state.to_dict(),
            "rng_state": capture_rng_state(),
            "config": config,
        }
        tmp = self.dir / "latest.tmp.pt"
        torch.save(payload, tmp)
        tmp.replace(self.latest)
        if is_best:
            tmp2 = self.dir / "best.tmp.pt"
            torch.save(payload, tmp2)
            tmp2.replace(self.best)
        meta = {"epoch": state.epoch, "global_step": state.global_step,
                "best_metric": state.best_metric, "best_step": state.best_step,
                "has_best": self.best.is_file(), "config": config}
        tmp3 = self.dir / "checkpoint_meta.tmp.json"
        tmp3.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        tmp3.replace(self.meta_path)
        return meta

    def load(self, model, optimizer=None, scheduler=None, path: Path | None = None
             ) -> TrainState | None:
        # Existence is checked BEFORE importing torch: "no checkpoint yet" is
        # the normal first-run case and must not require a torch install.
        p = Path(path) if path else self.latest
        if not p.is_file():
            return None

        import torch

        payload = torch.load(p, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state"):
            scheduler.load_state_dict(payload["scheduler_state"])
        restore_rng_state(payload.get("rng_state", {}))
        return TrainState.from_dict(payload.get("train_state", {}))

    def exists(self) -> bool:
        return self.latest.is_file()


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> float:
    """Macro-F1 without sklearn, so metrics work in any environment.

    A class with no true and no predicted instances contributes F1 = 0 and is
    still counted in the denominator; silently skipping it would inflate the
    score on the rare X-FACT classes.
    """
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def classification_report(y_true: Sequence[int], y_pred: Sequence[int],
                          labels: Sequence[str]) -> dict[str, Any]:
    n = len(labels)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    per: dict[str, dict[str, float]] = {}
    precs, recs = [], []
    for c in range(n):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[labels[c]] = {"precision": prec, "recall": rec, "f1": f1,
                          "support": sum(1 for t in y_true if t == c)}
        precs.append(prec)
        recs.append(rec)
    cm = [[sum(1 for t, p in zip(y_true, y_pred) if t == i and p == j) for j in range(n)]
          for i in range(n)]
    # Support-weighted F1. Reported ALONGSIDE macro-F1, never instead of it:
    # under imbalance the weighted score is dominated by the majority class and
    # can look healthy while a rare class scores 0.
    total = len(y_true)
    weighted_f1 = (sum(per[labels[c]]["f1"] * per[labels[c]]["support"] for c in range(n)) / total
                   if total and n else 0.0)
    return {"accuracy": acc, "macro_f1": macro_f1(y_true, y_pred, n),
            "weighted_f1": weighted_f1,
            "precision_macro": sum(precs) / n if n else 0.0,
            "recall_macro": sum(recs) / n if n else 0.0,
            "per_class": per, "confusion_matrix": cm, "labels": list(labels)}


class VerifierTrainer:
    """Training loop with per-step checkpointing and early stopping.

    `step_fn` and `eval_fn` are injected so the resume logic can be tested
    without torch, a GPU, or any downloaded weights.
    """

    def __init__(self, ckpt_dir: Path, config: dict[str, Any],
                 checkpoint_every_steps: int = 500,
                 early_stopping_patience: int = 2) -> None:
        self.ckpt = CheckpointManager(ckpt_dir)
        self.config = config
        self.checkpoint_every_steps = int(checkpoint_every_steps)
        self.patience = int(early_stopping_patience)

    def train(self, model, optimizer, scheduler, batches_fn: Callable[[int], Sequence[Any]],
              step_fn: Callable[[Any, Any], float], eval_fn: Callable[[Any], dict[str, Any]],
              epochs: int, resume: bool = True,
              log: Callable[[str], None] = print) -> dict[str, Any]:
        state = TrainState()
        if resume and self.ckpt.exists():
            loaded = self.ckpt.load(model, optimizer, scheduler)
            if loaded is not None:
                state = loaded
                log(f"  resumed at epoch {state.epoch}, step {state.global_step}")

        t0 = time.perf_counter()
        stopped_early = False
        for epoch in range(state.epoch, int(epochs)):
            state.epoch = epoch
            for batch in batches_fn(epoch):
                loss = step_fn(model, batch)
                state.global_step += 1
                if self.checkpoint_every_steps and \
                        state.global_step % self.checkpoint_every_steps == 0:
                    self.ckpt.save(model, optimizer, scheduler, state, self.config)

            metrics = eval_fn(model)
            score = float(metrics.get("macro_f1", 0.0))
            improved = score > state.best_metric
            if improved:
                state.best_metric, state.best_step = score, state.global_step
                state.epochs_without_improvement = 0
            else:
                state.epochs_without_improvement += 1
            state.history.append({"epoch": epoch, "step": state.global_step,
                                  "loss": float(loss), **metrics})
            self.ckpt.save(model, optimizer, scheduler, state, self.config, is_best=improved)
            log(f"  epoch {epoch}: macro_f1={score:.4f}"
                f"{' (best)' if improved else ''}")

            if state.epochs_without_improvement >= self.patience:
                log(f"  early stopping: no improvement for {self.patience} epoch(s)")
                stopped_early = True
                state.epoch = epoch + 1
                self.ckpt.save(model, optimizer, scheduler, state, self.config)
                break
        else:
            state.epoch = int(epochs)
            self.ckpt.save(model, optimizer, scheduler, state, self.config)

        return {"epochs_run": state.epoch, "global_step": state.global_step,
                "best_metric": state.best_metric, "best_step": state.best_step,
                "stopped_early": stopped_early, "history": state.history,
                "seconds": round(time.perf_counter() - t0, 2)}
