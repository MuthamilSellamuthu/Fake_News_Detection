"""Module 4 — two-stage training (SupCon projection head, then linear probe).

Stage 1 trains the projection head with supervised contrastive loss.
Stage 2 freezes it and fits a linear classifier on the 128-d output.

Why two stages rather than one joint objective: freezing the head means
any gain over the cross-entropy MLP baseline is attributable to the
learned *representation*, not to extra classifier capacity. A jointly
trained model would confound the two and weaken the paper's claim.

Model selection discipline, enforced structurally rather than by
convention:
  - Stage 1 early-stops on validation SupCon loss.
  - Stage 2 early-stops on validation macro-F1.
  - `evaluate_on_test()` is a separate call the training loops cannot
    reach. Test is touched once, after selection.

torch is imported lazily so this module can be inspected and partially
tested without it.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.contrastive.batch_sampler import build_sampler
from src.contrastive.embedding_dataset import SplitEmbeddings
from src.contrastive.evaluation import evaluate_predictions
from src.contrastive.projection_head import (
    build_ce_mlp_baseline,
    build_linear_classifier,
    build_projection_head,
)
from src.contrastive.supcon_loss import batch_positive_statistics, build_supcon_loss


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that affects a run, and record it in the config
    snapshot. Reproducibility is a verification criterion, not a nicety.
    """
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested: str = "auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContrastiveTrainer:
    """Runs one experiment: one pooling, one sampling strategy, one seed."""

    def __init__(self, config: dict[str, Any], device=None) -> None:
        self.config = config
        self.device = device if device is not None else select_device(config["runtime"]["device"])
        self.history: dict[str, list] = {"stage1": [], "stage2": []}
        self.telemetry: dict[str, Any] = {}

    # ---------------- stage 1: supervised contrastive ----------------

    def train_projection_head(
        self,
        train: SplitEmbeddings,
        val: SplitEmbeddings,
        sampling_strategy: str,
        seed: int,
    ):
        import torch

        cfg = self.config["supcon"]
        mcfg = self.config["model"]

        head = build_projection_head(
            input_dim=self.config["input"]["embedding_dim"],
            hidden_dim=mcfg["projection_hidden_dim"],
            output_dim=mcfg["projection_output_dim"],
            dropout=mcfg["dropout"],
            normalize=mcfg["normalize_projection"],
        ).to(self.device)

        criterion = build_supcon_loss(cfg["temperature"])
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"],
                                      weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

        sampler = build_sampler(
            strategy=sampling_strategy,
            languages=train.languages,
            batch_size=cfg["batch_size"],
            tamil_fraction=self.config["sampling"]["tamil_fraction"],
            seed=seed,
            replacement=self.config["sampling"]["replacement"],
        )

        x_train = torch.from_numpy(train.embeddings).float()
        y_train = torch.from_numpy(train.labels).long()
        x_val = torch.from_numpy(val.embeddings).float().to(self.device)
        y_val = torch.from_numpy(val.labels).long().to(self.device)

        best_val, best_state, patience = float("inf"), None, 0
        # Positive-count statistics from the first epoch: the direct
        # measurement that justifies language-aware sampling.
        first_epoch_positive_stats: list[dict] = []
        t0 = time.perf_counter()

        for epoch in range(cfg["epochs"]):
            head.train()
            sampler.set_epoch(epoch)
            losses = []
            for bi, idx in enumerate(sampler):
                xb = x_train[idx].to(self.device)
                yb = y_train[idx].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                z = head(xb)
                loss = criterion(z, yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if epoch == 0 and bi < 5:
                    first_epoch_positive_stats.append(
                        batch_positive_statistics(train.labels[idx], train.languages[idx])
                    )

            head.eval()
            with torch.no_grad():
                val_loss = float(criterion(head(x_val), y_val).cpu())

            scheduler.step()
            entry = {
                "epoch": epoch + 1,
                "train_supcon_loss": round(float(np.mean(losses)), 6),
                "val_supcon_loss": round(val_loss, 6),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            }
            self.history["stage1"].append(entry)

            if val_loss < best_val - 1e-6:
                best_val, patience = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg["early_stopping_patience"]:
                    break

        if best_state is not None:
            head.load_state_dict(best_state)

        self.telemetry["stage1"] = {
            "epochs_run": len(self.history["stage1"]),
            "best_val_supcon_loss": round(best_val, 6),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
            "sampler": sampler.describe(),
            "first_epoch_batch_positive_stats": first_epoch_positive_stats,
        }
        return head

    # ---------------- stage 2: linear probe on frozen projections ----

    def train_linear_classifier(self, head, train: SplitEmbeddings, val: SplitEmbeddings):
        import torch
        import torch.nn as nn

        cfg = self.config["classifier"]
        head.eval()
        for p in head.parameters():   # frozen: this is what makes the
            p.requires_grad = False   # comparison to CE-MLP meaningful

        with torch.no_grad():
            z_train = head(torch.from_numpy(train.embeddings).float().to(self.device)).cpu()
            z_val = head(torch.from_numpy(val.embeddings).float().to(self.device)).cpu()

        clf = build_linear_classifier(self.config["model"]["projection_output_dim"], 2).to(self.device)
        optimizer = torch.optim.AdamW(clf.parameters(), lr=cfg["learning_rate"],
                                      weight_decay=cfg["weight_decay"])
        criterion = nn.CrossEntropyLoss()

        y_train = torch.from_numpy(train.labels).long()
        x_val_d, y_val_d = z_val.to(self.device), torch.from_numpy(val.labels).long().to(self.device)

        best_f1, best_state, patience = -1.0, None, 0
        t0 = time.perf_counter()
        n = z_train.shape[0]
        rng = np.random.default_rng(self.config["reproducibility"]["seed"])

        for epoch in range(cfg["epochs"]):
            clf.train()
            order = rng.permutation(n)
            losses = []
            for s in range(0, n, cfg["batch_size"]):
                idx = order[s : s + cfg["batch_size"]]
                xb = z_train[idx].to(self.device)
                yb = y_train[idx].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(clf(xb), yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            clf.eval()
            with torch.no_grad():
                logits = clf(x_val_d)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_loss = float(criterion(logits, y_val_d).cpu())
            res = evaluate_predictions(val.labels, preds, val.languages)
            f1 = res["overall"]["macro_f1"]

            self.history["stage2"].append({
                "epoch": epoch + 1,
                "train_ce_loss": round(float(np.mean(losses)), 6),
                "val_ce_loss": round(val_loss, 6),
                "val_macro_f1": f1,
                "val_macro_f1_en": res["per_language"]["en"].get("macro_f1"),
                "val_macro_f1_ta": res["per_language"]["ta"].get("macro_f1"),
            })

            if f1 > best_f1 + 1e-6:
                best_f1, patience = f1, 0
                best_state = {k: v.detach().clone() for k, v in clf.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg["early_stopping_patience"]:
                    break

        if best_state is not None:
            clf.load_state_dict(best_state)

        self.telemetry["stage2"] = {
            "epochs_run": len(self.history["stage2"]),
            "best_val_macro_f1": round(best_f1, 6),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
        }
        return clf

    # ---------------- inference ----------------

    def predict(self, head, clf, data: SplitEmbeddings) -> tuple[np.ndarray, np.ndarray]:
        import torch
        import torch.nn.functional as F

        head.eval(); clf.eval()
        with torch.no_grad():
            x = torch.from_numpy(data.embeddings).float().to(self.device)
            logits = clf(head(x))
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds, probs

    def evaluate(self, head, clf, data: SplitEmbeddings) -> dict[str, Any]:
        preds, probs = self.predict(head, clf, data)
        return evaluate_predictions(data.labels, preds, data.languages, y_score=probs)

    # ---------------- CE-MLP baseline ----------------

    def train_ce_mlp(self, train: SplitEmbeddings, val: SplitEmbeddings, sampling_strategy: str, seed: int):
        """Capacity-matched cross-entropy baseline (768->512->128->2).

        Trained with the SAME sampler as the SupCon run, so the
        comparison isolates the objective rather than confounding it
        with a different data distribution.
        """
        import torch
        import torch.nn as nn

        bcfg = self.config["baselines"]
        model = build_ce_mlp_baseline(
            input_dim=self.config["input"]["embedding_dim"],
            hidden_dims=bcfg["mlp_hidden_dims"],
            n_classes=2,
            dropout=self.config["model"]["dropout"],
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=bcfg["mlp_learning_rate"])
        criterion = nn.CrossEntropyLoss()
        sampler = build_sampler(
            strategy=sampling_strategy, languages=train.languages,
            batch_size=self.config["supcon"]["batch_size"],
            tamil_fraction=self.config["sampling"]["tamil_fraction"],
            seed=seed, replacement=self.config["sampling"]["replacement"],
        )

        x_train = torch.from_numpy(train.embeddings).float()
        y_train = torch.from_numpy(train.labels).long()
        x_val_d = torch.from_numpy(val.embeddings).float().to(self.device)

        best_f1, best_state, patience, history = -1.0, None, 0, []
        t0 = time.perf_counter()
        for epoch in range(bcfg["mlp_epochs"]):
            model.train()
            sampler.set_epoch(epoch)
            losses = []
            for idx in sampler:
                xb, yb = x_train[idx].to(self.device), y_train[idx].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward(); optimizer.step()
                losses.append(float(loss.detach().cpu()))

            model.eval()
            with torch.no_grad():
                preds = model(x_val_d).argmax(dim=1).cpu().numpy()
            f1 = evaluate_predictions(val.labels, preds, val.languages)["overall"]["macro_f1"]
            history.append({"epoch": epoch + 1, "train_ce_loss": round(float(np.mean(losses)), 6),
                            "val_macro_f1": f1})
            if f1 > best_f1 + 1e-6:
                best_f1, patience = f1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.config["classifier"]["early_stopping_patience"]:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.telemetry["ce_mlp"] = {
            "epochs_run": len(history), "best_val_macro_f1": round(best_f1, 6),
            "runtime_seconds": round(time.perf_counter() - t0, 2), "history": history,
        }
        return model

    def evaluate_ce_mlp(self, model, data: SplitEmbeddings) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(data.embeddings).float().to(self.device))
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        return evaluate_predictions(data.labels, preds, data.languages, y_score=probs)

    def gpu_telemetry(self) -> dict[str, Any]:
        from src.embeddings.runtime_stats import collect_device_info, peak_gpu_memory

        return {"device": collect_device_info(self.device), "peak_gpu_memory": peak_gpu_memory()}
