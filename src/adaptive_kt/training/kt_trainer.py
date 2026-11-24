from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from adaptive_kt.evaluation.metrics import binary_metrics


@dataclass
class TrainResult:
    model: nn.Module
    best_epoch: int
    validation_metrics: dict[str, float]
    history: list[dict[str, float]]


def _batch_predictions(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device):
    skills = batch["skills"].to(device, non_blocking=True)
    items = batch["items"].to(device, non_blocking=True)
    correct = batch["correct"].to(device, non_blocking=True)
    behavior = batch.get("behavior")
    elapsed = batch.get("elapsed")
    lag = batch.get("lag")
    if behavior is not None:
        behavior = behavior.to(device, non_blocking=True)
    if elapsed is not None:
        elapsed = elapsed.to(device, non_blocking=True)
    if lag is not None:
        lag = lag.to(device, non_blocking=True)

    logits = model(
        skills=skills,
        items=items,
        correct=correct,
        behavior=behavior,
        elapsed=elapsed,
        lag=lag,
    )

    targets = correct[:, 1:].float()
    valid = items[:, 1:].ne(0)
    eligible = batch.get("eligible")
    if eligible is not None:
        eligible = eligible.to(device, non_blocking=True)
        valid = valid & eligible[:, 1:]
    return logits[valid], targets[valid]


def predict_kt(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits, y = _batch_predictions(model, batch, device)
            if y.numel() == 0:
                continue
            targets.append(y.detach().cpu().numpy())
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    if not targets:
        raise ValueError("No valid next-interaction targets were produced by the sequence loader.")
    return np.concatenate(targets), np.concatenate(probabilities)


def evaluate_kt(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true, y_prob = predict_kt(model, loader, device)
    return binary_metrics(y_true, y_prob, threshold=threshold)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_history(history: list[dict[str, float]], history_path: Path | None) -> None:
    if history_path is None:
        return
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)


def train_kt_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    *,
    checkpoint_path: str | Path | None = None,
    history_path: str | Path | None = None,
    model_name: str = "kt_model",
    seed: int = 42,
) -> TrainResult:
    """Train a KT model with live progress and epoch-level restart checkpoints.

    The restart checkpoint is written atomically after every completed epoch. Re-running the
    experiment resumes from the next epoch instead of starting that KT model from epoch 1.
    """
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    history_path = Path(history_path) if history_path is not None else None

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_score = -float("inf")
    best_epoch = 0
    best_state = _cpu_state_dict(model)
    history: list[dict[str, float]] = []
    stale_epochs = 0
    start_epoch = 1

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        best_score = float(checkpoint.get("best_score", best_score))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_state = checkpoint.get("best_state_dict", best_state)
        history = list(checkpoint.get("history", []))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        last_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = last_epoch + 1
        status = str(checkpoint.get("status", "in_progress"))
        print(
            f"[{model_name}] resume checkpoint found: epoch {last_epoch}/{epochs} "
            f"(best epoch {best_epoch}, status={status})",
            flush=True,
        )
        if status == "complete" or start_epoch > epochs or stale_epochs >= patience:
            model.load_state_dict(best_state)
            final_metrics = evaluate_kt(model, validation_loader, device, threshold=0.5)
            return TrainResult(model=model, best_epoch=best_epoch, validation_metrics=final_metrics, history=history)

    print(
        f"[{model_name}] training on {device} | epochs={epochs} | early-stopping patience={patience} | "
        f"starting_epoch={start_epoch}",
        flush=True,
    )

    last_completed_epoch = start_epoch - 1
    stopped_early = False

    for epoch in range(start_epoch, epochs + 1):
        # Epoch-specific seeds make a resumed epoch follow the same shuffle/dropout seed as a
        # clean uninterrupted run for that epoch.
        epoch_seed = int(seed) + int(epoch)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        loader_generator = getattr(train_loader, "generator", None)
        if loader_generator is not None:
            loader_generator.manual_seed(epoch_seed)

        model.train()
        running_loss = 0.0
        batch_count = 0
        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"{model_name} epoch {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            logits, targets = _batch_predictions(model, batch, device)
            if targets.numel() == 0:
                continue
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_count += 1
            running_loss += float(loss.detach().cpu())
            progress.set_postfix(train_loss=f"{running_loss / batch_count:.5f}")

        train_loss = running_loss / batch_count if batch_count else float("nan")
        print(f"[{model_name}] epoch {epoch}: validating...", flush=True)
        val_metrics = evaluate_kt(model, validation_loader, device, threshold=0.5)
        val_auc = float(val_metrics.get("auc", float("nan")))
        score = val_auc if np.isfinite(val_auc) else -float(val_metrics["rmse"])
        row = {"epoch": epoch, "train_loss": float(train_loss), **val_metrics}
        history.append(row)

        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1

        last_completed_epoch = epoch
        print(
            f"[{model_name}] epoch {epoch}/{epochs} complete | "
            f"loss={train_loss:.6f} | val_auc={val_metrics.get('auc', float('nan')):.6f} | "
            f"val_f1={val_metrics.get('f1', float('nan')):.6f} | "
            f"val_rmse={val_metrics.get('rmse', float('nan')):.6f} | "
            f"best_epoch={best_epoch} | patience={stale_epochs}/{patience}",
            flush=True,
        )

        _save_history(history, history_path)
        if checkpoint_path is not None:
            _atomic_torch_save(
                {
                    "status": "in_progress",
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_state_dict": best_state,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                    "history": history,
                    "seed": seed,
                },
                checkpoint_path,
            )
            print(f"[{model_name}] checkpoint saved -> {checkpoint_path}", flush=True)

        if stale_epochs >= patience:
            stopped_early = True
            print(
                f"[{model_name}] early stopping after epoch {epoch}; best epoch={best_epoch}",
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    final_metrics = evaluate_kt(model, validation_loader, device, threshold=0.5)
    _save_history(history, history_path)

    if checkpoint_path is not None:
        _atomic_torch_save(
            {
                "status": "complete",
                "epoch": last_completed_epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_state_dict": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "history": history,
                "seed": seed,
                "stopped_early": stopped_early,
            },
            checkpoint_path,
        )
        print(f"[{model_name}] training complete; final restart checkpoint -> {checkpoint_path}", flush=True)

    return TrainResult(model=model, best_epoch=best_epoch, validation_metrics=final_metrics, history=history)
