"""Protocol-locked training loop for probe models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .scheduler import WarmupCosineConfig


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str
    lr: float
    weight_decay: float
    grad_clip: float
    batch_size: int
    epochs: int
    schedule: WarmupCosineConfig


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    dtype_train: str = "fp32",
    log_every: int = 0,
) -> Dict[str, float]:
    """Train a classifier head for a fixed number of epochs.

    Notes:
    - Early stopping is forbidden by protocol.
    - Training dtype is fixed to FP32 for v1.1 (even for precision section).
    """
    if dtype_train.lower() != "fp32":
        raise ValueError("v1.1 training dtype is fixed to fp32 (PTQ uses fp32 training).")

    model.to(device)

    # ICON safety: training must be FP32

    model.to(dtype=torch.float32)


    model.train()

    if cfg.optimizer.lower() != "adamw":
        raise ValueError(f"Unsupported optimizer for protocol: {cfg.optimizer}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    last_loss = 0.0

    for epoch in range(cfg.epochs):
        mult = cfg.schedule.lr_multiplier(epoch)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * mult

        for xb, yb in train_loader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device)

            # ICON safety: enforce class-index labels
            if hasattr(yb, 'ndim') and yb.ndim > 1:
                yb = yb.argmax(dim=1)
            yb = yb.long()

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            last_loss = float(loss.item())
            global_step += 1
            if log_every and global_step % log_every == 0:
                print(f"[train] epoch={epoch+1}/{cfg.epochs} step={global_step} loss={last_loss:.4f} lr={cfg.lr*mult:.6f}")

    return {"final_loss": last_loss, "steps": float(global_step)}
