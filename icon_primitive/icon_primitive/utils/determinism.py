"""Determinism helpers (protocol-locked)."""

from __future__ import annotations

from typing import Dict

import torch


def apply_determinism(cfg: Dict) -> None:
    """Apply determinism-related flags from protocol config."""
    torch.backends.cudnn.deterministic = bool(cfg.get("cudnn_deterministic", True))
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
    use_det = cfg.get("torch_deterministic_algorithms", False)
    if use_det is not None:
        try:
            torch.use_deterministic_algorithms(bool(use_det))
        except Exception:
            # Some environments disallow full determinism. Record this in receipts.
            pass
