"""Post-training quantization (PTQ) utilities for Icon_primitive.

This module provides a lightweight *fake-quant* implementation designed
for protocol-locked experiments (v1.1). It is not intended to be a
production-grade quantization library.

Supported precisions:
  - fp16 / bf16: weight casting
  - int8 / int4: symmetric fake-quant (weights per-channel, activations per-tensor)

Calibration:
  - fixed indices from the training set (see data.make_calib_subset)
  - observer: minmax or percentile (for int4 default percentile=99.9)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _qrange(bits: int) -> Tuple[int, int]:
    if bits == 8:
        return -128, 127
    if bits == 4:
        return -8, 7
    raise ValueError(f"Unsupported bits: {bits}")


def _fake_quant(x: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    # symmetric quant, dequant back to float
    scale = torch.clamp(scale, min=1e-12)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q * scale


@dataclass
class PTQConfig:
    bits: int
    weight_scheme: str = "per_channel_symmetric"
    activation_scheme: str = "per_tensor_symmetric"
    observer: str = "minmax"  # minmax | percentile
    percentile: float = 99.9


class FakeQuantLinear(nn.Module):
    """Linear layer with symmetric fake-quant for weights and activations."""

    def __init__(self, base: nn.Linear, cfg: PTQConfig):
        super().__init__()
        if base.bias is not None:
            # Keep bias in fp32.
            self.bias = nn.Parameter(base.bias.detach().clone())
        else:
            self.bias = None
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.cfg = cfg
        self.weight_fp = nn.Parameter(base.weight.detach().clone())

        # Calibrated activation scale (scalar).
        self.act_scale: Optional[torch.Tensor] = None
        self._calib_samples: list[np.ndarray] = []
        self.calibrating: bool = False

        # Precompute weight scales per output channel.
        qmin, qmax = _qrange(cfg.bits)
        with torch.no_grad():
            # per-channel: scale[c] = max_abs(w[c,:]) / qmax
            max_abs = torch.amax(torch.abs(self.weight_fp), dim=1)
            self.weight_scale = (max_abs / float(qmax)).clamp(min=1e-12)

        self.qmin = qmin
        self.qmax = qmax

    def begin_calibration(self) -> None:
        self.calibrating = True
        self._calib_samples = []

    def end_calibration(self) -> Dict[str, float]:
        self.calibrating = False
        if not self._calib_samples:
            # Fallback: set a conservative scale.
            self.act_scale = torch.tensor(1.0)
            return {"act_scale": 1.0, "observer": "fallback"}
        arr = np.concatenate(self._calib_samples, axis=0)
        if self.cfg.observer == "percentile":
            bound = float(np.percentile(arr, self.cfg.percentile))
        else:
            bound = float(np.max(arr))
        bound = max(bound, 1e-6)
        self.act_scale = torch.tensor(bound / float(self.qmax))
        return {"act_scale": float(self.act_scale.item()), "observer": self.cfg.observer, "bound": bound}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collect calibration stats on input activation.
        if self.calibrating:
            # Sample to keep memory bounded.
            a = torch.abs(x.detach()).flatten().cpu().numpy()
            if a.size > 8192:
                # IMPORTANT: keep this deterministic.
                # Using np.random.choice here would introduce unseeded
                # nondeterminism into PTQ calibration.
                step = max(1, a.size // 8192)
                a = a[::step][:8192]
            self._calib_samples.append(a.astype(np.float64))

        # Fake-quant activation (per tensor)
        if self.act_scale is None:
            # Before calibration ends, treat scale as max abs in this batch
            scale = torch.amax(torch.abs(x), dim=None) / float(self.qmax)
        else:
            scale = self.act_scale.to(x.device)
        xq = _fake_quant(x, scale, self.qmin, self.qmax)

        # Fake-quant weights (per channel)
        w_scale = self.weight_scale.to(x.device).unsqueeze(1)
        wq = _fake_quant(self.weight_fp, w_scale, self.qmin, self.qmax)
        y = torch.matmul(xq, wq.t())
        if self.bias is not None:
            y = y + self.bias
        return y


def cast_model_precision(model: nn.Module, precision: str) -> nn.Module:
    """Cast model weights for fp16/bf16 evaluation."""
    p = precision.lower()
    if p == "fp16":
        return model.half()
    if p == "bf16":
        return model.to(dtype=torch.bfloat16)
    if p == "fp32":
        return model.float()
    raise ValueError(f"Unsupported cast precision: {precision}")


def apply_ptq_to_vector_probe(model: nn.Module, cfg: PTQConfig) -> Tuple[nn.Module, Dict[str, Dict[str, float]]]:
    """Replace key Linear layers in a VectorProbe with FakeQuantLinear."""
    stats: Dict[str, Dict[str, float]] = {}

    # We avoid touching the frozen stem.
    if hasattr(model, "linear") and isinstance(model.linear, nn.Linear):
        model.linear = FakeQuantLinear(model.linear, cfg)
        stats["linear"] = {}
    if hasattr(model, "concat_proj") and isinstance(model.concat_proj, nn.Linear):
        model.concat_proj = FakeQuantLinear(model.concat_proj, cfg)
        stats["concat_proj"] = {}
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = FakeQuantLinear(model.head, cfg)
        stats["head"] = {}
    return model, stats


def run_ptq_calibration(model: nn.Module, calib_loader, device: torch.device) -> Dict[str, Dict[str, float]]:
    """Run calibration pass and return observer stats per layer."""
    model.to(device)
    model.eval()

    qlayers: Dict[str, FakeQuantLinear] = {}
    for name in ["linear", "concat_proj", "head"]:
        m = getattr(model, name, None)
        if isinstance(m, FakeQuantLinear):
            qlayers[name] = m
            m.begin_calibration()

    with torch.no_grad():
        for xb, _ in calib_loader:
            xb = xb.to(device)
            _ = model(xb)

    stats: Dict[str, Dict[str, float]] = {}
    for name, m in qlayers.items():
        stats[name] = m.end_calibration()
    return stats
