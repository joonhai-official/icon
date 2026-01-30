"""ICON-Primitive model primitives.

This module defines the protocol-locked primitives used in the experiments:
  - activation
  - normalization
  - (skip connection helpers exist for reference, but the runner uses
    explicit wiring in the Probe modules)

Reproducibility notes
--------------------
- `get_normalization("none", ...)` returns `nn.Identity()` (never `None`).
- The API accepts both `dim=` and `num_features=` keywords for compatibility.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Activations
# ============================================================
class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))

class Identity(nn.Module):
    def forward(self, x): return x

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh, 
               "sigmoid": nn.Sigmoid, "mish": Mish, "identity": Identity}

def get_activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}")
    return ACTIVATIONS[name]()

# ============================================================
# Normalizations
# ============================================================
class RMSNorm(nn.Module):
    """RMSNorm for vector tensors shaped [B, D]."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize over last dim
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SpatialLayerNorm(nn.Module):
    """LayerNorm over channels for tensors shaped [B, C, H, W]."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        c = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        y = (x - mean) / torch.sqrt(var + self.eps)
        return y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SpatialRMSNorm(nn.Module):
    """RMSNorm over channels for tensors shaped [B, C, H, W]."""

    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        c = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)
        return x / rms * self.weight.view(1, -1, 1, 1)


def get_normalization(
    name: str,
    dim: int | None = None,
    num_features: int | None = None,
    *,
    spatial: bool = False,
    num_groups: int = 32,
    eps: float = 1e-5,
    **_: object,
) -> nn.Module:
    """Factory for normalization modules.

    Args:
        name: one of none|layernorm|rmsnorm|batchnorm|groupnorm
        dim / num_features: feature dimension (alias keywords)
        spatial: if True, create a normalization suitable for [B,C,H,W]
        num_groups: only for groupnorm
        eps: numerical stability
    """
    n = dim if dim is not None else num_features
    if n is None:
        raise TypeError("get_normalization requires dim= or num_features=")
    n = int(n)

    key = str(name).lower()
    if key == "none":
        return nn.Identity()

    if not spatial:
        if key == "layernorm":
            return nn.LayerNorm(n, eps=float(eps))
        if key == "rmsnorm":
            return RMSNorm(n, eps=float(eps))
        if key == "batchnorm":
            return nn.BatchNorm1d(n)
        if key == "groupnorm":
            ng = int(num_groups)
            if ng <= 0:
                ng = 1
            if n % ng != 0:
                # deterministic fallback heuristic
                ng = max(1, n // 8)
            return nn.GroupNorm(ng, n)

    # spatial
    if key == "layernorm":
        return SpatialLayerNorm(n, eps=float(eps))
    if key == "rmsnorm":
        return SpatialRMSNorm(n, eps=float(eps))
    if key == "batchnorm":
        return nn.BatchNorm2d(n)
    if key == "groupnorm":
        ng = int(num_groups)
        if ng <= 0:
            ng = 1
        if n % ng != 0:
            ng = max(1, n // 8)
        return nn.GroupNorm(ng, n)

    raise ValueError(f"Unknown normalization: {name}")

# ============================================================
# Skip Connections
# ============================================================
class ResidualSkip(nn.Module):
    def __init__(self, block: nn.Module, variance_preserving: bool = True):
        super().__init__()
        self.block, self.scale = block, 1/math.sqrt(2) if variance_preserving else 1.0
    def forward(self, x): return (x + self.block(x)) * self.scale

class DenseConcatSkip(nn.Module):
    def __init__(self, block: nn.Module, d_in: int, project: bool = False):
        super().__init__()
        self.block = block
        self.proj = nn.Linear(d_in*2, d_in) if project else None
    def forward(self, x):
        out = torch.cat([x, self.block(x)], dim=-1)
        return self.proj(out) if self.proj else out

def get_skip_connection(name: str, block: nn.Module, d_in: int = 256, 
                        variance_preserving: bool = True, project: bool = False) -> nn.Module:
    if name == "none": return block
    if name == "residual": return ResidualSkip(block, variance_preserving)
    if name == "dense_concat": return DenseConcatSkip(block, d_in, project)
    raise ValueError(f"Unknown skip: {name}")
