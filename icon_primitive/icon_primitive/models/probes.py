"""Probe networks used to measure primitive constants.

Design goal: make *only* the primitive change across runs.

VectorProbe:
  FrozenStem(E0) -> Trainable block (Linear -> Norm -> Act -> Skip) -> Head

SpatialProbe:
  FrozenStem(S0) -> Trainable op (Dense/Conv) -> Act -> Head

Both probes support capturing taps required for kappa measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn

from .primitives import get_activation, get_normalization
from .stems import create_vector_stem, create_spatial_stem, orthogonal_init


@dataclass
class ProbeConfig:
    width: int = 256
    num_classes: int = 10
    activation: str = "relu"
    normalization: str = "none"
    precision: str = "fp32"
    skip: str = "none"  # none|residual|dense_concat
    skip_scale: str = "none"  # none|variance_preserving
    concat_projection: str = "none"  # none|project_to_width
    linear_type: str = "dense"  # dense|conv1x1|conv3x3|depthwise (SpatialProbe)
    linear_budget: str = "shape_matched"  # shape_matched|param_matched|flops_matched


class FrozenStem(nn.Module):
    def __init__(self, stem: nn.Module):
        super().__init__()
        self.stem = stem
        for p in self.stem.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.stem(x)


class VectorProbe(nn.Module):
    """Vector probe (E0 frozen)."""

    def __init__(self, stem: nn.Module, cfg: ProbeConfig, *, init_seed: int):
        super().__init__()
        self.cfg = cfg
        self.stem = FrozenStem(stem)

        self.linear = nn.Linear(cfg.width, cfg.width)
        orthogonal_init(self.linear, init_seed)

        # Always store a module for normalization. For "none" this is nn.Identity().
        self.norm = get_normalization(cfg.normalization, dim=cfg.width)
        self.act = get_activation(cfg.activation)

        # concat changes dimension unless projected.
        head_in = cfg.width
        if cfg.skip == "dense_concat" and cfg.concat_projection == "none":
            head_in = cfg.width * 2
        self.head = nn.Linear(head_in, cfg.num_classes)
        orthogonal_init(self.head, init_seed + 1)

        self.concat_proj: Optional[nn.Linear] = None
        if cfg.skip == "dense_concat" and cfg.concat_projection == "project_to_width":
            self.concat_proj = nn.Linear(cfg.width * 2, cfg.width)
            orthogonal_init(self.concat_proj, init_seed + 2)

        self._taps: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, *, capture: bool = False, tap: str = "post_block") -> torch.Tensor:
        X = self.stem(x)

        # Match dtype to block params (fix fp16/bf16 matmul mismatch)

        X = X.to(dtype=self.linear.weight.dtype)
        # Match dtype to block params (fix fp16/bf16 matmul mismatch)
        X = X.to(dtype=self.linear.weight.dtype)
        # Match dtype to block params (fix fp16/bf16 matmul mismatch)
        X = X.to(dtype=self.linear.weight.dtype)
        pre_norm = self.linear(X)
        post_norm = self.norm(pre_norm)
        post_act = self.act(post_norm)

        # skip
        if self.cfg.skip == "none":
            post_skip = post_act
        elif self.cfg.skip == "residual":
            scale = 1.0
            if self.cfg.skip_scale == "variance_preserving":
                scale = 1.0 / math.sqrt(2.0)
            post_skip = (X + post_act) * scale
        elif self.cfg.skip == "dense_concat":
            cat = torch.cat([X, post_act], dim=-1)
            post_skip = self.concat_proj(cat) if self.concat_proj is not None else cat
        else:
            raise ValueError(f"Unknown skip: {self.cfg.skip}")

        if capture:
            self._taps = {
                "X": X.detach(),
                "pre_norm": pre_norm.detach(),
                "post_norm": post_norm.detach(),
                "post_activation": post_act.detach(),
                "post_skip": post_skip.detach(),
                "post_block": post_skip.detach(),
            }

        z_for_head = post_skip
        return self.head(z_for_head)

    def get_tap(self, name: str) -> torch.Tensor:
        if name not in self._taps:
            raise KeyError(f"Tap not captured: {name}. Call forward(..., capture=True) first.")
        return self._taps[name]

    def get_XZ_for_kappa(self, tap: str = "post_block") -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_tap("X"), self.get_tap(tap)


class DenseOp(nn.Module):
    """Dense op: flatten -> Linear -> reshape."""

    def __init__(self, c_in: int, c_out: int, h: int, w: int, *, init_seed: int):
        super().__init__()
        self.c_out, self.h, self.w = c_out, h, w
        self.linear = nn.Linear(c_in * h * w, c_out * h * w)
        orthogonal_init(self.linear, init_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        y = self.linear(x.view(b, -1))
        return y.view(b, self.c_out, self.h, self.w)


class SpatialProbe(nn.Module):
    """Spatial probe (S0 frozen)."""

    def __init__(self, stem: nn.Module, cfg: ProbeConfig, spatial_cfg: Dict[str, int], *, init_seed: int):
        super().__init__()
        self.cfg = cfg
        self.spatial_cfg = spatial_cfg
        self.stem = FrozenStem(stem)

        c_in, c_out, h, w = spatial_cfg["c_in"], spatial_cfg["c_out"], spatial_cfg["h"], spatial_cfg["w"]
        if cfg.linear_type == "dense":
            self.op = DenseOp(c_in, c_out, h, w, init_seed=init_seed)
        elif cfg.linear_type == "conv1x1":
            self.op = nn.Conv2d(c_in, c_out, 1)
            orthogonal_init(self.op, init_seed)
        elif cfg.linear_type == "conv3x3":
            self.op = nn.Conv2d(c_in, c_out, 3, padding=1)
            orthogonal_init(self.op, init_seed)
        elif cfg.linear_type == "depthwise":
            # depthwise expects groups=c_in and c_out=c_in
            self.op = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in)
            orthogonal_init(self.op, init_seed)
        else:
            raise ValueError(f"Unknown linear_type: {cfg.linear_type}")

        # Normalization/skip are supported for independence extensions.
        self.norm = get_normalization(cfg.normalization, dim=c_out, spatial=True)
        self.act = get_activation(cfg.activation)

        # Skip wiring
        self.concat_proj: Optional[nn.Module] = None
        head_in_channels = c_out
        if cfg.skip == "dense_concat":
            head_in_channels = c_in + c_out
            if cfg.concat_projection == "project_to_width":
                # Project channels back to c_out using a fixed 1x1 conv.
                self.concat_proj = nn.Conv2d(head_in_channels, c_out, kernel_size=1)
                orthogonal_init(self.concat_proj, init_seed + 2)
                head_in_channels = c_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(head_in_channels, cfg.num_classes)
        orthogonal_init(self.head, init_seed + 1)

        self._taps: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, *, capture: bool = False, tap: str = "post_block") -> torch.Tensor:
        X = self.stem(x)
        # Match dtype to op params (fix fp16/bf16 matmul mismatch)
        X = X.to(dtype=next(self.op.parameters()).dtype)
        # Match dtype to op params (fix fp16/bf16 matmul mismatch)
        X = X.to(dtype=next(self.op.parameters()).dtype)
        pre_norm = self.op(X)
        post_norm = self.norm(pre_norm)
        post_act = self.act(post_norm)

        # Skip
        if self.cfg.skip == "none":
            post_skip = post_act
        elif self.cfg.skip == "residual":
            # Require shape match
            if X.shape != post_act.shape:
                raise ValueError(
                    f"Residual skip requires matching shapes, got X={tuple(X.shape)} vs f(X)={tuple(post_act.shape)}"
                )
            scale = 1.0
            if getattr(self.cfg, "skip_scale", "none") == "variance_preserving":
                scale = 1.0 / math.sqrt(2.0)
            post_skip = (X + post_act) * scale
        elif self.cfg.skip == "dense_concat":
            cat = torch.cat([X, post_act], dim=1)
            post_skip = self.concat_proj(cat) if self.concat_proj is not None else cat
        else:
            raise ValueError(f"Unknown skip: {self.cfg.skip}")

        if capture:
            self._taps = {
                "X": X.detach(),
                "pre_norm": pre_norm.detach(),
                "post_norm": post_norm.detach(),
                "post_activation": post_act.detach(),
                "post_skip": post_skip.detach(),
                "post_block": post_skip.detach(),
            }

        z = self.gap(post_skip).squeeze(-1).squeeze(-1)
        return self.head(z)

    def get_tap(self, name: str) -> torch.Tensor:
        if name not in self._taps:
            raise KeyError(f"Tap not captured: {name}. Call forward(..., capture=True) first.")
        return self._taps[name]

    def get_XZ_for_kappa(self, tap: str = "post_block") -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_tap("X"), self.get_tap(tap)


def create_vector_probe(dataset: str, cfg: ProbeConfig, *, stem_seed: int, init_seed: int) -> VectorProbe:
    stem = create_vector_stem(dataset, output_dim=cfg.width, seed=stem_seed)
    return VectorProbe(stem, cfg, init_seed=init_seed)


def create_spatial_probe(
    dataset: str,
    cfg: ProbeConfig,
    spatial_cfg: Dict[str, int],
    *,
    stem_seed: int,
    init_seed: int,
) -> SpatialProbe:
    stem = create_spatial_stem(dataset, seed=stem_seed, out_channels=spatial_cfg["c_in"])
    return SpatialProbe(stem, cfg, spatial_cfg, init_seed=init_seed)
