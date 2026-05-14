# icon/measurements/rho.py
"""ρ — representational dispersion via participation ratio.

ρ(L) = PR(Z_L) / d_L,   PR(Z) = (Σ λ_i)² / Σ λ_i²

where λ_i are eigenvalues of Cov(Z). PR ∈ [1, d_L]; ρ ∈ [1/d_L, 1].

ρ is computed on the un-noised representation Z (Section 4 exempts ρ
from the noise channel). Sample-count requirement: N ≥ 4 d_L (Section 8).
"""

from __future__ import annotations

import torch


def participation_ratio(z: torch.Tensor, eigenvalue_floor: float = 1e-12) -> float:
    # Compute covariance in float64 for stability (Section 12.5).
    z64 = z.double()
    z_centered = z64 - z64.mean(dim=0, keepdim=True)
    n = z_centered.shape[0]
    cov = (z_centered.T @ z_centered) / (n - 1)

    eigenvalues = torch.linalg.eigvalsh(cov)
    # Numerical floor relative to the largest eigenvalue.
    floor = eigenvalue_floor * eigenvalues.max().clamp(min=1.0)
    eigenvalues = eigenvalues.clamp(min=floor)

    s1 = eigenvalues.sum()
    s2 = (eigenvalues ** 2).sum()
    return float((s1 * s1 / s2).item())


def compute_rho(
    z: torch.Tensor,
    eigenvalue_floor: float = 1e-12,
) -> tuple[float, bool]:
    n, d = z.shape
    pr = participation_ratio(z, eigenvalue_floor)
    rho = pr / d
    low_sample = n < 4 * d
    return rho, low_sample
