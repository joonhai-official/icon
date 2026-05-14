# icon/core/noise.py
"""Goldfeld noise channel (Section 4).

For deterministic Z = f(X), I(X; Z) reduces to H(Z) and loses dependence
on f. Adding small isotropic Gaussian noise turns the problem into a
noise-channel problem with finite, transformation-dependent MI
(Goldfeld et al., 2019):

    Z̃ = Z + σ · RMS(Z) · ε,   ε ~ N(0, I)

RMS is computed as a per-batch global scalar so the signal-to-noise
ratio is comparable across systems with different signal magnitudes.
"""

from __future__ import annotations

import torch


def rms(z: torch.Tensor, floor: float = 1e-6) -> float:
    # RMS(Z) = sqrt(E[||Z||^2 / d_L]) over the batch.
    # `z` has shape [N, d_L].
    value = torch.sqrt((z ** 2).mean()).item()
    return max(value, floor)


def inject_noise(
    z: torch.Tensor,
    sigma: float,
    seed: int,
    rms_floor: float = 1e-6,
) -> torch.Tensor:
    s = rms(z, rms_floor)
    generator = torch.Generator(device=z.device).manual_seed(seed)
    noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
    return z + sigma * s * noise
