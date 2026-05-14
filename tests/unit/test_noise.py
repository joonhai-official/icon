# tests/unit/test_noise.py
"""Unit tests for icon.core.noise — Section 4."""

import pytest
import torch

from icon.core.noise import inject_noise, rms


def test_rms_basic():
    # Tensor with known per-element scale.
    z = torch.full((100, 50), 3.0)
    assert rms(z) == pytest.approx(3.0, rel=1e-5)


def test_rms_floor_applied():
    # All-zero tensor must be clamped to floor.
    z = torch.zeros(10, 5)
    assert rms(z, floor=1e-6) == pytest.approx(1e-6)


def test_noise_scales_to_signal():
    # Noise std should match σ * RMS(Z).
    torch.manual_seed(0)
    z = torch.randn(500, 32) * 10.0
    sigma = 0.1
    z_tilde = inject_noise(z, sigma=sigma, seed=42)
    noise_std = (z_tilde - z).std().item()
    expected = sigma * rms(z)
    assert noise_std == pytest.approx(expected, rel=0.05)


def test_noise_deterministic_under_seed():
    z = torch.randn(20, 10)
    z1 = inject_noise(z, sigma=0.1, seed=42)
    z2 = inject_noise(z, sigma=0.1, seed=42)
    assert torch.equal(z1, z2)


def test_noise_different_seeds_differ():
    z = torch.randn(20, 10)
    z1 = inject_noise(z, sigma=0.1, seed=42)
    z2 = inject_noise(z, sigma=0.1, seed=43)
    assert not torch.equal(z1, z2)
