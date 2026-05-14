# tests/unit/test_rho.py
"""Unit tests for icon.measurements.rho — Sections 2, 8."""

import pytest
import torch

from icon.measurements.rho import compute_rho, participation_ratio


def test_rho_in_range():
    # ρ must always be in [1/d, 1].
    torch.manual_seed(0)
    z = torch.randn(200, 16)
    rho, _ = compute_rho(z)
    assert 1.0 / 16 - 1e-6 <= rho <= 1.0 + 1e-6


def test_rho_rank_one_collapse():
    # All variance on one axis → ρ ≈ 1/d.
    z = torch.zeros(200, 10)
    z[:, 0] = torch.randn(200)
    rho, _ = compute_rho(z)
    assert rho == pytest.approx(0.1, abs=0.02)


def test_rho_uniform_variance():
    # Isotropic Gaussian → ρ close to 1 (eigenvalues nearly equal).
    torch.manual_seed(0)
    z = torch.randn(2000, 10)  # large N for stable estimate
    rho, _ = compute_rho(z)
    assert rho > 0.8


def test_rho_low_sample_flag():
    # N < 4d should flag the result.
    z = torch.randn(20, 16)  # N=20 < 4*16=64
    _, low_sample = compute_rho(z)
    assert low_sample is True


def test_rho_sufficient_sample_no_flag():
    z = torch.randn(200, 16)  # N=200 > 4*16=64
    _, low_sample = compute_rho(z)
    assert low_sample is False


def test_participation_ratio_bounds():
    # PR ∈ [1, d].
    torch.manual_seed(0)
    z = torch.randn(500, 10)
    pr = participation_ratio(z)
    assert 1.0 <= pr <= 10.0 + 1e-6
