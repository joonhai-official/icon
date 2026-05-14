# tests/property/test_eta_t.py
"""Property test: η_t's d_L cancellation (Section A.7)."""

import math

import pytest
import torch

from icon.measurements.eta_t import compute_eta_t


def test_eta_t_basic():
    assert compute_eta_t(f_in=0.5, f_task=0.3) == pytest.approx(0.6)


def test_eta_t_zero_handling():
    assert math.isnan(compute_eta_t(f_in=0.0, f_task=0.3))


def test_eta_t_negative_f_in():
    # Defensive: negative F_in (estimator artifact) returns NaN.
    assert math.isnan(compute_eta_t(f_in=-0.1, f_task=0.3))


def test_eta_t_d_L_cancellation_exact():
    """The d_L in F_task/F_in cancels exactly.

    The empirical companion (Park, 2026a) verified at 6.15e-7 across n=633.
    Here we verify the algebraic identity directly.
    """
    # Simulate: I(Y;Z̃) = some value, I(X;Z̃) = some value, divide by d_L
    I_Y = 0.7345
    I_X = 1.2891
    d_L = 256

    f_task = I_Y / d_L
    f_in = I_X / d_L

    eta_from_F = compute_eta_t(f_in, f_task)
    eta_from_raw = I_Y / I_X

    assert abs(eta_from_F - eta_from_raw) < 1e-12
