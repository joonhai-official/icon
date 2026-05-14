# tests/property/test_saturation.py
"""Property test: InfoNCE saturation (Section 6, A.4).

These are slow tests (run real critic training); marked accordingly.
"""

import math

import pytest
import torch

from icon.core.infonce import InfoNCEConfig, estimate_mi, permutation_null


@pytest.mark.slow
def test_infonce_never_exceeds_log_batch():
    """Î_NCE ≤ log B by the structural saturation ceiling."""
    config = InfoNCEConfig(batch_size=32, n_steps=300)
    torch.manual_seed(0)

    # Perfect dependence: A = B.
    a = torch.randn(256, 8)
    mi = estimate_mi(a, a.clone(), config, seed=0)

    assert mi <= math.log(32) + 1e-3


@pytest.mark.slow
def test_independent_variables_give_low_mi():
    """Truly independent variables — estimator returns a finite-sample floor.

    Note: the floor is positive even at zero true MI (Section 6). We test
    that the floor stays well below the saturation ceiling — not that it
    is zero.
    """
    config = InfoNCEConfig(batch_size=64, n_steps=500)
    torch.manual_seed(0)

    a = torch.randn(512, 8)
    b = torch.randn(512, 8)  # independent
    mi = estimate_mi(a, b, config, seed=0)

    # The floor is real but should not saturate.
    assert mi < math.log(64)  # below ceiling
    assert mi >= 0  # bounded below
