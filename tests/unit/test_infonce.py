# tests/unit/test_infonce.py
"""Unit tests for icon.core.infonce — Appendix A.4 of the specification."""

import pytest


@pytest.mark.skip(reason="Phase 2: InfoNCE not yet implemented")
def test_infonce_returns_finite():
    """InfoNCE on synthetic data returns a finite estimate."""
    pass


@pytest.mark.skip(reason="Phase 2")
def test_infonce_bounded_by_log_batch():
    """Saturation ceiling: Î_NCE ≤ log B (Section 6)."""
    pass


@pytest.mark.skip(reason="Phase 2")
def test_permutation_null_positive():
    """Permutation null is positive (finite-sample bias floor)."""
    pass


@pytest.mark.skip(reason="Phase 2")
def test_critic_separable_cosine():
    """Critic is the separable cosine form specified in §A.4."""
    pass
