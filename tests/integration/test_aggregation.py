# tests/integration/test_aggregation.py
"""Integration test: aggregations (Section 10) on real profiles."""

import pytest
import torch

import icon


@pytest.fixture
def two_profiles():
    """Build two profiles for aggregation tests."""
    torch.manual_seed(0)
    N = 256
    D_IN = 16
    D_HID = 8

    X_all = torch.randn(N, D_IN)
    Y_all = torch.randint(0, 3, (N,))
    W1 = torch.randn(D_IN, D_HID) * 0.3

    class A(icon.AdapterBase):
        def layer_names(self): return ["h1"]
        def forward_with_taps(self, x, names): return {"h1": torch.relu(x @ W1)}
        def layer_dim(self, n): return D_HID

    class L(icon.DataLoaderBase):
        def train_batch(self, b, s): return X_all[:b], Y_all[:b]
        def val_batch(self, b, s): return X_all[:b], Y_all[:b]
        def num_classes(self): return 3

    protocol = icon.PROTOCOL()
    protocol.infonce.batch_size = 32
    protocol.infonce.n_steps = 100
    protocol.pool.sample_count = N

    p1 = icon.measure(A(), L(), "h1", protocol, master_seed=42)
    p2 = icon.measure(A(), L(), "h1", protocol, master_seed=43)
    return p1, p2


@pytest.mark.slow
def test_shift_basic(two_profiles):
    p1, p2 = two_profiles
    sh = icon.shift(p2, p1)
    # Delta is component-wise difference.
    assert sh.delta_f_in == pytest.approx(p2.f_in - p1.f_in)
    assert sh.delta_f_task == pytest.approx(p2.f_task - p1.f_task)


@pytest.mark.slow
def test_spatial_collects_profiles(two_profiles):
    p1, p2 = two_profiles
    sv = icon.spatial([p1, p2], time=0)
    assert len(sv.profiles) == 2


@pytest.mark.slow
def test_temporal_collects_profiles(two_profiles):
    p1, p2 = two_profiles
    tv = icon.temporal([p1, p2], layer_name="h1")
    assert len(tv.profiles) == 2
    assert tv.layer_name == "h1"


@pytest.mark.slow
def test_shift_trust_is_conservative(two_profiles):
    p1, p2 = two_profiles
    sh = icon.shift(p2, p1)
    # Trust τ of shift is at least as conservative as the worse of the two.
    from icon.core.trust import TrustClassification
    expected_at_least_as_strict = {
        TrustClassification.INVALID: [TrustClassification.INVALID],
        TrustClassification.SATURATED: [TrustClassification.SATURATED, TrustClassification.INVALID],
        TrustClassification.VALID: [
            TrustClassification.VALID,
            TrustClassification.SATURATED,
            TrustClassification.INVALID,
        ],
    }
    assert sh.trust_aggregate in expected_at_least_as_strict[p1.trust_aggregate]
