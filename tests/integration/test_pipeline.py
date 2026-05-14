# tests/integration/test_pipeline.py
"""Integration test: end-to-end pipeline (Section 9)."""

import pytest
import torch

import icon


@pytest.fixture
def tiny_pipeline_setup():
    """A minimal adapter + loader for fast integration tests."""
    torch.manual_seed(0)
    N = 256
    D_IN = 16
    D_HID = 8
    N_CLASSES = 3

    X_all = torch.randn(N, D_IN)
    W = torch.randn(D_IN, N_CLASSES)
    Y_all = (X_all @ W).argmax(dim=1)
    W1 = torch.randn(D_IN, D_HID) * 0.3

    class FastAdapter(icon.AdapterBase):
        def layer_names(self):
            return ["h1"]
        def forward_with_taps(self, x, names):
            return {"h1": torch.relu(x @ W1)}
        def layer_dim(self, name):
            return D_HID

    class FastLoader(icon.DataLoaderBase):
        def train_batch(self, b, s):
            return X_all[:b], Y_all[:b]
        def val_batch(self, b, s):
            return X_all[:b], Y_all[:b]
        def num_classes(self):
            return N_CLASSES

    protocol = icon.PROTOCOL()
    protocol.infonce.batch_size = 32
    protocol.infonce.n_steps = 100  # short for speed
    protocol.pool.sample_count = N

    return FastAdapter(), FastLoader(), protocol


@pytest.mark.slow
def test_pipeline_runs_on_synthetic(tiny_pipeline_setup):
    adapter, loader, protocol = tiny_pipeline_setup
    profile = icon.measure(adapter, loader, "h1", protocol, master_seed=42)
    assert profile is not None
    assert isinstance(profile.f_in, float)


@pytest.mark.slow
def test_profile_has_all_components(tiny_pipeline_setup):
    adapter, loader, protocol = tiny_pipeline_setup
    profile = icon.measure(adapter, loader, "h1", protocol, master_seed=42)
    # Last layer has F_layer = None.
    assert profile.f_in is not None
    assert profile.f_task is not None
    assert profile.f_self is not None
    assert profile.rho is not None
    # Single layer adapter → f_layer is None
    assert profile.f_layer is None


@pytest.mark.slow
def test_pipeline_deterministic(tiny_pipeline_setup):
    """Same master_seed → bit-identical Profile."""
    adapter, loader, protocol = tiny_pipeline_setup
    p1 = icon.measure(adapter, loader, "h1", protocol, master_seed=42)
    p2 = icon.measure(adapter, loader, "h1", protocol, master_seed=42)
    assert p1.f_in == p2.f_in
    assert p1.f_task == p2.f_task
    assert p1.rho == p2.rho


@pytest.mark.slow
def test_manifest_roundtrip_preserves_hash(tiny_pipeline_setup):
    adapter, loader, protocol = tiny_pipeline_setup
    profile = icon.measure(adapter, loader, "h1", protocol, master_seed=42)

    js = profile.manifest.to_json()
    from icon.io.manifest import Manifest
    m_restored = Manifest.from_json(js)
    assert profile.manifest.protocol_hash() == m_restored.protocol_hash()
