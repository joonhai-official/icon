# tests/unit/test_manifest.py
"""Unit tests for icon.io.manifest — Section 13."""

import pytest

from icon.io.manifest import Manifest, check_manifest_compat, collect_environment
from icon.io.protocol import PROTOCOL


def _make_manifest():
    return Manifest(
        icon_version="0.1.0",
        protocol_version="1.0",
        system={"adapter_class": "Test", "layer_name": "h1", "layer_dim": 16},
        data={"loader_class": "Test", "num_classes": 3, "n_samples": 256},
        protocol=PROTOCOL().to_dict(),
        seeds={"master": 42, "data": 1, "noise": 2, "critic": 3, "permutation": 4},
        environment=collect_environment(),
        results={"f_in": 0.3, "f_task": 0.2},
    )


def test_manifest_required_fields():
    m = _make_manifest()
    d = m.to_dict()
    for required in ["icon_version", "protocol_version", "system", "data",
                     "protocol", "seeds", "environment", "results"]:
        assert required in d


def test_manifest_json_roundtrip():
    m = _make_manifest()
    js = m.to_json()
    m2 = Manifest.from_json(js)
    assert m.protocol_hash() == m2.protocol_hash()


def test_protocol_hash_deterministic():
    m1 = _make_manifest()
    m2 = _make_manifest()
    assert m1.protocol_hash() == m2.protocol_hash()


def test_check_manifest_compat_same_version():
    assert check_manifest_compat({"protocol_version": "1.0"}) is True
    assert check_manifest_compat({"protocol_version": "1.5"}) is True  # same MAJOR
    assert check_manifest_compat({"protocol_version": "2.0"}) is False  # different MAJOR
    assert check_manifest_compat({}) is False
