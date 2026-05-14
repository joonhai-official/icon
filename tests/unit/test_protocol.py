# tests/unit/test_protocol.py
"""Unit tests for icon.io.protocol — Part 4."""

import pytest

import icon
from icon.io.protocol import PROTOCOL


def test_protocol_defaults():
    p = PROTOCOL()
    assert p.noise.sigma == 0.1
    assert p.infonce.batch_size == 512
    assert p.trust_tau.r_sat == 0.95
    assert p.statistics.fdr_level == 0.05
    assert p.protocol_version == "1.0"


def test_protocol_roundtrip():
    p1 = PROTOCOL()
    p1.noise.sigma = 0.05
    p2 = PROTOCOL.from_dict(p1.to_dict())
    assert p1.hash() == p2.hash()


def test_protocol_hash_deterministic():
    h1 = PROTOCOL().hash()
    h2 = PROTOCOL().hash()
    assert h1 == h2


def test_protocols_match_identical():
    assert icon.protocols_match(PROTOCOL(), PROTOCOL())


def test_protocols_dont_match_when_different():
    p = PROTOCOL()
    p.noise.sigma = 0.05
    assert not icon.protocols_match(PROTOCOL(), p)


def test_critic_hidden_tuple_preserved_through_roundtrip():
    p1 = PROTOCOL()
    p1.infonce.critic_hidden = (128, 128)
    p2 = PROTOCOL.from_dict(p1.to_dict())
    assert p2.infonce.critic_hidden == (128, 128)
