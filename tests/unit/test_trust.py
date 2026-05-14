# tests/unit/test_trust.py
"""Unit tests for icon.core.trust — Section 7."""

import math

import pytest

from icon.core.trust import (
    TrustClassification,
    TrustThresholds,
    aggregate_trust,
    classify_component,
    f_max,
)


def test_f_max_formula():
    # F_max = log(B) / d_L
    assert f_max(512, 10) == pytest.approx(math.log(512) / 10)


def test_classify_valid():
    # Well above r_abs * F_max, far from F_perm.
    fmax = 0.6
    c = classify_component(f=0.3, f_perm=0.05, f_max_value=fmax)
    assert c == TrustClassification.VALID


def test_classify_saturated():
    fmax = 0.6
    c = classify_component(f=0.59, f_perm=0.05, f_max_value=fmax)
    assert c == TrustClassification.SATURATED


def test_classify_invalid_near_perm():
    fmax = 0.6
    # Within epsilon_sep of perm.
    c = classify_component(f=0.052, f_perm=0.05, f_max_value=fmax)
    assert c == TrustClassification.INVALID


def test_classify_invalid_below_r_rel():
    fmax = 0.6
    # f < 3 * f_perm.
    c = classify_component(f=0.10, f_perm=0.05, f_max_value=fmax)
    assert c == TrustClassification.INVALID


def test_aggregate_invalid_wins():
    cls = [TrustClassification.VALID, TrustClassification.SATURATED, TrustClassification.INVALID]
    assert aggregate_trust(cls) == TrustClassification.INVALID


def test_aggregate_saturated_wins_over_valid():
    cls = [TrustClassification.VALID, TrustClassification.SATURATED, TrustClassification.VALID]
    assert aggregate_trust(cls) == TrustClassification.SATURATED


def test_aggregate_all_valid():
    cls = [TrustClassification.VALID] * 4
    assert aggregate_trust(cls) == TrustClassification.VALID


def test_custom_thresholds():
    # User-declared thresholds override defaults.
    fmax = 0.6
    custom = TrustThresholds(r_sat=0.8)  # lower saturation bar
    c = classify_component(f=0.50, f_perm=0.05, f_max_value=fmax, thresholds=custom)
    assert c == TrustClassification.SATURATED
