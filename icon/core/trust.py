# icon/core/trust.py
"""Trust τ classification (Section 7).

Three states for each F component:
    valid     — measurement reflects the actual relationship.
    saturated — at or near the estimator's ceiling (log B / d_L).
    invalid   — indistinguishable from the permutation floor.

Aggregate rule (Section 7): conservative.
    invalid   if any component is invalid;
    saturated if any saturated and none invalid;
    valid     if all four components are valid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class TrustClassification(str, Enum):
    VALID = "valid"
    SATURATED = "saturated"
    INVALID = "invalid"


@dataclass(frozen=True)
class TrustThresholds:
    r_sat: float = 0.95
    epsilon_sep: float = 0.005
    r_rel: float = 3.0
    r_abs: float = 0.20


def f_max(batch_size: int, dim: int) -> float:
    return math.log(batch_size) / dim


def classify_component(
    f: float,
    f_perm: float,
    f_max_value: float,
    thresholds: TrustThresholds | None = None,
) -> TrustClassification:
    t = thresholds or TrustThresholds()

    # Rules apply in order (Section 7).
    if f >= t.r_sat * f_max_value:
        return TrustClassification.SATURATED
    if abs(f - f_perm) < t.epsilon_sep or f < t.r_rel * f_perm:
        return TrustClassification.INVALID
    if f >= t.r_abs * f_max_value:
        return TrustClassification.VALID
    return TrustClassification.INVALID


def aggregate_trust(
    component_classifications: list[TrustClassification],
) -> TrustClassification:
    # Conservative: invalid wins over saturated wins over valid.
    if TrustClassification.INVALID in component_classifications:
        return TrustClassification.INVALID
    if TrustClassification.SATURATED in component_classifications:
        return TrustClassification.SATURATED
    return TrustClassification.VALID
