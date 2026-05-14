# icon/pipeline/aggregate.py
"""Four aggregation primitives (Section 10).

    Static   = P(L_0, t_0)              — single snapshot (identity)
    Spatial  = {P(L, t_0) : L}          — sequence over layers
    Temporal = {P(L_0, t) : t}          — sequence over time
    Shift    = P_b - P_a                — component-wise difference

Trust τ propagation is conservative (Section A.9): aggregated quantities
are no more trustworthy than their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from icon.core.trust import TrustClassification, aggregate_trust
from icon.pipeline.measure import Profile


@dataclass
class SpatialView:
    """A sequence of profiles at fixed time, indexed by layer."""

    profiles: list[Profile]
    time: int | float = 0


@dataclass
class TemporalView:
    """A sequence of profiles at fixed layer, indexed by time."""

    profiles: list[Profile]
    layer_name: str = ""


@dataclass
class Shift:
    """Component-wise difference between two profiles."""

    delta_f_in: float
    delta_f_task: float
    delta_f_self: float
    delta_f_layer: float | None
    delta_rho: float
    delta_eta_t: float
    trust_aggregate: TrustClassification


def static(profile: Profile) -> Profile:
    """Static aggregation — identity."""
    return profile


def spatial(profiles: list[Profile], time: int | float = 0) -> SpatialView:
    return SpatialView(profiles=list(profiles), time=time)


def temporal(profiles: list[Profile], layer_name: str = "") -> TemporalView:
    return TemporalView(profiles=list(profiles), layer_name=layer_name)


def shift(profile_b: Profile, profile_a: Profile) -> Shift:
    """Δ P = P_b - P_a. Trust τ is the conservative aggregate of both."""
    delta_f_layer = None
    if profile_b.f_layer is not None and profile_a.f_layer is not None:
        delta_f_layer = profile_b.f_layer - profile_a.f_layer

    return Shift(
        delta_f_in=profile_b.f_in - profile_a.f_in,
        delta_f_task=profile_b.f_task - profile_a.f_task,
        delta_f_self=profile_b.f_self - profile_a.f_self,
        delta_f_layer=delta_f_layer,
        delta_rho=profile_b.rho - profile_a.rho,
        delta_eta_t=profile_b.eta_t - profile_a.eta_t,
        trust_aggregate=aggregate_trust([profile_b.trust_aggregate, profile_a.trust_aggregate]),
    )
