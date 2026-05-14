# icon/pipeline/compare.py
"""Comparing profiles across systems (Section 11).

Within-PROTOCOL comparison: two profiles measured under identical
PROTOCOL are directly comparable.

Cross-PROTOCOL comparison: requires calibration. The framework recommends
comparing η_t (invariant under per-dimension normalization) or using a
shared anchor system.
"""

from __future__ import annotations

from dataclasses import dataclass

from icon.pipeline.aggregate import Shift, shift
from icon.pipeline.measure import Profile


@dataclass
class Comparison:
    protocol_match: bool
    direct_shift: Shift | None
    eta_t_pair: tuple[float, float]
    eta_t_diff: float
    notes: list[str]


def compare_profiles(p1: Profile, p2: Profile) -> Comparison:
    """Compare two profiles, with explicit handling of PROTOCOL match."""
    hash1 = p1.manifest.protocol_hash()
    hash2 = p2.manifest.protocol_hash()
    match = hash1 == hash2

    notes: list[str] = []
    direct_shift: Shift | None = None

    if match:
        direct_shift = shift(p2, p1)
    else:
        notes.append(
            "PROTOCOLs differ; direct component comparison requires calibration. "
            "Reporting η_t only (invariant under per-dimension normalization)."
        )

    return Comparison(
        protocol_match=match,
        direct_shift=direct_shift,
        eta_t_pair=(p1.eta_t, p2.eta_t),
        eta_t_diff=p2.eta_t - p1.eta_t,
        notes=notes,
    )
