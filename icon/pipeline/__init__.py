# icon/pipeline/__init__.py
"""The measurement pipeline (Part 3 of the specification).

This package orchestrates the components from `icon.core` and
`icon.measurements` into the nine-step pipeline of Section 9,
plus the four aggregation primitives of Section 10 and the
cross-system comparison logic of Section 11.
"""

from icon.pipeline.measure import measure, measure_layers, Profile
from icon.pipeline.aggregate import (
    static,
    spatial,
    temporal,
    shift,
    SpatialView,
    TemporalView,
    Shift,
)
from icon.pipeline.compare import compare_profiles

__all__ = [
    "measure",
    "measure_layers",
    "Profile",
    "static",
    "spatial",
    "temporal",
    "shift",
    "SpatialView",
    "TemporalView",
    "Shift",
    "compare_profiles",
]
