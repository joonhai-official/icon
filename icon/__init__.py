# icon/__init__.py
"""Icon — A framework for measuring information flow.

The public surface follows Section 17 of the framework specification.
This module re-exports the small set of functions and classes that users
interact with directly; everything else is implementation detail.
"""

# Versioning (Section 14)
__version__ = "0.1.0"
PROTOCOL_VERSION = "1.0"

# Contracts (Section 16)
from icon.contracts.adapter import AdapterBase
from icon.contracts.loader import DataLoaderBase

# PROTOCOL declaration (Part 4)
from icon.io.protocol import PROTOCOL, protocols_match

# Measurement entry points (Section 17)
from icon.pipeline.measure import measure, measure_layers

# Aggregations (Section 10)
from icon.pipeline.aggregate import (
    static,
    spatial,
    temporal,
    shift,
    SpatialView,
    TemporalView,
    Shift,
)

# Trust τ inspection (Section 7)
from icon.core.trust import (
    classify_component,
    aggregate_trust,
    TrustClassification,
)

# Versioning helpers
from icon.io.manifest import check_manifest_compat

# Profile container
from icon.pipeline.measure import Profile

__all__ = [
    # Version
    "__version__",
    "PROTOCOL_VERSION",
    # Contracts
    "AdapterBase",
    "DataLoaderBase",
    # PROTOCOL
    "PROTOCOL",
    "protocols_match",
    # Measurement
    "measure",
    "measure_layers",
    "Profile",
    # Aggregations
    "static",
    "spatial",
    "temporal",
    "shift",
    "SpatialView",
    "TemporalView",
    "Shift",
    # Trust
    "classify_component",
    "aggregate_trust",
    "TrustClassification",
    # Manifest
    "check_manifest_compat",
]
