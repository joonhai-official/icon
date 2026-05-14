# icon/measurements/__init__.py
"""The five measurements (Section 2 of the specification).

    F_in    = I(X; Z̃_L) / d_L         — input information density
    F_task  = I(Y; Z̃_L) / d_L         — task information density
    F_self  = I(Z_L; Z̃_L) / d_L       — self-consistency under noise
    F_layer = I(Z̃_L; Z̃_{L+1}) / d_{L+1}  — inter-layer transmission
    ρ       = PR(Z_L) / d_L           — representational dispersion

Plus the canonical ratio η_t = F_task / F_in, whose d_L cancels exactly
(Section A.7).

Each module is a thin wrapper around the core functions: it composes
noise injection, InfoNCE estimation, and dimension normalization to
produce the measurement.
"""

from icon.measurements.f_in import compute_f_in
from icon.measurements.f_task import compute_f_task
from icon.measurements.f_self import compute_f_self
from icon.measurements.f_layer import compute_f_layer
from icon.measurements.rho import compute_rho
from icon.measurements.eta_t import compute_eta_t

__all__ = [
    "compute_f_in",
    "compute_f_task",
    "compute_f_self",
    "compute_f_layer",
    "compute_rho",
    "compute_eta_t",
]
