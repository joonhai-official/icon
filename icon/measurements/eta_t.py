# icon/measurements/eta_t.py
"""η_t — the canonical ratio (Section 2, A.7).

η_t = F_task / F_in = I(Y; Z̃) / I(X; Z̃)

The d_L cancels exactly. Empirically verified to machine precision
(Park, 2026a; max abs diff 6.15e-7 across n = 633 records).
"""

from __future__ import annotations

import math


def compute_eta_t(f_in: float, f_task: float) -> float:
    if f_in <= 0:
        return math.nan
    return f_task / f_in
