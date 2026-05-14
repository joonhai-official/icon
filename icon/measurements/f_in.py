# icon/measurements/f_in.py
"""F_in — input information density.

    F_in(L) = I(X; Z̃_L) / d_L
"""

from __future__ import annotations

import torch

from icon.core.infonce import InfoNCEConfig, estimate_mi, permutation_null
from icon.core.trust import f_max


def compute_f_in(
    x: torch.Tensor,
    z_tilde: torch.Tensor,
    config: InfoNCEConfig,
    critic_seed: int,
    permutation_seed: int,
) -> tuple[float, float, float]:
    """Return (F_in, F_perm, F_max)."""
    d_L = z_tilde.shape[1]

    mi = estimate_mi(x, z_tilde, config, critic_seed)
    perm = permutation_null(x, z_tilde, config, permutation_seed)
    return mi / d_L, perm / d_L, f_max(config.batch_size, d_L)
