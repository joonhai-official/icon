# icon/measurements/f_layer.py
"""F_layer — inter-layer transmission.

    F_layer(L) = I(Z̃_L; Z̃_{L+1}) / d_{L+1}

Drops in F_layer mark bottlenecks: where the system loses information
between consecutive layers.
"""

from __future__ import annotations

import torch

from icon.core.infonce import InfoNCEConfig, estimate_mi, permutation_null
from icon.core.trust import f_max


def compute_f_layer(
    z_tilde_L: torch.Tensor,
    z_tilde_next: torch.Tensor,
    config: InfoNCEConfig,
    critic_seed: int,
    permutation_seed: int,
) -> tuple[float, float, float]:
    """Return (F_layer, F_perm, F_max).

    Normalized by d_{L+1} (the next layer's dimension), per Section 2.
    """
    d_next = z_tilde_next.shape[1]

    mi = estimate_mi(z_tilde_L, z_tilde_next, config, critic_seed)
    perm = permutation_null(z_tilde_L, z_tilde_next, config, permutation_seed)
    return mi / d_next, perm / d_next, f_max(config.batch_size, d_next)
