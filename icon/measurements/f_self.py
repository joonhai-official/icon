# icon/measurements/f_self.py
"""F_self — self-consistency under noise.

    F_self(L) = I(Z_L; Z̃_L) / d_L

Channel capacity of the noise channel applied to Z_L (Section 3).
"""

from __future__ import annotations

import torch

from icon.core.infonce import InfoNCEConfig, estimate_mi, permutation_null
from icon.core.trust import f_max


def compute_f_self(
    z: torch.Tensor,
    z_tilde: torch.Tensor,
    config: InfoNCEConfig,
    critic_seed: int,
    permutation_seed: int,
) -> tuple[float, float, float]:
    """Return (F_self, F_perm, F_max)."""
    d_L = z_tilde.shape[1]

    mi = estimate_mi(z, z_tilde, config, critic_seed)
    perm = permutation_null(z, z_tilde, config, permutation_seed)
    return mi / d_L, perm / d_L, f_max(config.batch_size, d_L)
