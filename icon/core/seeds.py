# icon/core/seeds.py
"""Deterministic seed derivation (Section 9 Step 1).

Given a master seed, derive sub-seeds for data shuffling, noise injection,
critic initialization, and permutation generation. The derivation is
deterministic so the same master_seed always yields the same sub-seeds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedBundle:
    master_seed: int
    data_seed: int
    noise_seed: int
    critic_seed: int
    permutation_seed: int


def _derive(master_seed: int, tag: str) -> int:
    # SHA-256 of "master_seed:tag", first 8 bytes as int.
    # Why SHA-256: deterministic, collision-resistant, stdlib.
    h = hashlib.sha256(f"{master_seed}:{tag}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def derive_seeds(master_seed: int) -> SeedBundle:
    return SeedBundle(
        master_seed=master_seed,
        data_seed=_derive(master_seed, "data"),
        noise_seed=_derive(master_seed, "noise"),
        critic_seed=_derive(master_seed, "critic"),
        permutation_seed=_derive(master_seed, "permutation"),
    )
