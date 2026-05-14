# icon/io/protocol.py
"""PROTOCOL declaration (Sections 12-14).

Eight categories of settings. Two measurements with identical canonicalized
PROTOCOLs are directly comparable; measurements with different PROTOCOLs
require calibration mapping.

The eight categories:
    1. Noise          (σ, RMS method, RMS floor)
    2. InfoNCE        (batch size, critic, optimizer, n_steps)
    3. Pool           (N, pool seed offset, permutation count)
    4. Trust τ        (r_sat, ε_sep, r_rel, r_abs)
    5. Numerical      (eigenvalue floor, precision)
    6. Training       (conditional)
    7. Perturbation   (conditional)
    8. Statistics     (bootstrap, FDR)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any


PROTOCOL_VERSION = "1.0"


@dataclass
class NoiseConfig:
    sigma: float = 0.1
    rms_method: str = "per_batch_global_scalar"
    rms_floor: float = 1e-6


@dataclass
class InfoNCEConfig:
    batch_size: int = 512
    n_steps: int = 2000
    critic_architecture: str = "separable_cosine_mlp_with_learnable_scale"
    critic_hidden: tuple[int, ...] = (256, 256)
    critic_proj_dim: int = 64
    optimizer: str = "adam"
    learning_rate: float = 1e-3


@dataclass
class PoolConfig:
    sample_count: int | None = None  # None → auto: max(8B, 4 d_L)
    pool_seed_offset: int = 0
    permutation_count: int = 1


@dataclass
class TrustConfig:
    r_sat: float = 0.95
    epsilon_sep: float = 0.005
    r_rel: float = 3.0
    r_abs: float = 0.20


@dataclass
class NumericalConfig:
    eigenvalue_floor: float = 1e-12
    negative_eigenvalue_tolerance: float = -1e-9
    forward_precision: str = "float32"
    covariance_precision: str = "float64"


@dataclass
class TrainingConfig:
    optimizer: str | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    epochs: int | None = None
    checkpoint_schedule: str | None = None


@dataclass
class PerturbationConfig:
    perturbation_type: str | None = None
    magnitude: float | None = None
    iterations: int | None = None


@dataclass
class StatisticsConfig:
    bootstrap_resamples: int = 0
    fdr_level: float = 0.05


@dataclass
class PROTOCOL:
    """The framework's settings declaration."""

    noise: NoiseConfig = field(default_factory=NoiseConfig)
    infonce: InfoNCEConfig = field(default_factory=InfoNCEConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    trust_tau: TrustConfig = field(default_factory=TrustConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    training: TrainingConfig | None = None
    perturbation: PerturbationConfig | None = None
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PROTOCOL:
        return cls(
            noise=NoiseConfig(**d.get("noise", {})),
            infonce=InfoNCEConfig(
                **{k: (tuple(v) if k == "critic_hidden" else v)
                   for k, v in d.get("infonce", {}).items()}
            ),
            pool=PoolConfig(**d.get("pool", {})),
            trust_tau=TrustConfig(**d.get("trust_tau", {})),
            numerical=NumericalConfig(**d.get("numerical", {})),
            training=TrainingConfig(**d["training"]) if d.get("training") else None,
            perturbation=PerturbationConfig(**d["perturbation"]) if d.get("perturbation") else None,
            statistics=StatisticsConfig(**d.get("statistics", {})),
            protocol_version=d.get("protocol_version", PROTOCOL_VERSION),
        )

    def hash(self) -> str:
        """SHA-256 of the canonicalized PROTOCOL — the PROTOCOL identifier (Section 13)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, default=_json_default)
        return hashlib.sha256(canonical.encode()).hexdigest()


def _json_default(o: Any) -> Any:
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON-serializable")


def protocols_match(p1: PROTOCOL, p2: PROTOCOL) -> bool:
    """Return True iff two PROTOCOLs would produce comparable measurements."""
    return p1.hash() == p2.hash()
