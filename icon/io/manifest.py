# icon/io/manifest.py
"""Manifest schema (Section 13).

One manifest = one measurement's full record. Contains everything needed
to reproduce the measurement, modulo hardware-specific numerical variation.

Required fields per Section 13:
    icon_version, protocol_version
    system     (adapter class, layer name, layer dim)
    data       (loader class, num_classes, n_samples)
    protocol   (full eight-category PROTOCOL)
    seeds      (master + four derived sub-seeds)
    environment (Python, framework, device, deterministic mode, UTC time)
    results    (the five components, F_perm, F_max, Trust τ, timing)
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


ICON_VERSION = "0.1.0"
PROTOCOL_VERSION = "1.0"


@dataclass
class Manifest:
    icon_version: str
    protocol_version: str
    system: dict[str, Any]
    data: dict[str, Any]
    protocol: dict[str, Any]
    seeds: dict[str, int]
    environment: dict[str, Any]
    results: dict[str, Any]
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "icon_version": self.icon_version,
            "protocol_version": self.protocol_version,
            "system": self.system,
            "data": self.data,
            "protocol": self.protocol,
            "seeds": self.seeds,
            "environment": self.environment,
            "results": self.results,
            "extensions": self.extensions,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=_json_default)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        return cls(
            icon_version=d["icon_version"],
            protocol_version=d["protocol_version"],
            system=d["system"],
            data=d["data"],
            protocol=d["protocol"],
            seeds=d["seeds"],
            environment=d["environment"],
            results=d["results"],
            extensions=d.get("extensions", {}),
        )

    @classmethod
    def from_json(cls, s: str) -> Manifest:
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> Manifest:
        with open(path) as f:
            return cls.from_json(f.read())

    def protocol_hash(self) -> str:
        """SHA-256 of the canonicalized protocol block — the PROTOCOL identifier."""
        canonical = json.dumps(self.protocol, sort_keys=True, default=_json_default)
        return hashlib.sha256(canonical.encode()).hexdigest()


def _json_default(o: Any) -> Any:
    # Handle tuples and numbers that JSON's default encoder rejects.
    if isinstance(o, tuple):
        return list(o)
    if hasattr(o, "value"):  # Enum
        return o.value
    raise TypeError(f"Object of type {type(o).__name__} is not JSON-serializable")


def check_manifest_compat(manifest_dict: dict[str, Any]) -> bool:
    """Check whether a manifest is compatible with the current framework version.

    A manifest written under any version v of the framework must remain
    readable under any later version v' with the same MAJOR number (Section 14).
    """
    if "protocol_version" not in manifest_dict:
        return False
    their_major = manifest_dict["protocol_version"].split(".")[0]
    our_major = PROTOCOL_VERSION.split(".")[0]
    return their_major == our_major


def collect_environment() -> dict[str, Any]:
    """Collect environment metadata for the manifest."""
    try:
        import torch
        framework = f"torch=={torch.__version__}"
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        deterministic = torch.are_deterministic_algorithms_enabled()
    except ImportError:
        framework = "unknown"
        device_count = 0
        device = "cpu"
        deterministic = False

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "framework": framework,
        "device": device,
        "device_count": device_count,
        "deterministic_mode": deterministic,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_manifest(
    adapter: Any,
    loader: Any,
    layer_name: str,
    layer_dim: int,
    sample_count: int,
    protocol: Any,  # PROTOCOL — avoid import cycle
    seeds: Any,     # SeedBundle
    results: dict[str, Any],
) -> Manifest:
    """Assemble a manifest from one measurement's components."""
    return Manifest(
        icon_version=ICON_VERSION,
        protocol_version=PROTOCOL_VERSION,
        system={
            "adapter_class": adapter.__class__.__name__,
            "layer_name": layer_name,
            "layer_dim": layer_dim,
        },
        data={
            "loader_class": loader.__class__.__name__,
            "num_classes": loader.num_classes(),
            "n_samples": sample_count,
        },
        protocol=protocol.to_dict(),
        seeds={
            "master": seeds.master_seed,
            "data": seeds.data_seed,
            "noise": seeds.noise_seed,
            "critic": seeds.critic_seed,
            "permutation": seeds.permutation_seed,
        },
        environment=collect_environment(),
        results=results,
    )
