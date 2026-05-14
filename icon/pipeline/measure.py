# icon/pipeline/measure.py
"""Main measurement pipeline (Section 9).

The nine steps:
    1. Initialize seeds (master → data, noise, critic, permutation)
    2. Sample collection (N from validation split)
    3. Forward pass with taps (via the adapter)
    4. Noise injection (Goldfeld channel)
    5. InfoNCE estimation (four F components)
    6. Permutation null estimation
    7. ρ computation
    8. Trust τ classification (per-component and aggregate)
    9. Manifest generation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from icon.contracts.adapter import AdapterBase
from icon.contracts.loader import DataLoaderBase
from icon.core.infonce import InfoNCEConfig
from icon.core.noise import inject_noise
from icon.core.seeds import SeedBundle, derive_seeds
from icon.core.trust import (
    TrustClassification,
    TrustThresholds,
    aggregate_trust,
    classify_component,
)
from icon.io.manifest import Manifest, build_manifest
from icon.io.protocol import PROTOCOL
from icon.measurements.eta_t import compute_eta_t
from icon.measurements.f_in import compute_f_in
from icon.measurements.f_layer import compute_f_layer
from icon.measurements.f_self import compute_f_self
from icon.measurements.f_task import compute_f_task
from icon.measurements.rho import compute_rho


@dataclass
class Profile:
    """The result of one measurement at one (layer, time).

    Contains the five components, their Trust τ classifications, the
    canonical ratio η_t, and the complete manifest.
    """

    # Five components
    f_in: float
    f_task: float
    f_self: float
    f_layer: float | None  # None when measuring the last layer
    rho: float

    # Canonical ratio
    eta_t: float

    # Per-component permutation nulls and saturation ceiling
    f_perm: dict[str, float]
    f_max: dict[str, float]

    # Per-component Trust τ classifications
    trust: dict[str, TrustClassification]

    # Aggregate Trust τ (over the four F components)
    trust_aggregate: TrustClassification

    # ρ has its own validity (Section 8)
    rho_low_sample: bool

    # Full manifest
    manifest: Manifest

    @property
    def components(self) -> tuple[float, float, float, float | None, float]:
        return (self.f_in, self.f_task, self.f_self, self.f_layer, self.rho)


def _collect_samples(
    loader: DataLoaderBase,
    sample_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step 2: collect N (X, Y) pairs from the validation split."""
    x, y = loader.val_batch(sample_count, seed)
    return x, y


def _resolve_pool_size(protocol: PROTOCOL, d_L: int) -> int:
    """Default N = max(8*B, 4*d_L) per Section A.8, unless overridden."""
    if protocol.pool.sample_count is not None:
        return protocol.pool.sample_count
    return max(8 * protocol.infonce.batch_size, 4 * d_L)


def _to_infonce_config(protocol: PROTOCOL) -> InfoNCEConfig:
    """Translate PROTOCOL.infonce → InfoNCEConfig used by the estimator."""
    return InfoNCEConfig(
        batch_size=protocol.infonce.batch_size,
        n_steps=protocol.infonce.n_steps,
        critic_hidden=protocol.infonce.critic_hidden,
        critic_proj_dim=protocol.infonce.critic_proj_dim,
        learning_rate=protocol.infonce.learning_rate,
    )


def measure(
    adapter: AdapterBase,
    loader: DataLoaderBase,
    layer_name: str,
    protocol: PROTOCOL | None = None,
    master_seed: int = 42,
) -> Profile:
    """Measure one layer of one system under one PROTOCOL.

    Implements the nine-step pipeline of Section 9.
    """
    protocol = protocol or PROTOCOL()
    seeds = derive_seeds(master_seed)
    config = _to_infonce_config(protocol)
    thresholds = TrustThresholds(
        r_sat=protocol.trust_tau.r_sat,
        epsilon_sep=protocol.trust_tau.epsilon_sep,
        r_rel=protocol.trust_tau.r_rel,
        r_abs=protocol.trust_tau.r_abs,
    )

    d_L = adapter.layer_dim(layer_name)
    n_samples = _resolve_pool_size(protocol, d_L)

    # Step 2: sample collection
    x, y = _collect_samples(loader, n_samples, seeds.data_seed)

    # Step 3: forward pass with taps. F_layer needs the next layer too.
    layer_names_ordered = adapter.layer_names()
    layer_idx = layer_names_ordered.index(layer_name)
    has_next = layer_idx + 1 < len(layer_names_ordered)
    requested = [layer_name]
    next_layer_name = None
    if has_next:
        next_layer_name = layer_names_ordered[layer_idx + 1]
        requested.append(next_layer_name)

    activations = adapter.forward_with_taps(x, requested)
    z = activations[layer_name]
    z_next = activations[next_layer_name] if has_next else None

    # Step 4: noise injection (only Z̃; ρ uses raw Z per Section 4)
    z_tilde = inject_noise(z, protocol.noise.sigma, seeds.noise_seed, protocol.noise.rms_floor)
    z_next_tilde = (
        inject_noise(z_next, protocol.noise.sigma, seeds.noise_seed, protocol.noise.rms_floor)
        if has_next
        else None
    )

    # Step 5+6: estimate the four F components (each computes MI + permutation null)
    f_in, f_in_perm, f_in_max = compute_f_in(
        x.flatten(start_dim=1) if x.dim() > 2 else x,
        z_tilde, config, seeds.critic_seed, seeds.permutation_seed,
    )
    f_task, f_task_perm, f_task_max = compute_f_task(
        y, z_tilde, config, seeds.critic_seed + 1, seeds.permutation_seed + 1,
    )
    f_self, f_self_perm, f_self_max = compute_f_self(
        z, z_tilde, config, seeds.critic_seed + 2, seeds.permutation_seed + 2,
    )
    if has_next:
        f_layer, f_layer_perm, f_layer_max = compute_f_layer(
            z_tilde, z_next_tilde, config, seeds.critic_seed + 3, seeds.permutation_seed + 3,
        )
    else:
        f_layer = f_layer_perm = f_layer_max = None

    # Step 7: ρ on un-noised Z
    rho, rho_low_sample = compute_rho(z, protocol.numerical.eigenvalue_floor)

    # Step 8: Trust τ
    trust = {
        "f_in": classify_component(f_in, f_in_perm, f_in_max, thresholds),
        "f_task": classify_component(f_task, f_task_perm, f_task_max, thresholds),
        "f_self": classify_component(f_self, f_self_perm, f_self_max, thresholds),
    }
    component_classifications = [trust["f_in"], trust["f_task"], trust["f_self"]]
    if has_next:
        trust["f_layer"] = classify_component(f_layer, f_layer_perm, f_layer_max, thresholds)
        component_classifications.append(trust["f_layer"])
    trust_agg = aggregate_trust(component_classifications)

    # η_t
    eta_t = compute_eta_t(f_in, f_task)

    # Step 9: manifest
    manifest = build_manifest(
        adapter=adapter,
        loader=loader,
        layer_name=layer_name,
        layer_dim=d_L,
        sample_count=n_samples,
        protocol=protocol,
        seeds=seeds,
        results={
            "f_in": f_in, "f_task": f_task, "f_self": f_self, "f_layer": f_layer, "rho": rho,
            "eta_t": eta_t,
            "f_perm": {"f_in": f_in_perm, "f_task": f_task_perm, "f_self": f_self_perm,
                       "f_layer": f_layer_perm},
            "f_max": {"f_in": f_in_max, "f_task": f_task_max, "f_self": f_self_max,
                      "f_layer": f_layer_max},
            "trust": {k: v.value for k, v in trust.items()},
            "trust_aggregate": trust_agg.value,
            "rho_low_sample": rho_low_sample,
        },
    )

    return Profile(
        f_in=f_in, f_task=f_task, f_self=f_self, f_layer=f_layer, rho=rho,
        eta_t=eta_t,
        f_perm={"f_in": f_in_perm, "f_task": f_task_perm, "f_self": f_self_perm,
                "f_layer": f_layer_perm},
        f_max={"f_in": f_in_max, "f_task": f_task_max, "f_self": f_self_max,
               "f_layer": f_layer_max},
        trust=trust,
        trust_aggregate=trust_agg,
        rho_low_sample=rho_low_sample,
        manifest=manifest,
    )


def measure_layers(
    adapter: AdapterBase,
    loader: DataLoaderBase,
    layer_names: list[str] | None = None,
    protocol: PROTOCOL | None = None,
    master_seed: int = 42,
) -> list[Profile]:
    """Measure multiple layers under a single PROTOCOL."""
    names = layer_names if layer_names is not None else adapter.layer_names()
    return [measure(adapter, loader, name, protocol, master_seed) for name in names]
