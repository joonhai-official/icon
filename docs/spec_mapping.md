# Specification Mapping

This document maps every section of the framework specification (Park, 2026) to the corresponding module, function, or test in this repository. The mapping is one-to-one wherever possible. When a single specification section spans multiple modules, all of them are listed.

## Part 1 — Foundations

| Section | Topic | Implementation |
|---|---|---|
| §1 | The Five Measurements (overview) | `icon/measurements/__init__.py` |
| §2 | Precise Definitions | `icon/measurements/` (one module per measurement) |
| §2 | Noise injection ($\widetilde{Z}_L = Z_L + \sigma \cdot \text{RMS}(Z_L) \cdot \varepsilon$) | `icon/core/noise.py` |
| §2 | Participation ratio | `icon/measurements/rho.py` |
| §2 | Canonical ratio $\eta_t$ | `icon/measurements/eta_t.py` |
| §3 | Closure derivation (informal) | Documented in code, formal proof in §A.6 |
| §4 | Why noise is required (Goldfeld) | `icon/core/noise.py` (docstring) |
| §5 | Empirical and conceptual origin | (companion papers, not code) |

## Part 2 — Statistical Validity

| Section | Topic | Implementation |
|---|---|---|
| §6 | Saturation ceiling ($\log B$) | `icon/core/trust.py` (constants) |
| §7 | Trust τ classification | `icon/core/trust.py` |
| §7 | Three states (valid/saturated/invalid) | `icon/core/trust.py:TrustClassification` |
| §7 | Aggregate Trust τ | `icon/core/trust.py:aggregate_trust` |
| §8 | Validity of ρ (sample-count requirement) | `icon/measurements/rho.py` |

## Part 3 — Algorithm

| Section | Topic | Implementation |
|---|---|---|
| §9 | Measurement pipeline (9 steps) | `icon/pipeline/measure.py:measure` |
| §9 Step 1 | Seed initialization | `icon/core/seeds.py` |
| §9 Step 2 | Sample collection | `icon/pipeline/measure.py:_collect_samples` |
| §9 Step 3 | Forward pass with taps | `icon/contracts/adapter.py` |
| §9 Step 4 | Noise injection | `icon/core/noise.py` |
| §9 Step 5 | InfoNCE estimation | `icon/core/infonce.py` |
| §9 Step 6 | Permutation null | `icon/core/infonce.py:permutation_null` |
| §9 Step 7 | ρ computation | `icon/measurements/rho.py` |
| §9 Step 8 | Trust τ classification | `icon/core/trust.py` |
| §9 Step 9 | Manifest generation | `icon/io/manifest.py` |
| §10 | Aggregations (4 primitives) | `icon/pipeline/aggregate.py` |
| §10 | Static / Spatial / Temporal / Shift | `icon/pipeline/aggregate.py` |
| §10 | Statistical reductions | `icon/pipeline/aggregate.py` |
| §11 | Comparing profiles across systems | `icon/pipeline/compare.py` |

## Part 4 — PROTOCOL Discipline

| Section | Topic | Implementation |
|---|---|---|
| §12 | The eight categories | `icon/io/protocol.py:PROTOCOL` |
| §12.1 | Noise settings | `icon/io/protocol.py:NoiseConfig` |
| §12.2 | InfoNCE settings | `icon/io/protocol.py:InfoNCEConfig` |
| §12.3 | Pool settings | `icon/io/protocol.py:PoolConfig` |
| §12.4 | Trust τ thresholds | `icon/io/protocol.py:TrustConfig` |
| §12.5 | Numerical safety | `icon/io/protocol.py:NumericalConfig` |
| §12.6 | Training (conditional) | `icon/io/protocol.py:TrainingConfig` |
| §12.7 | Perturbation (conditional) | `icon/io/protocol.py:PerturbationConfig` |
| §12.8 | Statistics | `icon/io/protocol.py:StatisticsConfig` |
| §13 | Manifest schema | `icon/io/manifest.py:Manifest` |
| §13 | SHA-256 hash for PROTOCOL identity | `icon/io/protocol.py:PROTOCOL.hash` |
| §14 | PROTOCOL versioning | `icon/io/protocol.py:PROTOCOL_VERSION` |

## Part 5 — Reference Implementation

| Section | Topic | Implementation |
|---|---|---|
| §15 | Code structure | (this document + directory tree) |
| §16 | AdapterBase contract | `icon/contracts/adapter.py` |
| §16 | DataLoaderBase contract | `icon/contracts/loader.py` |
| §17 | Public surface (`icon.measure`, etc.) | `icon/__init__.py` |

## Part 6 — What Can Be Done With This

(Application directions are described in the specification; this repository provides the substrate.)

| Direction | Where to start |
|---|---|
| Read — Seeing what a system does | `icon.measure`, `icon.measure_layers`, `icon.spatial`, `icon.temporal` |
| Shape — Building or changing a system | `icon.measure` + forward model fitting (user code) |
| Test — Perturbing and watching what changes | `icon.measure` (clean) + `icon.measure` (perturbed) + `icon.shift` |

## Part 7 — Community Model

(Community contributions are not part of the reference implementation; see `CONTRIBUTING.md` for guidance.)

## Part 8 — Limits and References

| Section | Topic | Implementation |
|---|---|---|
| §21 | Limitations | (documented in code where relevant; tests verify the structural ones) |

## Appendix A — Mathematical Definitions

| Section | Topic | Implementation |
|---|---|---|
| §A.4 | InfoNCE estimator definition | `icon/core/infonce.py:InfoNCEEstimator` |
| §A.4 | Reference critic (separable cosine) | `icon/core/infonce.py:SeparableCosineCritic` |
| §A.4 | Saturation ceiling proof | (verified in `tests/property/test_saturation.py`) |
| §A.5 | Trust τ formal classification | `icon/core/trust.py` |
| §A.6 | Closure derivation | (documentation; not executable) |
| §A.7 | $\eta_t$ identity ($d_L$ cancellation) | `icon/measurements/eta_t.py` + `tests/property/test_eta_t.py` |
| §A.8 | Sample complexity | (documented in `icon/pipeline/measure.py` default $N$) |
| §A.9 | Aggregation composition rules | `icon/pipeline/aggregate.py` |
| §A.10 | Identifiability | (documented; not executable) |

## Tests Mapping

| Test file | Verifies |
|---|---|
| `tests/unit/test_noise.py` | Noise injection correctness, RMS scaling |
| `tests/unit/test_infonce.py` | InfoNCE estimator basic behavior |
| `tests/unit/test_rho.py` | Participation ratio range [1/d, 1] |
| `tests/unit/test_trust.py` | Three-state classification rules |
| `tests/unit/test_protocol.py` | PROTOCOL hashing, serialization round-trip |
| `tests/unit/test_manifest.py` | Manifest schema, required fields |
| `tests/property/test_eta_t.py` | $d_L$ cancellation (machine precision) |
| `tests/property/test_saturation.py` | InfoNCE ≤ log B |
| `tests/property/test_determinism.py` | Same seed → same output |
| `tests/integration/test_pipeline.py` | End-to-end pipeline on synthetic data |
| `tests/integration/test_aggregation.py` | Composition of four aggregation primitives |

## Examples

| Example | Demonstrates |
|---|---|
| `examples/mnist_mlp.py` | Minimal end-to-end: small MLP on MNIST. Adapter wraps `nn.Module`, loader wraps a flat-vector dataset. |
| `examples/cifar10_resnet.py` | Realistic example: torchvision ResNet18 on CIFAR-10. Forward-hook-based adapter, global-average-pool tap convention for feature maps. |
