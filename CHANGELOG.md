# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). PROTOCOL versioning is separate and follows Section 14 of the framework specification.

## [0.1.0] — Initial alpha release

This alpha is released alongside the framework specification's initial submission. The reference implementation covers all five components defined in the specification.

### Added

- The five measurements: `F_in`, `F_task`, `F_self`, `F_layer`, `ρ` (Section 2).
- Goldfeld noise channel with RMS-scaled isotropic Gaussian (Section 4).
- InfoNCE estimator with separable cosine critic and learnable scale (Appendix A.4).
- Trust τ classification (valid / saturated / invalid) per component and aggregate (Section 7).
- The nine-step measurement pipeline (Section 9).
- Four aggregation primitives: static, spatial, temporal, shift (Section 10).
- Cross-system profile comparison with PROTOCOL-match detection (Section 11).
- PROTOCOL declaration with eight categories, JSON roundtrip, SHA-256 hashing (Sections 12–14).
- Manifest schema with all required fields, environment auto-collection (Section 13).
- Interface contracts: `AdapterBase`, `DataLoaderBase` (Section 16).
- Public surface: `measure`, `measure_layers`, `static`/`spatial`/`temporal`/`shift`, Trust τ inspection, manifest compatibility check (Section 17).
- Deterministic seed derivation from a single master seed (Section 9 Step 1, Section 12 universal requirement).
- Example adapters: MLP on MNIST, ResNet18 on CIFAR-10.
- Unit + property + integration tests; 44 tests passing.
- Specification-to-code mapping document.

### Status

Cross-architectural validation of `F_self`, `F_layer`, and `ρ` is described in the specification as follow-up work. The implementation itself is structurally complete and verified for determinism, saturation behavior, and the η_t identity at machine precision.
