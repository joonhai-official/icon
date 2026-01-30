# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-01-19

### Added
- **Measurement Channel**: κ definition now includes noise channel (Z̃ = Z + ε) to prevent MI divergence attacks
- **MI Estimator Split**: 8192 eval samples split into 4096 train / 4096 test to prevent critic overfitting
- **Three MI Estimators**: InfoNCE (primary), MINE (secondary), KSG (tertiary with 32-d projection)
- **Saturation Diagnostics**: InfoNCE saturation check (MI < log(batch) - 0.1)
- **Sanity Checks**: Permuted Z should yield MI ≈ 0
- **Skip Fairness**:
  - Residual: variance-preserving scaling (y = (x + f(x)) / sqrt(2))
  - Dense Concat: optional projection back to width
- **PTQ Protocol**: FP32 training → conversion → κ measurement
- **2-Track Linear Type**: shape_matched and param_matched budgets
- **Receipt Schema v1.1**: Strict JSON Schema validation with all required fields
- **Fixed Indices**: Evaluation (8192) and calibration (1024) indices are now fixed files
- **Frozen Stems**: E0 (vector) and S0 (spatial) stems with seed 123
- **Unit Tests**: Comprehensive test suite for core, models, data, and utils modules
- **Scripts**: 
  - `generate_stems.py`: Create frozen stems
  - `generate_eval_indices.py`: Create fixed evaluation and/or PTQ calibration indices

### Changed
- Protocol locked in `base_protocol.yaml` - no changes allowed
- Bootstrap CI now uses 2000 iterations with percentile method
- Aggregation uses mean(C_seed) instead of ratio of means

### Fixed
- Dense Concat (concat_projection=none): per-dim κ decreases as expected due to doubled d_z (C≈0.5039); this is a definition-consistent effect under κ=I/d_z.

- Optional import hardening for torchvision (dataset/preprocessing code no longer crashes on import)
- MINE EMA update detaches exp(T) to prevent backward graph reuse / memory leaks
- Default torch threads set to 1 for reproducibility (override via ICON_PRIMITIVE_NUM_THREADS)
- SHA256 uses full 64-hex digest (receipts become stronger audit artifacts)
- get_normalization() now accepts both `dim` and `num_features` keywords for API compatibility
- `none` normalization returns `nn.Identity()` (consistent module-based pattern)
- Runner records torch_num_threads / torch_num_interop_threads in receipt.extras for audit
- Scripts are import-safe as entry points (scripts/ is now a package)

### Security
- Receipt validation prevents arbitrary code injection through config
- Hash verification for configs, data, eval indices, and stems

## [1.0.0] - 2026-01-01 (Initial Design)

### Added
- Initial ICON-Primitive experiment framework
- Basic κ measurement pipeline
- Probe architectures (Vector, Spatial)
- Primitive components (Activation, Normalization, Skip)
