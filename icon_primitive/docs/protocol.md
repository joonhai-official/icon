# Protocol (Paper ↔ Code Map)

> This document is intentionally short and operational. It is meant to map the paper’s concepts to the repo’s code paths.

## 1) Core definitions

### Capacity proxy (per-dimension)
We report a capacity proxy:
\[
\kappa \,=\, \frac{I(X; \tilde Z)}{d_z}
\]
where \(d_z\) is the representation dimension.

### Measurement channel (noise channel)
To avoid MI divergence / estimator pathologies, we use a noise channel:
\[
\tilde Z = Z + \varepsilon, \quad \varepsilon \sim \mathcal N(0, (\sigma\cdot \mathrm{RMS}(Z))^2 I)
\]
This is implemented in `icon_primitive/core/noise_channel.py` and referenced by the runner.

### Primitive capacity factor (ratio)
For a primitive setting \(\pi\) and a fixed baseline \(\pi_{base}\):
\[
C(\pi) = \frac{\kappa(\pi)}{\kappa(\pi_{base})}
\]

### Compositional predictor (multiplicative law)
For a composition \(\Pi\) of primitives:
\[
\hat C(\Pi) = \prod_i C_i(\pi_i)
\]

## 2) What is “frozen protocol”?

We hold the following constant across comparisons:

- fixed evaluation indices (and PTQ calibration indices where applicable)
- fixed seeds (3 seeds per condition in the base regime)
- fixed training dtype (FP32 training; PTQ is applied afterwards for precision variants)
- fixed measurement estimator split (8192 eval samples split into 4096/4096 to prevent critic overfitting)
- strict receipt validation + hashes

These are primarily configured in:
- `configs/base_protocol.yaml`
- `assets/eval_indices/` and `assets/calib_indices/`
- `schemas/ICON_Primitive_Receipt_Schema_v1.1.json`

## 3) Paper sections ↔ repo sections

- **1A_activation**: activation factors C under `vector_probe`
- **1B_linear_type**: operator factors C under `spatial_probe`
- **1C_normalization**: normalization factors C under `vector_probe`
- **1D_precision**: FP32 train → PTQ → κ measurement (precision factors)
- **1E_skip**: skip variants (note: dense_concat has doubled dz under `concat_projection=none`)
- **1F_independence_core**: compositional law tests (prediction vs measurement)
- **1G_robustness**: robustness axis sweeps (reported under primary estimator policy)

See run definitions in:
- `configs/manifests/ICON_Primitive_Run_Manifest_Base153_v1.1.csv`
- `configs/sections/*.yaml`

## 4) Dense concat note (important)

`dense_concat` is run with `concat_projection=none`, which doubles \(d_z\) (e.g. 256→512).  
Since \(\kappa = I/d_z\), the per-dimension κ is expected to halve even if I stays similar.  
Therefore `C(dense_concat) ≈ 0.5039` is expected under this per-dim definition.

## 5) Code map

- Runner (CSV row → model/dataset → κ → receipt):
  - `icon_primitive/experiment/runner.py`
- κ computation + estimator wrappers:
  - `icon_primitive/core/kappa.py`
  - `icon_primitive/core/mi_estimators.py`
- Probes / primitives / stems:
  - `icon_primitive/models/probes.py`
  - `icon_primitive/models/primitives.py`
  - `icon_primitive/models/stems.py`
- CLI entry points:
  - `scripts/run_single.py`
  - `scripts/aggregate.py`
- Receipt schema:
  - `schemas/ICON_Primitive_Receipt_Schema_v1.1.json`
