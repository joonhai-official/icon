# Reproducibility

This repository is designed to make “paper numbers” reproducible from **receipts**.

## 1) What is a receipt?

A receipt is a per-run JSON record:
- `outputs/runs/<run_id>/receipt.json`

It includes:
- run identifiers (section/run_id)
- fixed protocol parameters
- hashes (config/data/indices/stems)
- environment info (torch threads, etc.)
- measured values (`results.kappa.*`, ratios)

Schema:
- `schemas/ICON_Primitive_Receipt_Schema_v1.1.json`

## 2) What must be fixed?

### Fixed indices
- Evaluation: `assets/eval_indices/{dataset}_test_8192_seed0.npy`
- PTQ calibration: `assets/calib_indices/{dataset}_train_1024_seed0.npy`

Generated once by:
- `scripts/generate_eval_indices.py --output assets --seed 0 --mode both`

### Frozen stems
Generated once by:
- `scripts/generate_stems.py --output assets/frozen_stems --seed 123`

### Seeds
Manifests contain seeds (model/data/mi). Base regime uses 3 seeds per condition.

### Training dtype
Training is fixed to FP32 for comparability.
Precision variants use PTQ after training.

## 3) How to reproduce tables/verification

### Aggregate constants YAML (patched)
```bash
python scripts/aggregate.py \
  --receipts outputs/runs \
  --template configs/packs/ICON_Primitive_Constants_Template_v1.1.yaml \
  --output outputs/constants/Icon_primitive_Constants.yaml
```

### Recompute from receipts (tools/)
- Constants table: `python tools/analyze_C.py`
- Independence (1F): `python tools/verify_1F_independence.py`
- Robustness (1G): `python tools/verify_1G_robustness.py`

## 4) Policy on exclusions

- No ad-hoc outlier removal.
- Estimator sanity-fail (e.g., non-finite ratios) must be reported and excluded by a declared rule.
- Robustness reporting uses the primary estimator policy (see estimator_policy.md).

## 5) Recommended artifacts for release

For a public release:
- keep code/configs/tools/schemas in git
- do **not** commit `data/` or `outputs/`
- optionally include a few sample receipts under `examples/`
- provide a “how to generate receipts” command block in README
