# ICON-Primitive v1.1 Config Packs (1A–1G)

This folder contains the remaining config packs needed to run ICON Series **Part 1 (ICON-Primitive)** end-to-end with strong fairness & reproducibility.

## New files generated
- ICON_Primitive_Sweep_Configs_1A-1E_v1.1.yaml  (activation/linear_type/norm/precision/skip sweeps)
- ICON_Primitive_Sweep_Configs_1A-1E_v1.1.csv
- ICON_Primitive_Robustness_1G_Configs_v1.1.yaml  (12-condition robustness matrix for C_gelu)
- ICON_Primitive_Robustness_1G_Configs_v1.1.csv
- ICON_Primitive_Run_Manifest_Base153_v1.1.csv   (all **base** runs for Part-1: 153 runs total)
- ICON_Primitive_Run_Manifest_Extensions_v1.1.csv (optional defense extensions)

## Counting (base 153)
- 1A: 7 (incl baseline) × 3 seeds = 21
- 1B Track-A: 4 (incl baseline) × 3 seeds = 12
- 1C: 5 (incl baseline) × 3 seeds = 15
- 1D: 5 (incl baseline) × 3 seeds = 15
- 1E: 3 (incl baseline) × 3 seeds = 9
- 1F: 15 combos × 3 seeds = 45 (baseline is a reference only)
- 1G: 12 conditions × 3 seeds = 36 (each condition is a paired-ratio run)
TOTAL = 153

## Optional extensions
- 1B Track-B budget-matched (B10–B13)
- 1E concat-projected variant (E03)
- 1F linear_type multiplicativity extension (T01–T08)

