# Icon_primitive

**Purpose.** Build a **reproducible primitive capacity-factor table** and validate a **near-multiplicative compositional law** under a frozen measurement protocol.

This repository is **protocol + receipts** driven:
- Every run produces an audit artifact: `outputs/runs/<run_id>/receipt.json`
- Tables/verification are regenerated deterministically from receipts (see `tools/` and `scripts/aggregate.py`)

---

## Key definitions

- **Capacity proxy (per-dimension)**  
  \[
  \kappa \,=\, \frac{I(X;\tilde Z)}{d_z}
  \]

- **Primitive factor (ratio)**  
  \[
  C(\pi) \,=\, \frac{\kappa(\pi)}{\kappa(\pi_{base})}
  \]

- **Compositional predictor (multiplicative law)**  
  \[
  \hat C(\Pi) \,=\, \prod_i C_i(\pi_i)
  \]

---

## Reproduction scope (default, paper-aligned)

The repo supports a larger manifest, but the **default paper-aligned scope** excludes any ImageNet-subset runs.

- **Datasets:** MNIST, CIFAR10  
- **Target:** **141 / 141** runs (non-imagenet subset of the Base manifest)  
- **Robustness (1G):** reported under **InfoNCE** measurement channel (see estimator policy)

---

## Important note: `dense_concat` yields **C ≈ 0.5039** by definition

`dense_concat` is run with `concat_projection = none`, which doubles representation dimension \(d_z\) (e.g., 256 → 512).  
Since \(\kappa = I(X;\tilde Z) / d_z\), even if \(I\) stays similar, per-dimension \(\kappa\) is expected to halve.

Therefore **C(dense_concat) ≈ 0.5039** is **expected** under this per-dimension κ definition.

---

## Quickstart (smoke)

### 1) Install
```bash
python -m pip install -e .
python -m pip install pytest  # optional
```

### 2) One-time prep (reproducibility-critical)
```bash
python scripts/generate_stems.py --output assets/frozen_stems --seed 123
python scripts/generate_eval_indices.py --output assets --seed 0 --mode both
```

### 3) Run one experiment
```bash
python scripts/run_single.py \
  --manifest configs/manifests/ICON_Primitive_Run_Manifest_Base153_v1.1.csv \
  --run_id 1A_A00_s0 \
  --output_root outputs \
  --data_root data
```

### 4) Aggregate receipts → constants YAML
```bash
python scripts/aggregate.py \
  --receipts outputs/runs \
  --template configs/packs/ICON_Primitive_Constants_Template_v1.1.yaml \
  --output outputs/constants/Icon_primitive_Constants.yaml
```

> Note: `data/` and `outputs/` are local artifacts and are not committed in a public repo. Create them locally or pass `--data_root` / `--output_root` as shown.

---

## Reproduce paper numbers from receipts (official path)

Primitive constants table (C):
```bash
python tools/reproduce_constants.py
```

Compositional law (1F):
```bash
python tools/verify_independence.py
```

Robustness (1G, **InfoNCE-only PASS criterion**):
```bash
python tools/verify_robustness.py
```

---

## Estimator policy (recommended)

- **InfoNCE is primary** for robustness reporting in this repo.
- KSG / MINE are treated as **secondary diagnostics** (may be high-variance or sanity-fail in this regime).
- Failures must be **counted + listed** and excluded only by a declared rule.  
See: `docs/estimator_policy.md`

---

## Examples (small, repo-friendly)

- `examples/sample_constants.yaml` — a sample constants YAML (paper-aligned fields)
- `examples/sample_receipts/` — a few example receipts (optional)

---

## Repo layout

- `icon_primitive/` core library
- `scripts/` execution + aggregation
- `configs/` protocol, sections, manifests, templates
- `schemas/` receipt schema + example
- `tools/` receipt → tables/verification scripts
- `assets/` fixed indices + frozen stems (small; reproducibility-critical)
- `docs/` protocol/reproducibility/estimator policy

Local artifacts (ignored):
- `data/`, `outputs/`, `snapshots/`, `_local/`, `_dev/`

---

## Tests
```bash
pytest -q
```

---

## Paper ↔ Code map
Start here:
- `docs/protocol.md`
- `docs/reproducibility.md`
- `docs/estimator_policy.md`

---

## License
See `LICENSE`.
