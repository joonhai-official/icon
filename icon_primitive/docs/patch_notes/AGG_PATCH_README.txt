Patch bundle for Icon_primitive:
- scripts/aggregate.py: fixes receipt schema validation bug and fills constants/verification from receipts (receipt-based logic).
- configs/packs/ICON_Primitive_Constants_Template_v1.1.yaml: sets dense_concat projection to 'none' and removes imagenet_subset from tested_axes.datasets.

Apply (from repo root):
  tar -xzf aggregate_patch_bundle.tgz
  python3 scripts/aggregate.py --receipts outputs/runs --template configs/packs/ICON_Primitive_Constants_Template_v1.1.yaml --output outputs/constants/Icon_primitive_Constants.yaml

Expected:
- activation/norm/precision/skip/linear_type filled (no nulls)
- skip.dense_concat mean around 0.504 (per-dim definition)
- independence.core passed with mean_error ~0.0095, max_error ~0.1007
- robustness passed (InfoNCE primary), tested_axes.datasets {mnist,cifar10}, widths {512,1024}
