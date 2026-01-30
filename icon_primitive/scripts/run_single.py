#!/usr/bin/env python3
"""Run one Icon_primitive job.

Recommended usage:
  python scripts/run_single.py \
    --manifest configs/manifests/ICON_Primitive_Run_Manifest_Base153_v1.1.csv \
    --run_id 1A_A01_s0
"""

from __future__ import annotations

# Allow `python scripts/...py` without requiring `pip install -e .`.
# (Entry points work via installation; this is for local/dev convenience.)
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _row_to_dict(row) -> Dict[str, Any]:
    d = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
    # Normalize empty strings to None
    for k, v in list(d.items()):
        if isinstance(v, str) and v.strip() == "":
            d[k] = None
    return d


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="CSV manifest (Base153 recommended)")
    p.add_argument("--run_id", required=True, help="Run ID to execute (must exist in manifest)")
    p.add_argument("--output_root", default="outputs", help="Output root directory")
    p.add_argument("--data_root", default="data", help="Torchvision dataset root")
    p.add_argument(
        "--imagenet_subset_root",
        default=None,
        help="Path to imagenet_subset folder (must contain train/val or train/test)",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = (repo_root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    df = pd.read_csv(manifest_path)
    matches = df[df["run_id"] == args.run_id]
    if len(matches) != 1:
        raise SystemExit(f"run_id not found or not unique in manifest: {args.run_id}")

    row = _row_to_dict(matches.iloc[0])

    # Import after sys.path is stable
    from icon_primitive.experiment.runner import run_manifest_row

    output_root = (repo_root / args.output_root).resolve()
    data_root = (repo_root / args.data_root).resolve()
    assets_root = (repo_root / "assets").resolve()
    protocol_path = (repo_root / "configs" / "base_protocol.yaml").resolve()
    schema_path = (repo_root / "schemas" / "ICON_Primitive_Receipt_Schema_v1.1.json").resolve()

    imagenet_subset_root = Path(args.imagenet_subset_root).resolve() if args.imagenet_subset_root else None

    receipt_path = run_manifest_row(
        row=row,
        repo_root=repo_root,
        output_root=output_root,
        data_root=data_root,
        assets_root=assets_root,
        protocol_path=protocol_path,
        schema_path=schema_path,
        imagenet_subset_root=imagenet_subset_root,
    )
    print(f"Saved receipt: {receipt_path}")


if __name__ == "__main__":
    main()
