#!/usr/bin/env python3
"""Run a manifest (CSV or YAML) under the Icon_primitive protocol.

CSV manifests are recommended (e.g., Base153). YAML manifests with a `jobs` list
are supported for backwards compatibility.
"""

from __future__ import annotations

# Allow `python scripts/...py` without requiring `pip install -e .`.
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


def _normalize_row(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
            continue
        if isinstance(v, float) and pd.isna(v):
            out[k] = None
            continue
        if isinstance(v, str) and v.strip() == "":
            out[k] = None
            continue
        out[k] = v
    return out


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return [_normalize_row(r) for r in df.to_dict(orient="records")]
    # YAML fallback
    with open(path, "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)
    jobs = m.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("YAML manifest must have a list at key: jobs")
    return [_normalize_row(j) for j in jobs]


def _run_one(row: Dict[str, Any], repo_root: str, output_root: str, data_root: str, assets_root: str, imagenet_subset_root: Optional[str]) -> str:
    from pathlib import Path

    from icon_primitive.experiment.runner import run_manifest_row

    repo = Path(repo_root)
    receipt = run_manifest_row(
        row=row,
        repo_root=repo,
        output_root=Path(output_root),
        data_root=Path(data_root),
        assets_root=Path(assets_root),
        protocol_path=repo / "configs" / "base_protocol.yaml",
        schema_path=repo / "schemas" / "ICON_Primitive_Receipt_Schema_v1.1.json",
        imagenet_subset_root=Path(imagenet_subset_root) if imagenet_subset_root else None,
    )
    return str(receipt)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="CSV (recommended) or YAML manifest")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel processes")
    p.add_argument("--filter_section", default=None, help="Optional filter like 1A, 1F, 1G")
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--data_root", default="data")
    p.add_argument("--imagenet_subset_root", default=None)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = (repo_root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    rows = _load_manifest(manifest_path)

    if args.filter_section:
        rows = [r for r in rows if str(r.get("section", "")).startswith(args.filter_section)]

    output_root = (repo_root / args.output_root).resolve()
    data_root = (repo_root / args.data_root).resolve()
    assets_root = (repo_root / "assets").resolve()

    if args.workers <= 1:
        for r in rows:
            receipt = _run_one(r, str(repo_root), str(output_root), str(data_root), str(assets_root), args.imagenet_subset_root)
            print(f"OK {r.get('run_id')} -> {receipt}")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run_one, r, str(repo_root), str(output_root), str(data_root), str(assets_root), args.imagenet_subset_root) for r in rows]
        for f in as_completed(futs):
            try:
                receipt = f.result()
                print(f"OK {receipt}")
            except Exception as e:
                print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
