#!/usr/bin/env python3
"""Generate fixed evaluation and/or PTQ-calibration indices.

Why this exists
---------------
The protocol requires that:
  - evaluation uses a *fixed* subset of the test split (n_eval)
  - PTQ calibration uses a *fixed* subset of the train split (n_calib)

We persist the indices as .npy files so that every run (and every machine)
uses exactly the same examples, ensuring fairness and reproducibility.

Usage examples
--------------
Generate eval indices only:
  python scripts/generate_eval_indices.py \
    --output assets/eval_indices --dataset cifar10 --seed 0 --mode eval

Generate calib indices only:
  python scripts/generate_eval_indices.py \
    --output assets/calib_indices --dataset cifar10 --seed 0 --mode calib

Generate both (to two subfolders under the given output root):
  python scripts/generate_eval_indices.py \
    --output assets --dataset cifar10 --seed 0 --mode both
"""

from __future__ import annotations

# Allow `python scripts/...py` without requiring `pip install -e .`.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from pathlib import Path

from icon_primitive.data.datasets import DatasetSpec, load_dataset
from icon_primitive.data.indices import FixedIndexSpec, get_or_create_fixed_indices


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_out(mode: str, output: Path) -> tuple[Path, Path] | tuple[Path, None]:
    """Return (eval_dir, calib_dir) for the requested mode."""
    m = mode.lower()
    if m == "eval":
        return (output, None)
    if m == "calib":
        return (None, output)  # type: ignore
    if m == "both":
        return (output / "eval_indices", output / "calib_indices")
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="Output directory (or base dir if --mode both)")
    p.add_argument("--dataset", default="cifar10", help="mnist|cifar10|imagenet_subset")
    p.add_argument("--seed", type=int, default=0, help="Indices seed")
    p.add_argument("--mode", choices=["eval", "calib", "both"], default="both")
    p.add_argument("--n_eval", type=int, default=8192)
    p.add_argument("--n_calib", type=int, default=1024)
    p.add_argument("--data_root", default="data", help="Dataset root (torchvision download cache)")
    p.add_argument("--imagenet_subset_root", default=None, help="Required for imagenet_subset")
    p.add_argument("--imagenet_resize", type=int, default=64)
    args = p.parse_args()

    repo = _repo_root()
    out = Path(args.output)
    if not out.is_absolute():
        out = (repo / out).resolve()

    eval_dir: Path | None
    calib_dir: Path | None
    if args.mode == "eval":
        eval_dir, calib_dir = out, None
    elif args.mode == "calib":
        eval_dir, calib_dir = None, out
    else:
        eval_dir, calib_dir = out / "eval_indices", out / "calib_indices"

    dataset = str(args.dataset)
    seed = int(args.seed)
    data_root = (repo / args.data_root).resolve() if not Path(args.data_root).is_absolute() else Path(args.data_root)
    imagenet_subset_root = Path(args.imagenet_subset_root).resolve() if args.imagenet_subset_root else None

    if eval_dir is not None:
        ds_test = load_dataset(
            DatasetSpec(
                name=dataset,
                split="test",
                root=data_root,
                imagenet_subset_root=imagenet_subset_root,
                imagenet_resize=int(args.imagenet_resize),
            )
        )
        spec = FixedIndexSpec(dataset=dataset, split="test", n=int(args.n_eval), seed=seed)
        path = (eval_dir / spec.filename()).resolve()
        get_or_create_fixed_indices(spec, dataset_length=len(ds_test), assets_dir=eval_dir)
        print(f"OK eval  -> {path}")

    if calib_dir is not None:
        ds_train = load_dataset(
            DatasetSpec(
                name=dataset,
                split="train",
                root=data_root,
                imagenet_subset_root=imagenet_subset_root,
                imagenet_resize=int(args.imagenet_resize),
            )
        )
        spec = FixedIndexSpec(dataset=dataset, split="train", n=int(args.n_calib), seed=seed)
        path = (calib_dir / spec.filename()).resolve()
        get_or_create_fixed_indices(spec, dataset_length=len(ds_train), assets_dir=calib_dir)
        print(f"OK calib -> {path}")


if __name__ == "__main__":
    main()
