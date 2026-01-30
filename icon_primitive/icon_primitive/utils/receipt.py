"""Receipt creation & JSONSchema validation.

Receipts are the reproducibility *gate*. Any run without a valid receipt is
considered invalid and must be excluded from aggregation.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema
import torch

from .hashing import hash_any


def _now_utc_iso() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def get_git_info(repo_root: Path) -> Dict[str, Any]:
    """Best-effort git metadata."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=str(repo_root), stderr=subprocess.DEVNULL) != 0
        return {"commit": commit, "dirty": bool(dirty), "repo": str(repo_root)}
    except Exception:
        return {"commit": "unknown", "dirty": False, "repo": str(repo_root)}


def get_environment() -> Dict[str, Any]:
    dev = {"name": "CPU", "count": 0}
    if torch.cuda.is_available():
        dev = {"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count()}
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cudnn": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
        "device": dev,
        "os": platform.platform(),
        "docker_image": "",
    }


def load_schema(schema_path: Path) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_receipt(receipt: Dict[str, Any], schema_path: Path) -> None:
    schema = load_schema(schema_path)
    jsonschema.validate(instance=receipt, schema=schema)


def save_receipt(receipt: Dict[str, Any], path: Path, schema_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_receipt(receipt, schema_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)


def build_receipt(
    *,
    schema_path: Path,
    repo_root: Path,
    run_id: str,
    section: str,
    experiment_name: str,
    probe: Dict[str, Any],
    data: Dict[str, Any],
    model: Dict[str, Any],
    training: Dict[str, Any],
    measurement: Dict[str, Any],
    results: Dict[str, Any],
    hashes: Dict[str, Any],
    artifacts: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    receipt: Dict[str, Any] = {
        "spec_version": "1.1",
        "run_id": run_id,
        "timestamp_utc": _now_utc_iso(),
        "git": get_git_info(repo_root),
        "section": section,
        "experiment": {"name": experiment_name, "tags": [], "notes": notes},
        "probe": probe,
        "data": data,
        "model": model,
        "training": training,
        "measurement": measurement,
        "results": results,
        "environment": get_environment(),
        "hashes": hashes,
    }
    if artifacts is not None:
        receipt["artifacts"] = artifacts
    if extras is not None:
        receipt["extras"] = extras

    validate_receipt(receipt, schema_path)
    return receipt


def compute_data_hash(dataset_name: str, split_sizes: Dict[str, int], preprocess: Dict[str, Any]) -> str:
    return hash_any({"dataset": dataset_name, "splits": split_sizes, "preprocess": preprocess})
