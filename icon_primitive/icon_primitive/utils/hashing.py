"""Hashing utilities for reproducibility.

Why full SHA256?
---------------
Receipts are an *audit artifact*. Truncating SHA256 increases the chance of
collisions and makes cross-run provenance weaker. For v1.1 we therefore use
the full 64-hex SHA256 digest everywhere.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import numpy as np
import torch


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_any(obj: Any) -> str:
    """Hash common Python/numpy/torch objects stably."""
    if obj is None:
        return _sha256_bytes(b"null")

    if isinstance(obj, (bytes, bytearray, memoryview)):
        return _sha256_bytes(bytes(obj))

    if isinstance(obj, str):
        return _sha256_bytes(obj.encode("utf-8"))

    if isinstance(obj, (int, float, bool)):
        return _sha256_bytes(repr(obj).encode("utf-8"))

    if isinstance(obj, dict):
        # JSON canonicalization
        return _sha256_bytes(
            json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )

    if isinstance(obj, (list, tuple)):
        # NOTE: sort_keys=True also sorts nested dict keys inside lists/tuples.
        return _sha256_bytes(
            json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )

    if isinstance(obj, np.ndarray):
        header = f"ndarray:{obj.dtype}:{obj.shape}".encode("utf-8")
        return _sha256_bytes(header + obj.tobytes(order="C"))

    if torch.is_tensor(obj):
        t = obj.detach().contiguous().cpu()
        header = f"tensor:{str(t.dtype)}:{tuple(t.shape)}".encode("utf-8")
        return _sha256_bytes(header + t.numpy().tobytes(order="C"))

    # Fallback: repr
    return _sha256_bytes(repr(obj).encode("utf-8"))


def hash_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    """Stable hash for a torch state_dict."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        h.update(k.encode("utf-8"))
        t = v.detach().contiguous().cpu()
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes(order="C"))
    return h.hexdigest()
