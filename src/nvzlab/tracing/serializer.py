"""Deterministic JSON encoding of traces: sorted keys, float32 little-endian base64."""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

MAX_ELEMENTS = 65_536


def encode_array(arr: np.ndarray) -> str:
    """Base64 of the array as little-endian float32, C order."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def decode_array(data: str, shape: list[int]) -> np.ndarray:
    """Inverse of :func:`encode_array`; returns float32."""
    return np.frombuffer(base64.b64decode(data), dtype="<f4").reshape(shape).copy()


def round6(v: float) -> float:
    """Round to 6 significant digits (stable across platforms)."""
    v = float(v)
    if v == 0.0 or not np.isfinite(v):
        return v
    return float(f"{v:.6g}")


def stats(arr: np.ndarray) -> dict[str, float]:
    """min/max/mean rounded to 6 significant digits."""
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": round6(arr.min()), "max": round6(arr.max()), "mean": round6(arr.mean())}


def to_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, compact separators, ASCII."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
