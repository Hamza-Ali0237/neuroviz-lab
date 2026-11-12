"""Helpers shared by op families."""

from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape`` (inverse of NumPy broadcasting)."""
    grad = np.asarray(grad)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def broadcastable(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    try:
        np.broadcast_shapes(a, b)
        return True
    except ValueError:
        return False
