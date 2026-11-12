"""Generate pretrained and initial weights for the built-in logistic regression.

Plain NumPy gradient descent on the ``blobs`` dataset, run offline. Writes
``src/nvzlab/weights/logistic_regression_f{1,2}.npz`` with keys ``w``, ``b``
(pretrained) and ``initial_w``, ``initial_b`` (fixed starting point for train traces).

Run from the repo root: ``python scripts/make_weights.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nvzlab import datasets

OUT = Path(__file__).resolve().parents[1] / "src" / "nvzlab" / "weights"
INITIAL = {1: (np.array([0.1]), np.array([0.0])), 2: (np.array([0.1, -0.1]), np.array([0.0]))}


def fit(
    x: np.ndarray, y: np.ndarray, lr: float = 0.5, epochs: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    w, b = np.zeros(x.shape[1]), np.zeros(1)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(x @ w + b)))
        err = (p - y) / len(y)
        w -= lr * x.T @ err
        b -= lr * err.sum(keepdims=True)
    return w, b


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in (1, 2):
        ds = datasets.load("blobs", features=f)
        w, b = fit(ds.X, ds.y.astype(float))
        w0, b0 = INITIAL[f]
        np.savez(OUT / f"logistic_regression_f{f}.npz", w=w, b=b, initial_w=w0, initial_b=b0)
        acc = np.mean(((ds.X @ w + b) > 0) == ds.y)
        print(f"features={f}: w={w.round(4)} b={b.round(4)} train accuracy={acc:.3f}")


if __name__ == "__main__":
    main()
