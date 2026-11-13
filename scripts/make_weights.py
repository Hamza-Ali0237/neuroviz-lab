"""Generate pretrained and initial weights for the built-in models.

Plain NumPy, run offline. Writes to ``src/nvzlab/weights/``:

- ``logistic_regression_f{1,2}.npz``: ``w``, ``b`` (gradient descent on ``blobs``) and
  ``initial_w``, ``initial_b`` (fixed starting point for train traces);
- ``mlp.npz``: ``W1``, ``b1``, ``W2``, ``b2`` (gradient descent on ``moons``) and their
  ``initial_*`` counterparts;
- ``decision_tree_{dataset}_f{features}_{criterion}.npz``: the tree grown to depth 3 by
  ``nvzlab.models.decision_tree.grow`` as arrays ``feature``, ``threshold``, ``left``,
  ``right``, ``counts``.

Run from the repo root: ``python scripts/make_weights.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nvzlab import datasets
from nvzlab.models.decision_tree import CRITERIA, MAX_DEPTH, grow

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


def mlp_initial(seed: int = 2) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 0.8, size=(2, 4)),
        "b1": np.zeros(4),
        "W2": rng.normal(0.0, 0.8, size=(4, 2)),
        "b2": np.zeros(2),
    }


def fit_mlp(
    x: np.ndarray, y: np.ndarray, p: dict[str, np.ndarray], lr: float = 0.5, epochs: int = 5000
) -> dict[str, np.ndarray]:
    p = {k: v.copy() for k, v in p.items()}
    onehot = np.eye(2)[y]
    for _ in range(epochs):
        z1 = x @ p["W1"] + p["b1"]
        h = np.maximum(z1, 0.0)
        z2 = h @ p["W2"] + p["b2"]
        e = np.exp(z2 - z2.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        g2 = (prob - onehot) / len(y)
        g1 = (g2 @ p["W2"].T) * (z1 > 0)
        p["W2"] -= lr * h.T @ g2
        p["b2"] -= lr * g2.sum(axis=0)
        p["W1"] -= lr * x.T @ g1
        p["b1"] -= lr * g1.sum(axis=0)
    return p


def mlp_accuracy(x: np.ndarray, y: np.ndarray, p: dict[str, np.ndarray]) -> float:
    h = np.maximum(x @ p["W1"] + p["b1"], 0.0)
    return float(np.mean((h @ p["W2"] + p["b2"]).argmax(axis=1) == y))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in (1, 2):
        ds = datasets.load("blobs", features=f)
        w, b = fit(ds.X, ds.y.astype(float))
        w0, b0 = INITIAL[f]
        np.savez(OUT / f"logistic_regression_f{f}.npz", w=w, b=b, initial_w=w0, initial_b=b0)
        acc = np.mean(((ds.X @ w + b) > 0) == ds.y)
        print(f"logistic_regression features={f}: w={w.round(4)} b={b.round(4)} accuracy={acc:.3f}")

    ds = datasets.load("moons")
    p0 = mlp_initial()
    p = fit_mlp(ds.X, ds.y, p0)
    np.savez(OUT / "mlp.npz", **p, **{"initial_" + k: v for k, v in p0.items()})
    print(f"mlp: train accuracy={mlp_accuracy(ds.X, ds.y, p):.3f}")

    for name, features in (("blobs", 1), ("blobs", 2), ("moons", 2)):
        ds = datasets.load(name, features=features)
        for criterion in CRITERIA:
            tree = grow(ds.X, ds.y, ds.feature_names, criterion, MAX_DEPTH)
            np.savez(OUT / f"decision_tree_{name}_f{features}_{criterion}.npz", **tree)
            acc = np.mean(_tree_predict(ds.X, tree).argmax(axis=1) == ds.y)
            n_nodes = len(tree["feature"])
            print(
                f"decision_tree {name} f{features} {criterion}: nodes={n_nodes} accuracy={acc:.3f}"
            )


def _tree_predict(x: np.ndarray, tree: dict[str, np.ndarray]) -> np.ndarray:
    from nvzlab import ops

    return ops.tree_predict(
        x, *(tree[k] for k in ("feature", "threshold", "left", "right", "counts"))
    )


if __name__ == "__main__":
    main()
