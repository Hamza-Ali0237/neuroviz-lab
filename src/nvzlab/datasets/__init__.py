"""Synthetic datasets, generated from fixed seeds. Read-only.

Examples
--------
>>> import nvzlab as nvz
>>> ds = nvz.datasets.load("blobs", features=2)
>>> ds.X.shape, ds.y.shape
((120, 2), (120,))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nvzlab.core.errors import NvzError, unknown_name


@dataclass(frozen=True)
class Dataset:
    """A small labelled dataset with read-only arrays."""

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    class_names: list[str]

    def sample(self, n: int = 1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Pick ``n`` rows at random (deterministic for a given ``seed``); arrays are read-only.

        Examples
        --------
        >>> x, y = nvz.datasets.load("blobs").sample(4)
        >>> x.shape
        (4, 2)
        """
        if not 1 <= n <= len(self.y):
            raise NvzError(f"sample: n must be between 1 and {len(self.y)}, received {n}.")
        idx = np.random.default_rng(seed).choice(len(self.y), size=n, replace=False)
        x, y = self.X[np.sort(idx)].copy(), self.y[np.sort(idx)].copy()
        x.setflags(write=False)
        y.setflags(write=False)
        return x, y

    def to_dict(self) -> dict:
        return {"name": self.name, "features": self.feature_names, "classes": self.class_names}


def blobs(features: int = 2, seed: int = 0) -> Dataset:
    """Two Gaussian classes, 120 rows, 1 or 2 features."""
    if features not in (1, 2):
        raise NvzError(f"blobs: features must be 1 or 2, received {features!r}.")
    rng = np.random.default_rng(seed)
    n = 60
    centers = np.array([[-1.5, -1.0], [1.5, 1.0]])[:, :features]
    x = np.concatenate([c + rng.normal(0.0, 1.0, size=(n, features)) for c in centers])
    y = np.repeat([0, 1], n)
    order = rng.permutation(2 * n)
    x, y = x[order], y[order]
    x.setflags(write=False)
    y.setflags(write=False)
    return Dataset("blobs", x, y, [f"x{i + 1}" for i in range(features)], ["0", "1"])


def moons(features: int = 2, seed: int = 0) -> Dataset:
    """Two interleaving half circles, 120 rows, 2 features; not linearly separable."""
    if features != 2:
        raise NvzError(f"moons: features must be 2, received {features!r}.")
    rng = np.random.default_rng(seed)
    n = 60
    t = np.linspace(0.0, np.pi, n)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    inner = np.stack([1.0 - np.cos(t), 1.0 - np.sin(t) - 0.5], axis=1)
    x = np.concatenate([outer, inner]) + rng.normal(0.0, 0.12, size=(2 * n, 2))
    y = np.repeat([0, 1], n)
    order = rng.permutation(2 * n)
    x, y = x[order], y[order]
    x.setflags(write=False)
    y.setflags(write=False)
    return Dataset("moons", x, y, ["x1", "x2"], ["0", "1"])


def shapes(seed: int = 0) -> Dataset:
    """90 RGB images of 8x8 pixels, layout (n, 3, 8, 8), values in [0, 1], three classes.

    Each image shows one shape (a filled 4x4 square, a 4x4 ring or a 5x5 cross) at a random
    position in a random colour over a dark, lightly noisy background.
    """
    rng = np.random.default_rng(seed)
    n_per_class, size = 30, 8
    masks = {
        "square": np.ones((4, 4), dtype=bool),
        "ring": np.pad(np.zeros((2, 2), dtype=bool), 1, constant_values=True),
        "cross": np.zeros((5, 5), dtype=bool),
    }
    masks["cross"][2, :] = masks["cross"][:, 2] = True
    x = np.zeros((3 * n_per_class, 3, size, size))
    y = np.repeat(np.arange(3), n_per_class)
    for i, label in enumerate(y):
        mask = masks[list(masks)[label]]
        r, c = rng.integers(0, size - mask.shape[0] + 1, size=2)
        colour = rng.uniform(0.5, 1.0, size=3)
        x[i, :, r : r + mask.shape[0], c : c + mask.shape[1]] = colour[:, None, None] * mask
    x = np.clip(x + rng.normal(0.0, 0.05, size=x.shape), 0.0, 1.0)
    order = rng.permutation(len(y))
    x, y = x[order], y[order]
    x.setflags(write=False)
    y.setflags(write=False)
    return Dataset("shapes", x, y, ["R", "G", "B"], list(masks))


_DATASETS = {"blobs": blobs, "moons": moons, "shapes": shapes}


def load(name: str, **options: int) -> Dataset:
    """Load a built-in dataset by name.

    Examples
    --------
    >>> nvz.datasets.load("blobs", features=1).feature_names
    ['x1']
    """
    if name not in _DATASETS:
        raise unknown_name("dataset", name, _DATASETS)
    return _DATASETS[name](**options)


def list_datasets() -> list[str]:
    """Names of the built-in datasets."""
    return sorted(_DATASETS)
