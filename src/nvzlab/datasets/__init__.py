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


_DATASETS = {"blobs": blobs}


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
