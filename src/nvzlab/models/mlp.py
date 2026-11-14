"""Two-layer perceptron, written with the public level-2 API."""

from __future__ import annotations

from importlib import resources

import numpy as np

from nvzlab import datasets, ops
from nvzlab.core.errors import NvzError
from nvzlab.core.tape import context, event, step
from nvzlab.models.base import Model

PROB_GRID = 41
GRID_MARGIN = 0.5
MIN_ACTIVE_HIDDEN = 2


class MLP(Model):
    """2 inputs -> 4 hidden (ReLU) -> 2 logits -> softmax, trained on ``moons``.

    Default sample rule (``x`` not given): the first row of ``moons`` in dataset order that
    the pretrained weights classify correctly and that activates at least
    ``MIN_ACTIVE_HIDDEN`` (2) of the 4 hidden ReLU units; ``samples="batch"`` takes the
    first 4 such rows. A random pick can land on a row where every hidden unit is 0, and
    then the network diagram shows nothing happening.
    """

    name = "mlp"
    display_name = "MLP"
    family = "mlp"
    description = "2 inputs -> 4 hidden (ReLU) -> 2 classes (softmax); cross-entropy on moons"
    supports = {"samples": ("single", "batch"), "features": (2,)}
    loss = "nll"

    def __init__(self, features: int = 2) -> None:
        self.check_option("features", features)
        self.features = features
        with resources.as_file(resources.files("nvzlab.weights") / "mlp.npz") as path:
            with np.load(path) as z:
                names = ("W1", "b1", "W2", "b2")
                self.params = {k: z[k] for k in names}
                self.initial_params = {k: z["initial_" + k] for k in names}
        for p in (*self.params.values(), *self.initial_params.values()):
            p.setflags(write=False)
        self.dataset = datasets.load("moons")

    def forward(self, x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.features:
            raise NvzError(
                f"mlp: expected x with shape (n_samples, {self.features}), received {x.shape}. "
                "Fix: reshape x to two columns (x1, x2)."
            )
        with step("Input"):
            event("input", {"n_samples": int(x.shape[0]), "n_features": int(x.shape[1])}, refs=[x])
        with step("Weighted sum", backward_title="Gradients reach W1 and b1", layer="Hidden layer"):
            z1 = ops.linear(x, p["W1"], p["b1"])
        with step("ReLU", backward_title="Gradient through ReLU", layer="Hidden layer"):
            h = ops.relu(z1)
        with step(
            "Weighted sum (logits)",
            backward_title="Gradients reach W2 and b2",
            layer="Output layer",
        ):
            z2 = ops.linear(h, p["W2"], p["b2"])
        with step("Softmax", backward_title="Gradient through softmax", layer="Output layer"):
            prob = ops.softmax(z2)
        with step("Prediction"):
            event(
                "prediction",
                {
                    "predicted_class": prob.argmax(axis=1).tolist(),
                    "probability": prob.max(axis=1).tolist(),
                },
                refs=[prob],
            )
        return prob

    def default_sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """The first ``n`` rows (dataset order) that are classified correctly by the
        pretrained weights and activate at least ``MIN_ACTIVE_HIDDEN`` hidden units."""
        ds, p = self.dataset, self.params
        h = np.maximum(ds.X @ p["W1"] + p["b1"], 0.0)
        correct = (h @ p["W2"] + p["b2"]).argmax(axis=1) == ds.y
        idx = np.flatnonzero(correct & ((h > 0).sum(axis=1) >= MIN_ACTIVE_HIDDEN))[:n]
        if len(idx) < n:
            raise NvzError(f"mlp: only {len(idx)} rows satisfy the default-sample rule, need {n}.")
        x, y = ds.X[idx].copy(), ds.y[idx].copy()
        x.setflags(write=False)
        y.setflags(write=False)
        return x, y

    # --- context -------------------------------------------------------------------
    def grid_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Grid values of x1 and x2: the data range of ``moons`` plus a 0.5 margin."""
        lo, hi = self.dataset.X.min(axis=0) - GRID_MARGIN, self.dataset.X.max(axis=0) + GRID_MARGIN
        return np.linspace(lo[0], hi[0], PROB_GRID), np.linspace(lo[1], hi[1], PROB_GRID)

    def prob_grid(self, p: dict[str, np.ndarray]) -> np.ndarray:
        """Class-1 probability on the grid for params ``p`` (plain NumPy, nothing recorded)."""
        g1, g2 = self.grid_axes()
        a1, a2 = np.meshgrid(g1, g2, indexing="ij")
        pts = np.stack([a1.ravel(), a2.ravel()], axis=1)
        h = np.maximum(pts @ p["W1"] + p["b1"], 0.0)
        z = h @ p["W2"] + p["b2"]
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return (e[:, 1] / e.sum(axis=1)).reshape(PROB_GRID, PROB_GRID)

    def _record_grid(self, k: int, p: dict[str, np.ndarray], when: str) -> None:
        context(
            f"prob_grid_{k}",
            self.prob_grid(p),
            f"Probability of class 1 on a {PROB_GRID}×{PROB_GRID} grid over (x1, x2) covering "
            f"the moons data range plus {GRID_MARGIN}, computed with the parameters {when}. "
            "Real values from the same forward function; the decision regions are drawn from "
            "this grid.",
        )

    def record_context(
        self, x: np.ndarray, y: np.ndarray | None, p: dict[str, np.ndarray], kind: str
    ) -> None:
        ds = self.dataset
        context(
            "dataset_X",
            ds.X,
            "The whole moons dataset, for orientation. Only the highlighted samples are "
            "computed in this trace.",
        )
        context("dataset_y", ds.y, "Labels of the whole moons dataset (0 or 1).")
        g1, g2 = self.grid_axes()
        context("prob_grid_x1", g1, "Grid values of x1 for the prob_grid tensors.")
        context("prob_grid_x2", g2, "Grid values of x2 for the prob_grid tensors.")
        self._record_grid(0, p, "pretrained" if kind == "predict" else "before training")

    def record_path(
        self,
        x: np.ndarray,
        y: np.ndarray,
        path: list[dict[str, np.ndarray]],
        losses: list[float],
    ) -> None:
        """One probability grid per visited parameter set after the first (``prob_grid_k``
        = after update ``k``), so the decision regions can be replayed as the model learns."""
        for k, p in enumerate(path[1:], start=1):
            self._record_grid(k, p, f"after update {k}")
