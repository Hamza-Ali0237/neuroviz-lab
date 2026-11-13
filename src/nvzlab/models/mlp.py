"""Two-layer perceptron, written with the public level-2 API."""

from __future__ import annotations

from importlib import resources

import numpy as np

from nvzlab import datasets, ops
from nvzlab.core.errors import NvzError
from nvzlab.core.tape import event, step
from nvzlab.models.base import Model


class MLP(Model):
    """2 inputs -> 4 hidden (ReLU) -> 2 logits -> softmax, trained on ``moons``."""

    name = "mlp"
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
