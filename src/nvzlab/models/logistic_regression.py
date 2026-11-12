"""Logistic regression, written with the public level-2 API."""

from __future__ import annotations

from importlib import resources

import numpy as np

from nvzlab import datasets, ops
from nvzlab.core.errors import NvzError
from nvzlab.core.tape import event, step
from nvzlab.models.base import Model


class LogisticRegression(Model):
    """Binary classifier: ``p = sigmoid(x @ w + b)``, trained on ``blobs``."""

    name = "logistic_regression"
    family = "linear"
    supports = {"samples": ("single", "batch"), "features": (1, 2)}
    loss = "bce"

    def __init__(self, features: int = 2) -> None:
        self.check_option("features", features)
        self.features = features
        with resources.as_file(
            resources.files("nvzlab.weights") / f"logistic_regression_f{features}.npz"
        ) as path:
            with np.load(path) as z:
                self.params = {"w": z["w"], "b": z["b"]}
                self.initial_params = {"w": z["initial_w"], "b": z["initial_b"]}
        for p in (*self.params.values(), *self.initial_params.values()):
            p.setflags(write=False)
        self.dataset = datasets.load("blobs", features=features)

    def forward(self, x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.features:
            raise NvzError(
                f"logistic_regression: expected x with shape (n_samples, {self.features}), "
                f"received {x.shape}. "
                f"Fix: reshape x, or load the model with features={x.shape[-1]}."
            )
        with step("Input"):
            event("input", {"n_samples": int(x.shape[0]), "n_features": int(x.shape[1])}, refs=[x])
        with step("Weighted sum z = Xw + b", backward_title="Gradients reach w and b"):
            z = ops.linear(x, p["w"], p["b"])
        with step("Sigmoid", backward_title="Gradient through the sigmoid"):
            prob = ops.sigmoid(z)
        with step("Prediction"):
            event(
                "prediction",
                {"threshold": 0.5, "predicted_class": (prob > 0.5).astype(int).tolist()},
                refs=[prob],
            )
        return prob
