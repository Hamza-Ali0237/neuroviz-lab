"""Bring your own model: a plain function built from nvz.ops plus a user-defined op."""

import numpy as np

import nvzlab as nvz
from nvzlab import ops


@nvz.op(category="activation", formula=r"x \cdot \sigma(x)", text="A smooth, ReLU-like gate.")
def swish(x):
    return x / (1.0 + np.exp(-x))


@swish.backward
def _(grad, x, out):
    s = 1.0 / (1.0 + np.exp(-x))
    return grad * (s + x * s * (1.0 - s))


params = {
    "w": np.array([[0.5, -0.3], [0.2, 0.8]]),
    "b": np.zeros(2),
    "v": np.array([1.0, -1.0]),
    "c": np.zeros(1),
}


def my_model(x, p):
    with nvz.step("Hidden layer"):
        h = swish(ops.linear(x, p["w"], p["b"]))
    with nvz.step("Output"):
        return ops.sigmoid(ops.linear(h, p["v"], p["c"]))


x = np.array([[1.0, 2.0], [-1.0, 0.5], [0.3, -0.7]])
y = np.array([1.0, 0.0, 1.0])

print("plain NumPy:", my_model(x, params))
tr = nvz.trace(my_model, x, y, params=params, kind="train", loss="bce", lr=0.1, iterations=2)
print(tr)
tr.save("02_custom_model.json")
print("params untouched:", params["w"].tolist())
