"""A user op with no formula, no text and no backward: the opaque, non-differentiable path.

The trace shows the op with its name, shapes and values only (``explain`` is null) and the
backward pass records a ``backward_stopped`` event at it instead of inventing a gradient.
"""

import numpy as np

import nvzlab as nvz
from nvzlab import ops


@nvz.op
def clamp_half(x):
    return np.minimum(x, 0.5)


params = {"w": np.array([0.5, -0.3]), "b": np.zeros(1)}


def model(x, p):
    return clamp_half(ops.sigmoid(ops.linear(x, p["w"], p["b"])))


x = np.array([[1.0, 2.0], [-1.0, 0.5]])
y = np.array([1.0, 0.0])

tr = nvz.trace(model, x, y, params=params, kind="train", loss="bce")
print(tr)
print([e.payload for e in tr.events if e.kind == "backward_stopped"])
tr.save("03_opaque_op.json")
