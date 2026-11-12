"""Built-in ops: plain NumPy functions that are recorded when a tape is active.

Examples
--------
>>> import numpy as np
>>> from nvzlab import ops
>>> ops.sigmoid(ops.linear(np.array([[1.0, 2.0]]), np.array([0.5, -0.5]), np.array([0.0])))
array([0.37754067])
"""

from nvzlab.ops.activations import relu, sigmoid, softmax
from nvzlab.ops.linalg import add, linear, matmul
from nvzlab.ops.losses import bce, cross_entropy
from nvzlab.ops.reduce import mean, sum

__all__ = [
    "add",
    "bce",
    "cross_entropy",
    "linear",
    "matmul",
    "mean",
    "relu",
    "sigmoid",
    "softmax",
    "sum",
]
