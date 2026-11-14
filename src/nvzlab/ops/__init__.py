"""Built-in ops: plain NumPy functions that are recorded when a tape is active.

Examples
--------
>>> import numpy as np
>>> from nvzlab import ops
>>> ops.sigmoid(ops.linear(np.array([[1.0, 2.0]]), np.array([0.5, -0.5]), np.array([0.0])))
array([0.37754067])
"""

from nvzlab.ops.activations import relu, sigmoid, softmax, tanh
from nvzlab.ops.attention import embedding, layernorm
from nvzlab.ops.conv import conv2d, flatten, maxpool2d
from nvzlab.ops.linalg import add, linear, matmul, mul, scale, transpose
from nvzlab.ops.losses import bce, cross_entropy, nll
from nvzlab.ops.reduce import mean, sum
from nvzlab.ops.tree import tree_predict

__all__ = [
    "add",
    "bce",
    "conv2d",
    "cross_entropy",
    "embedding",
    "flatten",
    "layernorm",
    "linear",
    "matmul",
    "maxpool2d",
    "mean",
    "mul",
    "nll",
    "relu",
    "scale",
    "sigmoid",
    "softmax",
    "sum",
    "tanh",
    "transpose",
    "tree_predict",
]
