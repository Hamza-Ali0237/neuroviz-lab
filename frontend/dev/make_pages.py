"""Write the viewer pages the screenshot script drives. Usage: python make_pages.py <out-dir>"""

import sys
from pathlib import Path

import numpy as np

import nvzlab as nvz
from nvzlab import ops

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
nvz.trace("logistic_regression", features=2).save(out / "lr-predict-f2.html")
nvz.trace("logistic_regression", kind="train", samples="batch", features=2, iterations=2).save(
    out / "lr-train-batch-f2.html"
)


@nvz.op
def clamp_half(x):
    return np.minimum(x, 0.5)


def opaque_model(x, p):
    return clamp_half(ops.sigmoid(ops.linear(x, p["w"], p["b"])))


params = {"w": np.array([0.5, -0.3]), "b": np.zeros(1)}
x = np.array([[1.0, 2.0], [-1.0, 0.5]])
nvz.trace(opaque_model, x, np.array([1.0, 0.0]), params=params, kind="train", loss="bce").save(
    out / "opaque.html"
)
print(sorted(p.name for p in out.iterdir()))
