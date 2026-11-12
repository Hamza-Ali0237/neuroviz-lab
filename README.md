# neuroviz-lab

**nvzlab** runs small machine-learning models step by step and records exactly what happens
to the data: the forward pass, the loss, the backward pass and the parameter update. Every
number in a trace comes from the engine, and a trace is plain JSON that any viewer can
render. Built for students and developers who want to *see* a model work, not just call it.
The runtime depends on NumPy only.

## Install

```bash
pip install neuroviz-lab
```

## Quickstart

```python
import nvzlab as nvz

tr = nvz.trace("logistic_regression", kind="train")  # forward, loss, backward, update
print(tr)  # step-by-step summary
tr.save("run.json")  # the full trace, ready for a viewer
```

`nvz.list_models()`, `nvz.list_datasets()` and `nvz.list_ops()` show what is built in.
The interactive HTML viewer (`nvz.show(...)`, `tr.save("run.html")`) arrives in the next
release step.

## Bring your own model

A model is an ordinary Python function on NumPy arrays. Build it from `nvz.ops` and it runs
as plain NumPy outside a trace and is recorded automatically inside one. Custom ops are one
decorator away.

```python
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


params = {"w": np.array([0.5, -0.3]), "b": np.zeros(1)}


def my_model(x, p):
    with nvz.step("Weighted sum"):
        z = ops.linear(x, p["w"], p["b"])
    return ops.sigmoid(swish(z))


x = np.array([[1.0, 2.0], [-1.0, 0.5]])
y = np.array([1.0, 0.0])

my_model(x, params)  # plain NumPy
tr = nvz.trace(my_model, x, y, params=params, kind="train", loss="bce", lr=0.1, iterations=2)
print(tr)
tr.save("my_model.json")
```

## Command line

```bash
nvzlab list
nvzlab demo logistic_regression --kind train --json run.json
```

## License

MIT, Hamza Ali.
