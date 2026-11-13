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

tr = nvz.show("logistic_regression")  # opens the step-by-step viewer
tr = nvz.show("logistic_regression", samples="batch", kind="train")  # forward, loss, backward, update
tr.save("run.html")  # one self-contained page; "run.json" for the raw trace
```

`nvz.show` displays inline in Jupyter and opens the browser from a script. Step with
←/→, play with space, click any number to see where it came from and the arithmetic
that produced it. The 2D/3D toggle switches renderer without losing the step or the
selection (`dim="3d"` opens in 3D); every 3D scene states what height, colour and
position mean. `nvz.list_models()`, `nvz.list_datasets()` and `nvz.list_ops()` show
what is built in.

Built-in models: `logistic_regression` (on `blobs`, 1 or 2 features), `mlp` (2 → 4 ReLU → 2
softmax on `moons`, steps grouped into layers) and `decision_tree` (depth 3, Gini or entropy,
`dataset="blobs"` or `"moons"`; a train trace grows the tree one level per iteration and
records every candidate split).

```python
nvz.show("mlp", kind="train", samples="batch")
nvz.show("decision_tree", kind="train", dataset="moons", criterion="entropy", iterations=2)
```

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
# nvz.context("dataset_X", X_all, "note") inside the model records a tensor drawn faded
# behind the plots for orientation (e.g. the whole dataset); it is never computed on.
print(tr)
tr.save("my_model.json")
```

## Command line

```bash
nvzlab list
nvzlab demo logistic_regression --kind train --samples batch   # opens the viewer
nvzlab demo logistic_regression --json run.json                # raw trace instead
nvzlab open run.json                                           # view a saved trace
```

## Development

The viewer lives in `frontend/` (Vite, React, TypeScript, KaTeX) and builds into
`src/nvzlab/viewer/static/`, which is committed so `pip install` needs no Node.

```bash
cd frontend && npm install
npm run dev        # live viewer on frontend/dev/sample-trace.json
npm run typecheck && npm run build
python dev/make_pages.py /tmp/pages && npm run screenshots -- /tmp/pages ../../nvz-artifacts/p03b
```

The screenshot script drives headless Chrome with `--use-angle=swiftshader
--enable-unsafe-swiftshader` so the 3D scenes render without a GPU.

## License

MIT, Hamza Ali.
