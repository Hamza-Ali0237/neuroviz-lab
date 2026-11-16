# neuroviz-lab

**See inside the model.**

<p align="center">
  <img src="https://github.com/user-attachments/assets/21ce3c06-9e13-4526-a380-d81d79421b0d" alt="neuroviz-lab demo" width="520">
</p>

neuroviz-lab runs a small machine-learning model one step at a time and shows what
happens to the data: the forward pass, the loss, the backward pass and the parameter
update, as interactive 2D and 3D views. Every number on screen was computed by the
engine; nothing is drawn from a formula alone. The runtime depends on NumPy only.

## Install

```bash
pip install neuroviz-lab
```

## Quickstart

```python
import nvzlab as nvz

tr = nvz.show("transformer")  # one line: opens the step-by-step viewer
tr = nvz.show("cnn", kind="train", samples="batch")  # forward, loss, backward, update
tr.save("run.html")  # one self-contained page; "run.json" for the raw trace
```

`nvz.show` works inline in Jupyter and opens the browser from a script. Step with ←/→,
play with space, click any number to see the tensor it came from and, where a rule exists,
the arithmetic that produced it. Swap the name for any built-in below; `dim="3d"` opens
the 3D renderer.

## What's in v1

Five built-in models, each with fixed weights and a small synthetic dataset, so the same
call always shows the same numbers:

| model | what it is | dataset |
| --- | --- | --- |
| `logistic_regression` | `sigmoid(xw + b)`, 1 or 2 features, with the loss landscape in 3D | `blobs` |
| `mlp` | 2 → 4 ReLU → 2 softmax, steps grouped into layers | `moons` |
| `decision_tree` | depth 3, Gini or entropy; train traces grow it one level per iteration and record every candidate split | `blobs` or `moons` |
| `cnn` | 3×8×8 image → conv 3→4 (3×3) → ReLU → max pool 2×2 → 64 → 3 classes; the pool records which position won each window | `shapes` |
| `transformer` | token + positional embeddings (d=8), one encoder block with single-head self-attention written out op by op, feed-forward, residuals → 2 classes | `tokens` |

Also in v1:

- Two trace kinds, `predict` and `train` (1 to 3 iterations; it is a visualizer, not a
  training loop). Backward steps mirror the forward steps in reverse, and gradient fan-in
  shows up as explicit accumulate nodes.
- Your own model as a plain Python function on NumPy arrays built from `nvz.ops`, and
  custom ops with `@nvz.op` (with or without a backward rule; the trace says which).
- A documented JSON trace format (`nvzlab.trace/0`) that renderers and other tools can
  read without Python.
- `trace.save("run.html")` writes one self-contained page; `nvzlab demo <model>` does it
  from the terminal.
- Scenes that move with the numbers: the logistic-regression decision boundary or sigmoid
  curve plus its loss landscape in 3D, the MLP network diagram with activations, weights
  and gradients and its decision regions, and the decision tree's partition of the plane
  with every candidate split.
- Capture mode for screen recording (`capture="square"` or `"vertical"`): controls hidden,
  a fixed 1080×1080 or 1080×1920 viewport, larger type, looping auto-play.
- Every op carries a plain-language explanation and a formula for both directions, and
  errors say what was expected, what was received and how to fix it.

## What's not yet

- No dedicated scenes for `cnn` and `transformer` yet: they show the step list, the
  tensors as tinted matrices (the attention weights included) and the op cards, but no
  image view and no attention heat-map picture.
- No worked-arithmetic cards for `conv2d`, `maxpool2d`, `layernorm` or attention in the
  viewer: those ops show their formula, shapes and values, not a per-cell derivation.
- 4-D tensors render as flattened grids.
- Multi-head attention (`heads=2`) is not built in.
- No PyTorch, TensorFlow or JAX adapters, no uploading your own data, no editing the
  built-in datasets.

## What's next

See [ROADMAP.md](ROADMAP.md). neuroviz-lab is the visual engine for a larger learning
project.

## Bring your own model

```python
import numpy as np
import nvzlab as nvz
from nvzlab import ops

params = {"w": np.array([0.5, -0.3]), "b": np.zeros(1)}


def my_model(x, p):
    with nvz.step("Weighted sum"):
        z = ops.linear(x, p["w"], p["b"])
    return ops.sigmoid(z)


x = np.array([[1.0, 2.0], [-1.0, 0.5]])
y = np.array([1.0, 0.0])
tr = nvz.trace(my_model, x, y, params=params, kind="train", loss="bce", iterations=2)
print(tr)  # a text table of steps, ops and shapes
tr.save("my_model.html")
```

Outside a trace the same function runs as ordinary NumPy. `nvz.list_models()`,
`nvz.list_datasets()` and `nvz.list_ops()` show what is built in; `examples/` has one
script per model plus a custom op and an opaque op.

## Command line

```bash
nvzlab list
nvzlab demo cnn --kind train --samples batch   # opens the viewer
nvzlab demo mlp --json run.json                # raw trace instead
nvzlab open run.json
```

## Contributing

Issues and pull requests are welcome; [CONTRIBUTING.md](CONTRIBUTING.md) covers the
development setup, the test suite, the frontend build and how to add an op or a model.
Issues labelled `good first issue` are a reasonable place to start.

## License

MIT, Humza Ali.
