# Contributing to neuroviz-lab

Thanks for helping. This page covers the development setup, the checks every change
must pass, and the two most common contributions: a new op and a new built-in model.

## Setup

Python 3.10 or newer. The runtime depends on NumPy only; everything else is a dev extra.

```bash
git clone https://github.com/hamzaali/neuroviz-lab
cd neuroviz-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running the checks

```bash
pytest                       # unit tests, gradient checks, schema validation, determinism
ruff check . && ruff format --check src tests scripts examples
python -m build              # the wheel must install and import with only numpy present
```

The PyTorch parity tests (`tests/test_cnn.py`, `tests/test_mlp.py`,
`tests/test_transformer.py`) skip automatically when `torch` is not installed. If you
change an op those models use, run them once in an environment that has a CPU build of
PyTorch.

## Frontend (the viewer)

The viewer lives in `frontend/` (Vite, React, TypeScript, three.js, KaTeX) and builds into
`src/nvzlab/viewer/static/`. The built bundle is committed so that `pip install` needs no
Node; rebuild it and commit the result whenever you change `frontend/src`.

```bash
cd frontend && npm install
npm run dev          # live viewer on frontend/dev/sample-trace.json
npm run typecheck && npm run build
```

Screenshots for review: `python dev/make_pages.py /tmp/pages && npm run screenshots -- /tmp/pages <out-dir>`
(headless Chrome; the 3D scenes render through SwiftShader without a GPU).

## Adding an op

Ops live in `src/nvzlab/ops/`, one module per family (`linalg.py`, `activations.py`,
`losses.py`, `conv.py`, `attention.py`, `reduce.py`). An op is a plain NumPy function
decorated with `@nvz.op`, plus a backward rule:

```python
@op(
    category="activation",
    formula=r"\mathrm{relu}(x) = \max(0, x)",  # KaTeX
    text="Keeps positive values and sets negative ones to zero.",  # 1 to 3 plain sentences
    shape_rule="same shape as the input",
    backward_formula=r"\frac{\partial L}{\partial x} = g \cdot \mathbb{1}[x > 0]",
    backward_text="The upstream gradient g passes where the input was positive and is blocked elsewhere.",
)
def relu(x: np.ndarray) -> np.ndarray:
    """One-line summary, then a runnable example in the docstring."""
    return np.maximum(x, 0.0)


@relu.backward
def _(grad, x, out):  # (upstream grad, *inputs, out, **attrs) -> one grad per input
    return grad * (x > 0)
```

Checklist:

1. Validate shapes with `check_shape(...)` so the error names the op, what was expected,
   what was received and how to fix it.
2. Export it from `src/nvzlab/ops/__init__.py` (keep `__all__` sorted).
3. Add a test in `tests/test_ops.py`: compare against a plain NumPy reference and call
   `gradcheck(op, inputs, **attrs)` for every differentiable input. `gradcheck` uses
   central differences in float64 with rtol 1e-5 and atol 1e-7.
4. Explanations must be true for every input the op accepts. If you cannot explain it,
   leave `formula` and `text` out and the op is shown as opaque; never invent text.
5. An op that must store extra data for the viewer (for example which position won a
   max pool) subclasses `nvz.Op` and returns it from `node_attrs(ctx)`.
6. Add a line to `CHANGELOG.md` under "Unreleased".

## Adding a built-in model

Built-in models use exactly the public API users have (see `src/nvzlab/models/mlp.py`,
the smallest one, and `cnn.py`):

1. A module-level `forward(x, p)` made of `nvz.ops` calls, wrapped in `nvz.step(title,
   backward_title=..., layer=...)` blocks whose titles read well in a step list.
   `nvz.event` records non-tensor facts (the prediction, a chosen split); `nvz.context`
   records orientation tensors; `nvz.name` labels an intermediate tensor.
2. A `Model` subclass with `name`, `family`, `description`, `supports` (the valid
   `samples`/`features`/other options), `loss`, `params`, `initial_params`, `dataset`.
3. A dataset generator in `src/nvzlab/datasets/__init__.py`: synthetic, seeded, small
   (tabular ≤ 200 rows, images ≤ 12×12×3, sequences ≤ 8 tokens), read-only arrays.
4. Weights produced by `scripts/make_weights.py` and committed as `.npz`
   (`fit_with_tape` trains any forward function through a `Tape`). Report the training
   accuracy in the pull request.
5. Register it in `src/nvzlab/models/registry.py` and `src/nvzlab/models/__init__.py`.
6. Tests in `tests/test_<model>.py`: accuracy of the pretrained weights, numerical
   gradient check over every parameter, a parity test against a reference (scikit-learn
   or PyTorch, skipped when absent), and the expected step titles and layers. Add the
   model to the schema cases in `tests/test_trace.py` and to the lists asserted in
   `tests/test_models.py`, `tests/test_cli.py` and `tests/test_errors.py`.
7. An example script in `examples/`, a README row and a CHANGELOG line.

## Commit style

Small, focused commits with conventional prefixes and a scope:
`feat(ops): ...`, `feat(models): ...`, `fix(core): ...`, `test(ops): ...`,
`docs: ...`, `chore(viewer): ...`. The subject says what changed, in the imperative.

## Releasing

Releases are published to PyPI by `.github/workflows/release.yml` when a tag `v*` is pushed.
It uses PyPI trusted publishing, so there is no API token to store.

One-time setup (maintainer):

1. On PyPI, open *Your account → Publishing* and add a pending publisher for the project
   `neuroviz-lab`: owner `hamzaali`, repository `neuroviz-lab`, workflow `release.yml`,
   environment `pypi`. After the first release it becomes a regular trusted publisher.
2. On GitHub, *Settings → Environments → New environment* named `pypi`. Optionally require a
   reviewer there so every publish needs a click.

Each release:

```bash
# 1. CHANGELOG.md: set the date of the version section; pyproject.toml: bump `version`
# 2. commit on main, wait for CI to pass
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
# 3. when the workflow is green, create the GitHub release for the tag and paste the
#    CHANGELOG section as its notes
```

## Ground rules

- Real values only. The viewer never shows a number the engine did not compute; a
  simplification (normalised colours, sampled values) is labelled as one.
- Runtime dependency: NumPy. Anything else goes in a dev or optional extra.
- No telemetry.
- Traces must be deterministic: the same call twice gives the same JSON bytes
  (`tests/test_trace.py::test_determinism`).
