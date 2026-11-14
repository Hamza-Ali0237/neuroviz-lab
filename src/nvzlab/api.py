"""Top-level ``trace`` and ``show``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from nvzlab.core.errors import NvzError
from nvzlab.core.tape import Tape
from nvzlab.models.base import Model
from nvzlab.models.registry import load as load_model
from nvzlab.ops import bce, cross_entropy, nll
from nvzlab.tracing.records import Trace

_LOSSES: dict[str, Callable[..., np.ndarray]] = {
    "bce": bce,
    "cross_entropy": cross_entropy,
    "nll": nll,
}
_BATCH = 4


def trace(
    model: str | Model | Callable[[np.ndarray, dict[str, np.ndarray]], np.ndarray],
    x: Any = None,
    y: Any = None,
    *,
    params: dict[str, np.ndarray] | None = None,
    kind: str = "predict",
    samples: str | None = None,
    features: int | None = None,
    loss: str | Callable[..., np.ndarray] | None = None,
    lr: float | None = None,
    iterations: int = 1,
    dim: str = "2d",
    name: str | None = None,
    **model_options: Any,
) -> Trace:
    """Run ``model`` step by step and return the recorded :class:`Trace`.

    ``model`` is a built-in name, a function ``f(x, params)`` built from ``nvz.ops``, or a
    ``nvz.Model`` instance. With a built-in name, ``x``/``y`` default to a sample from the
    model's dataset. ``kind="train"`` records forward, loss, backward and one SGD update per
    iteration (at most 3); ``lr`` defaults to the model's ``default_lr`` (0.1 for functions).
    User ``params`` are never modified. ``name`` overrides the display name (default: the
    built-in's ``display_name`` such as "Logistic Regression", the function's ``__name__``
    or the class name); the ``run_id`` always uses the built-in name.

    Examples
    --------
    >>> import nvzlab as nvz
    >>> tr = nvz.trace("logistic_regression")
    >>> tr = nvz.trace("logistic_regression", kind="train", samples="batch", iterations=2)
    """
    _choice("kind", kind, ("predict", "train"))
    _choice("dim", dim, ("2d", "3d"))
    if samples is not None:
        _choice("samples", samples, ("single", "batch"))
    if not isinstance(iterations, int) or not 1 <= iterations <= 3:
        raise NvzError(
            f"iterations must be 1, 2 or 3 (nvzlab shows a few steps, not a training loop), "
            f"received {iterations!r}."
        )

    provenance = "user"
    run_id_name = name
    if isinstance(model, str):
        if features is not None:
            model_options["features"] = features
        model = load_model(model, **model_options)
        provenance = "builtin-pretrained" if kind == "predict" else "builtin-initial"
    elif model_options:
        raise NvzError(
            f"Unexpected options {sorted(model_options)} for a user model. "
            "Model options apply to built-in names only."
        )

    forward: Callable[..., np.ndarray]
    self_trained = False  # the model records its own (non-gradient) training
    if isinstance(model, Model):
        if samples is not None:
            model.check_option("samples", samples)
        forward, family = model.forward, model.family
        if name is None:
            name = model.name if model.name != Model.name else type(model).__name__
            run_id_name, name = name, model.display_name or name
        self_trained = kind == "train" and model.trains_itself
        if params is None:
            params = (
                model.initial_params
                if kind == "train" and model.initial_params is not None
                else model.params
            )
        if loss is None:
            loss = model.loss
        if lr is None:
            lr = model.default_lr
        dataset = model.dataset
        if x is None and dataset is not None:
            if self_trained:
                if samples == "single":
                    raise NvzError(
                        f"{name}: kind='train' learns from the whole dataset, so "
                        "samples='single' does not apply. Fix: drop `samples` or use "
                        "kind='predict' to follow one sample."
                    )
                x, y_default = dataset.X, dataset.y
            else:
                x, y_default = model.default_sample(_BATCH if samples == "batch" else 1)
            y = y_default if y is None else y
        dataset_info = dataset.to_dict() if dataset is not None else None
    elif callable(model):
        forward, family = model, "custom"
        name = name or getattr(model, "__name__", type(model).__name__)
        run_id_name = run_id_name or name
        dataset_info = None
    else:
        raise NvzError(
            f"model must be a built-in name, a function f(x, params) or a nvz.Model, "
            f"received {type(model).__name__}."
        )
    if x is None:
        raise NvzError(
            f"{name}: x is required for a user model. "
            "Fix: pass x, e.g. nvz.trace(model, x, params=p)."
        )
    if lr is None:
        lr = 0.1

    x = np.asarray(x, dtype=np.float64)
    n = int(x.shape[0]) if x.ndim >= 2 else 1
    inferred = "single" if n == 1 else "batch"
    if samples is not None and samples != inferred:
        raise NvzError(
            f"samples={samples!r} but x has {n} row(s), which is {inferred!r}. "
            "Fix: drop `samples` or pass x with matching rows."
        )
    mode = {
        "samples": inferred,
        "n_samples": n,
        "n_features": int(np.prod(x.shape[1:])) if x.ndim >= 2 else int(x.size),
    }
    params = {k: np.array(v, dtype=np.float64) for k, v in (params or {}).items()}

    if kind == "train":
        if y is None:
            raise NvzError(
                "kind='train' needs labels. Fix: pass y (one label per sample), e.g. "
                "nvz.trace(model, x, y, params=p, kind='train', loss='bce')."
            )
        y = np.asarray(y, dtype=np.float64)
    if kind == "train" and not self_trained:
        if loss is None:
            raise NvzError(
                "kind='train' needs a loss. Fix: pass loss='bce' (probabilities, 0/1 labels), "
                "loss='cross_entropy' (logits, class indices) or a callable built from nvz.ops."
            )
        if isinstance(loss, str):
            if loss not in _LOSSES:
                raise NvzError(
                    f"Unknown loss {loss!r}. "
                    f"Valid names: {', '.join(sorted(_LOSSES))}, or a callable."
                )
            loss_fn = _LOSSES[loss]
        else:
            loss_fn = loss

    with Tape(name=name) as tape:
        tape.watch(x, role="input", name="x")
        if y is not None:
            tape.watch(y, role="label", name="y")
        for k, v in params.items():
            tape.watch(v, role="param", name=k)
        if isinstance(model, Model):
            model.record_context(x, y, params, kind)
        if self_trained:
            tape.phase = "update"
            for it in model.train(x, y, iterations):
                tape.iteration = it
        path = [params]  # the parameter sets visited: initial, then after each update
        loss_ids: list[str] = []
        for it in range(0 if self_trained else iterations if kind == "train" else 1):
            tape.iteration = it
            tape.phase = "forward"
            out = forward(x, params)
            out_id = tape.tensor_id(out)
            if out_id is None:
                raise NvzError(
                    f"{name} returned an array that no nvz.ops call produced. "
                    "Fix: build the output with nvz.ops (or a @nvz.op function) "
                    "so it can be traced."
                )
            tape.tensors[out_id].role = "output"
            tape.tensors[out_id].name = tape.tensors[out_id].name or "output"
            if kind == "predict":
                break
            tape.phase = "loss"
            with tape.step("Loss", backward_title="Gradient of the loss"):
                loss_val = loss_fn(out, y)
            loss_id = tape.tensor_id(loss_val)
            if loss_id is None:
                raise NvzError(
                    "The loss callable must return the output of an nvz.ops call "
                    "(e.g. ops.mean(...))."
                )
            tape.tensors[loss_id].role = "loss"
            tape.tensors[loss_id].name = "loss"
            loss_ids.append(loss_id)
            grads = tape.backward(loss_val)
            params = tape.sgd_update(params, grads, lr)
            path.append(params)

    if loss_ids:
        # Exact loss at every visited parameter set. The first ones are the recorded loss
        # tensors; the loss after the final update is computed here with the same function
        # and samples, outside the tape so nothing extra is recorded.
        losses = [float(tape.tensors[t].array) for t in loss_ids]
        losses.append(float(loss_fn(forward(x, path[-1]), y)))
        with tape:
            tape.context(
                "loss_path",
                np.array(losses),
                f"Exact loss at each visited parameter set: before training, then after "
                f"each of the {len(losses) - 1} update(s). Same loss function and the same "
                f"{mode['n_samples']} sample(s) as the trace; the last value is evaluated "
                "after the final update.",
            )
            if isinstance(model, Model):
                model.record_path(x, y, path, losses)

    return tape.to_trace(
        kind=kind,
        model={"name": name, "family": family, "provenance": provenance},
        run_id=f"{run_id_name}-{kind}-{mode['samples']}-f{mode['n_features']}",
        dataset=dataset_info,
        mode=mode,
        meta={
            "seed": 0,
            "iterations": iterations if kind == "train" else 1,
            "lr": lr if kind == "train" else None,
        },
    )


def show(model: Any, x: Any = None, y: Any = None, **options: Any) -> Trace:
    """Trace ``model`` and open the interactive viewer; same signature as :func:`trace`.

    Inline in Jupyter, otherwise in the default browser. Returns the trace so you can keep
    working with it (``tr.save("run.html")``). ``capture="square"`` or ``"vertical"`` opens
    capture mode for screen recording (``capture_step_seconds`` per step, default 2).

    Examples
    --------
    >>> import nvzlab as nvz
    >>> tr = nvz.show("logistic_regression")  # doctest: +SKIP
    >>> tr = nvz.show("logistic_regression", samples="batch", kind="train")  # doctest: +SKIP
    >>> tr = nvz.show("mlp", dim="3d", capture="vertical")  # doctest: +SKIP
    """
    from nvzlab import viewer

    capture = options.pop("capture", None)
    seconds = options.pop("capture_step_seconds", 2.0)
    viewer.check_capture(capture, seconds)
    tr = trace(model, x, y, **options)
    tr.show(dim=options.get("dim", "2d"), capture=capture, capture_step_seconds=seconds)
    return tr


def _choice(option: str, value: Any, valid: tuple[str, ...]) -> None:
    if value not in valid:
        raise NvzError(
            f"{option}={value!r} is not valid. Valid values: {', '.join(repr(v) for v in valid)}."
        )
