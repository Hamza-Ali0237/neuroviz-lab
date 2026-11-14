"""The ``Tape``: records every op call, runs reverse-mode backward, builds the trace.

Recording is implicit: while a tape is active (``with Tape(): ...``) every op call is
recorded. Outside a tape, ops are plain NumPy functions.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from nvzlab.core.errors import NvzError
from nvzlab.tracing.records import SCHEMA, Event, Node, Step, Tensor, Trace

if TYPE_CHECKING:
    from nvzlab.core.op import Op

_current: contextvars.ContextVar[Tape | None] = contextvars.ContextVar("nvzlab_tape", default=None)


def current_tape() -> Tape | None:
    """The active tape, or ``None`` outside a recording."""
    return _current.get()


class Tape:
    """Records op calls, events and steps; can run backward and export a trace.

    Examples
    --------
    >>> import numpy as np, nvzlab as nvz
    >>> from nvzlab import ops
    >>> x, w = np.array([[1.0, 2.0]]), np.array([0.5, -0.5])
    >>> with nvz.Tape(name="demo") as t:
    ...     t.watch(w, role="param", name="w")
    ...     loss = ops.sum(ops.matmul(x, w))
    ...     grads = t.backward(loss)
    >>> grads["w"]
    array([1., 2.])
    >>> trace = t.to_trace()
    """

    def __init__(self, name: str = "tape") -> None:
        self.name = name
        self.phase = "forward"
        self.iteration = 0
        self.tensors: dict[str, Tensor] = {}
        self.nodes: list[Node] = []
        self.events: list[Event] = []
        self.steps: list[Step] = []
        self._ids: dict[int, str] = {}  # id(array) -> tensor id
        self._refs: list[Any] = []  # keep arrays alive so ids are never reused
        self._ctx: dict[str, Any] = {}  # node id -> backward context
        self._ops: dict[str, Op] = {}
        self._backward_titles: dict[int, str] = {}  # forward step index -> backward title
        self._step_stack: list[Step] = []
        self._token: contextvars.Token | None = None
        self.did_backward = False

    # --- context -------------------------------------------------------------------
    def __enter__(self) -> Tape:
        if _current.get() is not None:
            raise NvzError("A tape is already active; nvzlab does not nest recordings.")
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        _current.reset(self._token)
        self._token = None

    # --- tensors -------------------------------------------------------------------
    def watch(
        self,
        arr: Any,
        role: str | None = None,
        name: str | None = None,
        computed: np.ndarray | None = None,
    ) -> str:
        """Return the tensor id of ``arr``, registering it on first sight.

        ``role`` and ``name`` are applied only when first registering (or when the tensor
        has no name yet). ``computed`` is the float64 view that ops actually use.
        """
        key = id(arr)
        tid = self._ids.get(key)
        if tid is None:
            tid = f"t{len(self.tensors)}"
            self._ids[key] = tid
            self._refs.append(arr)
            values = arr if computed is None else computed
            self.tensors[tid] = Tensor.from_array(tid, values, role or "input", name)
        else:
            t = self.tensors[tid]
            if name is not None and t.name is None:
                t.name = name
            if role is not None and role != "input":
                t.role = role
        return tid

    def context(self, name: str, arr: Any, note: str) -> str:
        """Record ``arr`` as a context tensor (shown for orientation, not computed).

        Recording the same ``name`` again (e.g. once per training iteration) is a no-op, and
        so is an array the tape already holds (e.g. a dataset that is also the traced ``x``):
        that tensor keeps its own name and role.
        """
        for tid, t in self.tensors.items():
            if t.role == "context" and t.name == name:
                return tid
        seen = self.tensor_id(arr)
        if seen is not None:
            return seen
        tid = self.watch(np.asarray(arr, dtype=np.float64), role="context", name=name)
        self.tensors[tid].note = note
        return tid

    def set_name(self, arr: Any, name: str) -> str:
        """Give a recorded array a display name (only if it has none yet); returns its id."""
        tid = self.watch(arr)
        if self.tensors[tid].name is None:
            self.tensors[tid].name = name
        return tid

    def tensor_id(self, arr: Any) -> str | None:
        """Tensor id of an array seen by this tape, or ``None``."""
        return self._ids.get(id(arr))

    # --- steps and events ----------------------------------------------------------
    @contextmanager
    def step(
        self, title: str, backward_title: str | None = None, layer: str | None = None
    ) -> Iterator[Step]:
        """Group everything recorded inside into one playback step.

        ``backward_title`` names the mirrored backward step (default
        ``"Backprop through: <title>"``); ``layer`` labels the step's layer (e.g.
        ``"Hidden layer 1"``) and is copied onto the mirrored backward step.
        """
        s = self._new_step(title, layer)
        if backward_title is not None:
            self._backward_titles[s.index] = backward_title
        self._step_stack.append(s)
        try:
            yield s
        finally:
            self._step_stack.pop()

    def _new_step(self, title: str, layer: str | None = None) -> Step:
        s = Step(len(self.steps), title, self.phase, iteration=self.iteration, layer=layer)
        self.steps.append(s)
        return s

    def _current_step(self, auto_title: str) -> Step:
        if self._step_stack:
            return self._step_stack[-1]
        return self._new_step(auto_title)

    def event(
        self, kind: str, payload: dict[str, Any] | None = None, refs: Sequence[Any] = ()
    ) -> Event:
        """Record a non-tensor algorithm step; ``refs`` are arrays or tensor ids."""
        step = self._current_step(kind)
        ids = [r if isinstance(r, str) else self.watch(r) for r in refs]
        e = Event(f"e{len(self.events)}", kind, self.phase, step.index, dict(payload or {}), ids)
        self.events.append(e)
        step.events.append(e.id)
        return e

    # --- nodes ---------------------------------------------------------------------
    def record(
        self, op: Op, in_ids: list[str], out: np.ndarray, attrs: dict[str, Any], ctx: Any
    ) -> Node:
        """Record one op call (called by ``Op.__call__``)."""
        step = self._current_step(op.name)
        out_id = self.watch(out, role="loss" if self.phase == "loss" else "activation")
        in_shapes = [self.tensors[t].shape for t in in_ids]
        node = Node(
            id=f"n{len(self.nodes)}",
            op=op.name,
            category=op.category,
            origin=op.origin,
            phase=self.phase,
            inputs=list(in_ids),
            outputs=[out_id],
            attrs=_jsonable({**attrs, **op.node_attrs(ctx)}),
            step=step.index,
            differentiable=op.differentiable,
            explain=op.explain(dict(attrs), in_shapes, tuple(out.shape)),
        )
        self.nodes.append(node)
        self._ctx[node.id] = ctx
        self._ops[node.id] = op
        step.nodes.append(node.id)
        return node

    # --- backward ------------------------------------------------------------------
    def backward(self, loss: Any, param_names: Sequence[str] = ()) -> dict[str, np.ndarray]:
        """Reverse-mode backward from ``loss`` over every recorded node.

        Returns gradients keyed by tensor id, plus by name for named tensors. Backward
        steps mirror, in reverse order, the forward steps of the current iteration whose
        nodes receive a gradient. When a tensor fed several ops, a ``grad_accumulate``
        node sums the partial gradients so fan-in stays visible.
        """
        loss_id = self.tensor_id(loss)
        if loss_id is None or not any(loss_id in n.outputs for n in self.nodes):
            raise NvzError(
                "Tape.backward: the loss must be the output of a recorded op. "
                "Fix: compute the loss with nvz.ops (e.g. ops.bce(pred, y)) inside the tape."
            )
        self.phase = "backward"
        self.did_backward = True
        grads: dict[str, np.ndarray] = {loss_id: np.ones(self.tensors[loss_id].shape)}
        grad_ids: dict[str, str] = {
            loss_id: self.watch(
                grads[loss_id], role="grad", name="grad_" + (self.tensors[loss_id].name or "loss")
            )
        }
        partials: dict[str, list[str]] = {loss_id: [grad_ids[loss_id]]}
        todo = [
            n
            for n in self.nodes
            if n.phase in ("forward", "loss") and self.steps[n.step].iteration == self.iteration
        ]
        # Global reverse order keeps the chain rule correct; a new backward step opens
        # whenever the forward step changes, so each forward step gets its mirror.
        current: Step | None = None
        for node in reversed(todo):
            if node.outputs[0] not in partials:
                continue
            if current is None or current.index != node.step:
                if current is not None:
                    self._step_stack.pop()
                current = self.steps[node.step]
                fwd = current
                title = self._backward_titles.get(fwd.index, f"Backprop through: {fwd.title}")
                self._step_stack.append(self._new_step(title, fwd.layer))
            self._backward_node(node, grads, grad_ids, partials)
        for tid, ids in list(partials.items()):
            if len(ids) > 1:  # leaves (inputs, params) with fan-in
                self._accumulate(tid, grads, grad_ids, partials)
        if current is not None:
            self._step_stack.pop()
        out = {tid: g for tid, g in grads.items()}
        for tid, g in grads.items():
            name = self.tensors[tid].name
            if name:
                out[name] = g
        return out

    def _accumulate(
        self,
        tid: str,
        grads: dict[str, np.ndarray],
        grad_ids: dict[str, str],
        partials: dict[str, list[str]],
    ) -> None:
        """Sum the partial gradients of ``tid`` into one tensor via a ``grad_accumulate`` node."""
        ids = partials[tid]
        if len(ids) < 2:
            return
        total = np.sum([self.tensors[g].array for g in ids], axis=0)
        name = self.tensors[tid].name
        total_id = self.watch(total, role="grad", name=f"grad_{name}" if name else None)
        step = self._current_step("grad_accumulate")
        node = Node(
            id=f"n{len(self.nodes)}",
            op="grad_accumulate",
            category="reduce",
            origin="builtin",
            phase="backward",
            inputs=list(ids),
            outputs=[total_id],
            attrs={"grad_of": [tid]},
            step=step.index,
            forward_of=None,
            differentiable=True,
            explain={
                "title": "Accumulate gradients",
                "formula": r"\frac{\partial L}{\partial x} = \sum_k g_k",
                "text": "Every path that used this value sends a gradient back; they add up.",
                "shape_rule": "all partial gradients share the tensor's shape",
            },
        )
        self.nodes.append(node)
        step.nodes.append(node.id)
        grads[tid] = total
        grad_ids[tid] = total_id
        partials[tid] = [total_id]

    def _backward_node(
        self,
        node: Node,
        grads: dict[str, np.ndarray],
        grad_ids: dict[str, str],
        partials: dict[str, list[str]],
    ) -> None:
        out_id = node.outputs[0]
        self._accumulate(out_id, grads, grad_ids, partials)
        op = self._ops[node.id]
        if not op.differentiable:
            self.event(
                "backward_stopped",
                {"node": node.id, "op": node.op, "reason": f"op {node.op!r} has no backward rule"},
                refs=[out_id],
            )
            return
        g_out = grads[out_id]
        in_grads = op.backward(self._ctx[node.id], g_out)
        step = self._current_step(node.op)
        new_ids, grad_of = [], []
        for tid, g in zip(node.inputs, in_grads, strict=True):
            if g is None:
                continue
            g = np.asarray(g, dtype=np.float64)
            if g.shape != self.tensors[tid].shape:
                raise NvzError(
                    f"backward of op {node.op!r} returned a gradient of shape {g.shape} for an "
                    f"input of shape {self.tensors[tid].shape}. "
                    "Fix: gradients must match their input shapes."
                )
            fwd_name = self.tensors[tid].name
            gid = self.watch(g, role="grad", name=f"grad_{fwd_name}" if fwd_name else None)
            new_ids.append(gid)
            grad_of.append(tid)
            partials.setdefault(tid, []).append(gid)
            grads[tid] = g if tid not in grads else grads[tid] + g
            grad_ids[tid] = gid
        in_shapes = [self.tensors[t].shape for t in node.inputs]
        bnode = Node(
            id=f"n{len(self.nodes)}",
            op=node.op,
            category=node.category,
            origin=node.origin,
            phase="backward",
            inputs=[grad_ids[out_id], *node.inputs],
            outputs=new_ids,
            attrs={"grad_of": grad_of},
            step=step.index,
            forward_of=node.id,
            differentiable=True,
            explain=op.explain_backward(dict(node.attrs), in_shapes, self.tensors[out_id].shape),
        )
        self.nodes.append(bnode)
        step.nodes.append(bnode.id)

    # --- update --------------------------------------------------------------------
    def sgd_update(
        self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float
    ) -> dict[str, np.ndarray]:
        """Record ``after = before - lr * grad`` per param with a gradient; return new params."""
        self.phase = "update"
        names = list(params)
        title = "Update " + (
            ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
        )
        new: dict[str, np.ndarray] = {}
        with self.step(title):
            for name, before in params.items():
                before_id = self.tensor_id(before)
                g = grads.get(before_id) if before_id else None
                if g is None:
                    new[name] = before
                    self.event(
                        "sgd_skipped",
                        {"param": name, "reason": "no gradient reached it"},
                        refs=[before],
                    )
                    continue
                after = before - lr * g
                after_id = self.watch(after, role="param", name=name)
                self.event(
                    "sgd_update",
                    {
                        "param": name,
                        "lr": lr,
                        "before": before_id,
                        "grad": self.tensor_id(g),
                        "after": after_id,
                    },
                    refs=[before_id, self.tensor_id(g), after_id],
                )
                new[name] = after
        return new

    # --- export --------------------------------------------------------------------
    def to_trace(
        self,
        *,
        kind: str | None = None,
        model: dict[str, Any] | None = None,
        dataset: dict[str, Any] | None = None,
        mode: dict[str, Any] | None = None,
        run_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Trace:
        """Build the :class:`Trace`. Everything not given is derived from what was recorded."""
        from nvzlab import __version__

        kind = kind or ("train" if self.did_backward else "predict")
        model = model or {"name": self.name, "family": "custom", "provenance": "user"}
        model = {**model, "architecture": self.architecture()}
        if mode is None:
            first = next((t for t in self.tensors.values() if t.role == "input"), None)
            shape = first.shape if first else ()
            n = int(shape[0]) if len(shape) >= 2 else 1
            f = int(np.prod(shape[1:])) if len(shape) >= 2 else int(np.prod(shape)) if shape else 0
            mode = {"samples": "single" if n == 1 else "batch", "n_samples": n, "n_features": f}
        meta = {"nvzlab_version": __version__, "seed": 0, **(meta or {})}
        return Trace(
            run_id=run_id or f"{model['name']}-{kind}-{mode['samples']}-f{mode['n_features']}",
            kind=kind,
            model=model,
            dataset=dataset,
            mode=mode,
            tensors=dict(self.tensors),
            nodes=list(self.nodes),
            events=list(self.events),
            steps=list(self.steps),
            meta=meta,
            schema=SCHEMA,
        )

    def architecture(self) -> list[dict[str, Any]]:
        """One entry per forward node of the first iteration: id, label, op, param names."""
        arch = []
        for n in self.nodes:
            if n.phase != "forward" or self.steps[n.step].iteration != 0:
                continue
            params = [self.tensors[t].name for t in n.inputs if self.tensors[t].role == "param"]
            arch.append(
                {
                    "id": n.id,
                    "label": self.steps[n.step].title,
                    "op": n.op,
                    "params": [p for p in params if p],
                }
            )
        return arch


def _jsonable(attrs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in attrs.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, tuple):
            v = list(v)
        out[k] = v
    return out


@contextmanager
def step(title: str, backward_title: str | None = None, layer: str | None = None) -> Iterator[None]:
    """Group the ops and events inside into one playback step (no-op outside a tape).

    In train traces each forward step gets a mirrored backward step titled
    ``backward_title`` (default ``"Backprop through: <title>"``). ``layer`` is an optional
    label (e.g. ``"Hidden layer 1"``) stored on the step and its mirror, so renderers can
    group steps into layers.

    Examples
    --------
    >>> import nvzlab as nvz
    >>> from nvzlab import ops
    >>> def model(x, p):
    ...     with nvz.step("Weighted sum", backward_title="Gradients reach w and b", layer="Output"):
    ...         return ops.linear(x, p["w"], p["b"])
    """
    tape = current_tape()
    if tape is None:
        yield
        return
    with tape.step(title, backward_title, layer):
        yield


def event(kind: str, payload: dict[str, Any] | None = None, refs: Sequence[Any] = ()) -> None:
    """Record a non-tensor algorithm step, e.g. a chosen split (no-op outside a tape).

    Examples
    --------
    >>> import nvzlab as nvz
    >>> nvz.event("split_chosen", {"feature": 0, "threshold": 0.5})
    """
    tape = current_tape()
    if tape is not None:
        tape.event(kind, payload, refs)


def name(array: Any, name: str) -> Any:
    """Name an intermediate array in the trace (e.g. ``"attention weights"``); returns it.

    Params, inputs and labels are named already; activations are anonymous unless named
    here. No-op outside a tape.

    Examples
    --------
    >>> import nvzlab as nvz
    >>> from nvzlab import ops
    >>> def model(x, p):
    ...     return nvz.name(ops.softmax(ops.matmul(x, p["W"])), "attention weights")
    """
    tape = current_tape()
    if tape is not None:
        tape.set_name(array, name)
    return array


def context(name: str, array: Any, note: str) -> None:
    """Record a tensor shown for orientation only, e.g. the whole dataset behind a sample.

    Context tensors are not part of the computation: renderers draw them faded and always
    show ``note``. No-op outside a tape.

    Examples
    --------
    >>> import numpy as np, nvzlab as nvz
    >>> nvz.context("dataset_X", np.zeros((120, 2)), "The whole dataset, for orientation.")
    """
    tape = current_tape()
    if tape is not None:
        tape.context(name, array, note)
