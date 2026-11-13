"""Trace dataclasses (schema ``nvzlab.trace/0``)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nvzlab.tracing import serializer

SCHEMA = "nvzlab.trace/0"


@dataclass
class Tensor:
    id: str
    array: np.ndarray | None
    role: str
    name: str | None = None
    shape: tuple[int, ...] = ()
    stats: dict[str, float] = field(default_factory=dict)
    truncated: bool = False
    note: str | None = None

    @classmethod
    def from_array(cls, tid: str, arr: np.ndarray, role: str, name: str | None) -> Tensor:
        arr = np.asarray(arr, dtype=np.float64)
        big = arr.size > serializer.MAX_ELEMENTS
        return cls(
            tid, None if big else arr, role, name, tuple(arr.shape), serializer.stats(arr), big
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": "float32",
            "role": self.role,
            "name": self.name,
            "stats": self.stats,
            "data": None if self.array is None else serializer.encode_array(self.array),
            "truncated": self.truncated,
            "note": self.note,
        }


@dataclass
class Node:
    id: str
    op: str
    category: str
    origin: str
    phase: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any]
    step: int
    forward_of: str | None = None
    differentiable: bool = True
    explain: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Event:
    id: str
    kind: str
    phase: str
    step: int
    payload: dict[str, Any]
    refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Step:
    index: int
    title: str
    phase: str
    nodes: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    iteration: int = 0
    layer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Trace:
    """A recorded run: tensors, nodes, events and playback steps.

    Examples
    --------
    >>> import nvzlab as nvz
    >>> tr = nvz.trace("logistic_regression")
    >>> tr.kind
    'predict'
    >>> tr.save("run.json")  # doctest: +SKIP
    """

    run_id: str
    kind: str
    model: dict[str, Any]
    dataset: dict[str, Any] | None
    mode: dict[str, Any]
    tensors: dict[str, Tensor]
    nodes: list[Node]
    events: list[Event]
    steps: list[Step]
    meta: dict[str, Any]
    schema: str = SCHEMA

    # --- export ---------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (keys sorted on serialization)."""
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "kind": self.kind,
            "model": self.model,
            "dataset": self.dataset,
            "mode": self.mode,
            "tensors": {k: t.to_dict() for k, t in self.tensors.items()},
            "nodes": [n.to_dict() for n in self.nodes],
            "events": [e.to_dict() for e in self.events],
            "steps": [s.to_dict() for s in self.steps],
            "meta": self.meta,
        }

    def to_json(self) -> str:
        """Deterministic JSON string: same trace, same bytes."""
        return serializer.to_json(self.to_dict())

    def save(self, path: str | Path) -> Path:
        """Write the trace; ``.json`` for the raw trace, ``.html`` for a self-contained viewer.

        Examples
        --------
        >>> tr = nvz.trace("logistic_regression")  # doctest: +SKIP
        >>> tr.save("run.json")  # doctest: +SKIP
        >>> tr.save("run.html")  # doctest: +SKIP
        """
        path = Path(path)
        if path.suffix == ".json":
            path.write_text(self.to_json(), encoding="utf-8")
            return path
        if path.suffix == ".html":
            from nvzlab import viewer

            path.write_text(viewer.to_html(self), encoding="utf-8")
            return path
        raise ValueError(
            f"Trace.save: unsupported extension {path.suffix!r} for {str(path)!r}. "
            "Fix: use '.json' (raw trace) or '.html' (interactive viewer)."
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        """Rebuild a trace from its JSON dict (tensor data decoded to float32 arrays)."""
        if d.get("schema") != SCHEMA:
            raise ValueError(
                f"Trace.load: expected schema {SCHEMA!r}, received {d.get('schema')!r}. "
                "Fix: load a file written by Trace.save of this nvzlab version."
            )
        tensors = {}
        for tid, t in d["tensors"].items():
            arr = None if t["data"] is None else serializer.decode_array(t["data"], t["shape"])
            tensors[tid] = Tensor(
                tid,
                arr,
                t["role"],
                t["name"],
                tuple(t["shape"]),
                t["stats"],
                t["truncated"],
                t.get("note"),
            )
        return cls(
            run_id=d["run_id"],
            kind=d["kind"],
            model=d["model"],
            dataset=d["dataset"],
            mode=d["mode"],
            tensors=tensors,
            nodes=[Node(**n) for n in d["nodes"]],
            events=[Event(**e) for e in d["events"]],
            steps=[Step(**s) for s in d["steps"]],
            meta=d["meta"],
            schema=d["schema"],
        )

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        """Load a ``.json`` trace written by :meth:`save`.

        Examples
        --------
        >>> tr = nvz.Trace.load("run.json")  # doctest: +SKIP
        """
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def show(self, dim: str = "2d") -> Path | None:
        """Open the interactive viewer: inline in Jupyter, otherwise in the browser.

        Returns the path of the written HTML file outside a notebook.

        Examples
        --------
        >>> nvz.trace("logistic_regression").show()  # doctest: +SKIP
        """
        if dim not in ("2d", "3d"):
            raise ValueError(f"dim must be '2d' or '3d', received {dim!r}.")
        from nvzlab import viewer

        return viewer.show(self)

    # --- inspection -----------------------------------------------------------------
    def summary(self) -> str:
        """A short text table of steps, ops and shapes."""
        mode = self.mode
        samples = f"{mode['n_samples']} sample" + ("s" if mode["n_samples"] != 1 else "")
        head = (
            f"nvzlab trace {self.run_id}: model={self.model['name']} kind={self.kind} "
            f"{samples} features={mode['n_features']} "
            f"tensors={len(self.tensors)} nodes={len(self.nodes)} events={len(self.events)}"
        )
        rows = []
        layered = any(s.layer for s in self.steps)
        for s in self.steps:
            parts = []
            for nid in s.nodes:
                n = self._node(nid)
                ins = "×".join(self._shape(t) for t in n.inputs)
                outs = ",".join(self._shape(t) for t in n.outputs)
                tag = "" if n.differentiable else " [non-differentiable]"
                parts.append(f"{n.op}{tag} {ins}→{outs}")
            for eid in s.events:
                parts.append(f"event:{self._event(eid).kind}")
            it = (
                f" it{s.iteration + 1}"
                if self.kind == "train" and self.meta.get("iterations", 1) > 1
                else ""
            )
            row = [str(s.index), s.phase + it, s.title, "; ".join(parts) or "-"]
            if layered:
                row.insert(2, s.layer or "")
            rows.append(row)
        header = ["#", "phase", "step", "ops"]
        if layered:
            header.insert(2, "layer")
        widths = [max(len(r[i]) for r in rows + [header]) for i in range(len(header) - 1)] + [0]
        lines = [head] + [
            "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
            for r in [header, *rows]
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def _node(self, nid: str) -> Node:
        return next(n for n in self.nodes if n.id == nid)

    def _event(self, eid: str) -> Event:
        return next(e for e in self.events if e.id == eid)

    def _shape(self, tid: str) -> str:
        return str(tuple(self.tensors[tid].shape))
