"""Registry of built-in models."""

from __future__ import annotations

from typing import Any

from nvzlab.core.errors import unknown_name
from nvzlab.models.base import Model
from nvzlab.models.cnn import CNN
from nvzlab.models.decision_tree import DecisionTree
from nvzlab.models.logistic_regression import LogisticRegression
from nvzlab.models.mlp import MLP

_MODELS: dict[str, type[Model]] = {
    "cnn": CNN,
    "decision_tree": DecisionTree,
    "logistic_regression": LogisticRegression,
    "mlp": MLP,
}


def load(name: str, **options: Any) -> Model:
    """Instantiate a built-in model by name with pretrained weights.

    Examples
    --------
    >>> import nvzlab as nvz
    >>> print(nvz.models.load("logistic_regression").describe())  # doctest: +SKIP
    """
    if name not in _MODELS:
        raise unknown_name("model", name, _MODELS)
    return _MODELS[name](**options)


def list_models() -> dict[str, str]:
    """Built-in model names with a one-line description and the options each supports.

    Examples
    --------
    >>> import nvzlab as nvz
    >>> sorted(nvz.list_models())
    ['cnn', 'decision_tree', 'logistic_regression', 'mlp']
    >>> print(nvz.list_models()["mlp"])  # doctest: +SKIP
    """
    out = {}
    for name, cls in sorted(_MODELS.items()):
        options = "; ".join(
            f"{k}: {', '.join(str(v) for v in vals)}" for k, vals in cls.supports.items()
        )
        out[name] = f"{cls.description} [{options}]"
    return out
