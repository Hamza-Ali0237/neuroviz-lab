"""Registry of built-in models."""

from __future__ import annotations

from typing import Any

from nvzlab.core.errors import unknown_name
from nvzlab.models.base import Model
from nvzlab.models.logistic_regression import LogisticRegression

_MODELS: dict[str, type[Model]] = {"logistic_regression": LogisticRegression}


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


def list_models() -> list[str]:
    """Names of the built-in models."""
    return sorted(_MODELS)
