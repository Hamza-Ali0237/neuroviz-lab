"""nvzlab: run small ML models step by step and see what happens to the data.

Examples
--------
>>> import nvzlab as nvz
>>> tr = nvz.trace("logistic_regression", kind="train")
>>> tr.save("run.json")  # doctest: +SKIP
"""

__version__ = "0.1.0"

from nvzlab import datasets, models, ops  # noqa: E402
from nvzlab.api import show, trace  # noqa: E402
from nvzlab.core.op import Op, op  # noqa: E402
from nvzlab.core.registry import names as _op_names  # noqa: E402
from nvzlab.core.tape import Tape, event, step  # noqa: E402
from nvzlab.datasets import list_datasets  # noqa: E402
from nvzlab.models import Model, list_models  # noqa: E402
from nvzlab.tracing.records import Trace  # noqa: E402


def list_ops() -> list[str]:
    """Names of every registered op (built-in and user-defined).

    Examples
    --------
    >>> import nvzlab as nvz
    >>> "sigmoid" in nvz.list_ops()
    True
    """
    return _op_names()


__all__ = [
    "Model",
    "Op",
    "Tape",
    "Trace",
    "__version__",
    "datasets",
    "event",
    "list_datasets",
    "list_models",
    "list_ops",
    "models",
    "op",
    "ops",
    "show",
    "step",
    "trace",
]
