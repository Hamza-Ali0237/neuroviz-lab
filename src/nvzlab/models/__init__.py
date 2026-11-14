"""Built-in models and the ``Model`` base class.

Examples
--------
>>> import nvzlab as nvz
>>> m = nvz.models.load("logistic_regression", features=2)
>>> sorted(m.params)
['b', 'w']
"""

from nvzlab.models.base import Model
from nvzlab.models.cnn import CNN
from nvzlab.models.decision_tree import DecisionTree
from nvzlab.models.logistic_regression import LogisticRegression
from nvzlab.models.mlp import MLP
from nvzlab.models.registry import list_models, load

__all__ = ["CNN", "MLP", "DecisionTree", "LogisticRegression", "Model", "list_models", "load"]
