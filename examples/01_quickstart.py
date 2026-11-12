"""One line to a trace: the built-in logistic regression, predict and train."""

import nvzlab as nvz

tr = nvz.trace("logistic_regression")
print(tr)
tr.save("01_quickstart_predict.json")

tr = nvz.trace("logistic_regression", kind="train", samples="batch", iterations=2)
print(tr)
tr.save("01_quickstart_train.json")
