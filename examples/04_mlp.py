"""Multi-layer perceptron: forward through ReLU and softmax, then real backprop."""

import nvzlab as nvz

tr = nvz.trace("mlp")
print(tr)
tr.save("04_mlp_predict.json")

tr = nvz.trace("mlp", kind="train", samples="batch", iterations=2)
print(tr)  # steps carry layer labels; backward steps mirror the layers in reverse
tr.save("04_mlp_train.json")
