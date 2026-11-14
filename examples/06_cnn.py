"""Tiny CNN on 8x8 RGB shapes: convolution, ReLU, max pooling with recorded routing, backprop."""

import nvzlab as nvz

tr = nvz.trace("cnn")
print(tr)
tr.save("06_cnn_predict.json")

tr = nvz.trace("cnn", kind="train", samples="batch")
print(tr)  # the max-pool node's attrs["argmax"] says where each gradient flows back to
tr.save("06_cnn_train.json")
