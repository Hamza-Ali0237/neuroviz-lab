"""Mini Transformer: one self-attention block with visible Q, K, V and attention weights."""

import nvzlab as nvz

tr = nvz.trace("transformer")
print(tr)
aw = next(t for t in tr.tensors.values() if t.name == "attention weights")
print("attention weights", aw.shape)  # (1, 6, 6): row i says how much position i looks at each j
print(aw.array[0].round(3))
tr.save("07_transformer_predict.json")

tr = nvz.trace("transformer", kind="train", samples="batch")
print(tr)  # residual adds show up as grad_accumulate nodes in the backward steps
tr.save("07_transformer_train.json")
