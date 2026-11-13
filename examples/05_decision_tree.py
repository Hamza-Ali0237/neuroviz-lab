"""Decision tree: an event-based algorithm (no gradients) on the same trace format."""

import nvzlab as nvz

tr = nvz.trace("decision_tree", samples="batch")  # each sample's path from root to leaf
print(tr)
tr.save("05_decision_tree_predict.json")

tr = nvz.trace("decision_tree", kind="train", dataset="moons", criterion="entropy", iterations=2)
print(tr)  # one step per node split: every candidate threshold, the chosen one, the children
tr.save("05_decision_tree_train.json")
