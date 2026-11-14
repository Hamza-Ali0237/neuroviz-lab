"""Built-in decision tree: scikit-learn parity, event payloads, growth by iteration."""

import json

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

import nvzlab as nvz
from conftest import validate
from nvzlab.models.decision_tree import ARRAYS, Growth, grow, impurity

CONFIGS = [("blobs", 1, "gini"), ("blobs", 1, "entropy"), ("blobs", 2, "gini"),
           ("blobs", 2, "entropy"), ("moons", 2, "gini"), ("moons", 2, "entropy")]  # fmt: skip


def events(tr, kind):
    return [e for e in tr.events if e.kind == kind]


@pytest.mark.parametrize("dataset,features,criterion", CONFIGS)
def test_parity_with_sklearn(dataset, features, criterion):
    m = nvz.models.load("decision_tree", dataset=dataset, features=features, criterion=criterion)
    ds = m.dataset
    sk = DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=0).fit(ds.X, ds.y)
    # same root split (feature and midpoint threshold)
    assert int(m.params["feature"][0]) == sk.tree_.feature[0]
    np.testing.assert_allclose(m.params["threshold"][0], sk.tree_.threshold[0], atol=1e-9)
    # identical predictions and class probabilities on the whole dataset; the structures
    # also agree (same node count), so no tie-break difference shows up in these datasets
    ours = m(ds.X)
    np.testing.assert_array_equal(ours.argmax(axis=1), sk.predict(ds.X))
    np.testing.assert_allclose(ours, sk.predict_proba(ds.X), atol=1e-12)
    assert len(m.params["feature"]) == sk.tree_.node_count
    assert np.mean(ours.argmax(axis=1) == ds.y) > 0.9


@pytest.mark.parametrize("dataset,features,criterion", CONFIGS)
def test_pretrained_tree_equals_tree_grown_by_a_train_trace(dataset, features, criterion):
    m = nvz.models.load("decision_tree", dataset=dataset, features=features, criterion=criterion)
    tr = nvz.trace(
        "decision_tree", kind="train", dataset=dataset, features=features, criterion=criterion,
        iterations=3,
    )  # fmt: skip
    final = events(tr, "tree_structure")[-1].payload
    grown = grow(m.dataset.X, m.dataset.y, m.dataset.feature_names, criterion)
    for k in ARRAYS:
        np.testing.assert_array_equal(grown[k], m.params[k], err_msg=k)
    assert final["n_nodes"] == len(m.params["feature"])
    for node in final["nodes"]:
        i = node["id"]
        assert node["is_leaf"] == (m.params["feature"][i] < 0)
        assert node["counts"] == m.params["counts"][i].tolist()


def test_impurity_formulas():
    assert impurity(np.array([5, 5]), "gini") == 0.5
    assert impurity(np.array([5, 5]), "entropy") == 1.0
    assert (
        impurity(np.array([0, 7]), "gini") == 0.0 and impurity(np.array([0, 7]), "entropy") == 0.0
    )
    assert impurity(np.array([0, 0]), "gini") == 0.0
    np.testing.assert_allclose(impurity(np.array([1, 3]), "gini"), 1 - 0.25**2 - 0.75**2)


def test_train_trace_events_recompute_from_counts():
    tr = nvz.trace("decision_tree", kind="train", iterations=2)
    validate(tr)
    searches = events(tr, "split_search")
    assert [e.payload["node"] for e in searches] == [0, 1, 2]
    root = searches[0].payload
    assert root["criterion"] == "gini" and root["explain"]["formula"] == r"G = 1 - \sum_k p_k^2"
    assert root["explain"]["text"] and root["n_samples"] == 120 and sum(root["counts"]) == 120
    assert len(root["candidates"]) == 2 * 119  # 120 distinct values per feature -> 119 midpoints
    # the explanation is in the step's first event
    step = tr.steps[searches[0].step]
    assert step.events[0] == searches[0].id and step.layer == "Depth 0" and step.phase == "update"
    for search in searches:
        n, parent = search.payload["n_samples"], search.payload["impurity"]
        np.testing.assert_allclose(parent, impurity(np.array(search.payload["counts"]), "gini"))
        cands = search.payload["candidates"]
        for c in cands:
            lc, rc = np.array(c["left_counts"]), np.array(c["right_counts"])
            assert lc.sum() + rc.sum() == n and lc.sum() > 0 and rc.sum() > 0
            np.testing.assert_allclose(c["left_impurity"], impurity(lc, "gini"))
            np.testing.assert_allclose(c["right_impurity"], impurity(rc, "gini"))
            weighted = (lc.sum() * c["left_impurity"] + rc.sum() * c["right_impurity"]) / n
            np.testing.assert_allclose(c["weighted_impurity"], weighted)
            np.testing.assert_allclose(c["gain"], parent - weighted)
        # thresholds are midpoints of sorted unique values, ascending per feature
        for j in range(2):
            ts = [c["threshold"] for c in cands if c["feature"] == j]
            assert ts == sorted(ts) and len(set(ts)) == len(ts)
    # the chosen split is the best candidate; ties -> lowest feature, then lowest threshold
    for search, chosen in zip(searches, events(tr, "split_chosen"), strict=True):
        cands = search.payload["candidates"]
        best = max(c["gain"] for c in cands)
        first = next(i for i, c in enumerate(cands) if c["gain"] == best)
        assert (
            chosen.payload["candidate"] == first
            and chosen.payload["node"] == search.payload["node"]
        )
        assert chosen.payload["gain"] == best
        assert tr.steps[chosen.step].index == search.step
    created = events(tr, "node_created")
    assert [(e.payload["id"], e.payload["parent"], e.payload["side"]) for e in created] == [
        (1, 0, "left"), (2, 0, "right"), (3, 1, "left"),
        (4, 1, "right"), (5, 2, "left"), (6, 2, "right"),
    ]  # fmt: skip
    for e in created:
        parent = next(
            c for c in events(tr, "split_chosen") if c.payload["node"] == e.payload["parent"]
        )
        side_counts = parent.payload[e.payload["side"] + "_counts"]
        assert e.payload["counts"] == side_counts and e.payload["n_samples"] == sum(side_counts)


def test_iterations_grow_one_depth_level_each():
    for iterations in (1, 2, 3):
        tr = nvz.trace("decision_tree", kind="train", iterations=iterations)
        validate(tr)
        assert tr.meta["iterations"] == iterations and tr.mode["n_samples"] == 120
        assert tr.nodes == [] and tr.model["architecture"] == []
        layers = {s.layer for s in tr.steps if s.title.startswith(("Split", "Node"))}
        assert layers == {f"Depth {d}" for d in range(iterations)}
        for s in tr.steps:
            if s.title.startswith("Split node"):
                assert s.iteration == int(s.layer.split()[1])
        final = events(tr, "tree_structure")[-1].payload
        assert max(n["depth"] for n in final["nodes"]) == iterations
        leaves = events(tr, "leaf")
        assert all(n["is_leaf"] for n in final["nodes"] if n["depth"] == iterations)
        assert {e.payload["node"] for e in leaves} == {
            n["id"] for n in final["nodes"] if n["is_leaf"]
        }
        for e in leaves:
            counts = e.payload["counts"]
            assert e.payload["predicted_class"] == int(np.argmax(counts))
            assert e.payload["probability"] == max(counts) / sum(counts)
            assert e.payload["reason"]
        assert tr.steps[-1].title == "Grown tree" and tr.steps[-1].iteration == iterations - 1
    reasons = {e.payload["reason"] for e in events(tr, "leaf")}
    assert "maximum depth reached" in reasons


def test_predict_trace_follows_each_sample_to_a_leaf():
    tr = nvz.trace("decision_tree", samples="batch", dataset="moons")
    validate(tr)
    structure = events(tr, "tree_structure")
    assert len(structure) == 1 and tr.steps[structure[0].step].title == "Tree"
    nodes = {n["id"]: n for n in structure[0].payload["nodes"]}
    assert structure[0].refs == [tr.tensors[r].id for r in structure[0].refs]
    assert {tr.tensors[r].name for r in structure[0].refs} == set(ARRAYS)
    x = tr.tensors["t0"].array
    out = next(t for t in tr.tensors.values() if t.role == "output").array
    for i in range(4):
        path = [
            e for e in tr.events if e.kind in ("node_test", "leaf") and e.payload["sample"] == i
        ]
        assert path[-1].kind == "leaf" and all(e.kind == "node_test" for e in path[:-1])
        node = 0
        for e in path[:-1]:
            n = nodes[node]
            assert e.payload["node"] == node and e.payload["feature"] == n["feature"]
            assert (
                e.payload["threshold"] == n["threshold"]
                and e.payload["feature_name"] == n["feature_name"]
            )
            assert e.payload["value"] == x[i, n["feature"]]
            expect_left = x[i, n["feature"]] <= n["threshold"]
            assert e.payload["direction"] == ("left" if expect_left else "right")
            assert tr.steps[e.step].title == f"Depth {n['depth']}"
            node = e.payload["next_node"]
            assert node == (n["left"] if expect_left else n["right"])
        leaf = path[-1].payload
        assert leaf["node"] == node and nodes[node]["is_leaf"]
        assert tr.steps[path[-1].step].title == f"Depth {nodes[node]['depth']}"
        np.testing.assert_allclose(out[i], np.array(leaf["counts"]) / sum(leaf["counts"]))
        assert leaf["predicted_class"] == out[i].argmax()
    pred = events(tr, "prediction")[0].payload
    assert pred["predicted_class"] == out.argmax(axis=1).tolist()
    titles = [s.title for s in tr.steps]
    assert titles[0] == "Tree" and titles[-1] == "Prediction"
    assert titles[1:-1] == [f"Depth {d}" for d in range(len(titles) - 2)]
    node = next(n for n in tr.nodes if n.op == "tree_predict")
    assert not node.differentiable and node.phase == "forward"
    assert tr.model["architecture"] == [
        {"id": "n0", "label": "Prediction", "op": "tree_predict",
         "params": ["feature", "threshold", "left", "right", "counts"]}
    ]  # fmt: skip


def test_tie_breaking_prefers_lowest_feature_then_lowest_threshold():
    # two identical features: every split ties across features -> feature 0 wins
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([0, 0, 1, 1])
    with nvz.Tape() as t:
        tree = grow(x, y, ["a", "b"], "gini", levels=1)
    chosen = next(e for e in t.events if e.kind == "split_chosen").payload
    assert chosen["feature"] == 0 and chosen["threshold"] == 1.5 and chosen["gain"] == 0.5
    assert tree["feature"].tolist() == [0, -1, -1] and tree["counts"].tolist() == [
        [2, 2],
        [2, 0],
        [0, 2],
    ]
    # a tie between thresholds of one feature -> the lowest threshold
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    with nvz.Tape() as t:
        grow(x, y, ["a"], "entropy", levels=1)
    search = next(e for e in t.events if e.kind == "split_search").payload
    gains = [round(c["gain"], 12) for c in search["candidates"]]
    assert gains[0] == gains[2] == max(gains)
    assert next(e for e in t.events if e.kind == "split_chosen").payload["threshold"] == 0.5


def test_leaf_reasons_and_unsplittable_nodes():
    x = np.array([[1.0], [1.0], [2.0]])
    y = np.array([0, 1, 1])
    with nvz.Tape() as t:
        tree = grow(x, y, ["a"], "gini", levels=3)
    leaves = [e.payload for e in t.events if e.kind == "leaf"]
    assert [leaf["reason"] for leaf in leaves] == [
        "no distinct values to split on",
        "pure: every sample has the same class",
    ]
    assert tree["feature"].tolist() == [0, -1, -1]
    g = Growth(x, y, ["a"], "gini", max_depth=0)
    g.level(0)
    assert g.nodes[0]["reason"] == "maximum depth reached"


def test_split_chosen_event_is_plain_json():
    tr = nvz.trace("decision_tree", kind="train")
    chosen = events(tr, "split_chosen")[0]
    d = json.loads(json.dumps(chosen.to_dict()))
    assert set(d["payload"]) == {
        "node", "candidate", "feature", "feature_name", "threshold", "left_counts",
        "right_counts", "left_impurity", "right_impurity", "weighted_impurity", "gain",
    }  # fmt: skip


@pytest.mark.parametrize("kind", ["predict", "train"])
def test_tree_traces_carry_the_dataset_as_context(kind):
    tr = nvz.trace("decision_tree", kind=kind)
    tensors = tr.to_dict()["tensors"].values()
    ctx = {t["name"]: t for t in tensors if t["role"] == "context"}
    if kind == "predict":
        assert set(ctx) == {"dataset_X", "dataset_y"}
        assert ctx["dataset_X"]["shape"] == [120, 2]
        assert "leaf regions" in ctx["dataset_X"]["note"]
    else:  # the tree is grown on the whole dataset, so the dataset is the traced x itself
        assert set(ctx) == {"dataset_y"}
        assert [t["shape"] for t in tensors if t["role"] == "input"] == [[120, 2]]
    assert ctx["dataset_y"]["shape"] == [120]
