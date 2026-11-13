# Changelog

## Unreleased

### Added

- Core engine: `Tape` recording with implicit autograd, `nvz.step`, `nvz.event`, SGD update events.
- Ops: `matmul`, `add`, `mul`, `linear`, `sigmoid`, `relu`, `softmax`, `tanh`, `bce`, `cross_entropy`, `mean`, `sum`, each with a forward and a backward explanation.
- `@nvz.op` decorator and `nvz.Op` class form for user-defined ops (with optional `backward_formula`/`backward_text`), with one shared registry.
- Backward steps mirror forward steps in reverse order (`nvz.step(title, backward_title=...)`); `grad_accumulate` nodes make gradient fan-in visible.
- Trace nodes carry the op's `category` and an `origin` (`builtin` or `user`).
- Trace schema v0 (`nvzlab.trace/0`), deterministic JSON serializer, `Trace.save`/`Trace.load`.
- `blobs` (1 or 2 features) and `moons` (two interleaving half circles, not linearly separable) synthetic datasets.
- Built-in `logistic_regression` model with pretrained and initial weights.
- Built-in `mlp` (2 -> 4 ReLU -> 2 softmax, cross-entropy on `moons`) with layer-labelled steps and pretrained and initial weights.
- Built-in `decision_tree` (depth 3, Gini or entropy, `blobs` or `moons`): predict traces follow each sample from the root to a leaf with `tree_structure`, `node_test` and `leaf` events; train traces grow the tree one depth level per iteration with `split_search` (every candidate), `split_chosen`, `node_created` and `leaf` events. Models can record their own non-gradient training by overriding `Model.train`.
- Ops `nll` (cross-entropy of probabilities) and `tree_predict` (non-differentiable).
- `nvz.step(..., layer=...)`: an optional layer label on steps, copied onto the mirrored backward steps and stored in the trace (`steps[].layer`).
- `nvz.list_models()` returns a one-line description and the supported options per model.
- CLI: `nvzlab list`, `nvzlab demo <model>` (opens the viewer; `--json`/`--html` to write a file), `nvzlab open trace.json`.
- Interactive 2D viewer (`nvz.show`, `Trace.show`, `Trace.save("x.html")`): single self-contained HTML page; step list grouped by iteration and tinted by phase, playback with speed and keyboard control, matrix grids with heatmap tint, op cards with KaTeX formulas and shape rules, worked arithmetic for any selected cell (forward, loss, backward, update), inspector with full tensor values, and a logistic-regression scene (decision boundary or sigmoid curve) that moves with the parameters.
