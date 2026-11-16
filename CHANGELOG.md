# Changelog

## 0.1.1 - 2026-11-16

### Fixed

- Project links on PyPI (homepage, repository, issues, changelog, roadmap) pointed to the wrong GitHub account.
- README "What's in v1" and "What's not yet" now match what 0.1.0 ships.

## 0.1.0 - 2026-11-16

First release.

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
- Ops `conv2d` (NCHW, stride and padding), `maxpool2d` (records the winning position of every window as `attrs["argmax"]` on the forward node, so the backward routing is visible) and `flatten`, each with forward and backward explanations and gradient checks.
- `shapes` dataset: 90 synthetic 8x8 RGB images (square, ring, cross) at random positions and colours.
- Built-in `cnn` (conv 3->4 3x3 -> ReLU -> max pool 2x2 -> flatten -> linear 64->3 -> softmax, cross-entropy on `shapes`) with layer-labelled steps, one context example image per class, pretrained and initial weights, parity with PyTorch.
- Ops can attach extra data to their recorded node through `Op.node_attrs(ctx)`.
- Ops `embedding`, `layernorm`, `transpose` and `scale`; `matmul` accepts batched (3-D) operands and `linear` any leading dimensions, with the old 2-D behaviour unchanged.
- `tokens` dataset: 120 sequences of 6 ids over the vocabulary A..H, label 1 when the first token equals the last.
- Built-in `transformer`: token and learned positional embeddings (d=8), one pre-norm encoder block with single-head self-attention written out as visible ops (Q, K, V, scores, scaling, softmax rows as the named tensor "attention weights", A·V, output projection), residual adds that show up as `grad_accumulate` nodes, feed-forward 8->16->8, mean over positions, 2-class softmax; pretrained and initial weights, parity with PyTorch.
- `nvz.name(array, "label")` names an intermediate tensor in the trace.
- `nvz.step(..., layer=...)`: an optional layer label on steps, copied onto the mirrored backward steps and stored in the trace (`steps[].layer`).
- `nvz.list_models()` returns a one-line description and the supported options per model.
- CLI: `nvzlab list`, `nvzlab demo <model>` (opens the viewer; `--json`/`--html` to write a file), `nvzlab open trace.json`.
- Interactive 2D viewer (`nvz.show`, `Trace.show`, `Trace.save("x.html")`): single self-contained HTML page; step list grouped by iteration and tinted by phase, playback with speed and keyboard control, matrix grids with heatmap tint, op cards with KaTeX formulas and shape rules, worked arithmetic for any selected cell (forward, loss, backward, update), inspector with full tensor values, and a logistic-regression scene (decision boundary or sigmoid curve) that moves with the parameters.
- `nvz.context(name, array, note)`: context tensors (role `context`, with a `note`) shown faded for orientation. Logistic regression records the whole `blobs` dataset and, for train traces, the real mean-BCE loss landscape on a 41×41 grid.
- `nvz.trace(..., name=...)` overrides the display name; user models are named after the function or class.
- 2D viewer: dataset context behind the plots with its note, axis ticks, class legend by colour and shape, diverging colour-scale legend on every grid, true minus signs, human labels for events, plot-first forward steps and a slim inspector rail when nothing is selected.
- 3D renderer (three.js): probability surface and loss landscape with the training path for logistic regression, and a generic tensor stack for every model; shares step, selection and sample with 2D; reset view; disabled with a reason when WebGL is unavailable.
- MLP scenes: network diagram with the real activations and weights (or gradients in the backward phase), decision regions from recorded probability grids, 3D network and surface.
- Decision-tree scenes: the partition of the plane and the split search, in 2D.
- Capture mode for screen recording: `trace.save("x.html", capture="square")`, `nvz.show(..., capture="vertical")` or a URL flag; controls hidden, fixed 1080×1080 or 1080×1920 viewport, larger type, auto-play that loops.
- Built-ins carry a `display_name` ("Logistic Regression", "MLP", "Decision Tree", "CNN", "Transformer") used as the trace title; `run_id` keeps the registry name. User models keep their function or class name.
- The MLP default sample is chosen by rule: the first `moons` row (first 4 rows for `samples="batch"`) that the pretrained weights classify correctly and that activates at least 2 of the 4 hidden units, so the network diagram never shows a dead layer.
- Update steps of models with more than 4 parameter arrays are titled "Update all parameters (N arrays)"; the step's first event lists every array.
- GitHub Actions: tests on Python 3.10 to 3.13 (Linux and macOS, one job with CPU PyTorch for the parity tests), a wheel smoke test in a clean venv, a frontend bundle freshness check, and a tag-triggered release workflow that publishes to PyPI with trusted publishing.

### Changed

- The pretrained Transformer is trained at learning rate 0.1 instead of 0.5. Accuracy on `tokens` stays at 100%, and the attention weights for the default sample now spread over several positions (largest weight per row between 0.39 and 0.76, scaled scores within ±7) instead of being rows of 0s and 1s.
