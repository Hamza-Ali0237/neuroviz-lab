# Roadmap

neuroviz-lab is a learning visualizer: it runs small models step by step and shows the
real numbers. It is not a training framework, an MLOps tool or an explainability toolkit,
and it will stay that way. neuroviz-lab is the visual engine for a larger learning project.

Items are grouped by theme, not by date. Releases are small and frequent once v1 is out,
with semantic versions and human-readable notes in `CHANGELOG.md`.

## v1 (in progress)

- Five built-in models: `logistic_regression`, `mlp`, `decision_tree`, `cnn`,
  `transformer`, each with fixed weights, a synthetic dataset and a demo scene.
- Predict and train traces (1 to 3 iterations), 2D and 3D renderers sharing one state.
- Capture mode: controls hidden, square (1080×1080) and vertical (1080×1920) sizes, larger
  type, clean looping playback, available from Python and from a URL flag.
- Worked-arithmetic cards for the remaining ops (`conv2d`, `maxpool2d`, `layernorm`,
  attention), so every selected cell can show the multiply-and-sum behind it.
- Image and heat-map views for 4-D tensors and attention weights.

## More models

- Multi-head attention (`heads=2`) for the transformer, with one heat map per head.
- Linear regression with mean squared error (the simplest possible loss surface).
- k-nearest neighbours and k-means as non-gradient models recorded through events, like
  the decision tree.
- A recurrent cell unrolled over a short sequence.

## Framework adapters

The trace format is the contract, so adapters can be built without touching the
renderers. Designed for, not yet built:

- PyTorch: record a `torch.nn.Module` forward and backward on CPU into a
  `nvzlab.trace/0` file.
- JAX and TensorFlow adapters following the same pattern.

## The trace format

- Keep `nvzlab.trace/0` stable through v1; document it as a standalone specification so
  other viewers and tools can read and write it.
- Integer tensors (token ids, routing indices) as first-class dtypes in a later schema
  version.
- A compact binary container for larger traces; the JSON form stays the reference.

## Teaching material

- A gallery of short scenes, one idea each (one convolution step, one gradient update, an
  attention heat map), with the exact command that reproduces every number shown.
- Examples of common beginner mistakes made visible: shape errors, sign errors, a
  learning rate that is too large.

## Not planned

Hosted services, image or dataset upload, editing built-in datasets, full training loops,
explainability methods, telemetry of any kind.
