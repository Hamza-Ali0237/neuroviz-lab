"""Convolution ops: conv2d, maxpool2d, flatten (layout NCHW)."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvzlab.core.errors import check_shape
from nvzlab.core.op import Op, op


def _check_image(name: str, x: np.ndarray) -> None:
    check_shape(
        x.ndim == 4,
        name,
        "x with 4 dimensions (n_samples, channels, height, width)",
        f"x {x.shape}",
        "reshape a single image as x.reshape(1, C, H, W)",
    )


def _out_size(size: int, k: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - k) // stride + 1


@op(
    category="conv",
    formula=(
        r"y_{o,i,j} = b_o + \sum_{c}\sum_{u,v} w_{o,c,u,v}\, x_{c,\,i s + u - p,\,j s + v - p}"
    ),
    text=(
        "Each output pixel is a weighted sum of one input patch: the kernel slides over the "
        "image, multiplies the patch under it element by element, sums everything and adds "
        "the bias. One kernel per output channel."
    ),
    shape_rule=(
        "x (n, C_in, H, W), w (C_out, C_in, kH, kW), b (C_out,) -> "
        "(n, C_out, (H + 2p - kH) / s + 1, (W + 2p - kW) / s + 1)"
    ),
    backward_formula=(
        r"\frac{\partial L}{\partial w_{o,c,u,v}} = \sum_{n,i,j} g_{n,o,i,j}\, "
        r"x_{n,c,\,is+u-p,\,js+v-p},\quad "
        r"\frac{\partial L}{\partial b_o} = \sum_{n,i,j} g_{n,o,i,j}"
    ),
    backward_text=(
        "The kernel gradient is the input patch weighted by the upstream gradient, summed over "
        "every position the kernel visited (and over the batch). The input gradient spreads each "
        "upstream value back over the patch it came from, through the kernel weights. The bias "
        "gradient is the upstream gradient summed over positions."
    ),
)
def conv2d(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, padding: int = 0
) -> np.ndarray:
    """2-D cross-correlation (what deep-learning frameworks call convolution), layout NCHW.

    Examples
    --------
    >>> from nvzlab import ops
    >>> x = np.arange(16.0).reshape(1, 1, 4, 4)
    >>> ops.conv2d(x, np.ones((1, 1, 2, 2)), np.zeros(1))[0, 0]
    array([[10., 14., 18.],
           [26., 30., 34.],
           [42., 46., 50.]])
    """
    _check_image("conv2d", x)
    check_shape(
        w.ndim == 4 and w.shape[1] == x.shape[1],
        "conv2d",
        f"w with shape (C_out, {x.shape[1]}, kH, kW) to match x {x.shape}",
        f"w {w.shape}",
        f"give w {x.shape[1]} input channels (second dimension)",
    )
    check_shape(
        b.shape == (w.shape[0],),
        "conv2d",
        f"b with shape ({w.shape[0]},), one bias per output channel",
        f"b {b.shape}",
        f"use b with shape ({w.shape[0]},)",
    )
    check_shape(
        isinstance(stride, int) and stride >= 1 and isinstance(padding, int) and padding >= 0,
        "conv2d",
        "stride >= 1 and padding >= 0 (integers)",
        f"stride={stride!r}, padding={padding!r}",
        "use e.g. stride=1, padding=1",
    )
    n, _, h, wd = x.shape
    c_out, _, kh, kw = w.shape
    ho, wo = _out_size(h, kh, stride, padding), _out_size(wd, kw, stride, padding)
    check_shape(
        ho >= 1 and wo >= 1,
        "conv2d",
        f"a kernel no larger than the padded image ({h + 2 * padding}x{wd + 2 * padding})",
        f"kernel {kh}x{kw}",
        "use a smaller kernel or more padding",
    )
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.zeros((n, c_out, ho, wo))
    for u in range(kh):
        for v in range(kw):
            patch = xp[:, :, u : u + stride * ho : stride, v : v + stride * wo : stride]
            out += np.einsum("nchw,oc->nohw", patch, w[:, :, u, v])
    return out + b[None, :, None, None]


@conv2d.backward
def _(grad, x, w, b, out, stride=1, padding=0):
    n, _, h, wd = x.shape
    kh, kw = w.shape[2:]
    ho, wo = grad.shape[2:]
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    gx = np.zeros_like(xp)
    gw = np.zeros_like(w)
    for u in range(kh):
        for v in range(kw):
            sl = (slice(None), slice(None), slice(u, u + stride * ho, stride))
            sl = (*sl, slice(v, v + stride * wo, stride))
            gw[:, :, u, v] = np.einsum("nohw,nchw->oc", grad, xp[sl])
            gx[sl] += np.einsum("nohw,oc->nchw", grad, w[:, :, u, v])
    if padding:
        gx = gx[:, :, padding : padding + h, padding : padding + wd]
    return gx, gw, grad.sum(axis=(0, 2, 3))


class MaxPool2d(Op):
    """Max pooling that records which input position won each window (``attrs["argmax"]``)."""

    name = "maxpool2d"
    category = "conv"
    backward_formula = (
        r"\frac{\partial L}{\partial x_{c,h,w}} = "
        r"\sum_{(i,j):\ \mathrm{argmax}_{i,j} = (h,w)} g_{c,i,j}"
    )
    backward_text = (
        "The gradient only flows back to the position that won each window (recorded as "
        "`argmax` on the forward node); every other input position in the window gets 0."
    )

    def forward(self, x: np.ndarray, size: int = 2, stride: int = 2) -> tuple[np.ndarray, Any]:
        _check_image("maxpool2d", x)
        n, c, h, w = x.shape
        check_shape(
            isinstance(size, int) and size >= 1 and isinstance(stride, int) and stride >= 1,
            "maxpool2d",
            "size >= 1 and stride >= 1 (integers)",
            f"size={size!r}, stride={stride!r}",
            "use e.g. size=2, stride=2",
        )
        check_shape(
            h >= size and w >= size and (h - size) % stride == 0 and (w - size) % stride == 0,
            "maxpool2d",
            f"windows of {size}x{size} with stride {stride} to tile the image exactly",
            f"image {h}x{w}",
            "pick size and stride so that (H - size) and (W - size) are multiples of stride",
        )
        ho, wo = _out_size(h, size, stride, 0), _out_size(w, size, stride, 0)
        windows = np.stack(
            [
                x[:, :, u : u + stride * ho : stride, v : v + stride * wo : stride]
                for u in range(size)
                for v in range(size)
            ],
            axis=-1,
        )  # (n, c, ho, wo, size*size); the first maximum wins a tie
        local = windows.argmax(axis=-1)
        rows = np.arange(ho)[:, None] * stride + local // size
        cols = np.arange(wo)[None, :] * stride + local % size
        argmax = rows * w + cols  # flat index into each (h, w) plane
        return windows.max(axis=-1), (x.shape, argmax)

    def backward(self, ctx: Any, grad_out: np.ndarray) -> tuple[np.ndarray]:
        shape, argmax = ctx
        n, c, h, w = shape
        gx = np.zeros((n, c, h * w))
        nn, cc = np.ogrid[:n, :c]
        np.add.at(gx, (nn[..., None, None], cc[..., None, None], argmax), grad_out)
        return (gx.reshape(shape),)

    def node_attrs(self, ctx: Any) -> dict[str, Any]:
        return {"argmax": ctx[1]}

    def explain(self, attrs, in_shapes, out_shape):
        size, stride = attrs.get("size", 2), attrs.get("stride", 2)
        return {
            "title": "Max pool",
            "formula": r"y_{c,i,j} = \max_{u,v < k} x_{c,\,is+u,\,js+v}",
            "text": (
                f"Each {size}x{size} window is replaced by its largest value, so the feature "
                "map shrinks and small shifts stop mattering. The winning position of every "
                "window is recorded (`argmax`) because only it receives gradient."
            ),
            "shape_rule": (
                f"(n, C, H, W) -> (n, C, (H - {size}) / {stride} + 1, (W - {size}) / {stride} + 1)"
            ),
        }


maxpool2d = MaxPool2d()
maxpool2d.__doc__ = """Max pooling over ``size`` x ``size`` windows, layout NCHW.

The forward node's ``attrs["argmax"]`` holds, per output cell, the flat index (row * W + col)
of the input position that won, so a viewer can draw where the gradient flows back.

Examples
--------
>>> from nvzlab import ops
>>> ops.maxpool2d(np.arange(16.0).reshape(1, 1, 4, 4))[0, 0]
array([[ 5.,  7.],
       [13., 15.]])
"""


@op(
    category="shape",
    formula=r"y = \mathrm{reshape}(x, (n, C \cdot H \cdot W))",
    text=(
        "Lays every sample's feature maps out as one long row, so a fully connected layer "
        "can read them. No values change, only the shape."
    ),
    shape_rule="(n, ...) -> (n, product of the remaining dimensions)",
    backward_formula=r"\frac{\partial L}{\partial x} = \mathrm{reshape}(g, \mathrm{shape}(x))",
    backward_text="The upstream gradient is reshaped back to the input's shape; nothing is scaled.",
)
def flatten(x: np.ndarray) -> np.ndarray:
    """Flatten every axis but the first (the batch axis).

    Examples
    --------
    >>> from nvzlab import ops
    >>> ops.flatten(np.zeros((2, 3, 4, 4))).shape
    (2, 48)
    """
    check_shape(
        x.ndim >= 2,
        "flatten",
        "x with at least 2 dimensions (n_samples, ...)",
        f"x {x.shape}",
        "add a batch axis, e.g. x[None]",
    )
    return x.reshape(x.shape[0], -1)


@flatten.backward
def _(grad, x, out):
    return grad.reshape(x.shape)
