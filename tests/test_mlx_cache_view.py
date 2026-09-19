# SPDX-License-Identifier: Apache-2.0
"""Native zero-copy cache view geometry, independent of shared storage adapters."""

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.metal import get_ops


@pytest.mark.parametrize("buffer", [None, 7, [], np.zeros(4, dtype=np.uint8)])
def test_cache_view_rejects_non_mlx_backing(buffer):
    # Nanobind rejects None before entering our explicit array-type guard.
    with pytest.raises(TypeError, match="cache_view"):
        get_ops().cache_view(buffer, (1,), (1,), 0)


@pytest.mark.parametrize(
    "shape,strides,offset",
    [
        ((2,), (), 0),
        ((0,), (1,), 0),
        ((-1,), (1,), 0),
        ((1,), (1 << 63,), 0),
        ((5,), (1 << 62,), 0),
        ((2, 2, 2, 2), (1 << 62,) * 4, 0),
        ((2,), (1,), (1 << 64) - 1),
        ((1,), (1,), 16),
        ((1 << 30, 1 << 30, 16), (0, 0, 0), 0),
    ],
    ids=[
        "rank",
        "zero-dimension",
        "negative-dimension",
        "signed-stride",
        "span-product-overflow",
        "span-sum-overflow",
        "offset-overflow",
        "out-of-bounds",
        "shape-product-overflow",
    ],
)
def test_cache_view_rejects_invalid_geometry_before_evaluation(shape, strides, offset):
    # Do not evaluate an invalid descriptor: overflow must be caught before
    # anything can consume an out-of-bounds GPU pointer.
    buffer = mx.zeros((16,), dtype=mx.uint8)
    with pytest.raises(ValueError, match="cache_view:"):
        get_ops().cache_view(buffer, shape, strides, offset)


@pytest.mark.parametrize("evaluate_input", [False, True], ids=["lazy", "evaluated"])
def test_cache_view_rejects_noncontiguous_typed_backing(evaluate_input):
    raw = mx.asarray(np.arange(32, dtype=np.uint8).reshape(4, 8), copy=False)
    buffer = raw.view(mx.uint32).T
    if evaluate_input:
        mx.eval(buffer)
    with pytest.raises(ValueError, match="cache_view:.*row.contiguous"):
        view = get_ops().cache_view(buffer, (2,), (1,), 0)
        mx.eval(view)


@pytest.mark.parametrize(
    "shape,strides,offset,indices",
    [((2, 2), (4, 1), 5, [[5, 6], [9, 10]]), ((), (), 15, 15)],
    ids=["strided", "scalar-at-end"],
)
def test_cache_view_preserves_typed_alias(shape, strides, offset, indices):
    raw = np.arange(64, dtype=np.uint8).reshape(4, 16)
    backing = mx.asarray(raw, copy=False).view(mx.uint32)
    view = get_ops().cache_view(backing, shape, strides, offset)
    mx.eval(view)
    typed = raw.view(np.uint32).reshape(-1)
    typed[offset] = 12345
    np.testing.assert_array_equal(np.array(view), typed[indices])


def test_cache_view_large_backing_does_not_flatten_or_evaluate():
    # Keep this metadata-only: a multidimensional arena may exceed MLX's
    # int32 dimension limit without requesting a multi-GiB allocation here.
    backing = mx.zeros((2, 1 << 30), dtype=mx.uint8)
    view = get_ops().cache_view(backing, (2, 2), (1 << 30, 1), 3)
    assert view.shape == (2, 2)
