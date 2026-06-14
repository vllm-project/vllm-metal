# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dense non-causal varlen encoder attention (RFC #333 PR1)."""

from __future__ import annotations

import platform

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal encoder kernels require Apple Silicon",
)

from vllm_metal.metal import encoder_varlen_attention  # noqa: E402

_HEAD_DIMS = [64, 80, 96, 128]
_DTYPES = [mx.float16, mx.bfloat16, mx.float32]
_SEGMENT_CONFIGS = [
    [32, 48],
    [16, 16, 16],
    [7, 13, 21, 5],
]


def _tolerance(dtype: mx.Dtype) -> tuple[float, float]:
    if dtype == mx.bfloat16:
        return 1e-2, 2e-2
    if dtype == mx.float16:
        return 1e-2, 1.5e-2
    return 1e-4, 1e-4


def _ref_encoder_varlen_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens: mx.array,
    *,
    scale: float,
) -> mx.array:
    """Pure-MLX split-loop bidirectional attention reference."""
    import mlx.core as mx

    q_np = np.array(q.astype(mx.float32))
    k_np = np.array(k.astype(mx.float32))
    v_np = np.array(v.astype(mx.float32))
    bounds = [int(x) for x in np.asarray(cu_seqlens)]
    out = np.zeros_like(q_np, dtype=np.float32)

    for seg in range(len(bounds) - 1):
        lo, hi = bounds[seg], bounds[seg + 1]
        q_seg = q_np[lo:hi]
        k_seg = k_np[lo:hi]
        v_seg = v_np[lo:hi]
        scores = scale * np.einsum("ihd,jhd->hij", q_seg, k_seg)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        out[lo:hi] = np.einsum("hij,jhd->ihd", weights, v_seg)

    return mx.array(out).astype(q.dtype)


def _make_inputs(
    segment_lens: list[int],
    *,
    num_heads: int,
    head_dim: int,
    dtype: mx.Dtype,
    seed: int = 0,
):
    mx.random.seed(seed)
    total_tokens = sum(segment_lens)
    cu = mx.array([0, *np.cumsum(segment_lens).tolist()], dtype=mx.int32)
    q = mx.random.normal(shape=(total_tokens, num_heads, head_dim)).astype(dtype)
    k = mx.random.normal(shape=(total_tokens, num_heads, head_dim)).astype(dtype)
    v = mx.random.normal(shape=(total_tokens, num_heads, head_dim)).astype(dtype)
    max_seqlen = max(segment_lens)
    scale = head_dim**-0.5
    return q, k, v, cu, max_seqlen, scale


@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("segment_lens", _SEGMENT_CONFIGS)
def test_encoder_varlen_matches_mlx_reference(
    head_dim: int,
    dtype: mx.Dtype,
    segment_lens: list[int],
) -> None:
    num_heads = 4
    q, k, v, cu, max_seqlen, scale = _make_inputs(
        segment_lens,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    out = encoder_varlen_attention(
        q,
        k,
        v,
        cu,
        max_seqlen=max_seqlen,
        softmax_scale=scale,
    )
    ref = _ref_encoder_varlen_attention(q, k, v, cu, scale=scale)

    mx.eval(out, ref)
    rtol, atol = _tolerance(dtype)
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        np.array(ref.astype(mx.float32)),
        rtol=rtol,
        atol=atol,
    )


def test_encoder_varlen_lazy_graph_self_chain() -> None:
    segment_lens = [24, 40]
    q, k, v, cu, max_seqlen, scale = _make_inputs(
        segment_lens,
        num_heads=2,
        head_dim=64,
        dtype=mx.float16,
        seed=1,
    )

    first = encoder_varlen_attention(
        q, k, v, cu, max_seqlen=max_seqlen, softmax_scale=scale
    )
    second = encoder_varlen_attention(
        first, k, v, cu, max_seqlen=max_seqlen, softmax_scale=scale
    )
    mx.eval(second)
    assert np.array(second).shape == (sum(segment_lens), 2, 64)


def test_encoder_varlen_rejects_bad_cu_seqlens() -> None:
    q, k, v, cu, max_seqlen, scale = _make_inputs(
        [16, 16],
        num_heads=2,
        head_dim=64,
        dtype=mx.float16,
    )
    bad = mx.array([0, 10, 31], dtype=mx.int32)
    with pytest.raises(ValueError, match="cu_seqlens\\[-1\\]"):
        encoder_varlen_attention(
            q, k, v, bad, max_seqlen=max_seqlen, softmax_scale=scale
        )


def test_encoder_varlen_rejects_unsupported_head_dim() -> None:
    q, k, v, cu, max_seqlen, scale = _make_inputs(
        [16],
        num_heads=2,
        head_dim=72,
        dtype=mx.float16,
    )
    with pytest.raises(ValueError, match="unsupported head_dim"):
        encoder_varlen_attention(
            q, k, v, cu, max_seqlen=max_seqlen, softmax_scale=scale
        )


def test_encoder_varlen_rejects_max_seqlen_too_small() -> None:
    q, k, v, cu, _max_seqlen, scale = _make_inputs(
        [32, 48],
        num_heads=2,
        head_dim=64,
        dtype=mx.float16,
    )
    with pytest.raises(ValueError, match="max_seqlen must be >="):
        encoder_varlen_attention(q, k, v, cu, max_seqlen=16, softmax_scale=scale)
