# SPDX-License-Identifier: Apache-2.0
"""Attention-sink parity for the paged attention kernel (issue #212).

GPT-OSS style attention sinks add one learned logit per query head to the
softmax denominator without contributing a value row.  The Metal kernel carries
``use_sinks`` (function constant 40) and reads the sink array at buffer 18
(buffer 6 in the reduce), but nothing could reach it until
``paged_attention_primitive`` grew a ``sinks`` argument.

The oracle is MLX's own ``mx.fast.scaled_dot_product_attention(..., sinks=...)``
(MLX >= 0.32), so these compare the kernel against the same reference the model
path would use.

Four kernel paths fold sinks, and each is covered here:

- non-partitioned: the sink joins the merged threadgroup state once, AFTER the
  per-warp merge.  Folding it before the merge counts it ``NUM_WARPS`` times;
  that regression is what ``test_sinks_not_multiplied_by_warp_count`` pins.
- split-KV: ``paged_attention_v2_reduce`` folds the sink into the global max and
  exp-sum, so it is counted once across partitions rather than once per
  partition.
- tiled prefill: same softmax update as non-partitioned, one sink per query row.
- window mode: same as non-partitioned, per window row.

Deterministic, no model load.

Run with:
    python -m pytest tests/test_paged_attention_sinks.py -v
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from vllm_metal.metal import get_ops

HEAD_SIZE = 128
BLOCK_SIZE = 16
GPT_OSS_HEAD_SIZE = 64
GPT_OSS_NUM_Q_HEADS = 64
GPT_OSS_NUM_KV_HEADS = 8
GPT_OSS_PREFILL_CTX = 64
GPT_OSS_PREFILL_LEN = 64

# Matched to the repo's existing paged-attention parity tolerances.
_TOLERANCES = {
    mx.bfloat16: (3e-2, 2e-2),
    mx.float16: (1.5e-2, 2e-2),
    mx.float32: (1e-3, 1e-3),
}


def _sinks(num_q_heads: int, value: float | None = None) -> mx.array:
    """Per-head sink logits; spread across a range when no value is given."""
    if value is not None:
        return mx.ones((num_q_heads,), dtype=mx.float32) * value
    return (mx.arange(num_q_heads).astype(mx.float32) - num_q_heads / 2.0) * 1.5


def _run_decode(
    ctx: int,
    num_q_heads: int,
    num_kv_heads: int,
    sinks: mx.array | None,
    dtype: mx.Dtype = mx.float32,
    head_size: int = HEAD_SIZE,
    seed: int = 0,
) -> tuple[mx.array, mx.array]:
    """One decode row against the paged kernel and the MLX oracle."""
    mx.random.seed(seed)
    blocks = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    key_cache = mx.random.normal(
        shape=(blocks, BLOCK_SIZE, num_kv_heads, head_size)
    ).astype(dtype)
    value_cache = mx.random.normal(
        shape=(blocks, BLOCK_SIZE, num_kv_heads, head_size)
    ).astype(dtype)
    query = mx.random.normal(shape=(1, num_q_heads, head_size)).astype(dtype)
    mx.eval(key_cache, value_cache, query)

    scale = head_size**-0.5
    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        0.0,
        mx.array([list(range(blocks))], dtype=mx.int32),
        mx.array([ctx], dtype=mx.int32),
        mx.array([0, 1], dtype=mx.int32),
        BLOCK_SIZE,
        ctx,
        -1,
        out,
        sinks=sinks,
    )
    mx.eval(out)

    # MLX wants [B, N, T, D]; the paged cache is [blocks, block, kv, D].
    flat_k = key_cache.reshape(blocks * BLOCK_SIZE, num_kv_heads, head_size)[:ctx]
    flat_v = value_cache.reshape(blocks * BLOCK_SIZE, num_kv_heads, head_size)[:ctx]
    # The kernel reads sinks as device float (fp32); MLX instead requires them
    # to promote to the output dtype, so the oracle takes a cast copy.
    ref = mx.fast.scaled_dot_product_attention(
        mx.transpose(query, (1, 0, 2))[None],
        mx.transpose(flat_k, (1, 0, 2))[None],
        mx.transpose(flat_v, (1, 0, 2))[None],
        scale=scale,
        sinks=None if sinks is None else sinks.astype(dtype),
    )
    mx.eval(ref)
    return out.reshape(num_q_heads, head_size), ref[0, :, 0, :]


def _run_prefill(
    ctx: int,
    query_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: mx.Dtype,
    sinks: mx.array,
    seed: int = 0,
) -> tuple[mx.array, mx.array]:
    """One ordinary multi-token prefill segment against the MLX oracle."""
    mx.random.seed(seed)
    kv_len = ctx + query_len
    blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    key_cache = mx.random.normal(
        shape=(blocks, BLOCK_SIZE, num_kv_heads, head_size)
    ).astype(dtype)
    value_cache = mx.random.normal(
        shape=(blocks, BLOCK_SIZE, num_kv_heads, head_size)
    ).astype(dtype)
    query = mx.random.normal(shape=(query_len, num_q_heads, head_size)).astype(dtype)
    mx.eval(key_cache, value_cache, query)

    scale = head_size**-0.5
    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        0.0,
        mx.array([list(range(blocks))], dtype=mx.int32),
        mx.array([kv_len], dtype=mx.int32),
        mx.array([0, query_len], dtype=mx.int32),
        BLOCK_SIZE,
        kv_len,
        -1,
        out,
        sinks=sinks,
    )
    mx.eval(out)

    flat_k = key_cache.reshape(blocks * BLOCK_SIZE, num_kv_heads, head_size)[:kv_len]
    flat_v = value_cache.reshape(blocks * BLOCK_SIZE, num_kv_heads, head_size)[:kv_len]
    ref = mx.fast.scaled_dot_product_attention(
        mx.transpose(query, (1, 0, 2))[None],
        mx.transpose(flat_k, (1, 0, 2))[None],
        mx.transpose(flat_v, (1, 0, 2))[None],
        scale=scale,
        mask="causal",
        sinks=sinks.astype(dtype),
    )
    mx.eval(ref)
    return out.reshape(query_len, num_q_heads, head_size), mx.transpose(
        ref[0], (1, 0, 2)
    )


def _assert_close(got: mx.array, ref: mx.array, dtype: mx.Dtype) -> None:
    atol, rtol = _TOLERANCES[dtype]
    assert mx.allclose(got, ref, atol=atol, rtol=rtol), (
        f"max abs diff {float(mx.max(mx.abs(got - ref))):.3e} exceeds atol={atol}"
    )


class TestSinkParity:
    """Kernel vs mx.fast.scaled_dot_product_attention with sinks."""

    @pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
    @pytest.mark.parametrize("sink_value", [0.0, 2.0, -5.0])
    def test_decode_matches_mlx_oracle(
        self, dtype: mx.Dtype, sink_value: float
    ) -> None:
        got, ref = _run_decode(64, 8, 8, _sinks(8, sink_value), dtype=dtype)
        _assert_close(got, ref, dtype)

    def test_per_head_sink_values(self) -> None:
        """Each head must use its OWN sink, not head 0's."""
        got, ref = _run_decode(64, 8, 8, _sinks(8))
        _assert_close(got, ref, mx.float32)

    @pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 8), (8, 2), (16, 4)])
    def test_gqa_head_ratios(self, num_q_heads: int, num_kv_heads: int) -> None:
        got, ref = _run_decode(64, num_q_heads, num_kv_heads, _sinks(num_q_heads))
        _assert_close(got, ref, mx.float32)

    @pytest.mark.parametrize("head_size", [64, 128])
    def test_head_sizes(self, head_size: int) -> None:
        got, ref = _run_decode(64, 8, 8, _sinks(8), head_size=head_size)
        _assert_close(got, ref, mx.float32)

    def test_split_kv_path(self) -> None:
        """Context past PARTITION_SIZE routes through the reduce kernel."""
        ctx = get_ops().PARTITION_SIZE * 3
        got, ref = _run_decode(ctx, 8, 8, _sinks(8))
        _assert_close(got, ref, mx.float32)

    def test_sinks_none_is_plain_softmax(self) -> None:
        """sinks=None must leave the existing path bit-for-bit unchanged."""
        got, ref = _run_decode(64, 8, 8, None)
        _assert_close(got, ref, mx.float32)

    def test_sinks_change_the_output(self) -> None:
        """Guards against the argument being accepted and then ignored."""
        without, _ = _run_decode(64, 8, 8, None)
        with_sinks, _ = _run_decode(64, 8, 8, _sinks(8, 2.0))
        assert float(mx.max(mx.abs(without - with_sinks))) > 1e-3

    def test_sinks_not_multiplied_by_warp_count(self) -> None:
        """Pins the bug this issue's kernel path shipped with.

        The vendored code folded the sink into every warp's state before the
        threadgroup merge summed all NUM_WARPS states, so the sink landed in the
        denominator NUM_WARPS times.  A large positive sink makes that
        unmissable: it is the case where exp(sink) dominates the denominator.
        """
        got, ref = _run_decode(64, 8, 8, _sinks(8, 2.0))
        # 8x over-counting showed up as ~2e-1 here, 6 orders above this bound.
        assert float(mx.max(mx.abs(got - ref))) < 1e-4


class TestSinkPrefillMode:
    """Ordinary prefill with sinks stays on the tiled sink path."""

    def test_prefill_matches_mlx_oracle(self) -> None:
        got, ref = _run_prefill(
            GPT_OSS_PREFILL_CTX,
            GPT_OSS_PREFILL_LEN,
            GPT_OSS_NUM_Q_HEADS,
            GPT_OSS_NUM_KV_HEADS,
            GPT_OSS_HEAD_SIZE,
            mx.bfloat16,
            _sinks(GPT_OSS_NUM_Q_HEADS),
        )
        _assert_close(got, ref, mx.bfloat16)


def _run_window(
    ctx: int,
    windows: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    sinks: mx.array | None,
    dtype: mx.Dtype = mx.float32,
    seed: int = 0,
) -> tuple[mx.array, mx.array]:
    """Same batch dispatched expanded (per-token) and windowed."""
    mx.random.seed(seed)
    total_q = sum(windows)
    lens = [ctx + w for w in windows]
    max_kv = max(lens)
    bps = (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    shape = (bps * len(windows), BLOCK_SIZE, num_kv_heads, HEAD_SIZE)
    key_cache = mx.random.normal(shape=shape).astype(dtype)
    value_cache = mx.random.normal(shape=shape).astype(dtype)
    query = mx.random.normal(shape=(total_q, num_q_heads, HEAD_SIZE)).astype(dtype)
    mx.eval(key_cache, value_cache, query)
    tables = [list(range(s * bps, (s + 1) * bps)) for s in range(len(windows))]

    def dispatch(tab, seq, cu, wq):
        out = mx.array(0)
        get_ops().paged_attention_primitive(
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            HEAD_SIZE**-0.5,
            0.0,
            mx.array(tab, dtype=mx.int32),
            mx.array(seq, dtype=mx.int32),
            mx.array(cu, dtype=mx.int32),
            BLOCK_SIZE,
            max_kv,
            -1,
            out,
            window_seqlen_q=wq,
            sinks=sinks,
        )
        mx.eval(out)
        return out

    exp_tab, exp_seq, exp_cu = [], [], [0]
    for s, w in enumerate(windows):
        for j in range(w):
            exp_tab.append(tables[s])
            exp_seq.append(ctx + j + 1)
            exp_cu.append(exp_cu[-1] + 1)

    win_cu = [0]
    for w in windows:
        win_cu.append(win_cu[-1] + w)

    return (
        dispatch(exp_tab, exp_seq, exp_cu, 1),
        dispatch(tables, lens, win_cu, max(windows)),
    )


class TestSinkWindowMode:
    """Spec-decode window mode folds sinks per window row, once each.

    The expanded layout drives the per-token kernel, which TestSinkParity
    already pins against the MLX oracle, so agreement between the layouts is
    what validates the window path.
    """

    @pytest.mark.parametrize("windows", [[2], [4], [2, 3]])
    @pytest.mark.parametrize("sink_value", [None, 2.0, -5.0])
    def test_windowed_matches_expanded(
        self, windows: list[int], sink_value: float | None
    ) -> None:
        sinks = None if sink_value is None else _sinks(8, sink_value)
        expanded, windowed = _run_window(128, windows, 8, 8, sinks)
        _assert_close(windowed, expanded, mx.float32)

    def test_windowed_matches_expanded_split_kv(self) -> None:
        ctx = get_ops().PARTITION_SIZE * 3
        expanded, windowed = _run_window(ctx, [4], 8, 8, _sinks(8))
        _assert_close(windowed, expanded, mx.float32)

    def test_window_sinks_are_applied(self) -> None:
        """Window mode must not silently drop the sink term."""
        without, _ = _run_window(128, [4], 8, 8, None)
        with_sinks, _ = _run_window(128, [4], 8, 8, _sinks(8, 2.0))
        assert float(mx.max(mx.abs(without - with_sinks))) > 1e-3


class TestSinkValidation:
    """Fail-fast contract for unsupported or malformed sinks."""

    def _call(self, sinks: mx.array, **kwargs: object) -> None:
        ctx, nq, nkv = 32, 8, 8
        blocks = ctx // BLOCK_SIZE
        kc = mx.zeros((blocks, BLOCK_SIZE, nkv, HEAD_SIZE))
        vc = mx.zeros((blocks, BLOCK_SIZE, nkv, HEAD_SIZE))
        q = mx.zeros((1, nq, HEAD_SIZE))
        out = mx.array(0)
        get_ops().paged_attention_primitive(
            q,
            kc,
            vc,
            nkv,
            HEAD_SIZE**-0.5,
            0.0,
            mx.array([list(range(blocks))], dtype=mx.int32),
            mx.array([ctx], dtype=mx.int32),
            mx.array([0, 1], dtype=mx.int32),
            BLOCK_SIZE,
            ctx,
            -1,
            out,
            sinks=sinks,
            **kwargs,
        )

    def test_rejects_wrong_head_count(self) -> None:
        with pytest.raises(ValueError, match="one entry per query head"):
            self._call(mx.zeros((4,), dtype=mx.float32))

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            self._call(mx.zeros((8, 1), dtype=mx.float32))

    def test_rejects_non_float32(self) -> None:
        with pytest.raises(ValueError, match="float32"):
            self._call(mx.zeros((8,), dtype=mx.float16))

    def test_rejects_turboquant_combination(self) -> None:
        """Upstream MLX refuses sinks with a quantized cache; so do we."""
        with pytest.raises(ValueError, match="TurboQuant"):
            self._call(mx.zeros((8,), dtype=mx.float32), use_turboquant=True)
