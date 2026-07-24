# SPDX-License-Identifier: Apache-2.0
"""Spec-decode window-mode paged attention parity (issue #465).

A K+1-token speculative-verification window can be dispatched two ways through
``paged_attention_primitive``:

- expanded: K+1 one-token ``cu_seqlens`` segments (``window_seqlen_q=1``), the
  per-token decode path spec verification used before window mode;
- windowed: one K+1-token segment (``window_seqlen_q=K+1``), where a threadgroup
  owns the window's rows and shares each KV read across them.

The window gate keys the split-KV partition decision on the split-equivalent
grid (``num_heads * total_q_tokens``), the same grid the expanded path uses, so
the two dispatch the same kernel family and are BITWISE identical on any GPU
core count -- the shape band that otherwise diverged on 16-core parts is covered
here (32q/8kv, single-sequence windows).  Both are also checked against the
pure-MLX reference.

Deterministic, no model load: runs in CI (``pytest -m "not slow" tests/``)
against the source-built kernel.  Coverage spans what the dispatch gate can
route into window mode: dtypes fp16/bf16/fp32 and TurboQuant, both GQA head
counts plus every admitted head size (64..256, incl. non-multiples of 32), the
kernel block sizes {8, 16, 32}, windows 2..12 (odd tails and past the old
8-row cap), sliding window, mixed decode+window batches, both split-KV kernel
families forced deterministically, the masked-±inf V skip, and the
window-hint contract fail-fasts.

Run with:
    python -m pytest tests/test_spec_window_parity.py -v
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tools.attention_bench_utils import ref_paged_attn
from vllm_metal.attention.caches.turboquant import (
    get_v_centroids,
    turbo_quant_encode,
)
from vllm_metal.metal import get_ops

HEAD_SIZE = 128
BLOCK_SIZE = 16
V_BITS = 3

# atol/rtol per dtype vs the pure-MLX reference (same as test_split_kv_decode).
_TOLERANCES = {
    mx.bfloat16: (3e-2, 2e-2),
    mx.float16: (1.5e-2, 2e-2),
    mx.float32: (1e-3, 1e-3),
}


def _cache(
    num_blocks: int,
    num_kv_heads: int,
    dtype: mx.Dtype,
    head_size: int = HEAD_SIZE,
    block_size: int = BLOCK_SIZE,
) -> mx.array:
    return mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)


def _run(
    query: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    num_kv_heads: int,
    tables: list[list[int]],
    seq_lens: list[int],
    cu_seqlens: list[int],
    max_kv: int,
    window_seqlen_q: int,
    sliding_window: int,
    head_size: int = HEAD_SIZE,
    block_size: int = BLOCK_SIZE,
) -> mx.array:
    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        head_size**-0.5,
        0.0,
        mx.array(tables, dtype=mx.int32),
        mx.array(seq_lens, dtype=mx.int32),
        mx.array(cu_seqlens, dtype=mx.int32),
        block_size,
        max_kv,
        sliding_window,
        out,
        window_seqlen_q=window_seqlen_q,
    )
    mx.eval(out)
    return out


def _expanded_and_windowed(
    num_q_heads: int,
    num_kv_heads: int,
    dtype: mx.Dtype,
    ctx: int,
    windows: list[int],
    sliding_window: int,
    seed: int,
    head_size: int = HEAD_SIZE,
    block_size: int = BLOCK_SIZE,
) -> tuple[mx.array, mx.array, list[int]]:
    """Run the same query/cache expanded vs windowed; return (expanded, windowed,
    per-sequence total lengths)."""
    mx.random.seed(seed)
    total_q = sum(windows)
    lens = [ctx + w for w in windows]
    max_kv = max(lens)
    blocks_per_seq = (max_kv + block_size - 1) // block_size
    key_cache = _cache(
        blocks_per_seq * len(windows), num_kv_heads, dtype, head_size, block_size
    )
    value_cache = _cache(
        blocks_per_seq * len(windows), num_kv_heads, dtype, head_size, block_size
    )
    query = mx.random.normal(shape=(total_q, num_q_heads, head_size)).astype(dtype)
    mx.eval(key_cache, value_cache, query)
    tables = [
        list(range(s * blocks_per_seq, (s + 1) * blocks_per_seq))
        for s in range(len(windows))
    ]

    exp_tables, exp_seq, exp_cu = [], [], [0]
    for s, w in enumerate(windows):
        for j in range(w):
            exp_tables.append(tables[s])
            exp_seq.append(ctx + j + 1)
            exp_cu.append(exp_cu[-1] + 1)
    expanded = _run(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        exp_tables,
        exp_seq,
        exp_cu,
        max_kv,
        1,
        sliding_window,
        head_size,
        block_size,
    )

    win_cu = [0]
    for w in windows:
        win_cu.append(win_cu[-1] + w)
    windowed = _run(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        tables,
        lens,
        win_cu,
        max_kv,
        max(windows),
        sliding_window,
        head_size,
        block_size,
    )
    return expanded, windowed, lens


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads",
    [(16, 8), (32, 8)],  # 0.6B and 8B GQA; 32q/8kv is the split-KV band
)
@pytest.mark.parametrize("ctx", [512, 8192])
@pytest.mark.parametrize("window", [2, 4, 5, 6, 8])  # K+1 for K in 1..7
def test_windowed_matches_expanded(
    dtype: mx.Dtype, num_q_heads: int, num_kv_heads: int, ctx: int, window: int
) -> None:
    """Windowed dispatch is bitwise identical to the per-token expanded path
    (correctness vs the reference is covered by test_windowed_matches_reference)."""
    expanded, windowed, _ = _expanded_and_windowed(
        num_q_heads, num_kv_heads, dtype, ctx, [window], sliding_window=-1, seed=0
    )
    assert mx.array_equal(expanded, windowed), (
        f"window vs expanded not bitwise: {num_q_heads}q/{num_kv_heads}kv "
        f"{dtype} ctx={ctx} window={window}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(16, 8), (32, 8)])
@pytest.mark.parametrize("ctx", [512, 8192])
def test_windowed_matches_reference(
    dtype: mx.Dtype, num_q_heads: int, num_kv_heads: int, ctx: int
) -> None:
    """Windowed output matches the pure-MLX gather-then-SDPA reference."""
    mx.random.seed(1)
    window = 6
    lens = [ctx + window]
    max_kv = lens[0]
    blocks = (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    key_cache = _cache(blocks, num_kv_heads, dtype)
    value_cache = _cache(blocks, num_kv_heads, dtype)
    query = mx.random.normal(shape=(window, num_q_heads, HEAD_SIZE)).astype(dtype)
    mx.eval(key_cache, value_cache, query)
    table = [list(range(blocks))]
    windowed = _run(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        table,
        lens,
        [0, window],
        max_kv,
        window,
        -1,
    )
    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[window],
        kv_lens=lens,
        block_tables=np.array(table),
        scale=HEAD_SIZE**-0.5,
    )
    mx.eval(ref)
    atol, rtol = _TOLERANCES[dtype]
    np.testing.assert_allclose(
        np.array(windowed.astype(mx.float32)),
        np.array(ref.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("head_size", [64, 80, 96, 112, 192, 256])
def test_windowed_matches_expanded_head_sizes(head_size: int) -> None:
    """Bitwise parity for every head size the dispatch gate admits (<= 256),
    including the non-multiple-of-32 sizes only the per-token kernel serves."""
    expanded, windowed, _ = _expanded_and_windowed(
        16, 8, mx.float16, 512, [6], sliding_window=-1, seed=5, head_size=head_size
    )
    assert mx.array_equal(expanded, windowed), f"head_size={head_size}"


@pytest.mark.parametrize("block_size", [8, 32])
def test_windowed_matches_expanded_block_sizes(block_size: int) -> None:
    """Bitwise parity for the other kernel-instantiated block sizes."""
    expanded, windowed, _ = _expanded_and_windowed(
        16, 8, mx.float16, 512, [6], sliding_window=-1, seed=6, block_size=block_size
    )
    assert mx.array_equal(expanded, windowed), f"block_size={block_size}"


def test_windowed_matches_expanded_large_window() -> None:
    """Windows past the old 8-row cap still take window mode (sub-window count
    is unbounded) and stay bitwise identical."""
    expanded, windowed, _ = _expanded_and_windowed(
        16, 8, mx.float16, 2048, [12], sliding_window=-1, seed=7
    )
    assert mx.array_equal(expanded, windowed)


def test_windowed_single_partition_family() -> None:
    """Force the single-pass (_ps0) family on every machine: total KV fits one
    partition, so the split gate disengages regardless of GPU core count."""
    ctx, window = 200, 6
    assert ctx + window <= get_ops().PARTITION_SIZE
    expanded, windowed, _ = _expanded_and_windowed(
        16, 8, mx.float16, ctx, [window], sliding_window=-1, seed=8
    )
    assert mx.array_equal(expanded, windowed)


def test_windowed_forced_split_family() -> None:
    """Force the partitioned (_ps512) family on every machine: 8 q-heads keep
    the split-equivalent gate grid under any realistic min_decode_grid (>= 56
    on the smallest Apple GPU; see test_split_kv_decode), and the context
    spans several partitions."""
    ops = get_ops()
    num_q_heads, window = 8, 6
    assert num_q_heads * window < ops.min_decode_grid()
    expanded, windowed, _ = _expanded_and_windowed(
        num_q_heads, 8, mx.float16, 2048, [window], sliding_window=-1, seed=9
    )
    assert mx.array_equal(expanded, windowed)


@pytest.mark.parametrize(
    "cu,total_q,hint,match",
    [
        (
            [0, 6],
            6,
            2,
            r"window_seqlen_q=2 does not match the longest cu_seqlens_q segment \(6\)",
        ),
        (
            [0, 1, 6],
            6,
            3,
            r"window_seqlen_q=3 does not match the longest cu_seqlens_q segment \(5\)",
        ),
        (
            [0, 4],
            4,
            6,
            r"window_seqlen_q=6 does not match the longest cu_seqlens_q segment \(4\)",
        ),
        (
            [0, 4],
            6,
            6,
            r"must start at 0 and end at the query row count \(6\), got 1 segment",
        ),
        (
            [1, 7],
            7,
            6,
            r"must start at 0 and end at the query row count \(7\), got 1 segment",
        ),
        (
            [0],
            6,
            6,
            r"must start at 0 and end at the query row count \(6\), got 0 segment",
        ),
        (
            [0, 0, 6],
            6,
            6,
            r"segment 0 is empty \(0 -> 0\); every window-mode segment needs at least one query row",
        ),
        (
            [0, 6],
            6,
            0,
            r"window_seqlen_q must be >= 1 \(1 = per-token decode\), got 0",
        ),
        (
            [0, 6],
            6,
            -1,
            r"window_seqlen_q must be >= 1 \(1 = per-token decode\), got -1",
        ),
    ],
    ids=[
        "understated",
        "understated-average-blind",
        "overstated",
        "cu-end-mismatch",
        "cu-start-nonzero",
        "cu-no-segments",
        "empty-segment",
        "hint-zero",
        "hint-negative",
    ],
)
def test_window_hint_contract_rejected(
    cu: list[int], total_q: int, hint: int, match: str
) -> None:
    """window_seqlen_q must equal the longest segment of a structurally valid
    cu_seqlens_q.  An understated hint leaves window rows unwritten — including
    the mixed-length batch a per-batch average check cannot see (segments
    [1, 5] with hint 3 satisfy total <= num_seqs * hint yet drop the fifth
    row) — so the primitive rejects the whole contract-violation family."""
    mx.random.seed(10)
    key_cache = _cache(4, 8, mx.float16)
    value_cache = _cache(4, 8, mx.float16)
    query = mx.random.normal(shape=(total_q, 16, HEAD_SIZE)).astype(mx.float16)
    mx.eval(key_cache, value_cache, query)
    num_segments = len(cu) - 1
    with pytest.raises(ValueError, match=match):
        _run(
            query,
            key_cache,
            value_cache,
            8,
            [list(range(4))] * num_segments,
            [56] * num_segments,
            cu,
            56,
            hint,
            -1,
        )


def test_window_hint_rejects_non_int32_cu() -> None:
    """Window-mode validation reads cu_seqlens_q host-side as int32; any
    other dtype is rejected before the bytes could be reinterpreted."""
    mx.random.seed(13)
    window = 4
    key_cache = _cache(4, 8, mx.float16)
    value_cache = _cache(4, 8, mx.float16)
    query = mx.random.normal(shape=(window, 16, HEAD_SIZE)).astype(mx.float16)
    mx.eval(key_cache, value_cache, query)
    out = mx.array(0)
    with pytest.raises(
        ValueError, match="cu_seqlens_q must be int32 for window-mode validation"
    ):
        get_ops().paged_attention_primitive(
            query,
            key_cache,
            value_cache,
            8,
            HEAD_SIZE**-0.5,
            0.0,
            mx.array([list(range(4))], dtype=mx.int32),
            mx.array([50 + window], dtype=mx.int32),
            mx.array([0, window], dtype=mx.int64),
            BLOCK_SIZE,
            50 + window,
            -1,
            out,
            window_seqlen_q=window,
        )


def test_window_hint_rejected_past_head_size_bound() -> None:
    """Heads past PA_WINDOW_MAX_HEAD_SIZE never take window mode (per-thread
    register state scales with rows * head_size); the binding rejects the
    hint outright so a merged wide-head window can never silently fall to
    the tiled kernel and break the bitwise contract.  prepare_unified keeps
    those models on the expanded per-token layout."""
    mx.random.seed(12)
    window, head_size = 4, 512
    key_cache = _cache(4, 8, mx.float16, head_size=head_size)
    value_cache = _cache(4, 8, mx.float16, head_size=head_size)
    query = mx.random.normal(shape=(window, 16, head_size)).astype(mx.float16)
    mx.eval(key_cache, value_cache, query)
    with pytest.raises(ValueError, match=r"requires head_size <= 256, got 512"):
        _run(
            query,
            key_cache,
            value_cache,
            8,
            [list(range(4))],
            [50 + window],
            [0, window],
            50 + window,
            window,
            -1,
            head_size=head_size,
        )


def test_windowed_skips_masked_inf_v() -> None:
    """A ±inf V value at a position only later window rows may attend must not
    leak into earlier rows: the w == 0 skip keeps their accumulation sequence
    identical to the expanded dispatch (0 * inf would otherwise inject NaN)."""
    mx.random.seed(11)
    ctx, window = 48, 4
    total_len = ctx + window
    blocks = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    key_cache = _cache(blocks, 8, mx.float16)
    value_cache = _cache(blocks, 8, mx.float16)
    poison_pos = ctx + 2  # visible only to window rows 2 and 3
    value_cache[poison_pos // BLOCK_SIZE, poison_pos % BLOCK_SIZE, :, :] = float("inf")
    query = mx.random.normal(shape=(window, 16, HEAD_SIZE)).astype(mx.float16)
    mx.eval(key_cache, value_cache, query)
    table = list(range(blocks))

    expanded = _run(
        query,
        key_cache,
        value_cache,
        8,
        [table] * window,
        [ctx + j + 1 for j in range(window)],
        list(range(window + 1)),
        total_len,
        1,
        -1,
    )
    windowed = _run(
        query,
        key_cache,
        value_cache,
        8,
        [table],
        [total_len],
        [0, window],
        total_len,
        window,
        -1,
    )
    # Rows 0-1 never see the poisoned position: bitwise parity must hold and
    # the outputs must stay finite.  Rows 2-3 legitimately read the inf in
    # both dispatches, so they are excluded (NaN compares unequal to itself).
    assert mx.array_equal(expanded[:2], windowed[:2])
    assert bool(mx.all(mx.isfinite(windowed[:2].astype(mx.float32))))


@pytest.mark.parametrize("sliding_window", [512])
@pytest.mark.parametrize("window", [2, 6])
def test_windowed_matches_expanded_sliding_window(
    sliding_window: int, window: int
) -> None:
    """Bitwise parity holds with a sliding window (Gemma-style)."""
    expanded, windowed, _ = _expanded_and_windowed(
        16, 8, mx.float16, 8192, [window], sliding_window, seed=2
    )
    assert mx.array_equal(expanded, windowed)


def test_windowed_matches_expanded_mixed_batch() -> None:
    """A batch mixing a plain decode and two different-length verify windows
    stays bitwise identical across dispatch."""
    expanded, windowed, _ = _expanded_and_windowed(
        16, 8, mx.float16, 8192, [1, 4, 6], sliding_window=-1, seed=3
    )
    assert mx.array_equal(expanded, windowed)


@pytest.mark.parametrize("quant", ["q8_0", "q4_0"])
@pytest.mark.parametrize("window", [4, 6])
def test_windowed_matches_expanded_turboquant(quant: str, window: int) -> None:
    """Bitwise parity holds for TurboQuant K/V caches."""
    mx.random.seed(4)
    num_kv_heads, num_q_heads = 8, 16
    ctx = 8192
    total_len = ctx + window
    nblocks = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_len = nblocks * BLOCK_SIZE
    k = mx.random.normal(shape=(kv_len, num_kv_heads, HEAD_SIZE)).astype(mx.float16)
    v = mx.random.normal(shape=(kv_len, num_kv_heads, HEAD_SIZE)).astype(mx.float16)
    query = mx.random.normal(shape=(window, num_q_heads, HEAD_SIZE)).astype(mx.float16)
    mx.eval(k, v, query)
    (k_packed, k_scale, k_zero), (v_packed, v_scale) = turbo_quant_encode(k, v, quant)
    sg = k_scale.shape[-1]
    kc = k_packed.reshape(nblocks, BLOCK_SIZE, num_kv_heads, -1)
    vc = v_packed.reshape(nblocks, BLOCK_SIZE, num_kv_heads, -1)
    ks = k_scale.reshape(nblocks, BLOCK_SIZE, num_kv_heads, sg)
    vs = v_scale.reshape(nblocks, BLOCK_SIZE, num_kv_heads, sg)
    kz = k_zero.reshape(nblocks, BLOCK_SIZE, num_kv_heads, sg)
    mx.eval(kc, vc, ks, vs, kz)
    blocks = list(range(nblocks))

    def run_tq(tables, seq_lens, cu, wq):
        out = mx.array(0)
        get_ops().paged_attention_primitive(
            query,
            kc,
            vc,
            num_kv_heads,
            HEAD_SIZE**-0.5,
            0.0,
            mx.array(tables, dtype=mx.int32),
            mx.array(seq_lens, dtype=mx.int32),
            mx.array(cu, dtype=mx.int32),
            BLOCK_SIZE,
            total_len,
            -1,
            out,
            key_scale_cache=ks,
            value_scale_cache=vs,
            key_zero_cache=kz,
            v_centroids=get_v_centroids(V_BITS),
            use_turboquant=True,
            quant_type=quant,
            v_bits=V_BITS,
            window_seqlen_q=wq,
        )
        mx.eval(out)
        return out

    expanded = run_tq(
        [blocks] * window,
        [ctx + j + 1 for j in range(window)],
        list(range(window + 1)),
        1,
    )
    windowed = run_tq([blocks], [total_len], [0, window], window)
    assert mx.array_equal(expanded, windowed)
