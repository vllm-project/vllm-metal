# SPDX-License-Identifier: Apache-2.0
"""Split-KV (flash-decoding) decode correctness.

``dispatch_paged_attention_v2_online`` routes pure-decode batches to the
partitioned kernel (``_ps512`` + ``paged_attention_v2_reduce``) when the base
grid (num_q_heads * num_seqs) underfills the GPU — fewer than 8 threadgroups
per core — and the longest context spans >= 2 partitions (> 512 tokens).

Every split case engages on any hardware: base_grid <= 32, while the gate
threshold is >= 56 (8 x 7 cores on the smallest Apple GPU) and 112 on the
IORegistry fallback used in VMs — and each test asserts that premise via the
exported ``min_decode_grid()`` so a gate regression fails loudly instead of
silently degrading every cell to single-pass-vs-reference.

Sliding-window and TurboQuant batches take the split too: windowed cells
cover window starts on and off partition boundaries (fully- and
partially-masked partitions), and TQ cells cover both packing families
(8-bit direct, 4-bit packed) with the inverse FWHT deferred to the reduce
pass.

Run with:
    python -m pytest tests/test_split_kv_decode.py -v
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tools.attention_bench_utils import ref_paged_attn
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.turboquant import (
    get_v_centroids,
    turbo_quant_decode,
    turbo_quant_encode,
)
from vllm_metal.metal import get_ops

NUM_QUERY_HEADS = 16  # Qwen3-0.6B GQA shape: 16 query / 8 KV heads
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 256

# atol/rtol per dtype.  float32: both paths compute in fp32, only
# kernel-order error is left.
_TOLERANCES = {
    mx.bfloat16: (3e-2, 2e-2),
    mx.float16: (1.5e-2, 2e-2),
    mx.float32: (1e-3, 1e-3),
}


def _run_paged_decode(
    kv_lens: list[int], dtype: mx.Dtype, seed: int
) -> tuple[mx.array, mx.array]:
    """Run one decode step through the primitive; return (out, reference)."""
    mx.random.seed(seed)
    num_seqs = len(kv_lens)
    max_kv_len = max(kv_lens)
    scale = HEAD_SIZE**-0.5

    key_cache = mx.random.normal(
        shape=(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    ).astype(dtype)
    value_cache = mx.random.normal(
        shape=(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    ).astype(dtype)
    query = mx.random.normal(shape=(num_seqs, NUM_QUERY_HEADS, HEAD_SIZE)).astype(dtype)

    max_blocks_per_seq = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = mx.random.randint(
        0, NUM_BLOCKS, shape=(num_seqs, max_blocks_per_seq)
    ).astype(mx.int32)
    kv_lens_arr = mx.array(kv_lens, dtype=mx.int32)
    cu_seqlens_q = mx.arange(num_seqs + 1, dtype=mx.int32)
    mx.eval(key_cache, value_cache, query, block_tables, kv_lens_arr, cu_seqlens_q)

    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        NUM_KV_HEADS,
        scale,
        0.0,  # softcap
        block_tables,
        kv_lens_arr,
        cu_seqlens_q,
        BLOCK_SIZE,
        max_kv_len,
        -1,  # sliding_window (disabled here; windowed cells below)
        out,
    )
    mx.eval(out)

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=kv_lens,
        block_tables=np.array(block_tables),
        scale=scale,
    )
    mx.eval(ref)
    return out, ref


def _assert_close(out: mx.array, ref: mx.array, dtype: mx.Dtype) -> None:
    atol, rtol = _TOLERANCES[dtype]
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        np.array(ref.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "kv_lens",
    [
        [8192],  # headline single-stream regime: 16 partitions
        [100, 2048],  # 1 valid partition inside a 4-partition grid
        [700, 4096],  # non-block-aligned lengths: 2 vs 8 valid partitions
        [1300],  # ODD partition count (3): reduce shmem 16-byte alignment
    ],
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_split_decode_vs_reference(kv_lens: list[int], dtype: mx.Dtype) -> None:
    """Partitioned decode matches the pure-MLX reference."""
    ops = get_ops()
    # Engagement premise: the gate is invisible from Python, so assert the
    # inputs that make it take the split path — otherwise a gate regression
    # would silently turn every cell into single-pass-vs-reference.
    assert NUM_QUERY_HEADS * len(kv_lens) < ops.min_decode_grid()
    assert max(kv_lens) > ops.PARTITION_SIZE

    out, ref = _run_paged_decode(kv_lens, dtype, seed=0)
    _assert_close(out, ref, dtype)


def test_high_occupancy_stays_single_pass() -> None:
    """The gate's OFF side: a batch whose base grid meets this machine's
    threshold must stay on the single-pass path and still match the reference.
    num_seqs is derived from the exported threshold, so the construction is
    machine-robust (engages nowhere)."""
    ops = get_ops()
    num_seqs = -(-ops.min_decode_grid() // NUM_QUERY_HEADS)  # ceil division
    kv_lens = [1024] * num_seqs  # > PARTITION_SIZE: only the grid term gates
    assert NUM_QUERY_HEADS * num_seqs >= ops.min_decode_grid()

    out, ref = _run_paged_decode(kv_lens, mx.float16, seed=1)
    _assert_close(out, ref, mx.float16)


def _ref_attention_np(
    query: mx.array, k_rows: np.ndarray, v_rows: np.ndarray
) -> np.ndarray:
    """float32 GQA attention over the given K/V rows (T, KV_HEADS, HEAD)."""
    q = np.array(query.astype(mx.float32))[0]
    rep = NUM_QUERY_HEADS // NUM_KV_HEADS
    k = np.repeat(k_rows, rep, axis=1)
    v = np.repeat(v_rows, rep, axis=1)
    scores = np.einsum("hd,thd->ht", q, k) * HEAD_SIZE**-0.5
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return np.einsum("ht,thd->hd", p, v)


@pytest.mark.parametrize(
    "kv_len,window",
    [
        (2048, 512),  # window start ON a partition boundary: 3 of 4 fully masked
        (2048, 700),  # window start mid-partition: partially masked partition
        (8192, 1300),  # long context: 13 fully-masked partitions
    ],
)
def test_split_decode_sliding_window(kv_len: int, window: int) -> None:
    """Partitioned decode with a sliding window matches a windowed reference.

    Fully-masked partitions must contribute exact zeros (epsilon-normalized
    partial, zero merge weight), never NaN."""
    mx.random.seed(2)
    ops = get_ops()
    assert NUM_QUERY_HEADS < ops.min_decode_grid()
    assert kv_len > ops.PARTITION_SIZE

    nblocks = kv_len // BLOCK_SIZE
    key_cache = mx.random.normal(
        shape=(nblocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    ).astype(mx.float16)
    value_cache = mx.random.normal(
        shape=(nblocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    ).astype(mx.float16)
    query = mx.random.normal(shape=(1, NUM_QUERY_HEADS, HEAD_SIZE)).astype(mx.float16)
    # identity block table: flat cache row t == token t, so the reference can
    # slice the window directly
    block_tables = mx.arange(nblocks, dtype=mx.int32).reshape(1, nblocks)
    seq_lens = mx.array([kv_len], dtype=mx.int32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
    mx.eval(key_cache, value_cache, query, block_tables, seq_lens, cu_seqlens_q)

    out = mx.array(0)
    ops.paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        NUM_KV_HEADS,
        HEAD_SIZE**-0.5,
        0.0,
        block_tables,
        seq_lens,
        cu_seqlens_q,
        BLOCK_SIZE,
        kv_len,
        window,
        out,
    )
    mx.eval(out)
    o = np.array(out.astype(mx.float32))[0]
    assert np.isfinite(o).all()

    start = kv_len - window
    k_rows = np.array(
        key_cache.reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[start:kv_len].astype(mx.float32)
    )
    v_rows = np.array(
        value_cache.reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[start:kv_len].astype(
            mx.float32
        )
    )
    ref = _ref_attention_np(query, k_rows, v_rows)
    np.testing.assert_allclose(o, ref, atol=1.5e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "quant,window",
    [
        ("q8_0", -1),
        ("q4_0", -1),
        ("q8_0", 700),  # TQ x sliding window: both gate-admitted features at once
    ],
)
def test_split_decode_turboquant(quant: str, window: int) -> None:
    """Partitioned TurboQuant decode matches the dequantized reference.

    kv_len=1300 also covers an odd partition count on the TQ reduce path
    (single inverse FWHT applied after the fp32 partition merge).  The
    windowed cell combines TQ with sliding-window masking — one fully and
    one partially masked partition over rotated-domain partials."""
    kv_len = 1300
    mx.random.seed(3)
    ops = get_ops()
    assert NUM_QUERY_HEADS < ops.min_decode_grid()
    assert kv_len > ops.PARTITION_SIZE

    nblocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache = MetalPagedKVCache(
        num_layers=1,
        num_blocks=nblocks,
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_SIZE,
        dtype=mx.float16,
        turboquant=True,
        k_quant=quant,
    )
    k = mx.random.normal(shape=(kv_len, NUM_KV_HEADS, HEAD_SIZE)).astype(mx.float16)
    v = mx.random.normal(shape=(kv_len, NUM_KV_HEADS, HEAD_SIZE)).astype(mx.float16)
    query = mx.random.normal(shape=(1, NUM_QUERY_HEADS, HEAD_SIZE)).astype(mx.float16)
    mx.eval(k, v, query)

    (k_packed, k_scale, k_zero), (v_packed, v_scale) = turbo_quant_encode(k, v, quant)
    slot = mx.arange(kv_len, dtype=mx.int64)
    layer = 0
    flat_k = cache.key_caches[layer].reshape(-1, NUM_KV_HEADS, cache.k_packed_dim)
    flat_k[slot] = k_packed
    cache.key_caches[layer] = flat_k.reshape(cache.key_caches[layer].shape)
    flat_v = cache.value_caches[layer].reshape(-1, NUM_KV_HEADS, cache.v_packed_dim)
    flat_v[slot] = v_packed
    cache.value_caches[layer] = flat_v.reshape(cache.value_caches[layer].shape)
    scale_groups = k_scale.shape[-1]
    for attr, data in [
        ("key_scale_caches", k_scale),
        ("value_scale_caches", v_scale),
        ("key_zero_caches", k_zero),
    ]:
        arr = getattr(cache, attr)[layer]
        flat = arr.reshape(-1, NUM_KV_HEADS, scale_groups)
        flat[slot] = data
        getattr(cache, attr)[layer] = flat.reshape(arr.shape)
    mx.eval(cache.key_caches[layer], cache.value_caches[layer])

    block_tables = mx.arange(nblocks, dtype=mx.int32).reshape(1, nblocks)
    seq_lens = mx.array([kv_len], dtype=mx.int32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
    out = mx.zeros((1, NUM_QUERY_HEADS, HEAD_SIZE), dtype=mx.float16)
    mx.eval(block_tables, seq_lens, cu_seqlens_q, out)

    ops.paged_attention_primitive(
        query,
        cache.key_caches[layer],
        cache.value_caches[layer],
        NUM_KV_HEADS,
        HEAD_SIZE**-0.5,
        0.0,
        block_tables,
        seq_lens,
        cu_seqlens_q,
        BLOCK_SIZE,
        kv_len,
        window,
        out,
        key_scale_cache=cache.key_scale_caches[layer],
        value_scale_cache=cache.value_scale_caches[layer],
        key_zero_cache=cache.key_zero_caches[layer],
        v_centroids=get_v_centroids(cache.v_bits),
        use_turboquant=True,
        quant_type=quant,
        v_bits=cache.v_bits,
    )
    mx.eval(out)
    o = np.array(out.astype(mx.float32))[0]
    assert np.isfinite(o).all()

    k_ref, v_ref = turbo_quant_decode(
        (k_packed, k_scale, k_zero),
        (v_packed, v_scale),
        output_dtype=mx.float16,
        key_quant_type=quant,
    )
    mx.eval(k_ref, v_ref)
    start = max(0, kv_len - window) if window > 0 else 0
    ref = _ref_attention_np(
        query,
        np.array(k_ref.astype(mx.float32))[start:kv_len],
        np.array(v_ref.astype(mx.float32))[start:kv_len],
    )
    np.testing.assert_allclose(o, ref, atol=1.5e-2, rtol=2e-2)
