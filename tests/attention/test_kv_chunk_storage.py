# SPDX-License-Identifier: Apache-2.0
"""Chunked KV pool (issue #1198): only written chunks become resident.

With ``VLLM_METAL_KV_CHUNK_MIB`` set, ``KVCacheStorage`` splits every layer
region into Metal buffers of that size and hands the attention kernels a table
of their GPU addresses instead of the cache itself. The contract pinned here:

- a chunk costs device memory only once a write lands in it, and is given
  back by ``reclaim_idle_chunks`` once no live request uses it;
- attention, block copies and block zeroing through the tables produce the
  same results as the plain single-buffer pool;
- consumers that need the pool bytes in one piece refuse to start.

Deterministic, no model load.
"""

from __future__ import annotations

import ctypes
import resource

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.metal import get_ops

BLOCK_SIZE = 16
KV_HEADS, HEAD_DIM, Q_HEADS = 2, 64, 4
LAYERS = ("l0", "l1", "l2")
CHUNK_MIB = 1
# fp16, K and V: one block of one layer is 2 * 16 * 2 * 64 * 2 bytes = 8 KiB,
# so a 1 MiB chunk holds 128 blocks and 300 blocks make three chunks per layer.
NUM_BLOCKS = 300
BLOCKS_PER_CHUNK = (CHUNK_MIB << 20) // (2 * BLOCK_SIZE * KV_HEADS * HEAD_DIM * 2)


def _resident_bytes(storage: KVCacheStorage) -> int:
    """Bytes of the pool's backing in physical memory (mincore)."""
    backing = next(iter(storage.tensors.values())).untyped_storage()
    page = resource.getpagesize()
    first = backing.data_ptr() - backing.data_ptr() % page
    pages = -(-(backing.data_ptr() + backing.nbytes() - first) // page)
    vec = (ctypes.c_char * pages)()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mincore(ctypes.c_void_p(first), ctypes.c_size_t(pages * page), vec):
        raise OSError(ctypes.get_errno(), "mincore failed")
    return sum(byte & 1 for byte in vec.raw) * page


def _config(num_blocks: int = NUM_BLOCKS) -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=KV_HEADS,
        head_size=HEAD_DIM,
        dtype=torch.float16,
    )
    page = spec.page_size_bytes
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=list(LAYERS), kv_cache_spec=spec)
        ],
        kv_cache_tensors=[
            KVCacheTensor(
                size=len(LAYERS) * num_blocks * page,
                layers=list(LAYERS),
                layer_stride=num_blocks * page,
                block_stride=page,
            )
        ],
        kv_cache_layout="LBNHC",
    )


@pytest.fixture
def chunked(monkeypatch) -> KVCacheStorage:
    monkeypatch.setenv("VLLM_METAL_KV_CHUNK_MIB", str(CHUNK_MIB))
    return KVCacheStorage(_config())


@pytest.fixture
def plain(monkeypatch) -> KVCacheStorage:
    monkeypatch.delenv("VLLM_METAL_KV_CHUNK_MIB", raising=False)
    return KVCacheStorage(_config())


def _write(storage, cache, layer, slots, key, value):
    """The write the attention layer makes, on either pool kind."""
    ops = get_ops()
    if storage.chunk_shift is None:
        k, v = ops.reshape_and_cache(
            key, value, cache.key_caches[layer], cache.value_caches[layer], slots
        )
    else:
        k, v = storage.write_chunked(
            key, value, slots, cache.key_caches[layer], BLOCK_SIZE
        )
    cache.replace_layer_cache(layer, k, v)


def _attend(storage, cache, layer, query, block_ids, ctx):
    out = get_ops().paged_attention_primitive(
        query,
        cache.key_caches[layer],
        cache.value_caches[layer],
        KV_HEADS,
        HEAD_DIM**-0.5,
        0.0,
        mx.array([block_ids], dtype=mx.int32),
        mx.array([ctx], dtype=mx.int32),
        mx.array([0, query.shape[0]], dtype=mx.int32),
        BLOCK_SIZE,
        ctx,
        -1,
        **storage.chunk_kwargs(),
    )
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def _fill(storage, block_ids, ctx, seed=0):
    """Write ``ctx`` random tokens into ``block_ids`` on every layer.

    Returns the cache and the decode/prefill queries used to read it back.
    """
    mx.random.seed(seed)
    cache = MetalPagedKVCache.from_upstream(storage, LAYERS)
    slots = mx.array(
        [block_ids[t // BLOCK_SIZE] * BLOCK_SIZE + t % BLOCK_SIZE for t in range(ctx)],
        dtype=mx.int64,
    )
    for layer in range(len(LAYERS)):
        key = mx.random.normal((ctx, KV_HEADS, HEAD_DIM)).astype(mx.float16)
        value = mx.random.normal((ctx, KV_HEADS, HEAD_DIM)).astype(mx.float16)
        _write(storage, cache, layer, slots, key, value)
    mx.eval(*storage.buffers)
    return cache


def test_chunked_pool_hands_kernels_address_tables(chunked):
    """Each region becomes a uint64 table with one entry per chunk."""
    chunks_per_layer = -(-NUM_BLOCKS // BLOCKS_PER_CHUNK)
    assert chunked.nbytes == _config().kv_cache_tensors[0].size
    assert len(chunked.buffers) == len(LAYERS)
    for table in chunked.buffers:
        assert table.dtype == mx.uint64
        assert table.shape == (chunks_per_layer,)
    assert chunked.chunk_kwargs() == {
        "kv_chunk_shift": BLOCKS_PER_CHUNK.bit_length() - 1,
        "kv_chunk_dtype": mx.float16,
        "kv_chunk_strides": [
            BLOCK_SIZE * KV_HEADS * 2 * HEAD_DIM,
            KV_HEADS * 2 * HEAD_DIM,
            2 * HEAD_DIM,
        ],
    }


def test_only_written_chunks_become_resident(chunked, plain):
    """Allocation backs nothing; a write backs exactly the chunks it lands in.

    The plain pool is one buffer per layer, and a dispatch makes every buffer
    it binds resident, so its first write commits the whole pool (#1198).
    """
    chunk_bytes = CHUNK_MIB << 20
    block_ids = [3, BLOCKS_PER_CHUNK + 5]  # chunks 0 and 1; chunk 2 never touched
    assert _resident_bytes(chunked) == 0
    _fill(chunked, block_ids, ctx=2 * BLOCK_SIZE)
    assert _resident_bytes(chunked) == 2 * len(LAYERS) * chunk_bytes

    _fill(plain, block_ids, ctx=2 * BLOCK_SIZE)
    assert _resident_bytes(plain) == plain.nbytes


@pytest.mark.parametrize("ctx", [20, 3 * BLOCK_SIZE])
def test_attention_through_tables_matches_the_plain_pool(chunked, plain, ctx):
    """Decode (one query row) and prefill (ctx rows) over blocks in two chunks."""
    block_ids = [7, 2 * BLOCKS_PER_CHUNK + 1, 11][: -(-ctx // BLOCK_SIZE)]
    caches = {
        name: _fill(storage, block_ids, ctx)
        for name, storage in (("chunked", chunked), ("plain", plain))
    }
    mx.random.seed(1)
    decode = mx.random.normal((1, Q_HEADS, HEAD_DIM)).astype(mx.float16)
    prefill = mx.random.normal((ctx, Q_HEADS, HEAD_DIM)).astype(mx.float16)
    for layer in range(len(LAYERS)):
        for query in (decode, prefill):
            outs = [
                _attend(storage, caches[name], layer, query, block_ids, ctx)
                for name, storage in (("chunked", chunked), ("plain", plain))
            ]
            np.testing.assert_array_equal(outs[0], outs[1])


def test_block_copies_and_zeroing_match_the_plain_pool(chunked, plain):
    """Scheduler copy-on-write and zeroing cross chunk boundaries correctly."""
    src, dst, zeroed = 7, 2 * BLOCKS_PER_CHUNK + 9, 42
    caches = {}
    for name, storage in (("chunked", chunked), ("plain", plain)):
        caches[name] = _fill(storage, [src, zeroed], ctx=2 * BLOCK_SIZE)
        storage.copy_blocks([(src, dst)])
        storage.zero_blocks([zeroed])
        mx.eval(*storage.buffers)
    query = mx.ones((1, Q_HEADS, HEAD_DIM)).astype(mx.float16)
    for layer in range(len(LAYERS)):
        copied, original = (
            _attend(chunked, caches["chunked"], layer, query, [block], BLOCK_SIZE)
            for block in (dst, src)
        )
        np.testing.assert_array_equal(copied, original)
        np.testing.assert_array_equal(
            copied, _attend(plain, caches["plain"], layer, query, [dst], BLOCK_SIZE)
        )
        # All-zero values attended over any weights give zero.
        assert not _attend(
            chunked, caches["chunked"], layer, query, [zeroed], BLOCK_SIZE
        ).any()


def test_idle_chunks_are_reclaimed_and_rewired_on_demand(chunked):
    """Memory goes back after the idle threshold and returns on the next write."""
    chunk_bytes = CHUNK_MIB << 20
    block_ids = [1, BLOCKS_PER_CHUNK + 1]
    _fill(chunked, block_ids, ctx=2 * BLOCK_SIZE)
    wired = _resident_bytes(chunked)
    assert wired >= 2 * len(LAYERS) * chunk_bytes

    # Chunk 0 stays live; chunk 1 idles past the threshold.
    freed = [chunked.reclaim_idle_chunks({0}, min_idle_steps=3) for _ in range(3)]
    assert freed == [0, 0, len(LAYERS) * chunk_bytes]
    assert _resident_bytes(chunked) <= wired - len(LAYERS) * chunk_bytes

    # The table learns the new chunk's address: a fresh write reads back.
    cache = _fill(chunked, block_ids, ctx=2 * BLOCK_SIZE, seed=5)
    query = mx.ones((1, Q_HEADS, HEAD_DIM)).astype(mx.float16)
    out = _attend(chunked, cache, 0, query, block_ids, 2 * BLOCK_SIZE)
    assert np.isfinite(out).all() and out.any()
    assert _resident_bytes(chunked) >= 2 * len(LAYERS) * chunk_bytes


def test_release_and_reallocate_keep_the_chunked_layout(chunked):
    """Sleep mode frees the chunks; wake rebuilds tables of the same shape."""
    shapes = [table.shape for table in chunked.buffers]
    chunked.release()
    assert chunked.released
    chunked.reallocate()
    assert [table.shape for table in chunked.buffers] == shapes
    assert chunked.chunk_kwargs()["kv_chunk_shift"] == BLOCKS_PER_CHUNK.bit_length() - 1


def test_consumers_that_need_whole_pool_bytes_refuse(chunked):
    with pytest.raises(NotImplementedError, match="kv chunk mode"):
        chunked.require_plain_pool("KV offloading")
    with pytest.raises(NotImplementedError, match="kv chunk mode"):
        chunked.scatter_rows([], [])


def test_plain_pool_has_no_chunk_arguments(plain):
    assert plain.chunk_shift is None
    assert plain.chunk_kwargs() == {}
    plain.require_plain_pool("anything")
