# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the KV offloading tests.

The snapshot/zero helpers mirror the cache kernels' write-then-rebind idiom
(see vllm_metal/attention/caches/kv_cache.py): mutate through the current
list entry, then rebind it, so later readers get provenance.
"""

import mmap
import uuid

import mlx.core as mx
import numpy as np
from vllm.v1.kv_offload.base import GPULoadStoreSpec

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion
from vllm_metal.v1.kv_offload.worker import CACHE_ATTRS, MetalKVOffloadWorker


def make_cache(dtype: mx.Dtype = mx.float16, **kwargs) -> MetalPagedKVCache:
    kwargs.setdefault("num_layers", 2)
    kwargs.setdefault("num_kv_heads", 2)
    kwargs.setdefault("head_dim", 32)
    kwargs.setdefault("num_blocks", 8)
    kwargs.setdefault("block_size", 4)
    return MetalPagedKVCache(dtype=dtype, **kwargs)


def gpu_block_bytes(cache: MetalPagedKVCache) -> int:
    """Bytes of one GPU block summed across every cache array."""
    total = 0
    for attr in CACHE_ATTRS:
        for arr in getattr(cache, attr):
            total += int(np.prod(arr.shape[1:])) * arr.dtype.size
    return total


def make_region(
    cache: MetalPagedKVCache,
    *,
    block_size_factor: int,
    num_cpu_blocks: int,
    rank: int | None = 0,
    engine_id: str | None = None,
) -> MetalSharedOffloadRegion:
    """One attachment to a shared region sized for ``cache``.

    Rows are rounded up to the page size, as upstream's SharedOffloadRegion
    does. Pass the same ``engine_id`` twice (rank 0 and rank None) to get
    the worker-side and scheduler-side views of one region."""
    raw_row = gpu_block_bytes(cache) * block_size_factor
    aligned_row = -(-raw_row // mmap.PAGESIZE) * mmap.PAGESIZE
    return MetalSharedOffloadRegion(
        engine_id=engine_id or f"test-{uuid.uuid4().hex[:8]}",
        num_blocks=num_cpu_blocks,
        rank=rank,
        kv_bytes_per_block=aligned_row,
        cpu_page_size=raw_row,
    )


def make_worker(
    cache: MetalPagedKVCache,
    *,
    block_size_factor: int = 1,
    num_cpu_blocks: int = 4,
    **kwargs,
) -> MetalKVOffloadWorker:
    """A worker over a fresh region; ``kwargs`` pass through."""
    region = make_region(
        cache, block_size_factor=block_size_factor, num_cpu_blocks=num_cpu_blocks
    )
    return MetalKVOffloadWorker(
        cache,
        block_size_factor=block_size_factor,
        num_cpu_blocks=num_cpu_blocks,
        region=region,
        **kwargs,
    )


def randomize(cache: MetalPagedKVCache, seed: int = 0) -> None:
    """Fill every cache array with distinct values."""
    mx.random.seed(seed)
    for attr in CACHE_ATTRS:
        arrays = getattr(cache, attr)
        for i, arr in enumerate(arrays):
            if arr.dtype in (mx.float16, mx.bfloat16, mx.float32):
                arrays[i] = mx.random.normal(arr.shape).astype(arr.dtype)
            else:
                arrays[i] = mx.random.randint(0, 255, arr.shape).astype(arr.dtype)
        mx.eval(*arrays)


def snapshot_blocks(cache: MetalPagedKVCache, block_ids: list[int]) -> list[np.ndarray]:
    """Copy the given blocks of every cache array to numpy (bf16 as uint16)."""
    out = []
    for attr in CACHE_ATTRS:
        for arr in getattr(cache, attr):
            sl = mx.take(arr, mx.array(block_ids), axis=0)
            if sl.dtype == mx.bfloat16:
                sl = mx.view(sl, mx.uint16)
            out.append(np.array(sl))
    return out


def zero_blocks(cache: MetalPagedKVCache, block_ids: list[int]) -> None:
    for attr in CACHE_ATTRS:
        arrays = getattr(cache, attr)
        for i, arr in enumerate(arrays):
            arr[mx.array(block_ids)] = mx.zeros(
                (len(block_ids), *arr.shape[1:]), dtype=arr.dtype
            )
            arrays[i] = arr
            mx.eval(arr)


def gpu_spec(block_ids: list[int], block_index: int = 0) -> GPULoadStoreSpec:
    return GPULoadStoreSpec(
        block_ids, group_sizes=[len(block_ids)], block_indices=[block_index]
    )
