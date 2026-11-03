# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the KV offloading tests.

Caches are vLLM-allocated ``KVCacheStorage``, as the Metal runtime binds them.
The randomize/zero helpers write through the native scatter and assign the
result back through ``storage.pages``, the same idiom the worker and
``KVCacheStorage.copy_blocks`` use (see vllm_metal/attention/caches/storage.py).
"""

import mmap
import uuid

import mlx.core as mx
import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import record_kv_cache_layout
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_offload.base import GPULoadStoreSpec

from vllm_metal.attention.caches.placement import KV_CACHE_LAYOUT
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.metal import get_ops
from vllm_metal.v1.cache_policy import TurboQuantAttentionSpec
from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion
from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker


def make_storage(
    dtype: torch.dtype = torch.float16,
    *,
    num_layers: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 32,
    num_blocks: int = 8,
    block_size: int = 4,
    turboquant: bool = False,
    k_quant: str = "q8_0",
    v_quant: str = "q3_0",
) -> KVCacheStorage:
    """A single-group full-attention cache allocated by vLLM."""
    if turboquant:
        spec: FullAttentionSpec = TurboQuantAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            dtype=torch.int8,
            k_quant=k_quant,
            v_quant=v_quant,
        )
    else:
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            dtype=dtype,
        )
    names = [f"layers.{i}" for i in range(num_layers)]
    vllm_config = VllmConfig()
    record_kv_cache_layout(vllm_config.cache_config, KV_CACHE_LAYOUT)
    config = get_kv_cache_config_from_groups(
        vllm_config,
        [KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)],
        spec.page_size_bytes * num_blocks * num_layers,
    )
    config.kv_cache_layout = vllm_config.cache_config.kv_cache_layout
    assert config.num_blocks == num_blocks, (config.num_blocks, num_blocks)
    return KVCacheStorage(config)


def gpu_block_bytes(storage: KVCacheStorage) -> int:
    """Bytes of one GPU block summed across every page region."""
    return sum(int(page.shape[1]) for page in storage.pages)


def make_turboquant_storage(**kwargs) -> KVCacheStorage:
    return make_storage(head_dim=64, turboquant=True, **kwargs)


def chunk_bytes(storage: KVCacheStorage, blocks_per_chunk: int) -> int:
    """Bytes per chunk as the spec sizes it: rounded up to the page size."""
    raw = gpu_block_bytes(storage) * blocks_per_chunk
    return -(-raw // mmap.PAGESIZE) * mmap.PAGESIZE


def make_region(
    storage: KVCacheStorage,
    *,
    blocks_per_chunk: int = 1,
    num_cpu_chunks: int = 4,
    rank: int | None = 0,
    engine_id: str | None = None,
) -> MetalSharedOffloadRegion:
    """One attachment to a region sized for ``storage``.

    The same ``engine_id`` with rank 0 and rank None gives the worker-side and
    scheduler-side views of one region."""
    return MetalSharedOffloadRegion(
        engine_id=engine_id or f"test-{uuid.uuid4().hex[:8]}",
        num_chunks=num_cpu_chunks,
        rank=rank,
        kv_bytes_per_chunk=chunk_bytes(storage, blocks_per_chunk),
        cpu_page_size=gpu_block_bytes(storage) * blocks_per_chunk,
    )


def make_worker(
    storage: KVCacheStorage,
    *,
    blocks_per_chunk: int = 1,
    num_cpu_chunks: int = 4,
    kv_bytes_per_chunk: int | None = None,
    region: MetalSharedOffloadRegion | None = None,
) -> MetalKVOffloadWorker:
    """A worker over a fresh region, sized as the spec would size it."""
    if region is None:
        region = make_region(
            storage, blocks_per_chunk=blocks_per_chunk, num_cpu_chunks=num_cpu_chunks
        )
    if kv_bytes_per_chunk is None:
        kv_bytes_per_chunk = chunk_bytes(storage, blocks_per_chunk)
    return MetalKVOffloadWorker(
        storage,
        blocks_per_chunk=blocks_per_chunk,
        num_cpu_chunks=num_cpu_chunks,
        region=region,
        kv_bytes_per_chunk=kv_bytes_per_chunk,
    )


def _all_blocks(storage: KVCacheStorage) -> mx.array:
    return mx.arange(storage.config.num_blocks, dtype=mx.int32)


def randomize(storage: KVCacheStorage, seed: int = 0) -> None:
    """Fill every page with distinct bytes."""
    mx.random.seed(seed)
    ids = _all_blocks(storage)
    for i, page in enumerate(storage.pages):
        rows = mx.random.randint(0, 256, page.shape).astype(mx.uint8)
        storage.pages[i] = get_ops().gdn_state_scatter(page, rows, ids)
    mx.eval(*storage.buffers)


def snapshot_blocks(storage: KVCacheStorage, block_ids: list[int]) -> list[np.ndarray]:
    """Copy the given blocks of every page region to numpy."""
    ids = mx.array(block_ids, dtype=mx.int32)
    return [np.array(page[ids]) for page in storage.pages]


def zero_blocks(storage: KVCacheStorage, block_ids: list[int]) -> None:
    storage.zero_blocks(block_ids)
    mx.eval(*storage.buffers)


def gpu_spec(block_ids: list[int], block_index: int = 0) -> GPULoadStoreSpec:
    return GPULoadStoreSpec(
        block_ids, group_sizes=[len(block_ids)], block_indices=[block_index]
    )
