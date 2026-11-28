# SPDX-License-Identifier: Apache-2.0
"""Uninitialized KV backing: layout parity with vLLM, and slot masking.

``KVCacheStorage`` allocates its backing store uninitialized whenever
vLLM says the pool does not have to be zeroed (uniform-precision attention
caches; ``KVCacheConfig.needs_kv_cache_zeroing`` is False). On unified memory
zero-filling a multi-GB pool commits every page, so a serving run that touches
a fraction of its blocks pays RSS for all of them.

Two properties have to hold for that to be safe, and both are pinned here:

- the allocation must be byte-for-byte the layout
  ``vllm.v1.worker.utils.allocate_kv_cache`` produces, or block indexing drifts;
- the attention kernels must never let a slot past the sequence length reach an
  output, so an unwritten slot may hold anything at all (including NaN, which a
  stale or recycled page happily contains).

Deterministic, no model load.
"""

from __future__ import annotations

import ctypes
import resource

import mlx.core as mx
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_metal.attention.caches import storage as storage_module
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.metal import get_ops

BLOCK_SIZE = 16


def _attention_config(num_blocks: int = 4, layer_names=("k0", "k1")) -> KVCacheConfig:
    """A uniform-precision (attention-only) config: zeroing is not required."""
    attention = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_size=32,
        dtype=torch.float16,
    )
    page = attention.page_size_bytes
    group = KVCacheGroupSpec(layer_names=list(layer_names), kv_cache_spec=attention)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=[group],
        kv_cache_tensors=[
            KVCacheTensor(
                size=len(layer_names) * num_blocks * page,
                layers=list(layer_names),
                layer_stride=num_blocks * page,
                block_stride=page,
            )
        ],
        kv_cache_layout="LBNHC",
    )


def _hybrid_config(num_blocks: int = 4) -> KVCacheConfig:
    """The Mamba-bearing config from test_shared_cache_storage: zeroing required."""
    attention = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=32, dtype=torch.float16
    )
    state = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((2, 4), (1, 4, 32)),
        dtypes=(torch.float16, torch.float32),
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode="align",
    )
    page = attention.page_size_bytes
    groups = [
        KVCacheGroupSpec(layer_names=["a0", "a1"], kv_cache_spec=attention),
        KVCacheGroupSpec(layer_names=["s0", "s1"], kv_cache_spec=state),
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * num_blocks * page,
                layers=group.layer_names,
                layer_stride=num_blocks * page,
                block_stride=page,
            )
            for group in groups
        ],
        kv_cache_layout="LBNHC",
    )


def test_attention_only_config_is_allocated_uninitialized():
    """The lazy path applies exactly where vLLM says zeroing is not needed."""
    config = _attention_config()
    assert not config.needs_kv_cache_zeroing
    storage = KVCacheStorage(config)
    assert (
        storage.nbytes
        == next(iter(storage.tensors.values())).untyped_storage().nbytes()
    )


def test_no_config_goes_through_vllms_zero_filling_allocator(monkeypatch):
    """A revert of the lazy path must fail here, not merely cost memory.

    The layout assertions below hold for both allocations, so they cannot tell
    the two apart. This pins the contract itself: every cache delegates to
    vLLM's allocator with its own ``allocate`` callback. A Mamba cache still
    reads as zeros, but the zeros come from the OS instead of a fill.
    """
    allocators: list[object] = []
    original = storage_module.allocate_kv_cache

    def spy(config, *args, **kwargs):
        allocators.append(kwargs.get("allocate"))
        return original(config, *args, **kwargs)

    monkeypatch.setattr(storage_module, "allocate_kv_cache", spy)

    KVCacheStorage(_attention_config())
    assert len(allocators) == 1
    assert allocators[0] is not None

    KVCacheStorage(_hybrid_config())
    assert len(allocators) == 2
    assert allocators[1] is not None


def test_uninitialized_backing_matches_upstream_layout():
    """Byte-for-byte the layout ``allocate_kv_cache`` would have produced."""
    config = _attention_config(num_blocks=6)
    storage = KVCacheStorage(config)
    reference = allocate_kv_cache(
        config,
        torch.device("cpu"),
        KVCacheLayout[config.kv_cache_layout],
    )
    assert set(storage.tensors) == set(reference)
    assert storage.nbytes == next(iter(reference.values())).untyped_storage().nbytes()
    base = min(tensor.storage_offset() for tensor in reference.values())
    for name, expected in reference.items():
        actual = storage.tensors[name]
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert actual.dtype == expected.dtype
        # Same placement inside the shared backing, which is what makes a
        # block id address the same bytes on both allocations.
        assert actual.storage_offset() - base == expected.storage_offset() - base


def test_mamba_config_keeps_vllms_zero_filled_allocation():
    """State is read before it is written, so that path still gets zeros."""
    config = _hybrid_config()
    assert config.needs_kv_cache_zeroing
    storage = KVCacheStorage(config)
    assert not storage.tensors["s0"].any()
    assert not storage.tensors["s1"].any()


def _resident_bytes(tensor: torch.Tensor) -> int:
    """Bytes of the tensor's storage that are in physical memory (mincore)."""
    page = resource.getpagesize()
    start = tensor.untyped_storage().data_ptr()
    first = start - start % page
    pages = -(-(start + tensor.untyped_storage().nbytes() - first) // page)
    vec = (ctypes.c_char * pages)()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mincore(ctypes.c_void_p(first), ctypes.c_size_t(pages * page), vec):
        raise OSError(ctypes.get_errno(), "mincore failed")
    return sum(byte & 1 for byte in vec.raw) * page


def test_pool_is_not_resident_under_the_workers_wired_limit():
    """The worker sets an MLX wired limit, and MLX wires the buffers it adopts
    from torch while one is set: the whole pool used to be resident before the
    first request (#1198). Only what the GPU actually uses may be committed."""
    # Large enough that the allocator hands out fresh, never-touched pages.
    pool_bytes = 256 << 20
    layers = 2
    num_blocks = pool_bytes // (
        layers * _attention_config().kv_cache_tensors[0].block_stride
    )
    previous = mx.set_wired_limit(
        int(mx.device_info()["max_recommended_working_set_size"])
    )
    try:
        storage = KVCacheStorage(_attention_config(num_blocks=num_blocks))
        mx.eval(mx.ones(16) + 1)  # an unrelated command buffer
        backing = next(iter(storage.tensors.values()))
        assert storage.nbytes == pool_bytes
        assert _resident_bytes(backing) == 0
    finally:
        mx.set_wired_limit(previous)


# A Mamba-bearing pool of 128 MiB. Attention and state layers share one
# allocation: 2 layers x 2048-byte pages.
_HYBRID_POOL_BYTES = 128 << 20
_HYBRID_POOL_BLOCKS = _HYBRID_POOL_BYTES // (2 * 2048)


def _partial_pages_bytes() -> int:
    """Zeroing writes at most the two partial pages at the ends of the pool."""
    return 2 * resource.getpagesize()


def test_zero_filled_pools_read_zero_without_being_resident():
    """State is read before it is written, so the pool must read as zeros, but
    a fill would commit all of it before the first request."""
    storage = KVCacheStorage(_hybrid_config(num_blocks=_HYBRID_POOL_BLOCKS))
    backing = next(iter(storage.tensors.values()))
    assert storage.config.needs_kv_cache_zeroing
    assert storage.nbytes == _HYBRID_POOL_BYTES
    assert _resident_bytes(backing) <= _partial_pages_bytes()
    raw = torch.empty(0, dtype=torch.uint8).set_(backing.untyped_storage())
    assert not raw.any()


def test_wake_zeroes_the_pool_without_committing_it():
    storage = KVCacheStorage(_hybrid_config(num_blocks=_HYBRID_POOL_BLOCKS))
    backing = next(iter(storage.tensors.values()))
    raw = torch.empty(0, dtype=torch.uint8).set_(backing.untyped_storage())
    raw.fill_(7)

    storage.release()
    storage.reallocate()

    assert _resident_bytes(backing) <= _partial_pages_bytes()
    assert not raw.any()


def _run_decode(ctx: int, tail_is_nan: bool) -> mx.array:
    """One paged-attention decode row with the block tail filled how we ask."""
    num_q_heads, num_kv_heads, head_size = 8, 2, 128
    blocks = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    mx.random.seed(0)
    key_cache = mx.random.normal(
        shape=(blocks, BLOCK_SIZE, num_kv_heads, head_size), dtype=mx.float32
    )
    value_cache = mx.random.normal(
        shape=(blocks, BLOCK_SIZE, num_kv_heads, head_size), dtype=mx.float32
    )
    query = mx.random.normal(shape=(1, num_q_heads, head_size), dtype=mx.float32)
    if tail_is_nan:
        # Slots [ctx, blocks * BLOCK_SIZE) are never written by the runtime.
        # MLX hands back a *new* array from reshape, so an in-place write to the
        # flattened view stays in that temporary and never reaches the buffers
        # the kernel reads: carry it back explicitly, then assert the premise so
        # this arm cannot silently stop testing anything.
        filled = []
        for cache in (key_cache, value_cache):
            flat = cache.reshape(blocks * BLOCK_SIZE, num_kv_heads, head_size)
            flat[ctx:] = float("nan")
            filled.append(flat.reshape(cache.shape))
        key_cache, value_cache = filled
    mx.eval(key_cache, value_cache, query)
    if tail_is_nan:
        for name, cache in (("key", key_cache), ("value", value_cache)):
            flat = cache.reshape(blocks * BLOCK_SIZE, num_kv_heads, head_size)
            assert bool(mx.all(mx.isnan(flat[ctx:]))), f"{name} NaN tail missing"
            assert bool(mx.all(mx.isfinite(flat[:ctx]))), f"{name} written rows changed"

    out = get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        head_size**-0.5,
        0.0,
        mx.array([list(range(blocks))], dtype=mx.int32),
        mx.array([ctx], dtype=mx.int32),
        mx.array([0, 1], dtype=mx.int32),
        BLOCK_SIZE,
        ctx,
        -1,
    )
    mx.eval(out)
    return out.reshape(num_q_heads, head_size)


@pytest.mark.parametrize("ctx", [20, 33, 17])
def test_unwritten_slots_never_reach_the_output(ctx: int):
    """A NaN-filled block tail must not change a decode result.

    ``ctx`` is deliberately never a multiple of ``BLOCK_SIZE``, so the last
    block is only partially written — the case the pool-level zero fill used to
    cover.
    """
    zeroed = _run_decode(ctx, tail_is_nan=False)
    nan_filled = _run_decode(ctx, tail_is_nan=True)
    assert bool(mx.all(mx.isfinite(nan_filled)))
    assert bool(mx.array_equal(zeroed, nan_filled))
