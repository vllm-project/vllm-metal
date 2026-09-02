# SPDX-License-Identifier: Apache-2.0
"""Merge/extract helpers for the non-paged KV cache path."""

from typing import TypeAlias

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)

# Minimum requests to use BatchKVCache for batched decode
_MIN_BATCH_SIZE_FOR_BATCHING = 2

# Per-layer cache types used by non-paged decode.  ``CacheList`` is mlx_lm's
# per-layer container for models whose layer owns several stateful modules,
# e.g. Falcon-H1's parallel Mamba-2 + attention layer, which ``make_cache``
# shapes as ``CacheList(ArraysCache(size=2), KVCache())``.
AnyCache: TypeAlias = KVCache | RotatingKVCache | ArraysCache | CacheList
BatchedCache: TypeAlias = BatchKVCache | BatchRotatingKVCache | ArraysCache | CacheList


def _merge_arrays_caches(caches: list[ArraysCache]) -> ArraysCache:
    """Merge ArraysCache while preserving entries that are all ``None``."""
    if not caches:
        raise ValueError("caches must be non-empty")

    num_entries = len(caches[0].state)
    batch_size = len(caches)

    merged = ArraysCache(num_entries)
    for entry_idx in range(num_entries):
        values = [cache.state[entry_idx] for cache in caches]
        template = next((value for value in values if value is not None), None)
        if template is None:
            continue

        shape = list(template.shape)
        shape[0] = batch_size
        merged_state = mx.zeros(tuple(shape), template.dtype)
        for batch_idx, value in enumerate(values):
            if value is None:
                continue
            merged_state[batch_idx : batch_idx + 1] = value

        merged[entry_idx] = merged_state

    return merged


def _extract_arrays_cache(batch_cache: ArraysCache, idx: int) -> ArraysCache:
    """Extract one request's ArraysCache, preserving all-``None`` entries."""
    state = batch_cache.state
    extracted = ArraysCache(len(state))
    extracted.state = [
        None if value is None else value[idx : idx + 1] for value in state
    ]
    return extracted


def _merge_layer_caches(layer_caches: list[AnyCache]) -> BatchedCache:
    """Merge one layer's per-request caches into a single batched cache."""
    first = layer_caches[0]
    if isinstance(first, CacheList):
        # Recurse member-wise so each nested cache takes its own merge path
        # (ArraysCache densification, BatchKVCache left padding, ...), then
        # rebuild the container the model indexes as ``cache[i]``.
        cache_lists: list[CacheList] = []
        for cache in layer_caches:
            if not isinstance(cache, CacheList):
                raise TypeError(
                    "Mixed cache types in a single layer: expected CacheList"
                )
            cache_lists.append(cache)
        width = len(first.caches)
        for cache_list in cache_lists:
            if len(cache_list.caches) != width:
                raise TypeError(
                    "CacheList width mismatch in a single layer: "
                    f"expected {width}, got {len(cache_list.caches)}"
                )
        return CacheList(
            *(
                _merge_layer_caches(
                    [cache_list.caches[i] for cache_list in cache_lists]
                )
                for i in range(width)
            )
        )
    if isinstance(first, ArraysCache):
        arrays_caches: list[ArraysCache] = []
        for cache in layer_caches:
            if not isinstance(cache, ArraysCache):
                raise TypeError(
                    "Mixed cache types in a single layer: expected ArraysCache"
                )
            arrays_caches.append(cache)
        return _merge_arrays_caches(arrays_caches)
    if isinstance(first, RotatingKVCache):
        rotating_caches: list[RotatingKVCache] = []
        for cache in layer_caches:
            if not isinstance(cache, RotatingKVCache):
                raise TypeError(
                    "Mixed cache types in a single layer: expected RotatingKVCache"
                )
            rotating_caches.append(cache)
        return BatchRotatingKVCache.merge(rotating_caches)
    if isinstance(first, KVCache):
        kv_caches: list[KVCache] = []
        for cache in layer_caches:
            if not isinstance(cache, KVCache):
                raise TypeError("Mixed cache types in a single layer: expected KVCache")
            kv_caches.append(cache)
        return BatchKVCache.merge(kv_caches)
    cache_type = type(first).__name__
    raise TypeError(f"Unsupported cache type for batching: {cache_type}")


def _merge_kv_caches(
    caches_list: list[list[AnyCache]],
) -> list[BatchedCache]:
    """Merge per-request layer caches into batched layer caches."""
    if not caches_list:
        return []

    num_layers = len(caches_list[0])
    return [
        _merge_layer_caches([caches[layer_idx] for caches in caches_list])
        for layer_idx in range(num_layers)
    ]


def _extract_layer_cache(cache: BatchedCache, idx: int) -> AnyCache:
    """Extract one request's cache for a single layer."""
    if isinstance(cache, CacheList):
        return CacheList(*(_extract_layer_cache(sub, idx) for sub in cache.caches))
    if isinstance(cache, ArraysCache):
        return _extract_arrays_cache(cache, idx)
    c = cache.extract(idx)
    # Pad sliced rotating buffers so later decode can update in place.
    if (
        isinstance(c, RotatingKVCache)
        and c.keys is not None
        and c.offset > c.max_size
        and c.keys.shape[2] < c.max_size
    ):
        pad = c.max_size - c.keys.shape[2]
        z_k = mx.zeros(
            (1, c.keys.shape[1], pad, c.keys.shape[3]),
            dtype=c.keys.dtype,
        )
        z_v = mx.zeros(
            (1, c.values.shape[1], pad, c.values.shape[3]),
            dtype=c.values.dtype,
        )
        c.keys = mx.concatenate([c.keys, z_k], axis=2)
        c.values = mx.concatenate([c.values, z_v], axis=2)
    return c


def _extract_kv_cache(batch_caches: list[BatchedCache], idx: int) -> list[AnyCache]:
    """Extract one request's layer caches from batched layer caches."""
    return [_extract_layer_cache(cache, idx) for cache in batch_caches]
