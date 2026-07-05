# SPDX-License-Identifier: Apache-2.0
"""Merge/extract helpers for the non-paged KV cache path."""

from typing import TypeAlias

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)

# Minimum requests to use BatchKVCache for batched decode
_MIN_BATCH_SIZE_FOR_BATCHING = 2

# Per-layer cache types used by non-paged decode.
AnyCache: TypeAlias = KVCache | RotatingKVCache | ArraysCache


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


def _merge_kv_caches(
    caches_list: list[list[AnyCache]],
) -> list[BatchKVCache | BatchRotatingKVCache | ArraysCache]:
    """Merge per-request layer caches into batched layer caches."""
    if not caches_list:
        return []

    num_layers = len(caches_list[0])
    merged: list[BatchKVCache | BatchRotatingKVCache | ArraysCache] = []

    for layer_idx in range(num_layers):
        layer_caches = [caches[layer_idx] for caches in caches_list]
        if isinstance(layer_caches[0], ArraysCache):
            arrays_caches: list[ArraysCache] = []
            for cache in layer_caches:
                if not isinstance(cache, ArraysCache):
                    raise TypeError(
                        "Mixed cache types in a single layer: expected ArraysCache"
                    )
                arrays_caches.append(cache)
            batch_cache = _merge_arrays_caches(arrays_caches)
        elif isinstance(layer_caches[0], RotatingKVCache):
            rotating_caches: list[RotatingKVCache] = []
            for cache in layer_caches:
                if not isinstance(cache, RotatingKVCache):
                    raise TypeError(
                        "Mixed cache types in a single layer: expected RotatingKVCache"
                    )
                rotating_caches.append(cache)
            batch_cache = BatchRotatingKVCache.merge(rotating_caches)
        elif isinstance(layer_caches[0], KVCache):
            kv_caches: list[KVCache] = []
            for cache in layer_caches:
                if not isinstance(cache, KVCache):
                    raise TypeError(
                        "Mixed cache types in a single layer: expected KVCache"
                    )
                kv_caches.append(cache)
            batch_cache = BatchKVCache.merge(kv_caches)
        else:
            cache_type = type(layer_caches[0]).__name__
            raise TypeError(f"Unsupported cache type for batching: {cache_type}")
        merged.append(batch_cache)

    return merged


def _extract_kv_cache(
    batch_caches: list[BatchKVCache | BatchRotatingKVCache | ArraysCache], idx: int
) -> list[AnyCache]:
    """Extract one request's layer caches from batched layer caches."""
    extracted: list[AnyCache] = []
    for cache in batch_caches:
        if isinstance(cache, ArraysCache):
            extracted.append(_extract_arrays_cache(cache, idx))
        else:
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
            extracted.append(c)
    return extracted
