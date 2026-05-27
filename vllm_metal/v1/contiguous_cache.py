# SPDX-License-Identifier: Apache-2.0
"""Contiguous KV cache utilities (non-paged code path).

Batched KV cache merge/extract helpers for the contiguous KV cache path,
used when paged attention is disabled. KV state is stored as contiguous
mx.array tensors per request, in contrast to the fixed-block layout used
by paged attention.
"""

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

# Type alias for any per-layer cache type supported by the model.
#
# Notes:
# - Some models (e.g. gpt_oss) use `RotatingKVCache` for sliding-window attention.
# - Hybrid models use `ArraysCache` for non-attention state.
AnyCache: TypeAlias = KVCache | RotatingKVCache | ArraysCache


# ---------------------------------------------------------------------------
# Batched KV cache merge / extract helpers (legacy non-paged decode path)
# ---------------------------------------------------------------------------


def _merge_arrays_caches(caches: list[ArraysCache]) -> ArraysCache:
    """Merge per-request ArraysCache objects into a single batched ArraysCache.

    This mirrors the behavior of `mlx_lm.models.cache.ArraysCache.merge` but is
    implemented here for compatibility with older mlx-lm versions that do not
    provide `merge()` / `extract()`.
    """
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
    """Extract a single request's ArraysCache from a batched ArraysCache."""
    state = batch_cache.state
    extracted = ArraysCache(len(state))
    extracted.state = [
        None if value is None else value[idx : idx + 1] for value in state
    ]
    return extracted


def _merge_rotating_kv_caches(
    caches: list[RotatingKVCache],
) -> BatchRotatingKVCache:
    """Merge per-request RotatingKVCache objects into a single BatchRotatingKVCache.

    This mirrors ``BatchRotatingKVCache.merge`` but pre-computes the temporal-ordered
    keys/values, trims them to ``len(cache)`` (the effective sliding-window length),
    and uses that length for the copy width.  The upstream implementation in
    mlx-lm <= 0.29.1 uses ``c.offset`` which can exceed the underlying array size
    after the cache has rotated, causing a broadcast shape error.

    This workaround can be removed once vllm-metal can depend on an mlx-lm version
    that includes the upstream fix (ml-explore/mlx-lm#738) and has been verified
    to work with gpt-oss models end-to-end.
    """
    if not caches:
        raise ValueError("caches must be non-empty")

    if any(c.keys is None or c.values is None for c in caches):
        raise ValueError(
            "Cannot merge unpopulated RotatingKVCache (keys/values is None)"
        )

    if not all(c.max_size == caches[0].max_size for c in caches):
        raise ValueError(
            "BatchRotatingKVCache can only merge caches with the same maximum size"
        )

    # Pre-compute temporal-ordered keys/values and trim to the effective
    # sliding-window length.  ``_temporal_order`` may return an array larger
    # than ``len(cache)`` when the internal buffer has not been trimmed yet
    # (e.g. after a large prefill), so we trim via ``_trim`` to preserve
    # the ``keep`` prefix semantics used by RotatingKVCache internally.
    ordered: list[tuple[mx.array, mx.array]] = []
    for c in caches:
        effective_len = c.size() if hasattr(c, "size") else len(c)
        ordered_keys = c._temporal_order(c.keys)
        ordered_values = c._temporal_order(c.values)
        if ordered_keys.shape[2] > effective_len:
            trim_size = ordered_keys.shape[2] - effective_len
            ordered_keys = c._trim(trim_size, ordered_keys)
            ordered_values = c._trim(trim_size, ordered_values)
        else:
            ordered_keys = ordered_keys[..., :effective_len, :]
            ordered_values = ordered_values[..., :effective_len, :]
        ordered.append((ordered_keys, ordered_values))

    lengths = [k.shape[2] for k, _ in ordered]
    max_length = max(lengths)
    padding = [max_length - length for length in lengths]
    batch_size = len(caches)
    n_heads = max(k.shape[1] for k, _ in ordered)
    k_dim = max(k.shape[3] for k, _ in ordered)
    v_dim = max(v.shape[3] for _, v in ordered)
    dtype = next(iter(k.dtype for k, _ in ordered))

    keys = mx.zeros((batch_size, n_heads, max_length, k_dim), dtype=dtype)
    values = mx.zeros((batch_size, n_heads, max_length, v_dim), dtype=dtype)
    for i, (pad, (k, v)) in enumerate(zip(padding, ordered, strict=True)):
        n = k.shape[2]
        keys[i : i + 1, :, pad : pad + n] = k
        values[i : i + 1, :, pad : pad + n] = v

    cache = BatchRotatingKVCache(caches[0].max_size, padding)
    cache.keys = keys
    cache.values = values
    cache.offset = mx.array([c.offset for c in caches])
    cache._idx = keys.shape[2]
    cache._offset = keys.shape[2]

    return cache


def _merge_kv_caches(
    caches_list: list[list[AnyCache]],
) -> list[BatchKVCache | BatchRotatingKVCache | ArraysCache]:
    """Merge multiple per-request caches into batched caches.

    Args:
        caches_list: List of per-request caches, each is a list of per-layer caches

    Returns:
        List of batched caches, one per layer
    """
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
            batch_cache = _merge_rotating_kv_caches(rotating_caches)
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
    """Extract a single request's cache from batched caches.

    Args:
        batch_caches: List of batched caches, one per layer
        idx: Index of the request in the batch

    Returns:
        List of caches for the request, one per layer
    """
    extracted: list[AnyCache] = []
    for cache in batch_caches:
        if isinstance(cache, ArraysCache):
            extracted.append(_extract_arrays_cache(cache, idx))
        else:
            c = cache.extract(idx)
            # After extract, RotatingKVCache may have offset > max_size but
            # keys.shape[2] < max_size (buffer was sliced).  Pad the buffer
            # back to max_size so _update_in_place won't try to grow it
            # (which would compute a negative new_size).  The padded region
            # is dead space that will be overwritten on the next rotation.
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
