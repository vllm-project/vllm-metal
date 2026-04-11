# SPDX-License-Identifier: Apache-2.0
"""Contiguous KV cache utilities (non-paged code path).

Prefix caching (PrefixCacheManager) and batched KV cache merge/extract
helpers for the contiguous KV cache path, used when paged attention is
disabled. KV state is stored as contiguous mx.array tensors per request,
in contrast to the fixed-block layout used by paged attention.
"""

import hashlib
import math
from array import array
from dataclasses import dataclass
from typing import Any, TypeAlias

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
    make_prompt_cache,
)
from vllm.logger import init_logger

import vllm_metal.envs as envs

logger = init_logger(__name__)

# Minimum requests to use BatchKVCache for batched decode
_MIN_BATCH_SIZE_FOR_BATCHING = 2

# Type alias for any per-layer cache type supported by the model.
#
# Notes:
# - Some models (e.g. gpt_oss) use `RotatingKVCache` for sliding-window attention.
# - Hybrid models use `ArraysCache` for non-attention state.
AnyCache: TypeAlias = KVCache | RotatingKVCache | ArraysCache


# ---------------------------------------------------------------------------
# Prefix cache configuration — enabled by setting VLLM_METAL_PREFIX_CACHE
# in the environment (any value; unset to disable).
# ---------------------------------------------------------------------------


def _prefix_cache_enabled() -> bool:
    """Check whether prefix caching is enabled via environment variable."""
    return envs.VLLM_METAL_PREFIX_CACHE


_PREFIX_CACHE_ENABLED = _prefix_cache_enabled()
_PREFIX_CACHE_DEFAULT_FRACTION = 0.05  # 5% of MLX working set


def _get_prefix_cache_max_bytes() -> int:
    """Get prefix cache memory limit based on MLX recommended working set."""
    fraction_str = envs.VLLM_METAL_PREFIX_CACHE_FRACTION
    if fraction_str:
        try:
            fraction = float(fraction_str)
            if not math.isfinite(fraction) or fraction <= 0 or fraction > 1:
                logger.warning(
                    "VLLM_METAL_PREFIX_CACHE_FRACTION=%r out of range (0, 1], "
                    "using default %.2f",
                    fraction_str,
                    _PREFIX_CACHE_DEFAULT_FRACTION,
                )
                fraction = _PREFIX_CACHE_DEFAULT_FRACTION
        except ValueError:
            logger.warning(
                "Invalid VLLM_METAL_PREFIX_CACHE_FRACTION=%r, using default %.2f",
                fraction_str,
                _PREFIX_CACHE_DEFAULT_FRACTION,
            )
            fraction = _PREFIX_CACHE_DEFAULT_FRACTION
    else:
        fraction = _PREFIX_CACHE_DEFAULT_FRACTION

    fallback_bytes = 8 * 1024 * 1024 * 1024  # 8 GB
    try:
        device_info = mx.device_info()
        total = int(device_info.get("max_recommended_working_set_size", 0))
    except (AttributeError, RuntimeError):
        total = 0

    if total == 0:
        total = fallback_bytes
        logger.warning("Could not get MLX working set size, using 8GB fallback")

    max_bytes = int(total * fraction)
    logger.info(
        "Prefix cache: %.1fGB limit (%.1f%% of %.1fGB MLX working set)",
        max_bytes / (1024 * 1024 * 1024),
        fraction * 100,
        total / (1024 * 1024 * 1024),
    )
    return max_bytes


def _compute_prefix_hash(token_ids: list[int]) -> bytes:
    """Compute content hash for a token sequence."""
    h = hashlib.sha256()
    h.update(array("I", token_ids).tobytes())
    return h.digest()


def _compute_entry_bytes(cache_state: list[tuple[mx.array, mx.array] | None]) -> int:
    """Compute memory usage of a cache entry in bytes."""
    total = 0
    for pair in cache_state:
        if pair is not None:
            total += pair[0].nbytes + pair[1].nbytes
    return total


@dataclass
class CachedPrefix:
    """Cached KV state for a token prefix.

    cache_state contains (k, v) tuples for KVCache layers, or None for
    ArraysCache layers in hybrid models.
    """

    token_ids: list[int]
    cache_state: list[tuple[mx.array, mx.array] | None]
    size_bytes: int = 0
    ref_count: int = 0


class PrefixCacheManager:
    """Manager for prefix KV cache reuse with memory-based eviction."""

    def __init__(self, max_bytes: int | None = None):
        self._cache: dict[bytes, CachedPrefix] = {}
        self._max_bytes = (
            max_bytes if max_bytes is not None else _get_prefix_cache_max_bytes()
        )
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0

    def lookup(self, token_ids: list[int]) -> CachedPrefix | None:
        """Look up cached prefix by token IDs."""
        prefix_hash = _compute_prefix_hash(token_ids)
        cached = self._cache.get(prefix_hash)
        if cached is not None:
            self._hits += 1
            cached.ref_count += 1
            logger.debug(
                "Prefix cache HIT: %d hits, %d misses, rate=%.1f%%",
                self._hits,
                self._misses,
                self.hit_rate * 100,
            )
            return cached
        self._misses += 1
        logger.debug(
            "Prefix cache MISS: %d hits, %d misses, rate=%.1f%%",
            self._hits,
            self._misses,
            self.hit_rate * 100,
        )
        return None

    def _evict_until_fits(self, needed_bytes: int) -> None:
        """Evict entries until we have room for needed_bytes."""
        while self._current_bytes + needed_bytes > self._max_bytes and self._cache:
            min_hash, min_entry = min(self._cache.items(), key=lambda x: x[1].ref_count)
            self._current_bytes -= min_entry.size_bytes
            del self._cache[min_hash]
            logger.debug(
                "Prefix cache eviction: freed %.1fMB",
                min_entry.size_bytes / (1024 * 1024),
            )

    def insert(self, token_ids: list[int], cache: list[KVCache]) -> None:
        """Insert a prefix cache entry with memory-based eviction.

        Only KVCache layers are cached. ArraysCache layers are skipped (stored as
        None) for hybrid model compatibility.
        """
        prefix_hash = _compute_prefix_hash(token_ids)
        if prefix_hash in self._cache:
            return

        cache_state = []
        for layer_cache in cache:
            if isinstance(layer_cache, KVCache):
                k = layer_cache.state[0]
                v = layer_cache.state[1]
                cache_state.append((mx.array(k), mx.array(v)))
            else:
                cache_state.append(None)

        entry_bytes = _compute_entry_bytes(cache_state)

        # Skip if single entry exceeds memory limit
        if entry_bytes > self._max_bytes:
            logger.debug(
                "Prefix cache skip: entry %.1fMB exceeds limit %.1fGB",
                entry_bytes / (1024 * 1024),
                self._max_bytes / (1024 * 1024 * 1024),
            )
            return

        self._evict_until_fits(entry_bytes)

        self._cache[prefix_hash] = CachedPrefix(
            token_ids=list(token_ids),
            cache_state=cache_state,
            size_bytes=entry_bytes,
            ref_count=1,
        )
        self._current_bytes += entry_bytes

    def restore_cache(
        self, cached: CachedPrefix, model: Any, is_vlm: bool
    ) -> list[AnyCache]:
        """Restore a cached prefix to a fresh KVCache.

        Only KVCache layers are restored. RotatingKVCache / ArraysCache layers
        remain in their fresh state.
        """
        cache_model = (
            model.language_model
            if is_vlm and hasattr(model, "language_model")
            else model
        )
        cache = make_prompt_cache(cache_model)
        for i, layer_cache in enumerate(cache):
            if i < len(cached.cache_state) and cached.cache_state[i] is not None:
                if isinstance(layer_cache, KVCache):
                    k, v = cached.cache_state[i]
                    layer_cache.state = [mx.array(k), mx.array(v)]
                    # Keep RoPE position correct even if KVCache.state setter
                    # behavior changes in future mlx-lm versions.
                    layer_cache.offset = int(k.shape[2])
        return cache

    @property
    def hit_rate(self) -> float:
        """Return prefix cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Return prefix cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "cached_entries": len(self._cache),
            "current_bytes": self._current_bytes,
            "max_bytes": self._max_bytes,
        }


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
