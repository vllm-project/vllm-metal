# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import mlx.core as mx
import pytest

from vllm_metal.v1 import contiguous_cache


class TestHybridCacheMergeExtract:
    """Regression tests for hybrid (KV + ArraysCache) batching.

    Background:
    - `mlx-lm==0.30.6` removed `MambaCache` and hybrid models now use `ArraysCache`.
    - Older mlx-lm versions don't provide `ArraysCache.merge()` / `extract()`.

    These tests validate that vllm-metal can merge per-request caches into a batched
    cache, run a batched forward pass, and then extract per-request caches back,
    without depending on `MambaCache` or new mlx-lm APIs.
    """

    _ARRAYS_CACHE_ENTRIES = 2
    _ARRAYS_CACHE_FEATURES = 4

    _KV_NUM_HEADS = 1
    _KV_HEAD_DIM = 2

    def _make_arrays_cache(
        self, v0: float | None, v1: float | None
    ) -> contiguous_cache.ArraysCache:
        cache = contiguous_cache.ArraysCache(self._ARRAYS_CACHE_ENTRIES)
        if v0 is not None:
            cache[0] = mx.full((1, self._ARRAYS_CACHE_FEATURES), v0, dtype=mx.float32)
        if v1 is not None:
            cache[1] = mx.full((1, self._ARRAYS_CACHE_FEATURES), v1, dtype=mx.float32)
        return cache

    def _make_kv_cache(self, seq_len: int, value: float) -> contiguous_cache.KVCache:
        kv = contiguous_cache.KVCache()
        kv.keys = mx.full(
            (1, self._KV_NUM_HEADS, seq_len, self._KV_HEAD_DIM),
            value,
            dtype=mx.float32,
        )
        kv.values = mx.full(
            (1, self._KV_NUM_HEADS, seq_len, self._KV_HEAD_DIM),
            value + 0.5,
            dtype=mx.float32,
        )
        kv.offset = seq_len
        return kv

    def _make_rotating_kv_cache(
        self, *, max_size: int, total_tokens: int, value: float
    ) -> contiguous_cache.RotatingKVCache:
        cache = contiguous_cache.RotatingKVCache(max_size=max_size)
        keys = mx.full(
            (1, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM),
            value,
            dtype=mx.float32,
        )
        values = mx.full(
            (1, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM),
            value + 0.5,
            dtype=mx.float32,
        )
        for _ in range(total_tokens):
            cache.update_and_fetch(keys, values)
        return cache

    def test_arrays_cache_merge_extract_roundtrip(self) -> None:
        """Merging then extracting ArraysCache round-trips per request."""
        arrays_cache_req0 = self._make_arrays_cache(1.0, 11.0)
        arrays_cache_req1 = self._make_arrays_cache(2.0, 22.0)

        merged = contiguous_cache._merge_kv_caches(
            [[arrays_cache_req0], [arrays_cache_req1]]
        )
        extracted_req0 = contiguous_cache._extract_kv_cache(merged, 0)[0]
        extracted_req1 = contiguous_cache._extract_kv_cache(merged, 1)[0]

        assert isinstance(merged[0], contiguous_cache.ArraysCache)
        assert isinstance(extracted_req0, contiguous_cache.ArraysCache)
        assert isinstance(extracted_req1, contiguous_cache.ArraysCache)
        assert bool(mx.allclose(extracted_req0.state[0], arrays_cache_req0.state[0]))
        assert bool(mx.allclose(extracted_req0.state[1], arrays_cache_req0.state[1]))
        assert bool(mx.allclose(extracted_req1.state[0], arrays_cache_req1.state[0]))
        assert bool(mx.allclose(extracted_req1.state[1], arrays_cache_req1.state[1]))

    def test_arrays_cache_merge_extract_handles_missing_entries(self) -> None:
        """Missing per-request entries become zeros after merging.

        ArraysCache merging densifies per-entry state into a batch array when at
        least one request has that entry populated. Requests that had `None`
        for the entry are represented as zeros in the merged state.
        """
        arrays_cache_req0 = self._make_arrays_cache(1.0, 11.0)
        arrays_cache_req1 = self._make_arrays_cache(2.0, None)

        merged = contiguous_cache._merge_kv_caches(
            [[arrays_cache_req0], [arrays_cache_req1]]
        )

        extracted_req0 = contiguous_cache._extract_kv_cache(merged, 0)[0]
        extracted_req1 = contiguous_cache._extract_kv_cache(merged, 1)[0]

        assert isinstance(extracted_req0, contiguous_cache.ArraysCache)
        assert isinstance(extracted_req1, contiguous_cache.ArraysCache)

        assert bool(mx.allclose(extracted_req0.state[0], arrays_cache_req0.state[0]))
        assert bool(mx.allclose(extracted_req0.state[1], arrays_cache_req0.state[1]))
        assert bool(mx.allclose(extracted_req1.state[0], arrays_cache_req1.state[0]))

        missing = extracted_req1.state[1]
        assert missing is not None
        assert missing.shape == (1, self._ARRAYS_CACHE_FEATURES)
        assert bool(mx.allclose(missing, mx.zeros_like(missing)))

    def test_mixed_kv_and_arrays_cache_merge_extract_roundtrip(self) -> None:
        """Merging/extracting preserves both KVCache and ArraysCache layers."""
        kv_cache_req0 = self._make_kv_cache(seq_len=2, value=1.0)
        kv_cache_req1 = self._make_kv_cache(seq_len=4, value=2.0)
        arrays_cache_req0 = self._make_arrays_cache(3.0, 33.0)
        arrays_cache_req1 = self._make_arrays_cache(4.0, 44.0)

        merged = contiguous_cache._merge_kv_caches(
            [[kv_cache_req0, arrays_cache_req0], [kv_cache_req1, arrays_cache_req1]]
        )
        extracted_req0 = contiguous_cache._extract_kv_cache(merged, 0)
        extracted_req1 = contiguous_cache._extract_kv_cache(merged, 1)

        assert isinstance(merged[0], contiguous_cache.BatchKVCache)
        assert isinstance(merged[1], contiguous_cache.ArraysCache)

        kv_req0_out, arrays_req0_out = extracted_req0
        kv_req1_out, arrays_req1_out = extracted_req1

        assert isinstance(kv_req0_out, contiguous_cache.KVCache)
        assert isinstance(kv_req1_out, contiguous_cache.KVCache)
        assert isinstance(arrays_req0_out, contiguous_cache.ArraysCache)
        assert isinstance(arrays_req1_out, contiguous_cache.ArraysCache)

        assert kv_req0_out.offset == kv_cache_req0.offset
        assert kv_req1_out.offset == kv_cache_req1.offset
        assert bool(mx.allclose(kv_req0_out.keys, kv_cache_req0.keys))
        assert bool(mx.allclose(kv_req0_out.values, kv_cache_req0.values))
        assert bool(mx.allclose(kv_req1_out.keys, kv_cache_req1.keys))
        assert bool(mx.allclose(kv_req1_out.values, kv_cache_req1.values))

        assert bool(mx.allclose(arrays_req0_out.state[0], arrays_cache_req0.state[0]))
        assert bool(mx.allclose(arrays_req0_out.state[1], arrays_cache_req0.state[1]))
        assert bool(mx.allclose(arrays_req1_out.state[0], arrays_cache_req1.state[0]))
        assert bool(mx.allclose(arrays_req1_out.state[1], arrays_cache_req1.state[1]))

    def test_rotating_kvcache_merge_extract_preserves_offsets(self) -> None:
        cache_req0 = self._make_rotating_kv_cache(
            max_size=8, total_tokens=20, value=1.0
        )
        cache_req1 = self._make_rotating_kv_cache(max_size=8, total_tokens=5, value=2.0)

        merged = contiguous_cache._merge_kv_caches([[cache_req0], [cache_req1]])
        extracted_req0 = contiguous_cache._extract_kv_cache(merged, 0)[0]
        extracted_req1 = contiguous_cache._extract_kv_cache(merged, 1)[0]

        assert isinstance(merged[0], contiguous_cache.BatchRotatingKVCache)
        assert isinstance(extracted_req0, contiguous_cache.RotatingKVCache)
        assert isinstance(extracted_req1, contiguous_cache.RotatingKVCache)
        assert extracted_req0.offset == cache_req0.offset
        assert extracted_req1.offset == cache_req1.offset

    def test_rotating_kvcache_merge_handles_offset_exceeding_max_size(self) -> None:
        """Merging works when offset > max_size (cache has rotated).

        This is a regression test for a bug in ``BatchRotatingKVCache.merge``
        (mlx-lm <= 0.29.1) where using ``c.offset`` instead of ``len(c)`` caused
        a broadcast shape error after the cache rotated past its maximum size.
        """
        # offset=300 >> max_size=8 — the cache has rotated many times
        cache_req0 = self._make_rotating_kv_cache(
            max_size=8, total_tokens=300, value=1.0
        )
        cache_req1 = self._make_rotating_kv_cache(
            max_size=8, total_tokens=150, value=2.0
        )

        assert cache_req0.offset > cache_req0.max_size
        assert cache_req1.offset > cache_req1.max_size

        merged = contiguous_cache._merge_kv_caches([[cache_req0], [cache_req1]])
        extracted_req0 = contiguous_cache._extract_kv_cache(merged, 0)[0]
        extracted_req1 = contiguous_cache._extract_kv_cache(merged, 1)[0]

        assert isinstance(merged[0], contiguous_cache.BatchRotatingKVCache)
        assert isinstance(extracted_req0, contiguous_cache.RotatingKVCache)
        assert isinstance(extracted_req1, contiguous_cache.RotatingKVCache)
        assert extracted_req0.offset == cache_req0.offset
        assert extracted_req1.offset == cache_req1.offset

    def test_rotating_kvcache_merge_handles_prefill_exceeding_max_size(self) -> None:
        """Merging works when prefill length exceeds max_size.

        After a large prefill the internal buffer may temporarily be larger than
        ``max_size``.  The merge must trim to the effective sliding-window length.
        """
        # Prefill 128 tokens into a cache with max_size=70
        cache_req0 = contiguous_cache.RotatingKVCache(max_size=70)
        big_k = mx.full(
            (1, self._KV_NUM_HEADS, 128, self._KV_HEAD_DIM), 1.0, dtype=mx.float32
        )
        big_v = mx.full(
            (1, self._KV_NUM_HEADS, 128, self._KV_HEAD_DIM), 1.5, dtype=mx.float32
        )
        cache_req0.update_and_fetch(big_k, big_v)

        cache_req1 = self._make_rotating_kv_cache(
            max_size=70, total_tokens=30, value=2.0
        )

        merged = contiguous_cache._merge_kv_caches([[cache_req0], [cache_req1]])
        extracted_req0 = contiguous_cache._extract_kv_cache(merged, 0)[0]
        extracted_req1 = contiguous_cache._extract_kv_cache(merged, 1)[0]

        assert isinstance(merged[0], contiguous_cache.BatchRotatingKVCache)
        assert isinstance(extracted_req0, contiguous_cache.RotatingKVCache)
        assert isinstance(extracted_req1, contiguous_cache.RotatingKVCache)
        assert extracted_req0.offset == cache_req0.offset
        assert extracted_req1.offset == cache_req1.offset

    def test_rotating_kvcache_merge_decode_extract_roundtrip(self) -> None:
        """Merged cache can be used for a batched decode step and extracted back.

        This verifies that the internal state (_idx, _offset) set by
        ``_merge_rotating_kv_caches`` is compatible with
        ``BatchRotatingKVCache.update_and_fetch`` and ``extract``.
        """
        cache_req0 = self._make_rotating_kv_cache(
            max_size=8, total_tokens=20, value=1.0
        )
        cache_req1 = self._make_rotating_kv_cache(max_size=8, total_tokens=5, value=2.0)

        merged = contiguous_cache._merge_kv_caches([[cache_req0], [cache_req1]])
        batch_cache = merged[0]
        assert isinstance(batch_cache, contiguous_cache.BatchRotatingKVCache)

        # Simulate one batched decode step (S=1)
        decode_k = mx.ones((2, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM))
        decode_v = mx.ones((2, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM))
        batch_cache.update_and_fetch(decode_k, decode_v)

        # Extract back to per-request caches
        extracted_req0 = batch_cache.extract(0)
        extracted_req1 = batch_cache.extract(1)

        assert isinstance(extracted_req0, contiguous_cache.RotatingKVCache)
        assert isinstance(extracted_req1, contiguous_cache.RotatingKVCache)
        assert extracted_req0.offset == cache_req0.offset + 1
        assert extracted_req1.offset == cache_req1.offset + 1

    def test_extracted_rotating_cache_can_decode_after_rotation(self) -> None:
        """Extracted RotatingKVCache can continue decoding after offset > max_size.

        After merge -> extract, the extracted cache may have offset > max_size
        but keys.shape[2] < max_size (buffer sliced by extract).  Without the
        buffer padding fix in ``_extract_kv_cache``, the next
        ``_update_in_place`` call would compute a negative ``new_size`` and
        crash with ``ValueError: [full] Negative dimensions not allowed``.
        """
        cache_req0 = self._make_rotating_kv_cache(
            max_size=8, total_tokens=300, value=1.0
        )
        cache_req1 = self._make_rotating_kv_cache(
            max_size=8, total_tokens=150, value=2.0
        )

        merged = contiguous_cache._merge_kv_caches([[cache_req0], [cache_req1]])
        extracted_req0 = contiguous_cache._extract_kv_cache(merged, 0)[0]

        assert isinstance(extracted_req0, contiguous_cache.RotatingKVCache)
        assert extracted_req0.offset > extracted_req0.max_size
        assert extracted_req0.keys.shape[2] == extracted_req0.max_size

        # This would crash without the buffer padding fix
        decode_k = mx.ones(
            (1, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM), dtype=mx.float32
        )
        decode_v = mx.ones(
            (1, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM), dtype=mx.float32
        )
        extracted_req0.update_and_fetch(decode_k, decode_v)

        assert extracted_req0.offset == cache_req0.offset + 1

    def test_merge_kv_caches_rejects_mixed_cache_types_within_layer(self) -> None:
        arrays_cache = self._make_arrays_cache(1.0, 2.0)
        kv_cache = contiguous_cache.KVCache()
        with pytest.raises(TypeError, match="Mixed cache types in a single layer"):
            contiguous_cache._merge_kv_caches([[arrays_cache], [kv_cache]])
