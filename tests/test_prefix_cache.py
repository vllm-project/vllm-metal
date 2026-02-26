# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

import vllm_metal.v1.model_runner as mr


class StubArraysCache:
    @property
    def state(self):
        return []


class TestPrefixCacheHybridGuard:
    def _make_runner(self) -> mr.MetalModelRunner:
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.model = MagicMock()
        runner._is_vlm = False
        runner._prefix_cache = mr.PrefixCacheManager(max_bytes=1024 * 1024)
        return runner

    def test_hybrid_model_skips_prefix_cache(self, monkeypatch) -> None:
        lookup_spy = MagicMock(return_value=None)
        insert_spy = MagicMock()

        def fake_make_prompt_cache(model):
            kv = mr.KVCache()
            kv.keys = mx.zeros((1, 8, 0, 64))
            kv.values = mx.zeros((1, 8, 0, 64))
            kv.offset = 0
            return [kv, StubArraysCache(), kv]

        monkeypatch.setattr(mr, "make_prompt_cache", fake_make_prompt_cache)

        runner = self._make_runner()
        monkeypatch.setattr(runner._prefix_cache, "lookup", lookup_spy)
        monkeypatch.setattr(runner._prefix_cache, "insert", insert_spy)

        fake_logits = mx.zeros((1, 5, 100))
        runner.model.return_value = MagicMock(logits=fake_logits)

        token_ids = [1, 2, 3, 4, 5]
        sampling_params = MagicMock(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            frequency_penalty=0,
            presence_penalty=0,
            repetition_penalty=1.0,
        )

        runner._prefill_single("req-1", token_ids, sampling_params)

        lookup_spy.assert_not_called()
        insert_spy.assert_not_called()

    def test_pure_kvcache_uses_prefix_cache(self, monkeypatch) -> None:
        lookup_spy = MagicMock(return_value=None)
        insert_spy = MagicMock()

        def fake_make_prompt_cache(model):
            kv = mr.KVCache()
            kv.state = [mx.zeros((1, 4, 8, 64)), mx.zeros((1, 4, 8, 64))]
            return [kv, kv]

        monkeypatch.setattr(mr, "make_prompt_cache", fake_make_prompt_cache)

        runner = self._make_runner()
        monkeypatch.setattr(runner._prefix_cache, "lookup", lookup_spy)
        monkeypatch.setattr(runner._prefix_cache, "insert", insert_spy)

        fake_logits = mx.zeros((1, 1, 100))
        runner.model.return_value = MagicMock(logits=fake_logits)

        token_ids = [1, 2, 3, 4, 5]
        sampling_params = MagicMock(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            frequency_penalty=0,
            presence_penalty=0,
            repetition_penalty=1.0,
        )

        runner._prefill_single("req-1", token_ids, sampling_params)

        lookup_spy.assert_called_once_with([1, 2, 3, 4])
        insert_spy.assert_called_once()


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

    def _make_arrays_cache(self, v0: float | None, v1: float | None) -> mr.ArraysCache:
        cache = mr.ArraysCache(self._ARRAYS_CACHE_ENTRIES)
        if v0 is not None:
            cache[0] = mx.full((1, self._ARRAYS_CACHE_FEATURES), v0, dtype=mx.float32)
        if v1 is not None:
            cache[1] = mx.full((1, self._ARRAYS_CACHE_FEATURES), v1, dtype=mx.float32)
        return cache

    def _make_kv_cache(self, seq_len: int, value: float) -> mr.KVCache:
        kv = mr.KVCache()
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
    ) -> mr.RotatingKVCache:
        cache = mr.RotatingKVCache(max_size=max_size)
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

        merged = mr._merge_kv_caches([[arrays_cache_req0], [arrays_cache_req1]])
        extracted_req0 = mr._extract_kv_cache(merged, 0)[0]
        extracted_req1 = mr._extract_kv_cache(merged, 1)[0]

        assert isinstance(merged[0], mr.ArraysCache)
        assert isinstance(extracted_req0, mr.ArraysCache)
        assert isinstance(extracted_req1, mr.ArraysCache)
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

        merged = mr._merge_kv_caches([[arrays_cache_req0], [arrays_cache_req1]])

        extracted_req0 = mr._extract_kv_cache(merged, 0)[0]
        extracted_req1 = mr._extract_kv_cache(merged, 1)[0]

        assert isinstance(extracted_req0, mr.ArraysCache)
        assert isinstance(extracted_req1, mr.ArraysCache)

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

        merged = mr._merge_kv_caches(
            [[kv_cache_req0, arrays_cache_req0], [kv_cache_req1, arrays_cache_req1]]
        )
        extracted_req0 = mr._extract_kv_cache(merged, 0)
        extracted_req1 = mr._extract_kv_cache(merged, 1)

        assert isinstance(merged[0], mr.BatchKVCache)
        assert isinstance(merged[1], mr.ArraysCache)

        kv_req0_out, arrays_req0_out = extracted_req0
        kv_req1_out, arrays_req1_out = extracted_req1

        assert isinstance(kv_req0_out, mr.KVCache)
        assert isinstance(kv_req1_out, mr.KVCache)
        assert isinstance(arrays_req0_out, mr.ArraysCache)
        assert isinstance(arrays_req1_out, mr.ArraysCache)

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

        merged = mr._merge_kv_caches([[cache_req0], [cache_req1]])
        extracted_req0 = mr._extract_kv_cache(merged, 0)[0]
        extracted_req1 = mr._extract_kv_cache(merged, 1)[0]

        assert isinstance(merged[0], mr.BatchRotatingKVCache)
        assert isinstance(extracted_req0, mr.RotatingKVCache)
        assert isinstance(extracted_req1, mr.RotatingKVCache)
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

        merged = mr._merge_kv_caches([[cache_req0], [cache_req1]])
        extracted_req0 = mr._extract_kv_cache(merged, 0)[0]
        extracted_req1 = mr._extract_kv_cache(merged, 1)[0]

        assert isinstance(merged[0], mr.BatchRotatingKVCache)
        assert isinstance(extracted_req0, mr.RotatingKVCache)
        assert isinstance(extracted_req1, mr.RotatingKVCache)
        assert extracted_req0.offset == cache_req0.offset
        assert extracted_req1.offset == cache_req1.offset

    def test_rotating_kvcache_merge_handles_prefill_exceeding_max_size(self) -> None:
        """Merging works when prefill length exceeds max_size.

        After a large prefill the internal buffer may temporarily be larger than
        ``max_size``.  The merge must trim to the effective sliding-window length.
        """
        # Prefill 128 tokens into a cache with max_size=70
        cache_req0 = mr.RotatingKVCache(max_size=70)
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

        merged = mr._merge_kv_caches([[cache_req0], [cache_req1]])
        extracted_req0 = mr._extract_kv_cache(merged, 0)[0]
        extracted_req1 = mr._extract_kv_cache(merged, 1)[0]

        assert isinstance(merged[0], mr.BatchRotatingKVCache)
        assert isinstance(extracted_req0, mr.RotatingKVCache)
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

        merged = mr._merge_kv_caches([[cache_req0], [cache_req1]])
        batch_cache = merged[0]
        assert isinstance(batch_cache, mr.BatchRotatingKVCache)

        # Simulate one batched decode step (S=1)
        decode_k = mx.ones((2, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM))
        decode_v = mx.ones((2, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM))
        batch_cache.update_and_fetch(decode_k, decode_v)

        # Extract back to per-request caches
        extracted_req0 = batch_cache.extract(0)
        extracted_req1 = batch_cache.extract(1)

        assert isinstance(extracted_req0, mr.RotatingKVCache)
        assert isinstance(extracted_req1, mr.RotatingKVCache)
        assert extracted_req0.offset == cache_req0.offset + 1
        assert extracted_req1.offset == cache_req1.offset + 1

    def test_merge_kv_caches_rejects_mixed_cache_types_within_layer(self) -> None:
        arrays_cache = self._make_arrays_cache(1.0, 2.0)
        kv_cache = mr.KVCache()
        with pytest.raises(TypeError, match="Mixed cache types in a single layer"):
            mr._merge_kv_caches([[arrays_cache], [kv_cache]])


class TestPrefixCacheEviction:
    def test_eviction_under_max_bytes(self) -> None:
        # 1KB limit
        mgr = mr.PrefixCacheManager(max_bytes=1024)

        # Create fake KVCache with known size
        kv = mr.KVCache()
        k = mx.zeros((1, 4, 8, 64))  # 8192 bytes (float32)
        v = mx.zeros((1, 4, 8, 64))
        kv.state = [k, v]

        # Insert should be skipped (entry > max_bytes)
        mgr.insert([1, 2, 3], [kv])
        assert len(mgr._cache) == 0
        assert mgr._current_bytes == 0

    def _make_kv(self, seq_len: int = 1) -> mr.KVCache:
        kv = mr.KVCache()
        kv.state = [
            mx.zeros((1, 1, seq_len, 8)),
            mx.zeros((1, 1, seq_len, 8)),
        ]
        return kv

    def test_eviction_triggers_on_full(self) -> None:
        kv = self._make_kv()
        entry_bytes = kv.state[0].nbytes + kv.state[1].nbytes
        mgr = mr.PrefixCacheManager(max_bytes=entry_bytes * 2 + 1)

        mgr.insert([1], [self._make_kv()])
        mgr.insert([2], [self._make_kv()])
        assert len(mgr._cache) == 2

        # Third insert should evict one entry
        mgr.insert([3], [self._make_kv()])
        assert len(mgr._cache) == 2

    def test_duplicate_insert_skipped(self) -> None:
        mgr = mr.PrefixCacheManager(max_bytes=1024 * 1024)

        mgr.insert([1, 2], [self._make_kv()])
        bytes_after_first = mgr._current_bytes

        mgr.insert([1, 2], [self._make_kv()])
        assert mgr._current_bytes == bytes_after_first


class TestPrefixCacheEnableFlag:
    """Verify VLLM_METAL_PREFIX_CACHE presence-based enabling."""

    def test_enabled_when_env_set(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE", "1")
        assert mr._prefix_cache_enabled() is True

    def test_disabled_when_env_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("VLLM_METAL_PREFIX_CACHE", raising=False)
        assert mr._prefix_cache_enabled() is False


_TEN_GB = 10 * 1024**3


def _mock_device_info():
    return {"max_recommended_working_set_size": _TEN_GB}


class TestPrefixCacheFractionParsing:
    def test_valid_fraction(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE_FRACTION", "0.1")
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * 0.1)

    def test_default_fraction(self, monkeypatch) -> None:
        monkeypatch.delenv("VLLM_METAL_PREFIX_CACHE_FRACTION", raising=False)
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * mr._PREFIX_CACHE_DEFAULT_FRACTION)

    def test_invalid_string_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE_FRACTION", "abc")
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * mr._PREFIX_CACHE_DEFAULT_FRACTION)

    def test_out_of_range_zero_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE_FRACTION", "0")
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * mr._PREFIX_CACHE_DEFAULT_FRACTION)

    def test_out_of_range_above_one_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE_FRACTION", "2")
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * mr._PREFIX_CACHE_DEFAULT_FRACTION)

    def test_nan_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE_FRACTION", "nan")
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * mr._PREFIX_CACHE_DEFAULT_FRACTION)

    def test_inf_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE_FRACTION", "inf")
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * mr._PREFIX_CACHE_DEFAULT_FRACTION)

    def test_device_info_fallback(self, monkeypatch) -> None:
        monkeypatch.delenv("VLLM_METAL_PREFIX_CACHE_FRACTION", raising=False)
        monkeypatch.setattr(
            mr.mx.metal,
            "device_info",
            lambda: {},
        )
        result = mr._get_prefix_cache_max_bytes()
        fallback = 8 * 1024**3
        assert result == int(fallback * mr._PREFIX_CACHE_DEFAULT_FRACTION)
