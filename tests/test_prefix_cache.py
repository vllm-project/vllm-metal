# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx

import vllm_metal.v1.model_runner as mr


class StubMambaCache:
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
            return [kv, StubMambaCache(), kv]

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
