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
        runner._prefix_cache = mr.PrefixCacheManager()
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
