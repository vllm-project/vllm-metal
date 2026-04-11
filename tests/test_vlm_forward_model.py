# SPDX-License-Identifier: Apache-2.0
"""Unit tests for VLM text-only forward dispatch.

Covers:
- _vlm_text_model() helper (contiguous_cache module)
- MetalModelRunner._forward_model property
- PrefixCacheManager.restore_cache routing
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.stub_runner import make_stub_runner
from vllm_metal.v1.vlm_utils import _vlm_text_model

# ---------------------------------------------------------------------------
# _vlm_text_model helper
# ---------------------------------------------------------------------------


class TestVlmTextModel:
    def test_returns_language_model_when_present(self) -> None:
        lang = object()
        vlm = SimpleNamespace(language_model=lang)
        assert _vlm_text_model(vlm) is lang

    def test_returns_model_when_no_language_model(self) -> None:
        model = object()
        assert _vlm_text_model(model) is model


# ---------------------------------------------------------------------------
# MetalModelRunner._forward_model property
# ---------------------------------------------------------------------------


class TestForwardModelProperty:
    def test_text_model_returns_self(self) -> None:
        model = MagicMock()
        runner = make_stub_runner(model=model, _is_vlm=False)
        assert runner._forward_model is model

    def test_vlm_with_language_model_returns_language_model(self) -> None:
        lang = MagicMock()
        vlm = MagicMock()
        vlm.language_model = lang
        runner = make_stub_runner(model=vlm, _is_vlm=True)
        assert runner._forward_model is lang

    def test_vlm_without_language_model_returns_model(self) -> None:
        # Edge case: mlx-vlm model that doesn't have .language_model
        model = MagicMock(spec=[])  # no attributes
        runner = make_stub_runner(model=model, _is_vlm=True)
        assert runner._forward_model is model

    def test_vlm_flag_false_bypasses_language_model(self) -> None:
        # _is_vlm=False: language_model attribute is ignored even if present
        lang = MagicMock()
        model = MagicMock()
        model.language_model = lang
        runner = make_stub_runner(model=model, _is_vlm=False)
        assert runner._forward_model is model


# ---------------------------------------------------------------------------
# PrefixCacheManager.restore_cache routing
# ---------------------------------------------------------------------------


class TestRestoreCacheRouting:
    def _make_cached_prefix(self, num_layers: int = 2):
        from vllm_metal.v1.contiguous_cache import CachedPrefix

        cp = CachedPrefix.__new__(CachedPrefix)
        cp.token_ids = [1, 2, 3]
        cp.cache_state = [None] * num_layers
        cp.size_bytes = 0
        cp.ref_count = 1
        return cp

    def test_non_vlm_uses_model_directly(self, monkeypatch) -> None:
        from vllm_metal.v1.contiguous_cache import PrefixCacheManager

        model = MagicMock()
        captured = {}

        def fake_make_prompt_cache(m):
            captured["model"] = m
            return []

        monkeypatch.setattr(
            "vllm_metal.v1.contiguous_cache.make_prompt_cache", fake_make_prompt_cache
        )
        mgr = PrefixCacheManager.__new__(PrefixCacheManager)
        mgr.restore_cache(self._make_cached_prefix(), model=model, is_vlm=False)
        assert captured["model"] is model

    def test_vlm_uses_language_model(self, monkeypatch) -> None:
        from vllm_metal.v1.contiguous_cache import PrefixCacheManager

        lang = MagicMock()
        vlm = MagicMock()
        vlm.language_model = lang
        captured = {}

        def fake_make_prompt_cache(m):
            captured["model"] = m
            return []

        monkeypatch.setattr(
            "vllm_metal.v1.contiguous_cache.make_prompt_cache", fake_make_prompt_cache
        )
        mgr = PrefixCacheManager.__new__(PrefixCacheManager)
        mgr.restore_cache(self._make_cached_prefix(), model=vlm, is_vlm=True)
        assert captured["model"] is lang
