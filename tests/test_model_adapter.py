# SPDX-License-Identifier: Apache-2.0
"""Tests for model adapter behavior."""

from types import SimpleNamespace

import pytest

from vllm_metal.v1.model_adapter import DefaultModelAdapter


class TestShouldForceTextBackbone:
    """Tests for should_force_text_backbone()."""

    def test_gemma4_model_type_is_overridden(self) -> None:
        hf_config = SimpleNamespace(model_type="gemma4")
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is True

    def test_non_overridden_model_type_is_not_forced(self) -> None:
        hf_config = SimpleNamespace(model_type="qwen3_5")
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_missing_model_type_is_not_forced(self) -> None:
        hf_config = SimpleNamespace()
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False


class TestTextModel:
    def test_returns_language_model_when_present(self) -> None:
        language_model = object()
        vlm = SimpleNamespace(language_model=language_model)
        adapter = DefaultModelAdapter()
        assert adapter.text_model(vlm) is language_model

    def test_returns_model_when_no_language_model(self) -> None:
        model = object()
        adapter = DefaultModelAdapter()
        assert adapter.text_model(model) is model


class TestResolveMaxHeadDim:
    """Tests for resolve_max_head_dim()."""

    def test_returns_global_when_larger(self) -> None:
        args = {"global_head_dim": 512}
        head_dim = 256
        adapter = DefaultModelAdapter()
        result = adapter.resolve_max_head_dim(args, head_dim)
        assert result == 512

    def test_returns_head_dim_when_larger(self) -> None:
        args = {"global_head_dim": 128}
        head_dim = 256
        adapter = DefaultModelAdapter()
        result = adapter.resolve_max_head_dim(args, head_dim)
        assert result == 256

    def test_returns_head_dim_when_global_missing(self) -> None:
        args = {}
        head_dim = 128
        adapter = DefaultModelAdapter()
        result = adapter.resolve_max_head_dim(args, head_dim)
        assert result == 128

    def test_returns_none_when_head_dim_none(self) -> None:
        args = {"global_head_dim": 512}
        head_dim = None
        adapter = DefaultModelAdapter()
        result = adapter.resolve_max_head_dim(args, head_dim)
        assert result is None


class TestRequireUniformKvHeads:
    """Tests for require_uniform_kv_heads()."""

    def test_allows_uniform_heads(self) -> None:
        args = {"num_global_key_value_heads": 8}
        num_kv_heads = 8
        adapter = DefaultModelAdapter()
        adapter.require_uniform_kv_heads(args, num_kv_heads)

    def test_allows_missing_global(self) -> None:
        args = {}
        num_kv_heads = 8
        adapter = DefaultModelAdapter()
        adapter.require_uniform_kv_heads(args, num_kv_heads)

    def test_rejects_gemma4_31b_config(self) -> None:
        args = {"num_global_key_value_heads": 4}
        num_kv_heads = 16
        adapter = DefaultModelAdapter()
        with pytest.raises(ValueError, match="variable KV head count"):
            adapter.require_uniform_kv_heads(args, num_kv_heads)

    def test_rejects_gemma4_26b_config(self) -> None:
        args = {"num_global_key_value_heads": 2}
        num_kv_heads = 8
        adapter = DefaultModelAdapter()
        with pytest.raises(ValueError, match="VLLM_METAL_USE_PAGED_ATTENTION=0"):
            adapter.require_uniform_kv_heads(args, num_kv_heads)
