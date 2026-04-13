# SPDX-License-Identifier: Apache-2.0
"""Tests for model adapter behavior."""

from types import SimpleNamespace

import pytest

from vllm_metal.v1.model_adapter import DefaultModelAdapter


class TestShouldForceTextBackbone:
    """Tests for should_force_text_backbone()."""

    def test_gemma4_model_type_is_overridden(self) -> None:
        # Arrange
        hf_config = SimpleNamespace(model_type="gemma4")
        adapter = DefaultModelAdapter()

        # Act
        result = adapter.should_force_text_backbone(hf_config)

        # Assert
        assert result is True

    def test_non_overridden_model_type_is_not_forced(self) -> None:
        # Arrange
        hf_config = SimpleNamespace(model_type="qwen3_5")
        adapter = DefaultModelAdapter()

        # Act
        result = adapter.should_force_text_backbone(hf_config)

        # Assert
        assert result is False

    def test_missing_model_type_is_not_forced(self) -> None:
        # Arrange
        hf_config = SimpleNamespace()
        adapter = DefaultModelAdapter()

        # Act
        result = adapter.should_force_text_backbone(hf_config)

        # Assert
        assert result is False


class TestResolveMaxHeadDim:
    """Tests for resolve_max_head_dim()."""

    def test_returns_global_when_larger(self) -> None:
        # Arrange — Gemma4-style: sliding=256, full=512
        args = {"global_head_dim": 512}
        head_dim = 256
        adapter = DefaultModelAdapter()

        # Act
        result = adapter.resolve_max_head_dim(args, head_dim)

        # Assert
        assert result == 512

    def test_returns_head_dim_when_larger(self) -> None:
        # Arrange — hypothetical inverse case
        args = {"global_head_dim": 128}
        head_dim = 256
        adapter = DefaultModelAdapter()

        # Act
        result = adapter.resolve_max_head_dim(args, head_dim)

        # Assert
        assert result == 256

    def test_returns_head_dim_when_global_missing(self) -> None:
        # Arrange — uniform-head_dim models (most models)
        args = {}
        head_dim = 128
        adapter = DefaultModelAdapter()

        # Act
        result = adapter.resolve_max_head_dim(args, head_dim)

        # Assert
        assert result == 128

    def test_returns_none_when_head_dim_none(self) -> None:
        # Arrange
        args = {"global_head_dim": 512}
        head_dim = None
        adapter = DefaultModelAdapter()

        # Act
        result = adapter.resolve_max_head_dim(args, head_dim)

        # Assert
        assert result is None


class TestRequireUniformKvHeads:
    """Tests for require_uniform_kv_heads()."""

    def test_allows_uniform_heads(self) -> None:
        # Arrange — typical model
        args = {"num_global_key_value_heads": 8}
        num_kv_heads = 8
        adapter = DefaultModelAdapter()

        # Act — should not raise
        adapter.require_uniform_kv_heads(args, num_kv_heads)

    def test_allows_missing_global(self) -> None:
        # Arrange — no global KV head count set (most models)
        args = {}
        num_kv_heads = 8
        adapter = DefaultModelAdapter()

        # Act — should not raise
        adapter.require_uniform_kv_heads(args, num_kv_heads)

    def test_rejects_gemma4_31b_config(self) -> None:
        # Arrange — Gemma4 31B: sliding=16, full=4
        args = {"num_global_key_value_heads": 4}
        num_kv_heads = 16
        adapter = DefaultModelAdapter()

        # Act / Assert
        with pytest.raises(ValueError, match="variable KV head count"):
            adapter.require_uniform_kv_heads(args, num_kv_heads)

    def test_rejects_gemma4_26b_config(self) -> None:
        # Arrange — Gemma4 26B: sliding=8, full=2
        args = {"num_global_key_value_heads": 2}
        num_kv_heads = 8
        adapter = DefaultModelAdapter()

        # Act / Assert
        with pytest.raises(ValueError, match="VLLM_METAL_USE_PAGED_ATTENTION=0"):
            adapter.require_uniform_kv_heads(args, num_kv_heads)
