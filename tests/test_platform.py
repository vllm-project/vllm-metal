# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal platform."""

import platform

import pytest
import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

from vllm_metal.platform import MetalPlatform


class TestMetalPlatform:
    """Tests for MetalPlatform class."""

    def test_device_name(self) -> None:
        """Test device name retrieval."""
        name = MetalPlatform.get_device_name()
        assert "Apple Silicon" in name

    def test_device_count(self) -> None:
        """Test device count."""
        count = MetalPlatform.get_device_count()
        assert count == 1

    def test_current_device(self) -> None:
        """Test current device."""
        device = MetalPlatform.current_device()
        assert device == 0

    def test_set_device_valid(self) -> None:
        """Test setting valid device."""
        MetalPlatform.set_device(0)  # Should not raise

    def test_set_device_invalid(self) -> None:
        """Test setting invalid device."""
        with pytest.raises(ValueError, match="only supports device 0"):
            MetalPlatform.set_device(1)

    def test_device_capability(self) -> None:
        """Test device capability."""
        major, minor = MetalPlatform.get_device_capability()
        assert isinstance(major, int)
        assert isinstance(minor, int)

    def test_get_attn_backend_cls_returns_cpu_backend(self) -> None:
        """Metal platform should return a concrete backend path."""
        cfg = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
        )
        backend = MetalPlatform.get_attn_backend_cls(AttentionBackendEnum.CPU_ATTN, cfg)
        assert backend == AttentionBackendEnum.CPU_ATTN.get_path()

    def test_get_attn_backend_cls_rejects_mla(self) -> None:
        """MLA is not supported on Metal/MLX."""
        cfg = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=True,
        )
        with pytest.raises(NotImplementedError, match="MLA is not supported"):
            MetalPlatform.get_attn_backend_cls(AttentionBackendEnum.CPU_ATTN, cfg)

    def test_get_attn_backend_cls_rejects_sparse(self) -> None:
        """Sparse attention is not supported on Metal/MLX."""
        cfg = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_sparse=True,
        )
        with pytest.raises(
            NotImplementedError, match="Sparse Attention is not supported"
        ):
            MetalPlatform.get_attn_backend_cls(AttentionBackendEnum.CPU_ATTN, cfg)

    def test_memory_info(self) -> None:
        """Test memory information."""
        total = MetalPlatform.get_device_total_memory()
        available = MetalPlatform.get_device_available_memory()

        assert total > 0
        assert available > 0
        assert available <= total

    @pytest.mark.skipif(
        platform.machine() != "arm64" or platform.system() != "Darwin",
        reason="Only runs on Apple Silicon",
    )
    def test_is_available(self) -> None:
        """Test platform availability on Apple Silicon."""
        assert MetalPlatform.is_available() is True

    def test_is_available_does_not_mutate_default_device(self) -> None:
        """Availability check should not change the MLX default device."""
        mx = pytest.importorskip("mlx.core")

        before = mx.default_device()
        MetalPlatform.is_available()
        after = mx.default_device()

        assert before == after

    def test_torch_device(self) -> None:
        """Test PyTorch device retrieval."""

        device = MetalPlatform.get_torch_device()
        assert device.type in ("mps", "cpu")

    def test_verify_quantization_supported(self) -> None:
        """Test that verify_quantization allows all methods to pass through."""
        # All quantization methods should pass through - actual support depends
        # on model implementation, not the platform
        MetalPlatform.verify_quantization("none")
        MetalPlatform.verify_quantization(None)
        MetalPlatform.verify_quantization("fp16")
        MetalPlatform.verify_quantization("bfloat16")
        MetalPlatform.verify_quantization("int8")
        MetalPlatform.verify_quantization("awq")
        MetalPlatform.verify_quantization("compressed-tensors")

    def test_synchronize_runs_mlx_barrier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Platform synchronize should use MX barrier when present."""
        mx = pytest.importorskip("mlx.core")
        if not hasattr(mx, "synchronize"):
            pytest.skip("mlx.core.synchronize not available")

        called = False

        def fake_sync() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(mx, "synchronize", fake_sync)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        MetalPlatform.synchronize()
        assert called is True

    def test_synchronize_falls_back_to_eval_when_missing_barrier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fallback to evaluation when MX barrier is unavailable."""
        mx = pytest.importorskip("mlx.core")

        monkeypatch.delattr(mx, "synchronize", raising=False)

        called = False

        def fake_eval(value: mx.array) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(mx, "eval", fake_eval)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        MetalPlatform.synchronize()
        assert called is True

    def test_synchronize_falls_back_to_eval_when_barrier_signature_incompatible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fallback if MX barrier exists but can't be called with no args."""
        mx = pytest.importorskip("mlx.core")

        def fake_sync(_stream: object) -> None:
            return None

        monkeypatch.setattr(mx, "synchronize", fake_sync)

        called = False

        def fake_eval(value: mx.array) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(mx, "eval", fake_eval)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        MetalPlatform.synchronize()
        assert called is True
