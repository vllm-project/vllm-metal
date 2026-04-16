# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal platform."""

import platform
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

from vllm_metal.config import reset_config
from vllm_metal.platform import MetalPlatform
from vllm_metal.v1.worker import MetalWorker


class TestMetalPlatform:
    """Tests for MetalPlatform class."""

    @staticmethod
    def _patch_stt_resolution(
        monkeypatch: pytest.MonkeyPatch,
        *,
        is_stt: bool,
    ) -> None:
        monkeypatch.setattr(
            "vllm_metal.utils.get_model_download_path",
            lambda model: model,
        )
        monkeypatch.setattr(
            "vllm_metal.stt.detection.is_stt_model", lambda _model: is_stt
        )

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

    def test_get_attn_backend_cls_accepts_mla(self) -> None:
        """MLA is handled by the vllm-metal model runner; CPU_ATTN is returned."""
        cfg = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=True,
        )
        backend = MetalPlatform.get_attn_backend_cls(AttentionBackendEnum.CPU_ATTN, cfg)
        assert backend == AttentionBackendEnum.CPU_ATTN.get_path()

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

    def test_is_available_propagates_unexpected_mlx_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unexpected MLX errors should surface instead of looking unavailable."""
        monkeypatch.setattr("vllm_metal.platform.py_platform.machine", lambda: "arm64")
        monkeypatch.setattr("vllm_metal.platform.py_platform.system", lambda: "Darwin")

        mlx_module = ModuleType("mlx")
        mlx_core = ModuleType("mlx.core")

        class _BrokenMetal:
            @staticmethod
            def is_available() -> bool:
                raise ValueError("unexpected mlx regression")

        mlx_core.metal = _BrokenMetal()
        mlx_module.core = mlx_core
        monkeypatch.setitem(sys.modules, "mlx", mlx_module)
        monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

        with pytest.raises(ValueError, match="unexpected mlx regression"):
            MetalPlatform.is_available()

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

    def test_check_and_update_config_disables_chunked_prefill_non_paged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-paged path should disable chunked prefill.

        When chunked prefill is disabled, max_num_batched_tokens must be at
        least max_model_len so the scheduler can schedule the entire prompt
        in a single step.
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "0")
        reset_config()
        try:
            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(block_size=None),
                model_config=SimpleNamespace(
                    model="test-model",
                    disable_cascade_attn=False,
                    tokenizer=None,
                    max_model_len=32768,
                    multimodal_config=None,
                    hf_config=SimpleNamespace(model_type="qwen3"),
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=True,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=None,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert vllm_config.scheduler_config.enable_chunked_prefill is False
            assert vllm_config.scheduler_config.max_num_batched_tokens == 32768
            assert (
                vllm_config.parallel_config.worker_cls
                == "vllm_metal.v1.worker.MetalWorker"
            )
            assert vllm_config.parallel_config.distributed_executor_backend == "uni"
            assert vllm_config.parallel_config.disable_custom_all_reduce is True
        finally:
            reset_config()

    def test_check_and_update_config_keeps_chunked_prefill_for_paged_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Paged path should keep chunked prefill enabled.

        The unified varlen Metal kernel handles mixed prefill + decode,
        so chunked prefill works correctly on the paged path.
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    block_size=None, enable_prefix_caching=False
                ),
                model_config=SimpleNamespace(
                    model="test-model",
                    disable_cascade_attn=False,
                    tokenizer=None,
                    max_model_len=32768,
                    multimodal_config=None,
                    hf_config=SimpleNamespace(model_type="qwen3"),
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=True,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=None,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert vllm_config.scheduler_config.enable_chunked_prefill is True
            # max_num_batched_tokens should NOT be bumped (chunked prefill handles it)
            assert vllm_config.scheduler_config.max_num_batched_tokens == 2048
        finally:
            reset_config()

    def test_check_and_update_config_increases_max_num_scheduled_tokens_below_max_model_len(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_num_scheduled_tokens below max_model_len should be bumped up to max_model_len.

        When max_num_scheduled_tokens is explicitly set to a value smaller
        than max_model_len, it must be raised to match max_model_len so that
        the scheduler can schedule the full prompt in a single step.
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "0")
        reset_config()
        try:
            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(block_size=None),
                model_config=SimpleNamespace(
                    model="test-model",
                    disable_cascade_attn=False,
                    tokenizer=None,
                    max_model_len=32768,
                    multimodal_config=None,
                    hf_config=SimpleNamespace(model_type="qwen3"),
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=True,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=2048,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert vllm_config.scheduler_config.enable_chunked_prefill is False
            assert vllm_config.scheduler_config.max_num_batched_tokens == 32768
            assert vllm_config.scheduler_config.max_num_scheduled_tokens == 32768
        finally:
            reset_config()

    def test_check_and_update_config_does_not_reduce_large_max_num_batched_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_num_batched_tokens must not be lowered when already >= max_model_len.

        If the user has explicitly set a token budget larger than max_model_len,
        that setting must be preserved.
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "0")
        reset_config()
        try:
            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(block_size=None),
                model_config=SimpleNamespace(
                    model="test-model",
                    disable_cascade_attn=False,
                    tokenizer=None,
                    max_model_len=32768,
                    multimodal_config=None,
                    hf_config=SimpleNamespace(model_type="qwen3"),
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=True,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=65536,
                    max_num_scheduled_tokens=None,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert vllm_config.scheduler_config.enable_chunked_prefill is False
            # 65536 > 32768, so the value must stay at 65536
            assert vllm_config.scheduler_config.max_num_batched_tokens == 65536
        finally:
            reset_config()

    @pytest.mark.parametrize("max_num_scheduled_tokens", [32768, 65536])
    def test_check_and_update_config_does_not_reduce_max_num_scheduled_tokens_when_at_least_max_model_len(
        self,
        monkeypatch: pytest.MonkeyPatch,
        max_num_scheduled_tokens: int,
    ) -> None:
        """max_num_scheduled_tokens must not be lowered when already >= max_model_len.

        If the user has explicitly set a scheduled-token budget at least
        max_model_len, that setting must be preserved (only values strictly
        below max_model_len are bumped up).
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "0")
        reset_config()
        try:
            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(block_size=None),
                model_config=SimpleNamespace(
                    model="test-model",
                    disable_cascade_attn=False,
                    tokenizer=None,
                    max_model_len=32768,
                    multimodal_config=None,
                    hf_config=SimpleNamespace(model_type="qwen3"),
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=True,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=65536,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert vllm_config.scheduler_config.enable_chunked_prefill is False
            assert vllm_config.scheduler_config.max_num_batched_tokens == 65536
            assert (
                vllm_config.scheduler_config.max_num_scheduled_tokens
                == max_num_scheduled_tokens
            )
        finally:
            reset_config()

    def test_check_and_update_config_applies_stt_scheduler_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """STT models should get tokenizer fallback and async scheduling disabled."""
        self._patch_stt_resolution(monkeypatch, is_stt=True)
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                disable_custom_all_reduce=False,
            ),
            cache_config=SimpleNamespace(block_size=None),
            model_config=SimpleNamespace(
                model="openai/whisper-tiny",
                disable_cascade_attn=False,
                tokenizer=None,
                multimodal_config=None,
                hf_config=SimpleNamespace(model_type="whisper"),
            ),
            scheduler_config=SimpleNamespace(
                async_scheduling=True,
                enable_chunked_prefill=False,
            ),
        )

        MetalPlatform.check_and_update_config(vllm_config)

        assert vllm_config.model_config.tokenizer == "openai/whisper-tiny"
        assert vllm_config.scheduler_config.async_scheduling is False

    def test_check_and_update_config_preserves_existing_tokenizer_for_stt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """STT policy should not overwrite an explicitly configured tokenizer."""
        self._patch_stt_resolution(monkeypatch, is_stt=True)
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                disable_custom_all_reduce=False,
            ),
            cache_config=SimpleNamespace(block_size=None),
            model_config=SimpleNamespace(
                model="openai/whisper-tiny",
                disable_cascade_attn=False,
                tokenizer="custom-tokenizer",
                multimodal_config=None,
                hf_config=SimpleNamespace(model_type="whisper"),
            ),
            scheduler_config=SimpleNamespace(
                async_scheduling=True,
                enable_chunked_prefill=False,
            ),
        )

        MetalPlatform.check_and_update_config(vllm_config)

        assert vllm_config.model_config.tokenizer == "custom-tokenizer"
        assert vllm_config.scheduler_config.async_scheduling is False

    def test_check_and_update_config_clears_multimodal_for_text_backbone_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gemma4-style multimodal configs must be cleared for the text-only path.

        Gemma4 MLX checkpoints are flagged multimodal in the HF config but
        ship without the vision/audio preprocessor files that vLLM's input
        processor tries to load. Clearing ``multimodal_config`` at the
        platform layer makes ``is_multimodal_model`` False so the input
        processor skips feature-extractor loading.
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            model_config = SimpleNamespace(
                model="test-model",
                disable_cascade_attn=False,
                tokenizer=None,
                max_model_len=128,
                multimodal_config=SimpleNamespace(language_model_only=False),
                hf_config=SimpleNamespace(model_type="gemma4"),
            )
            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    block_size=None, enable_prefix_caching=False
                ),
                model_config=model_config,
                scheduler_config=SimpleNamespace(
                    async_scheduling=False,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=None,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert model_config.multimodal_config is None

        finally:
            reset_config()

    def test_check_and_update_config_preserves_multimodal_for_non_gemma4_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-overridden multimodal models must keep multimodal_config set."""
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            sentinel = SimpleNamespace(language_model_only=False)
            model_config = SimpleNamespace(
                model="test-model",
                disable_cascade_attn=False,
                tokenizer=None,
                max_model_len=128,
                multimodal_config=sentinel,
                hf_config=SimpleNamespace(model_type="qwen3_vl"),
            )
            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    block_size=None, enable_prefix_caching=False
                ),
                model_config=model_config,
                scheduler_config=SimpleNamespace(
                    async_scheduling=False,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=None,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert model_config.multimodal_config is sentinel

        finally:
            reset_config()

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

        def fake_eval(_value: object) -> None:
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

        def fake_eval(_value: object) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(mx, "eval", fake_eval)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        MetalPlatform.synchronize()
        assert called is True


class TestKvBudgetBytes:
    """Tests for MetalWorker._kv_budget_bytes.

    Numbers mirror a real M2 Max with GLM-4.7-Flash-4bit loaded:
      metal_limit = 22.9 GB (max_recommended_working_set_size)
      model_memory = 16.85 GB (mx.get_active_memory() after load)
    """

    _METAL_LIMIT = int(22.9e9)
    _MODEL_MEM = int(16.85e9)
    # Simulated measured overhead (matches what profile_run would return).
    _OVERHEAD = 200 * 1024 * 1024  # 200 MB

    def test_normal_case(self) -> None:
        budget = MetalWorker._kv_budget_bytes(
            self._METAL_LIMIT,
            self._MODEL_MEM,
            fraction=0.9,
            overhead=self._OVERHEAD,
        )

        assert budget == int(self._METAL_LIMIT * 0.9) - self._MODEL_MEM - self._OVERHEAD
        assert budget > 0

    def test_fraction_too_low_yields_negative_budget(self) -> None:
        # fraction=0.3 → usable=6.9 GB < model(16.85 GB) → negative
        budget = MetalWorker._kv_budget_bytes(
            self._METAL_LIMIT,
            self._MODEL_MEM,
            fraction=0.3,
            overhead=self._OVERHEAD,
        )

        assert budget < 0

    def test_boundary_zero(self) -> None:
        # Craft inputs so budget lands exactly at zero.
        limit = self._MODEL_MEM + self._OVERHEAD

        budget = MetalWorker._kv_budget_bytes(
            limit, self._MODEL_MEM, fraction=1.0, overhead=self._OVERHEAD
        )

        assert budget == 0

    def test_custom_overhead(self) -> None:
        budget_zero_overhead = MetalWorker._kv_budget_bytes(
            self._METAL_LIMIT, self._MODEL_MEM, fraction=0.9, overhead=0
        )
        budget_with_overhead = MetalWorker._kv_budget_bytes(
            self._METAL_LIMIT,
            self._MODEL_MEM,
            fraction=0.9,
            overhead=self._OVERHEAD,
        )

        assert budget_zero_overhead - budget_with_overhead == self._OVERHEAD

    def test_large_model_has_positive_budget_at_default_fraction(self) -> None:
        # GLM-4.7-Flash-4bit at fraction=0.9 must yield > 1 GB for KV cache.
        budget = MetalWorker._kv_budget_bytes(
            self._METAL_LIMIT,
            self._MODEL_MEM,
            fraction=0.9,
            overhead=self._OVERHEAD,
        )

        assert budget > 1e9
