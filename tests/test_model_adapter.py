# SPDX-License-Identifier: Apache-2.0
"""Tests for model adapter behavior."""

from types import SimpleNamespace

import pytest

import vllm_metal.envs as envs
from vllm_metal.config import reset_config
from vllm_metal.v1.model_adapter import DefaultModelAdapter


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch):
    for var in envs.environment_variables:
        monkeypatch.delenv(var, raising=False)
    reset_config()
    yield
    reset_config()


class TestShouldForceTextBackbone:
    """Tests for should_force_text_backbone()."""

    def test_gemma4_model_type_is_overridden(self) -> None:
        hf_config = SimpleNamespace(model_type="gemma4")
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is True

    def test_qwen35_fp8_conditional_generation_uses_auto_override(self) -> None:
        hf_config = SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
            quantization_config={"quant_method": "fp8"},
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is True

    def test_qwen36_fp8_conditional_generation_uses_auto_override(self) -> None:
        hf_config = SimpleNamespace(
            model_type="qwen3_6",
            architectures=["Qwen3_6ForConditionalGeneration"],
            quantization_config={"quant_method": "fp8"},
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is True

    def test_qwen35_non_fp8_conditional_generation_skips_auto_override(self) -> None:
        hf_config = SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
            quantization_config={"quant_method": "mxfp4"},
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_non_overridden_model_type_is_not_forced_in_auto_mode(self) -> None:
        hf_config = SimpleNamespace(model_type="qwen3_5")
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_text_only_compat_mode_forces_allowlisted_text_backbone(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "text-only-compat")
        reset_config()

        hf_config = SimpleNamespace(
            model_type="qwen3_6",
            architectures=["Qwen3_6MoeForConditionalGeneration"],
            quantization_config={"quant_method": "fp8"},
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is True

    def test_text_only_compat_mode_does_not_force_generic_vlm(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "text-only-compat")
        reset_config()

        hf_config = SimpleNamespace(model_type="phi3_v")
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_multimodal_native_mode_disables_auto_override(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "multimodal-native")
        reset_config()

        hf_config = SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
            quantization_config={"quant_method": "fp8"},
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_missing_model_type_is_not_forced(self) -> None:
        hf_config = SimpleNamespace()
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_none_hf_config_is_not_forced_in_auto_mode(self) -> None:
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(None)
        assert result is False


class TestNormalizeModelConfig:
    """Tests for normalize_model_config()."""

    def test_clears_multimodal_config_for_gemma4(self) -> None:
        model_config = SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=False),
            hf_config=SimpleNamespace(model_type="gemma4"),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is None

    def test_clears_multimodal_config_for_qwen35_fp8_in_auto_mode(self) -> None:
        model_config = SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=False),
            hf_config=SimpleNamespace(
                model_type="qwen3_5",
                architectures=["Qwen3_5ForConditionalGeneration"],
                quantization_config={"quant_method": "fp8"},
            ),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is None

    def test_text_only_compat_mode_clears_multimodal_for_allowlisted_vlm(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "text-only-compat")
        reset_config()

        model_config = SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=False),
            hf_config=SimpleNamespace(
                model_type="qwen3_6",
                architectures=["Qwen3_6ForConditionalGeneration"],
                quantization_config={"quant_method": "fp8"},
            ),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is None

    def test_text_only_compat_mode_preserves_generic_vlm(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "text-only-compat")
        reset_config()

        sentinel = SimpleNamespace(language_model_only=False)
        model_config = SimpleNamespace(
            multimodal_config=sentinel,
            hf_config=SimpleNamespace(model_type="phi3_v"),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is sentinel

    def test_preserves_multimodal_config_for_other_models(self) -> None:
        sentinel = SimpleNamespace(language_model_only=False)
        model_config = SimpleNamespace(
            multimodal_config=sentinel,
            hf_config=SimpleNamespace(model_type="qwen3_vl"),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is sentinel

    def test_multimodal_native_mode_preserves_qwen35_fp8_multimodal_config(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "multimodal-native")
        reset_config()

        sentinel = SimpleNamespace(language_model_only=False)
        model_config = SimpleNamespace(
            multimodal_config=sentinel,
            hf_config=SimpleNamespace(
                model_type="qwen3_5",
                architectures=["Qwen3_5ForConditionalGeneration"],
                quantization_config={"quant_method": "fp8"},
            ),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is sentinel

    def test_noop_when_multimodal_config_already_none(self) -> None:
        model_config = SimpleNamespace(
            multimodal_config=None,
            hf_config=SimpleNamespace(model_type="gemma4"),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is None

    def test_text_only_compat_mode_preserves_missing_hf_config(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "text-only-compat")
        reset_config()

        sentinel = SimpleNamespace(language_model_only=False)
        model_config = SimpleNamespace(
            multimodal_config=sentinel,
            hf_config=None,
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is sentinel


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


class TestYocoCacheIntegration:
    """Integration tests for YOCO KV cache sharing.

    Verifies reduced cache-byte accounting, _make_backend wiring, and
    shared layers reusing the remapped cache slot.
    """

    # Gemma4-like config: 8 layers, 3 shared, alternating types
    _LAYER_TYPES = [
        "sliding",
        "sliding",
        "full",
        "sliding",
        "full",
        "sliding",
        "full",
        "sliding",
    ]
    _NUM_HIDDEN = len(_LAYER_TYPES)
    _NUM_SHARED = 3
    _NUM_UNIQUE = _NUM_HIDDEN - _NUM_SHARED
    _NUM_KV_HEADS = 2
    _HEAD_DIM = 4
    _BLOCK_SIZE = 16
    _VOCAB_SIZE = 100

    def _gemma4_args(self) -> dict:
        return {
            "vocab_size": self._VOCAB_SIZE,
            "num_hidden_layers": self._NUM_HIDDEN,
            "num_kv_shared_layers": self._NUM_SHARED,
            "layer_types": list(self._LAYER_TYPES),
            "num_key_value_heads": self._NUM_KV_HEADS,
            "num_attention_heads": self._NUM_KV_HEADS,
            "head_dim": self._HEAD_DIM,
        }

    def test_num_kv_cache_layers_reduced(self) -> None:
        """Runner's num_kv_cache_layers uses unique count, not total."""
        from tests.stub_runner import make_stub_runner

        args = self._gemma4_args()
        runner = make_stub_runner(model_args=args)
        runner.num_layers = self._NUM_HIDDEN
        runner._model_adapter = DefaultModelAdapter()

        yoco = runner._model_adapter.build_yoco_cache_mapping(args)
        assert yoco is not None
        num_unique, _ = yoco
        assert num_unique == self._NUM_UNIQUE

    def test_cache_block_bytes_uses_unique_layers(self) -> None:
        """get_cache_block_size_bytes should use num_kv_cache_layers, not num_layers."""
        import mlx.core as mx

        from tests.stub_runner import make_stub_runner

        args = self._gemma4_args()
        runner = make_stub_runner(model_args=args)
        runner.num_layers = self._NUM_HIDDEN
        runner.num_kv_cache_layers = self._NUM_UNIQUE
        runner.num_kv_heads = self._NUM_KV_HEADS
        runner.head_dim = self._HEAD_DIM
        runner.kv_cache_dtype = mx.float16
        runner.cache_config = SimpleNamespace(block_size=self._BLOCK_SIZE)

        block_bytes = runner.get_cache_block_size_bytes()

        # 2 (K+V) * num_unique * block_size * kv_heads * head_dim * dtype
        dtype_size = mx.float16.size
        expected = (
            2
            * self._NUM_UNIQUE
            * self._BLOCK_SIZE
            * self._NUM_KV_HEADS
            * self._HEAD_DIM
            * dtype_size
        )
        assert block_bytes == expected

    def test_make_backend_uses_compact_layer_count(self) -> None:
        """_make_backend should create MHA backend with reduced num_layers."""
        import mlx.core as mx

        from tests.stub_runner import make_stub_runner
        from vllm_metal.v1.worker import MetalWorker

        adapter = DefaultModelAdapter()
        args = self._gemma4_args()
        yoco = adapter.build_yoco_cache_mapping(args)

        runner = make_stub_runner(
            model_args=args,
            num_layers=self._NUM_HIDDEN,
            num_kv_heads=self._NUM_KV_HEADS,
            head_dim=self._HEAD_DIM,
            kv_cache_dtype=mx.float16,
            _model_adapter=adapter,
            _yoco_cache_mapping=yoco,
            num_kv_cache_layers=yoco[0],
        )

        backend = MetalWorker._make_backend(runner, block_size=self._BLOCK_SIZE)

        assert backend._num_layers == self._NUM_UNIQUE
        assert backend._cache_idx_map is not None
        # Shared layers map to a unique layer of same type
        for i in range(self._NUM_UNIQUE, self._NUM_HIDDEN):
            ref = backend._cache_idx_map[i]
            assert ref < self._NUM_UNIQUE
            assert self._LAYER_TYPES[ref] == self._LAYER_TYPES[i]

    def test_shared_layer_reuses_cache_slot(self) -> None:
        """Shared layers should get same cache_idx as their reference layer."""
        adapter = DefaultModelAdapter()
        args = self._gemma4_args()
        result = adapter.build_yoco_cache_mapping(args)
        assert result is not None
        _, mapping = result

        # Layer 5 (sliding, shared) should map to the same cache_idx as
        # the last unique sliding layer in 0..4
        shared_sliding = 5
        ref = mapping[shared_sliding]
        assert self._LAYER_TYPES[ref] == "sliding"
        assert ref < self._NUM_UNIQUE

        # Layer 6 (full, shared) → same cache_idx as last unique full layer
        shared_full = 6
        ref_full = mapping[shared_full]
        assert self._LAYER_TYPES[ref_full] == "full"
        assert ref_full < self._NUM_UNIQUE


class TestRequireUniformKvHeads:
    """Tests for require_uniform_kv_heads().

    The method still raises on mismatched KV head counts under the uniform
    cache path.  For models whose adapter populates ``kv_heads_per_layer``
    via :meth:`build_per_layer_kv_shapes` (Gemma4 26B/31B), the uniform
    guard is gated out at the call-site in ``ModelCachePolicy`` — see
    :class:`tests.test_model_lifecycle.TestResolveModelDims` for that path.
    """

    def test_allows_uniform_heads(self) -> None:
        args = {"num_global_key_value_heads": 8}
        num_kv_heads = 8
        adapter = DefaultModelAdapter()
        adapter.require_uniform_kv_heads(args, num_kv_heads)

    def test_allows_missing_global(self) -> None:
        args: dict[str, int] = {}
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


class TestBuildPerLayerKVShapes:
    """Tests for :meth:`DefaultModelAdapter.build_per_layer_kv_shapes`."""

    def test_returns_none_for_uniform_model(self) -> None:
        adapter = DefaultModelAdapter()
        result = adapter.build_per_layer_kv_shapes(
            args={},
            num_layers=4,
            num_kv_heads=8,
            head_dim=128,
        )
        assert result is None

    def test_returns_none_when_layer_types_length_mismatches(self) -> None:
        adapter = DefaultModelAdapter()
        args = {
            "layer_types": ["sliding_attention", "full_attention"],
            "global_head_dim": 512,
            "num_global_key_value_heads": 4,
        }
        result = adapter.build_per_layer_kv_shapes(
            args=args,
            num_layers=4,
            num_kv_heads=16,
            head_dim=256,
        )
        assert result is None

    def test_returns_none_without_global_head_dim(self) -> None:
        """No ``global_head_dim`` → uniform path, even with layer_types present."""
        adapter = DefaultModelAdapter()
        args = {"layer_types": ["full_attention", "full_attention"]}
        result = adapter.build_per_layer_kv_shapes(
            args=args,
            num_layers=2,
            num_kv_heads=8,
            head_dim=128,
        )
        assert result is None

    def test_gemma4_31b_style_full_override(self) -> None:
        """31B-style configs with distinct full-attention KV heads and head_dim."""
        adapter = DefaultModelAdapter()
        args = {
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            "global_head_dim": 512,
            "num_global_key_value_heads": 4,
        }
        result = adapter.build_per_layer_kv_shapes(
            args=args,
            num_layers=4,
            num_kv_heads=16,
            head_dim=256,
        )

        assert result == ([16, 4, 16, 4], [256, 512, 256, 512])

    def test_gemma4_e2b_style_head_dim_only_override(self) -> None:
        """E2B-style configs omit ``num_global_key_value_heads`` entirely.

        Full-attention layers must fall back to the sliding-layer KV-head
        count instead of collapsing to a uniform shape, which would cause
        full-attention writes to overflow the allocated cache head_dim.
        """
        adapter = DefaultModelAdapter()
        args = {
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            "global_head_dim": 512,
        }
        result = adapter.build_per_layer_kv_shapes(
            args=args,
            num_layers=4,
            num_kv_heads=1,
            head_dim=256,
        )

        assert result == ([1, 1, 1, 1], [256, 512, 256, 512])

    def test_unknown_layer_type_raises(self) -> None:
        """Unknown layer_type surfaces as a ValueError pinpointing the index
        rather than silently collapsing to full-attention shapes."""
        adapter = DefaultModelAdapter()
        args = {
            "layer_types": ["sliding_attention", "linear_attention"],
            "global_head_dim": 512,
            "num_global_key_value_heads": 4,
        }
        with pytest.raises(
            ValueError,
            match=r"Unsupported Gemma4 layer_type at index 1: 'linear_attention'",
        ):
            adapter.build_per_layer_kv_shapes(
                args=args,
                num_layers=2,
                num_kv_heads=16,
                head_dim=256,
            )


class TestBuildSlidingWindowPerLayer:
    """Tests for build_sliding_window_per_layer()."""

    def test_returns_none_for_model_without_layer_types(self) -> None:
        adapter = DefaultModelAdapter()
        result = adapter.build_sliding_window_per_layer({}, num_layers=4)
        assert result is None

    def test_returns_none_without_sliding_window_config(self) -> None:
        args = {"layer_types": ["sliding_attention", "full_attention"]}
        adapter = DefaultModelAdapter()
        result = adapter.build_sliding_window_per_layer(args, num_layers=2)
        assert result is None

    def test_gemma4_sliding_layers_get_window_full_layers_disabled(self) -> None:
        args = {
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
            "sliding_window": 1024,
        }
        adapter = DefaultModelAdapter()
        result = adapter.build_sliding_window_per_layer(args, num_layers=4)
        assert result == [1024, 1024, -1, 1024]

    def test_length_mismatch_returns_none(self) -> None:
        args = {
            "layer_types": ["sliding_attention"],
            "sliding_window": 1024,
        }
        adapter = DefaultModelAdapter()
        result = adapter.build_sliding_window_per_layer(args, num_layers=4)
        assert result is None
