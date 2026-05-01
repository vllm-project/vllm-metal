# SPDX-License-Identifier: Apache-2.0
"""Tests for per-layer KV cache shape support."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from tests.stub_runner import make_stub_runner
from vllm_metal.config import AUTO_MEMORY_FRACTION, MetalConfig
from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.paged_attention_backend.mha import (
    MHAPagedAttentionBackend,
    warm_up_paged_cache,
)


class TestMetalPagedKVCachePerLayer:
    """MetalPagedKVCache with heterogeneous per-layer shapes."""

    def test_heterogeneous_shapes_allocate_correctly(self) -> None:
        """Each layer's cache tensor matches its requested shape."""
        kv_heads = [16, 4]
        head_dims = [256, 512]
        num_blocks = 4
        block_size = 16

        cache = MetalPagedKVCache(
            num_layers=2,
            num_kv_heads=kv_heads[0],
            head_dim=head_dims[0],
            num_blocks=num_blocks,
            block_size=block_size,
            dtype=mx.bfloat16,
            kv_heads_per_layer=kv_heads,
            head_dim_per_layer=head_dims,
        )

        assert cache.key_caches[0].shape == (num_blocks, block_size, 16, 256)
        assert cache.key_caches[1].shape == (num_blocks, block_size, 4, 512)
        assert cache.value_caches[0].shape == (num_blocks, block_size, 16, 256)
        assert cache.value_caches[1].shape == (num_blocks, block_size, 4, 512)

    def test_uniform_per_layer_matches_scalar(self) -> None:
        """Uniform per-layer lists produce identical shapes to scalar params."""
        num_layers = 4
        num_kv_heads = 8
        head_dim = 128
        num_blocks = 2
        block_size = 16

        scalar_cache = MetalPagedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
        )
        per_layer_cache = MetalPagedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            kv_heads_per_layer=[num_kv_heads] * num_layers,
            head_dim_per_layer=[head_dim] * num_layers,
        )

        for i in range(num_layers):
            assert (
                scalar_cache.key_caches[i].shape == per_layer_cache.key_caches[i].shape
            )

    def test_length_mismatch_raises(self) -> None:
        """Mismatched list lengths are caught early."""
        with pytest.raises(ValueError, match="kv_heads_per_layer length"):
            MetalPagedKVCache(
                num_layers=2,
                num_kv_heads=8,
                head_dim=128,
                num_blocks=1,
                block_size=16,
                kv_heads_per_layer=[8, 8, 8],
            )


class TestMHABackendPerLayer:
    """MHAPagedAttentionBackend passes per-layer shapes to cache."""

    def test_backend_propagates_per_layer_shapes(self) -> None:
        kv_heads = [16, 4]
        head_dims = [256, 512]

        backend = MHAPagedAttentionBackend(
            num_layers=2,
            num_kv_heads=kv_heads[0],
            head_dim=head_dims[0],
            block_size=16,
            dtype=mx.bfloat16,
            kv_heads_per_layer=kv_heads,
            head_dim_per_layer=head_dims,
        )
        backend.initialize(num_blocks=4)

        cache = backend._cache
        assert cache is not None
        assert cache.key_caches[0].shape[-2:] == (16, 256)
        assert cache.key_caches[1].shape[-2:] == (4, 512)

    def test_warm_up_uses_layer_zero_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kernel warm-up should compile against the concrete layer-0 shape."""
        cache = MetalPagedKVCache(
            num_layers=2,
            num_kv_heads=8,
            head_dim=128,
            num_blocks=1,
            block_size=16,
            dtype=mx.bfloat16,
            kv_heads_per_layer=[16, 4],
            head_dim_per_layer=[256, 512],
        )
        seen: dict[str, tuple[int, ...]] = {}

        class _FakeOps:
            def reshape_and_cache(
                self,
                dummy_k: mx.array,
                dummy_v: mx.array,
                key_cache: mx.array,
                value_cache: mx.array,
                dummy_slot: mx.array,
            ) -> None:
                seen["dummy_k"] = dummy_k.shape
                seen["dummy_v"] = dummy_v.shape
                seen["key_cache"] = key_cache.shape
                seen["value_cache"] = value_cache.shape

        monkeypatch.setattr(
            "vllm_metal.paged_attention_backend.mha.get_ops",
            lambda: _FakeOps(),
        )

        warm_up_paged_cache(cache)

        assert seen["dummy_k"] == (1, 16, 256)
        assert seen["dummy_v"] == (1, 16, 256)
        assert seen["key_cache"] == (1, 16, 16, 256)
        assert seen["value_cache"] == (1, 16, 16, 256)


class TestCachePolicyPerLayerBytes:
    """Block-size-bytes and one-sequence-bytes with per-layer shapes."""

    _BLOCK_SIZE = 16
    _DTYPE = mx.bfloat16

    def _make_runner(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        kv_heads_per_layer: list[int] | None = None,
        head_dim_per_layer: list[int] | None = None,
        model_args: dict | None = None,
    ) -> object:
        return make_stub_runner(
            model_args=model_args,
            num_layers=num_layers,
            num_kv_cache_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_cache_dtype=self._DTYPE,
            cache_config=SimpleNamespace(block_size=self._BLOCK_SIZE),
            kv_heads_per_layer=kv_heads_per_layer,
            head_dim_per_layer=head_dim_per_layer,
        )

    def test_uniform_per_layer_matches_scalar_block_bytes(self) -> None:
        """Uniform per-layer lists give identical block bytes to scalar path."""
        num_layers = 4
        num_kv_heads = 8
        head_dim = 128

        scalar_runner = self._make_runner(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        per_layer_runner = self._make_runner(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_heads_per_layer=[num_kv_heads] * num_layers,
            head_dim_per_layer=[head_dim] * num_layers,
        )

        assert (
            scalar_runner.get_cache_block_size_bytes()
            == per_layer_runner.get_cache_block_size_bytes()
        )

    def test_heterogeneous_block_bytes_equals_hand_computed_sum(self) -> None:
        """Per-layer bytes sum matches hand computation."""
        kv_heads = [16, 4]
        head_dims = [256, 512]
        dtype_size = self._DTYPE.size
        kv_factor = 2

        runner = self._make_runner(
            num_layers=2,
            num_kv_heads=kv_heads[0],
            head_dim=head_dims[0],
            kv_heads_per_layer=kv_heads,
            head_dim_per_layer=head_dims,
        )

        expected = (
            kv_factor
            * self._BLOCK_SIZE
            * dtype_size
            * sum(h * d for h, d in zip(kv_heads, head_dims, strict=True))
        )
        assert runner.get_cache_block_size_bytes() == expected

    def test_hybrid_per_layer_shapes_raise_early(self) -> None:
        """Unsupported hybrid + per-layer combos should fail at public APIs."""
        runner = self._make_runner(
            num_layers=4,
            num_kv_heads=4,
            head_dim=256,
            kv_heads_per_layer=[4, 4, 4, 4],
            head_dim_per_layer=[256, 256, 256, 256],
            model_args={"full_attention_interval": 2},
        )

        with pytest.raises(
            NotImplementedError, match="Per-layer KV shapes with hybrid models"
        ):
            runner.get_kv_cache_spec()

        with pytest.raises(
            NotImplementedError, match="Per-layer KV shapes with hybrid models"
        ):
            runner.build_paged_attention_backend(block_size=self._BLOCK_SIZE)

    def test_turboquant_per_layer_shapes_raise_early(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unsupported TurboQuant + per-layer combos should fail at public APIs."""
        runner = self._make_runner(
            num_layers=2,
            num_kv_heads=4,
            head_dim=256,
            kv_heads_per_layer=[4, 2],
            head_dim_per_layer=[256, 512],
        )
        monkeypatch.setattr(
            "vllm_metal.v1.cache_policy.get_config",
            lambda: MetalConfig(
                memory_fraction=AUTO_MEMORY_FRACTION,
                use_mlx=True,
                mlx_device="gpu",
                debug=False,
                turboquant=True,
            ),
        )

        with pytest.raises(
            NotImplementedError,
            match="TurboQuant with per-layer KV shapes is not yet supported",
        ):
            runner.validate_paged_attention_support()

        with pytest.raises(
            NotImplementedError,
            match="TurboQuant with per-layer KV shapes is not yet supported",
        ):
            runner.get_kv_cache_spec()
