# SPDX-License-Identifier: Apache-2.0
"""Tests for per-layer KV cache shape support."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)

from tests.stub_runner import make_stub_runner
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.mha_layout import MHAKVCacheLayout
from vllm_metal.attention.runtime.mha import (
    MHAPagedAttentionRuntime,
)
from vllm_metal.config import AUTO_MEMORY_FRACTION, MetalConfig


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
    """MHAPagedAttentionRuntime passes per-layer shapes to cache."""

    def test_backend_propagates_per_layer_shapes(self) -> None:
        kv_heads = [16, 4]
        head_dims = [256, 512]

        backend = MHAPagedAttentionRuntime(
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
            runner.build_paged_attention_runtime(block_size=self._BLOCK_SIZE)

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


class TestMHAKVCacheLayout:
    """vLLM-managed standard-MHA cache layout contracts."""

    @staticmethod
    def _mixed_mha_config() -> tuple[KVCacheConfig, tuple[str, ...]]:
        full = FullAttentionSpec(
            block_size=32,
            num_kv_heads=4,
            head_size=512,
            dtype=torch.bfloat16,
        )
        sliding = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=16,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=1024,
        )
        names = tuple(f"layers.{index}.self_attn" for index in range(4))
        groups = [
            KVCacheGroupSpec([names[0], names[2]], full),
            KVCacheGroupSpec([names[1], names[3]], sliding),
        ]
        assert full.page_size_bytes == sliding.page_size_bytes
        size = full.page_size_bytes * 3
        config = KVCacheConfig(
            num_blocks=3,
            kv_cache_tensors=[
                KVCacheTensor(size=size, shared_by=[names[0], names[1]]),
                KVCacheTensor(size=size, shared_by=[names[2], names[3]]),
            ],
            kv_cache_groups=groups,
        )
        return config, names

    def test_translates_upstream_slots_and_groups(self) -> None:
        config, names = self._mixed_mha_config()

        layout = MHAKVCacheLayout.from_config(config, names)

        assert layout.group_block_sizes == (32, 16)
        assert layout.slot_layers == ((0, 1), (2, 3))
        assert [layer.tensor_index for layer in layout.layers] == [0, 0, 1, 1]
        assert [layer.group_index for layer in layout.layers] == [0, 1, 0, 1]
        assert [layer.sliding_window for layer in layout.layers] == [-1, 1024, -1, 1024]
        assert layout.total_bytes == sum(
            tensor.size for tensor in config.kv_cache_tensors
        )

    def test_allocates_shared_slots_from_layout(self) -> None:
        config, names = self._mixed_mha_config()
        layout = MHAKVCacheLayout.from_config(config, names)

        cache = MetalPagedKVCache.from_layout(layout, mx.bfloat16)

        assert cache.key_caches[0].shape == (3, 32, 4, 512)
        assert cache.key_caches[1].shape == (3, 16, 16, 256)
        assert cache.group_index_for_layer(1) == 1
        assert cache.block_size_for_layer(1) == 16

    def test_rebind_updates_every_layer_sharing_the_slot(self) -> None:
        config, names = self._mixed_mha_config()
        cache = MetalPagedKVCache.from_layout(
            MHAKVCacheLayout.from_config(config, names), mx.bfloat16
        )

        new_key = cache.key_caches[1] + mx.array(1, dtype=mx.bfloat16)
        new_value = cache.value_caches[1] + mx.array(2, dtype=mx.bfloat16)

        cache.replace_layer_cache(1, new_key, new_value)

        assert mx.all(cache.key_caches[1].reshape(-1) == new_key.reshape(-1)).item()
        assert mx.all(cache.value_caches[1].reshape(-1) == new_value.reshape(-1)).item()
        assert mx.all(cache.key_caches[0].reshape(-1) == new_key.reshape(-1)).item()
        assert mx.all(cache.value_caches[0].reshape(-1) == new_value.reshape(-1)).item()

    def test_rejects_packed_upstream_tensors(self) -> None:
        config, names = self._mixed_mha_config()
        config.kv_cache_tensors[0].block_stride = 256

        with pytest.raises(NotImplementedError, match="offset and block_stride"):
            MHAKVCacheLayout.from_config(config, names)
