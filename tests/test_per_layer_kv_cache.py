# SPDX-License-Identifier: Apache-2.0
"""Tests for per-layer KV cache shape support."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mlx.core as mx
import pytest
import torch
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_configs,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

from tests.stub_runner import make_gemma4_mixed_mha_runner, make_stub_runner
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.mha_layout import MHAKVCacheLayout
from vllm_metal.attention.impls.sdpa_wrapper import SDPAPagedAttentionWrapper
from vllm_metal.attention.runtime.mha import (
    MHAPagedAttentionRuntime,
)
from vllm_metal.config import (
    AUTO_MEMORY_FRACTION,
    PAGED_ATTENTION_MIN_BLOCKS,
    MetalConfig,
)
from vllm_metal.v1.cache_policy import WorkerCachePlanner


def vllm_config_for_kv_grouping() -> VllmConfig:
    return VllmConfig()


def config_from_vllm_groups(
    groups: list[KVCacheGroupSpec], num_blocks: int
) -> KVCacheConfig:
    if len(groups) == 1:
        available = groups[0].kv_cache_spec.page_size_bytes * num_blocks
    else:
        group_size = max(len(group.layer_names) for group in groups)
        available = groups[0].kv_cache_spec.page_size_bytes * num_blocks * group_size
    return get_kv_cache_config_from_groups(
        vllm_config_for_kv_grouping(), groups, available
    )


def merged_full_mha_config() -> tuple[KVCacheConfig, tuple[str, ...]]:
    names = tuple(f"layers.{index}.self_attn" for index in range(4))
    specs = {
        names[0]: FullAttentionSpec(
            block_size=16,
            num_kv_heads=16,
            head_size=256,
            dtype=torch.bfloat16,
        ),
        names[1]: FullAttentionSpec(
            block_size=16,
            num_kv_heads=16,
            head_size=256,
            dtype=torch.bfloat16,
        ),
        names[2]: FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=512,
            dtype=torch.bfloat16,
        ),
        names[3]: FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=512,
            dtype=torch.bfloat16,
        ),
    }
    groups = get_kv_cache_groups(vllm_config_for_kv_grouping(), specs)
    assert len(groups) == 1
    return config_from_vllm_groups(groups, 3), names


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

    def test_heterogeneous_dense_log_reports_per_layer_bytes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """E4B-shaped unique layers log 2.8 MB, not the old all-512 4.7 MB."""
        head_dims = [512 if (index + 1) % 6 == 0 else 256 for index in range(24)]
        with caplog.at_level(logging.INFO):
            MetalPagedKVCache(
                num_layers=24,
                num_kv_heads=2,
                head_dim=512,
                num_blocks=3,
                block_size=16,
                dtype=mx.bfloat16,
                kv_heads_per_layer=[2] * 24,
                head_dim_per_layer=head_dims,
            )
        messages = [
            record.message
            for record in caplog.records
            if record.message.startswith("KV cache:")
        ]
        assert len(messages) == 1, messages
        assert "KV cache: 2.8 MB" in messages[0]
        assert "4.7 MB" not in messages[0]


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
                mlx_device="gpu",
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

    def gemma4_mixed_runner(
        self,
        *,
        num_layers: int,
        sliding_kv_heads: int,
        full_kv_heads: int,
        disable_hybrid_manager: bool = False,
        num_gpu_blocks_override: int | None = None,
    ):
        return make_gemma4_mixed_mha_runner(
            num_layers=num_layers,
            sliding_kv_heads=sliding_kv_heads,
            full_kv_heads=full_kv_heads,
            disable_hybrid_manager=disable_hybrid_manager,
            num_gpu_blocks_override=num_gpu_blocks_override,
        )

    def _mixed_mha_config(self) -> tuple[KVCacheConfig, tuple[str, ...]]:
        names = tuple(f"layers.{index}.self_attn" for index in range(4))
        specs = {
            names[0]: FullAttentionSpec(
                block_size=32,
                num_kv_heads=4,
                head_size=512,
                dtype=torch.bfloat16,
            ),
            names[1]: SlidingWindowSpec(
                block_size=16,
                num_kv_heads=16,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
            ),
            names[2]: FullAttentionSpec(
                block_size=32,
                num_kv_heads=4,
                head_size=512,
                dtype=torch.bfloat16,
            ),
            names[3]: SlidingWindowSpec(
                block_size=16,
                num_kv_heads=16,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
            ),
        }
        groups = get_kv_cache_groups(vllm_config_for_kv_grouping(), specs)
        assert len(groups) == 2
        return config_from_vllm_groups(groups, 3), names

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

    def test_cache_policy_adopts_engine_layout(self, monkeypatch) -> None:
        config, _ = self._mixed_mha_config()
        runner = make_stub_runner(
            num_layers=4,
            num_kv_cache_layers=4,
            num_kv_heads=4,
            head_dim=512,
            kv_cache_dtype=mx.bfloat16,
            cache_config=SimpleNamespace(block_size=32),
        )
        backend = MHAPagedAttentionRuntime(
            num_layers=4,
            num_kv_heads=4,
            head_dim=512,
            block_size=32,
            dtype=mx.bfloat16,
        )
        backend.initialize(num_blocks=config.num_blocks)
        runner.install_paged_attention_runtime(backend, block_size=32)

        patched: dict[str, object] = {}

        def patch_model(model):
            patched["model"] = model
            return 4

        monkeypatch.setattr(backend, "patch_model", patch_model)
        monkeypatch.setattr(
            "vllm_metal.v1.cache_policy.get_config",
            lambda: MetalConfig(
                memory_fraction=AUTO_MEMORY_FRACTION,
                mlx_device="gpu",
                turboquant=False,
            ),
        )

        runner.initialize_kv_cache(config)

        assert patched["model"] is runner.model
        assert backend.kv_cache.group_index_for_layer(1) == 1
        assert backend.kv_cache.block_size_for_layer(1) == 16
        assert runner._paged_scheduler_group_indices == (0, 1)
        assert runner._paged_group_block_sizes == (32, 16)

    def test_cache_policy_keeps_merged_single_group_on_existing_runtime(
        self, monkeypatch
    ) -> None:
        config, _ = merged_full_mha_config()
        runner = make_stub_runner(
            num_layers=4,
            num_kv_cache_layers=4,
            num_kv_heads=16,
            head_dim=256,
            kv_cache_dtype=mx.bfloat16,
            cache_config=SimpleNamespace(block_size=16),
            kv_heads_per_layer=[16, 16, 4, 4],
            head_dim_per_layer=[256, 256, 512, 512],
        )
        backend = MHAPagedAttentionRuntime(
            num_layers=4,
            num_kv_heads=16,
            head_dim=256,
            block_size=16,
            dtype=mx.bfloat16,
            kv_heads_per_layer=[16, 16, 4, 4],
            head_dim_per_layer=[256, 256, 512, 512],
        )
        backend.initialize(num_blocks=config.num_blocks)
        original_cache = backend.kv_cache
        runner.install_paged_attention_runtime(backend, block_size=16)

        monkeypatch.setattr(
            "vllm_metal.v1.cache_policy.get_config",
            lambda: MetalConfig(
                memory_fraction=AUTO_MEMORY_FRACTION,
                mlx_device="gpu",
                turboquant=False,
            ),
        )
        monkeypatch.setattr(
            backend,
            "adopt_layout",
            lambda layout: pytest.fail("single scheduler group should not reallocate"),
        )

        runner.initialize_kv_cache(config)

        assert backend.kv_cache is original_cache
        assert runner._paged_scheduler_group_indices == (0,)
        assert runner._paged_group_block_sizes == (16,)

    @pytest.mark.parametrize(
        ("num_layers", "sliding_kv_heads", "full_kv_heads", "expected_slots"),
        (
            (60, 16, 4, 10),
            (30, 8, 2, 5),
        ),
    )
    def test_cache_policy_initializes_gemma4_grouped_layout_from_budget(
        self,
        monkeypatch,
        num_layers: int,
        sliding_kv_heads: int,
        full_kv_heads: int,
        expected_slots: int,
    ) -> None:
        runner = self.gemma4_mixed_runner(
            num_layers=num_layers,
            sliding_kv_heads=sliding_kv_heads,
            full_kv_heads=full_kv_heads,
        )
        metal_config = MetalConfig(
            memory_fraction=1.0,
            mlx_device="gpu",
            turboquant=False,
        )
        monkeypatch.setattr(
            "vllm_metal.v1.cache_policy.get_config",
            lambda: metal_config,
        )
        reported_dense_blocks = 2
        dense_block_bytes = runner.get_cache_block_size_bytes()
        worker = SimpleNamespace(
            model_runner=runner,
            metal_config=metal_config,
            vllm_config=runner.vllm_config,
            get_cache_block_size_bytes=runner.get_cache_block_size_bytes,
        )
        planner = WorkerCachePlanner(worker)
        monkeypatch.setattr(runner, "profile_run", lambda: 0)
        monkeypatch.setattr(planner, "get_model_memory_usage", lambda: 0)
        monkeypatch.setattr(
            planner,
            "_metal_limit_bytes",
            lambda: dense_block_bytes * reported_dense_blocks,
        )

        available = planner.determine_available_memory()

        specs = runner.get_kv_cache_spec()
        config = get_kv_cache_configs(runner.vllm_config, [specs], [available])[0]

        assert runner.paged_attention_runtime is None
        assert available // dense_block_bytes == reported_dense_blocks
        assert type(specs["layers.0.self_attn"]) is SlidingWindowSpec
        assert type(specs["layers.5.self_attn"]) is FullAttentionSpec
        assert len(config.kv_cache_groups) == 6
        assert config.num_blocks > reported_dense_blocks
        assert len(config.kv_cache_tensors) == expected_slots

        runner.initialize_kv_cache(config)

        backend = runner.paged_attention_runtime
        assert isinstance(backend, MHAPagedAttentionRuntime)
        assert backend.num_blocks() == config.num_blocks
        assert runner._paged_scheduler_group_indices == (0, 1, 2, 3, 4, 5)
        assert runner._paged_group_block_sizes == (16, 16, 16, 16, 16, 32)
        assert backend.kv_cache.group_index_for_layer(5) == 5
        assert backend.kv_cache.block_size_for_layer(5) == 32
        assert backend.kv_cache.sliding_window_per_layer[0] == 1024
        assert backend.kv_cache.sliding_window_per_layer[5] == -1
        assert len(backend.kv_cache.key_caches) == num_layers
        assert all(
            isinstance(layer.self_attn, SDPAPagedAttentionWrapper)
            for layer in runner.model.layers
        )

    def test_disabled_hybrid_manager_stays_on_dense_path(self, monkeypatch) -> None:
        runner = self.gemma4_mixed_runner(
            num_layers=60,
            sliding_kv_heads=16,
            full_kv_heads=4,
            disable_hybrid_manager=True,
        )
        metal_config = MetalConfig(
            memory_fraction=1.0,
            mlx_device="gpu",
            turboquant=False,
        )
        monkeypatch.setattr(
            "vllm_metal.v1.cache_policy.get_config",
            lambda: metal_config,
        )
        dense_block_bytes = runner.get_cache_block_size_bytes()
        worker = SimpleNamespace(
            model_runner=runner,
            metal_config=metal_config,
            vllm_config=runner.vllm_config,
            get_cache_block_size_bytes=runner.get_cache_block_size_bytes,
        )
        planner = WorkerCachePlanner(worker)
        monkeypatch.setattr(runner, "profile_run", lambda: 0)
        monkeypatch.setattr(planner, "get_model_memory_usage", lambda: 0)
        monkeypatch.setattr(
            planner,
            "_metal_limit_bytes",
            lambda: dense_block_bytes * PAGED_ATTENTION_MIN_BLOCKS,
        )

        available = planner.determine_available_memory()
        specs = runner.get_kv_cache_spec()
        config = get_kv_cache_configs(runner.vllm_config, [specs], [available])[0]
        runner.initialize_kv_cache(config)

        assert all(type(spec) is FullAttentionSpec for spec in specs.values())
        assert runner._paged_scheduler_group_indices == (0,)
        assert runner.paged_attention_runtime is not None
        assert runner.paged_attention_runtime.num_blocks() == config.num_blocks

    def test_block_override_uses_existing_capacity_guard(self, monkeypatch) -> None:
        runner = self.gemma4_mixed_runner(
            num_layers=60,
            sliding_kv_heads=16,
            full_kv_heads=4,
            num_gpu_blocks_override=100,
        )
        metal_config = MetalConfig(
            memory_fraction=1.0,
            mlx_device="gpu",
            turboquant=False,
        )
        monkeypatch.setattr(
            "vllm_metal.v1.cache_policy.get_config",
            lambda: metal_config,
        )
        dense_block_bytes = runner.get_cache_block_size_bytes()
        worker = SimpleNamespace(
            model_runner=runner,
            metal_config=metal_config,
            vllm_config=runner.vllm_config,
            get_cache_block_size_bytes=runner.get_cache_block_size_bytes,
        )
        planner = WorkerCachePlanner(worker)
        monkeypatch.setattr(runner, "profile_run", lambda: 0)
        monkeypatch.setattr(planner, "get_model_memory_usage", lambda: 0)
        monkeypatch.setattr(
            planner,
            "_metal_limit_bytes",
            lambda: dense_block_bytes * PAGED_ATTENTION_MIN_BLOCKS,
        )

        available = planner.determine_available_memory()
        specs = runner.get_kv_cache_spec()
        config = get_kv_cache_configs(runner.vllm_config, [specs], [available])[0]

        assert config.num_blocks == 100
        with pytest.raises(ValueError, match="Engine KV cache config requests 100"):
            runner.initialize_kv_cache(config)

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
