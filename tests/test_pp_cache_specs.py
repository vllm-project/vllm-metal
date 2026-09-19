# SPDX-License-Identifier: Apache-2.0
"""Pipeline stages retain global cache names and local array indices."""

from __future__ import annotations

import pytest
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from tests.stub_runner import make_gemma4_mixed_attention_runner
from vllm_metal.attention.caches.attention_layout import AttentionKVCacheLayout
from vllm_metal.config import MetalConfig


def _stage(start: int, end: int, *, mixed: bool = True):
    """Build generic explicit-shape metadata to exercise deferred KV grouping."""
    runner = make_gemma4_mixed_attention_runner(
        num_layers=end - start,
        sliding_kv_heads=8,
        full_kv_heads=8,
        max_model_len=256,
        original_max_model_len=256,
        max_in_flight_tokens=32,
    )
    runner._pp_layer_start = start
    runner.head_dim = 64
    runner.kv_heads_per_layer = [8] * (end - start)
    runner.head_dim_per_layer = [64] * (end - start)
    runner.sliding_window_per_layer = [
        128 if mixed and index % 2 == 0 else -1 for index in range(start, end)
    ]
    return runner


@pytest.mark.parametrize("split,total", [(3, 8), (13, 36), (14, 36)])
def test_deferred_mixed_pipeline_cache_specs_merge_and_map_locally(
    monkeypatch, split: int, total: int
) -> None:
    """vLLM must see every layer once, including odd stage boundaries."""
    monkeypatch.setattr(
        "vllm_metal.v1.cache_policy.get_config",
        lambda: MetalConfig(mlx_device="gpu", turboquant=False),
    )
    stages = [_stage(0, split), _stage(split, total)]
    specs = [stage.get_kv_cache_spec() for stage in stages]

    # Exercise the actual cross-worker merge/group/projection, which rejects
    # repeated stage-local names when their full/sliding attention kinds differ.
    configs = get_kv_cache_configs(
        stages[0].vllm_config, specs, [32 * 1024**2, 48 * 1024**2]
    )

    assert len(set(specs[0]) | set(specs[1])) == total
    assert set(specs[0]).isdisjoint(specs[1])
    assert configs[0].num_blocks == configs[1].num_blocks

    for stage, spec, config, start, end in zip(
        stages, specs, configs, (0, split), (split, total), strict=True
    ):
        names = tuple(f"layers.{index}.self_attn" for index in range(start, end))
        assert tuple(spec) == names
        assert stage._cache_policy._attention_layer_names() == names
        layout = AttentionKVCacheLayout.from_config(config, names)
        assert len(layout.layers) == end - start
        assert sorted(index for slot in layout.slot_layers for index in slot) == list(
            range(end - start)
        )
        for local_index, name in enumerate(names):
            window = 128 if (start + local_index) % 2 == 0 else -1
            expected_type = SlidingWindowSpec if window >= 0 else FullAttentionSpec
            assert type(spec[name]) is expected_type
            assert layout.layers[local_index].sliding_window == window
            assert layout.layers[local_index].num_kv_heads == 8
            assert layout.layers[local_index].head_dim == 64
        groups = stage._cache_policy._scheduler_group_indices_for_layers(config, names)
        assert groups == tuple(
            dict.fromkeys(layer.group_index for layer in layout.layers)
        )


def test_uniform_pipeline_cache_names_are_distinct(monkeypatch) -> None:
    """Uniform attention also describes eight global layers, not five duplicates."""
    monkeypatch.setattr(
        "vllm_metal.v1.cache_policy.get_config",
        lambda: MetalConfig(mlx_device="gpu", turboquant=False),
    )
    stages = [_stage(0, 3, mixed=False), _stage(3, 8, mixed=False)]
    specs = [stage.get_kv_cache_spec() for stage in stages]
    configs = get_kv_cache_configs(
        stages[0].vllm_config, specs, [32 * 1024**2, 48 * 1024**2]
    )
    global_names = [name for spec in specs for name in spec]
    assert len(set(global_names)) == 8
    assert global_names == [f"layers.{index}.self_attn" for index in range(8)]
    for stage, config, spec in zip(stages, configs, specs, strict=True):
        names = stage._cache_policy._attention_layer_names()
        assert set(names) == set(spec)
        assert set(config.kv_cache_groups[0].layer_names) == set(names)


@pytest.mark.parametrize("split,total", [(3, 8), (13, 36), (14, 36)])
def test_gpt_oss_dense_pipeline_cache_preserves_local_windows(
    monkeypatch, split: int, total: int
) -> None:
    """GPT-OSS uses dense cache capacity while kernels enforce local windows."""
    monkeypatch.setattr(
        "vllm_metal.v1.cache_policy.get_config",
        lambda: MetalConfig(mlx_device="gpu", turboquant=False),
    )
    stages = [_stage(0, split), _stage(split, total)]
    for stage in stages:
        # Actual GPT-OSS metadata has scalar KV dimensions. In particular, it
        # does not use the explicit-shape/deferred allocation path above.
        stage.kv_heads_per_layer = None
        stage.head_dim_per_layer = None
        stage.model_args = {"model_type": "gpt_oss", "sliding_window": 128}
        assert stage.scheduler_memory_reporting_mode() == "paged_attention_capacity"
        stage.validate_paged_attention_support()

    specs = [stage.get_kv_cache_spec() for stage in stages]
    pool_blocks = 32
    available = [pool_blocks * stage.get_cache_block_size_bytes() for stage in stages]
    configs = get_kv_cache_configs(stages[0].vllm_config, specs, available)

    assert set(specs[0]).isdisjoint(specs[1])
    assert set(specs[0]) | set(specs[1]) == {
        f"layers.{index}.self_attn" for index in range(total)
    }
    for stage, spec, config, start, end in zip(
        stages, specs, configs, (0, split), (split, total), strict=True
    ):
        names = tuple(f"layers.{index}.self_attn" for index in range(start, end))
        assert tuple(spec) == names
        assert all(
            type(layer_spec) is FullAttentionSpec for layer_spec in spec.values()
        )
        assert config.num_blocks == pool_blocks
        assert len(config.kv_cache_groups) == 1
        assert tuple(config.kv_cache_groups[0].layer_names) == names

        # Use the real backend factory/allocation and engine config adoption.
        # FullAttentionSpec reserves capacity; it must not replace the window
        # metadata used by each local attention kernel.
        runtime = stage.build_paged_attention_runtime(
            block_size=stage.cache_config.block_size
        )
        runtime.initialize(pool_blocks)
        stage.install_paged_attention_runtime(
            runtime, block_size=stage.cache_config.block_size
        )
        stage.initialize_kv_cache(config)
        cache = runtime.kv_cache
        assert stage.paged_attention_runtime is runtime
        assert len(cache.key_caches) == end - start
        assert stage._paged_scheduler_group_indices == (0,)
        expected_windows = [
            128 if index % 2 == 0 else -1 for index in range(start, end)
        ]
        assert cache.sliding_window_per_layer == expected_windows
        for local_index in range(end - start):
            assert cache.group_index_for_layer(local_index) == 0
            assert cache.key_caches[local_index].shape == (pool_blocks, 16, 8, 64)
