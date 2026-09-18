# SPDX-License-Identifier: Apache-2.0
"""Physical sharing and native writes using vLLM-created cache views."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)

from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.metal import get_ops


def make_storage(num_blocks=4):
    attention = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float16
    )
    state = MambaSpec(
        block_size=16,
        shapes=((2, 4), (1, 4, 32)),
        dtypes=(torch.float16, torch.float32),
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode="align",
    )
    page = attention.page_size_bytes
    groups = [
        KVCacheGroupSpec(layer_names=["a0", "a1"], kv_cache_spec=attention),
        KVCacheGroupSpec(layer_names=["s0", "s1"], kv_cache_spec=state),
    ]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * num_blocks * page,
                layers=group.layer_names,
                layer_stride=num_blocks * page,
                block_stride=page,
            )
            for group in groups
        ],
        kv_cache_layout="LBNHC",
    )
    return KVCacheStorage(config)


def test_strided_state_writes_share_upstream_backing():
    storage = make_storage()
    conv, recurrent = storage.state_views(["s0", "s1"])
    for views, value in ((conv, 7), (recurrent, 9)):
        rows = mx.full((1, *views[1].shape[1:]), value, dtype=views[1].dtype)
        views[1] = get_ops().gdn_state_scatter(views[1], rows, mx.array([3]))
    mx.eval(storage.buffer)
    np.testing.assert_array_equal(np.array(conv[1][3]), 7)
    np.testing.assert_array_equal(np.array(recurrent[1][3]), 9)
    # Source Torch views see the same bytes, without a reverse conversion.
    page = storage.tensors["s1"][3].flatten()
    assert torch.all(page[:16].view(torch.float16) == 7)
    assert torch.all(page[16:528].view(torch.float32) == 9)
    assert not storage.tensors["s0"].any()
    assert storage.nbytes == 16384


def test_engine_can_limit_usable_blocks_without_resizing_the_backing():
    from dataclasses import replace

    planned = make_storage(num_blocks=5).config
    storage = KVCacheStorage(replace(planned, num_blocks=3))
    assert storage.nbytes == planned.kv_cache_tensors[0].size
    conv, recurrent = storage.state_views(["s0", "s1"])
    assert conv[0].shape[0] == recurrent[1].shape[0] == 3


def test_copy_cycles_snapshot_sources_and_deduplicate_aliases(monkeypatch):
    def no_runtime_bridge(*args, **kwargs):
        raise AssertionError("cache runtime must not cross the Torch/MLX bridge")

    monkeypatch.setattr(mx, "from_dlpack", no_runtime_bridge)
    storage = make_storage()
    monkeypatch.setattr(torch.Tensor, "numpy", no_runtime_bridge)
    storage.tensors["a0"][1].fill_(1)
    storage.tensors["a0"][2].fill_(2)
    storage.tensors["a1"][1].fill_(3)
    storage.tensors["a1"][2].fill_(4)
    storage.copy_blocks([(1, 2), (2, 1)])
    mx.eval(storage.buffer)
    for layer, expected in (("a0", (2, 1)), ("a1", (4, 3))):
        assert torch.all(storage.tensors[layer][1] == expected[0])
        assert torch.all(storage.tensors[layer][2] == expected[1])
    storage.zero_blocks([2])
    mx.eval(storage.buffer)
    assert not storage.tensors["a0"][2].any()
    assert not storage.tensors["s1"][2].any()
    assert torch.all(storage.tensors["a0"][1] == 2)


def test_packed_kv_store_preserves_strides_and_shared_backing():
    storage = make_storage()
    tensor = storage.tensors["a0"].transpose(1, 2)
    key, value = tensor.split(32, dim=-1)
    keys, values = storage.views([key]), storage.views([value])
    new_k, new_v = get_ops().reshape_and_cache(
        mx.full((1, 1, 32), 3, dtype=mx.float16),
        mx.full((1, 1, 32), 5, dtype=mx.float16),
        keys[0],
        values[0],
        mx.array([35], dtype=mx.int64),
    )
    keys[0], values[0] = new_k, new_v
    mx.eval(storage.buffer)
    assert torch.all(key[2, 3] == 3)
    assert torch.all(value[2, 3] == 5)
    assert torch.all(key[2, 2] == 0)


@pytest.mark.parametrize("layout", ["LBHNC", "BLHNC", "LHBNC", "BHLNC"])
def test_attention_rejects_heads_outer_layouts(layout):
    from dataclasses import replace

    from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache

    config = replace(make_storage().config, kv_cache_layout=layout)
    storage = KVCacheStorage(config)
    with pytest.raises(ValueError, match=f"token-major.*{layout}"):
        MetalPagedKVCache.from_upstream(storage, ["a0", "a1"])


def test_budget_above_buffer_limit_can_plan_and_initialize_shared_cache(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups

    from tests.stub_runner import make_cache_config
    from vllm_metal.v1.cache_policy import WorkerCachePlanner

    groups = make_storage().config.kv_cache_groups
    buffer_limit = 16_384
    config = SimpleNamespace(cache_config=make_cache_config(gpu_memory_utilization=0.5))
    runner = SimpleNamespace(
        is_hybrid=True,
        scheduler_memory_reporting_mode=lambda: "paged_attention_layout_budget",
        profile_run=lambda: 0,
        draft_scratch_reserve_bytes=lambda: 0,
    )
    planner = WorkerCachePlanner(
        SimpleNamespace(model_runner=runner, vllm_config=config)
    )
    monkeypatch.setattr(planner, "get_model_memory_usage", lambda: 0)
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {
            "max_recommended_working_set_size": 4 * buffer_limit,
            "max_buffer_length": buffer_limit,
        },
    )

    budget = planner.determine_available_memory()
    planned = get_kv_cache_config_from_groups(config, groups, budget)
    storage = KVCacheStorage(planned)

    assert budget == buffer_limit
    assert 0 < storage.nbytes <= buffer_limit
    assert storage.state_views(["s0", "s1"])[0][0].shape[0] == planned.num_blocks
