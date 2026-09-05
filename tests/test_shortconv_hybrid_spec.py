# SPDX-License-Identifier: Apache-2.0
"""LFM2 family routing, scheduler layout, and physical cache integration."""

from __future__ import annotations

from dataclasses import asdict
from importlib import import_module
from types import SimpleNamespace

import mlx.core as mx
import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from tests.stub_runner import make_stub_runner
from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
from vllm_metal.config import MetalConfig
from vllm_metal.v1.model_lifecycle import ModelLifecycle

BLOCK_SIZE = 16
NUM_BLOCKS = 12
ATTENTION_INDICES = (1, 4)
STATE_INDICES = (0, 2, 3, 5)


@pytest.fixture(params=["lfm2", "lfm2_moe"])
def lfm_model(request):
    module = import_module(f"mlx_lm.models.{request.param}")
    args = {
        "model_type": request.param,
        "vocab_size": 32,
        "hidden_size": 64,
        "num_hidden_layers": 6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "norm_eps": 1e-5,
        "conv_bias": False,
        "conv_L_cache": 3,
        # The nonalternating layout catches implicit interval assumptions.
        "layer_types": [
            "conv",
            "full_attention",
            "conv",
            "conv",
            "full_attention",
            "conv",
        ],
    }
    if request.param == "lfm2":
        args.update(
            block_dim=64,
            block_ff_dim=128,
            block_multiple_of=16,
            block_ffn_dim_multiplier=1.0,
            block_auto_adjust_ff_dim=False,
        )
    else:
        args.update(
            intermediate_size=128,
            moe_intermediate_size=64,
            num_experts=2,
            num_experts_per_tok=1,
            norm_topk_prob=True,
            use_expert_bias=False,
            num_dense_layers=1,
        )
    return module.Model(module.ModelArgs(**args))


@pytest.fixture(autouse=True)
def _ordinary_paged_cache(monkeypatch):
    monkeypatch.setattr(
        "vllm_metal.v1.cache_policy.get_config",
        lambda: MetalConfig(
            memory_fraction=-1.0,
            mlx_device="gpu",
            use_paged_attention=True,
            turboquant=False,
        ),
    )


def _runner(model, *, mode="align"):
    runner = make_stub_runner(
        model=model,
        model_args=asdict(model.args),
        is_hybrid=True,
        kv_cache_dtype=mx.bfloat16,
        cache_config=SimpleNamespace(
            block_size=BLOCK_SIZE,
            mamba_block_size=BLOCK_SIZE if mode == "align" else 128,
            mamba_page_size_padded=None,
            mamba_cache_mode=mode,
            num_gpu_blocks_override=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=3,
            disable_hybrid_kv_cache_manager=False,
        ),
    )
    ModelLifecycle(runner, runner._model_adapter).resolve_model_dims()
    return runner


def _runtime(runner):
    runtime = runner.build_paged_attention_runtime(block_size=BLOCK_SIZE)
    assert isinstance(runtime, HybridPagedAttentionRuntime)
    runtime.initialize(NUM_BLOCKS)
    runner.install_paged_attention_runtime(runtime, block_size=BLOCK_SIZE)
    return runtime


@pytest.mark.parametrize("mode", ["none", "align"])
def test_scheduler_spec_bills_only_the_convolution_tail(lfm_model, mode):
    runner = _runner(lfm_model, mode=mode)
    specs = runner.get_kv_cache_spec()
    expected_names = {
        *(f"layers.{i}.conv" for i in STATE_INDICES),
        *(f"layers.{i}.self_attn" for i in ATTENTION_INDICES),
    }
    assert set(specs) == expected_names
    for index in STATE_INDICES:
        spec = specs[f"layers.{index}.conv"]
        assert isinstance(spec, MambaSpec)
        assert spec.shapes == ((2, 64),)
        assert spec.dtypes == (torch.bfloat16,)
        assert spec.mamba_type == MambaAttentionBackendEnum.SHORT_CONV
        assert spec.mamba_cache_mode == mode
        assert spec.block_size == (BLOCK_SIZE if mode == "align" else 128)
        assert spec.page_size_bytes == 256
    for index in ATTENTION_INDICES:
        spec = specs[f"layers.{index}.self_attn"]
        assert isinstance(spec, FullAttentionSpec)
        assert spec.page_size_bytes == 2048
    assert runner.linear_cache_bytes_per_slot() == 4 * 256
    assert runner.get_cache_block_size_bytes() == 2 * 2048


def test_upstream_scheduler_groups_adopt_conv_names_and_shared_state_pools(lfm_model):
    """Round-trip real vLLM grouping, including its shared_by physical layout."""
    runner = _runner(lfm_model)
    runtime = _runtime(runner)
    engine_config = SimpleNamespace(
        cache_config=runner.cache_config,
        scheduler_config=runner.scheduler_config,
        kv_transfer_config=None,
    )
    groups = get_kv_cache_groups(engine_config, runner.get_kv_cache_spec())
    scheduler_cache = get_kv_cache_config_from_groups(
        engine_config,
        groups,
        available_memory=NUM_BLOCKS * runner.get_cache_block_size_bytes(),
    )
    assert scheduler_cache.num_blocks == NUM_BLOCKS
    runner.initialize_kv_cache(scheduler_cache)

    expected_state_groups = tuple(
        i
        for i, group in enumerate(groups)
        if isinstance(group.kv_cache_spec, MambaSpec)
    )
    expected_attention_groups = tuple(
        i
        for i, group in enumerate(groups)
        if isinstance(group.kv_cache_spec, FullAttentionSpec)
    )
    assert len(expected_state_groups) == 2
    assert len(expected_attention_groups) == 1
    assert runtime.state_scheduler_group_indices() == expected_state_groups
    assert runtime.kv_scheduler_group_indices() == expected_attention_groups
    assert runner._paged_state_group_indices == expected_state_groups

    cache = runtime.state_cache
    assert cache.allocated_seqs == 0
    cache.ensure_capacity(NUM_BLOCKS)
    assert cache.num_state_pools == 2
    for tensor in scheduler_cache.kv_cache_tensors:
        indices = [
            STATE_INDICES.index(int(name.split(".")[1]))
            for name in tensor.shared_by
            if name.endswith(".conv")
        ]
        assert len(indices) == 2
        assert cache.conv_states[indices[0]] is cache.conv_states[indices[1]]
        assert cache.layer_group_ordinal(indices[0]) != cache.layer_group_ordinal(
            indices[1]
        )
    assert (
        sum(array.nbytes for array in cache.updated_state_arrays())
        == NUM_BLOCKS * 2 * 256
    )
    assert runner._cache_policy.hybrid_align_state_bytes_per_block() == 2 * 256
    assert runner._cache_policy.hybrid_align_growth_bytes_per_block() == 256
