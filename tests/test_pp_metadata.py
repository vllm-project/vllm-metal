# SPDX-License-Identifier: Apache-2.0
"""Stage-local GPT-OSS attention metadata and scheduler cache specifications."""

from types import SimpleNamespace

import pytest
import torch
from mlx_lm.models.gpt_oss import Model, ModelArgs
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from tests.stub_runner import make_stub_runner
from vllm_metal.distributed.pipeline import PipelineGroup
from vllm_metal.v1.model_lifecycle import ModelLifecycle


def runner_for_stage(rank: int, size: int = 2):
    args = ModelArgs(
        num_hidden_layers=6,
        hidden_size=32,
        intermediate_size=64,
        num_local_experts=2,
        num_experts_per_tok=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=64,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"] * 3,
    )
    pp = PipelineGroup(SimpleNamespace(rank=lambda: rank, size=lambda: size))
    runner = make_stub_runner(model=Model(args), model_args=vars(args), pp=pp)
    return runner, ModelLifecycle(runner, runner._model_adapter)


@pytest.mark.parametrize("partition", ["1,5", "3,3", "4,2"])
@pytest.mark.parametrize("rank", [0, 1])
def test_gpt_oss_stage_owns_matching_attention_and_cache_specs(
    monkeypatch, partition, rank
):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    runner, lifecycle = runner_for_stage(rank)
    global_windows = [4, -1] * 3
    counts = [int(value) for value in partition.split(",")]
    start = sum(counts[:rank])
    end = start + counts[rank]

    lifecycle.resolve_model_dims()
    runner.apply_pipeline_split(runner.pp)

    expected = global_windows[start:end]
    assert runner.num_layers == runner.num_kv_cache_layers == counts[rank]
    assert runner._pp_layer_start == start
    assert runner.sliding_window_per_layer == expected
    assert runner.model_args["num_hidden_layers"] == 6
    assert len(runner.model_args["layer_types"]) == 6
    for index, window in enumerate(expected):
        spec = runner._cache_policy._build_attention_spec(
            index, 16, 1, 16, torch.float32
        )
        if window == -1:
            assert type(spec) is FullAttentionSpec
        else:
            assert isinstance(spec, SlidingWindowSpec)
            assert spec.sliding_window == window


def test_single_stage_preserves_gpt_oss_metadata():
    runner, lifecycle = runner_for_stage(0, size=1)
    lifecycle.resolve_model_dims()
    runner.apply_pipeline_split(runner.pp)
    assert runner.num_layers == 6
    assert runner.sliding_window_per_layer == [4, -1] * 3


@pytest.mark.parametrize("default_layout", [None, []])
def test_gpt_oss_default_attention_layout_comes_from_constructed_model(default_layout):
    runner, lifecycle = runner_for_stage(0)
    runner.model.args.layer_types = default_layout
    # MLX-LM supplies its default alternating layout on the built backbone.
    extracted = lifecycle._extract_model_args(runner.model, False)
    assert extracted["layer_types"] == ["sliding_attention", "full_attention"] * 3
    assert runner.model.args.layer_types is default_layout


def test_gpt_oss_does_not_enable_heterogeneous_kv_shapes():
    runner, lifecycle = runner_for_stage(0)
    runner.model_args["global_head_dim"] = 32
    with pytest.raises(NotImplementedError, match="non-uniform per-layer KV"):
        lifecycle.resolve_model_dims()
