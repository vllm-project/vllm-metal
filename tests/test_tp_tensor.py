from types import SimpleNamespace

import mlx.core as mx
import pytest

from tests.test_pp_gpt_oss import _Group, _model
from vllm_metal.distributed.tensor import (
    TensorGroup,
    apply_tensor_shard,
    validate_tensor_config,
)


def config(**overrides):
    values = {
        "parallel_config": SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="ray",
            enable_expert_parallel=False,
        ),
        "model_config": SimpleNamespace(
            hf_config=SimpleNamespace(model_type="gpt_oss"),
            quantization=None,
            multimodal_config=None,
            runner_type="generate",
        ),
        "scheduler_config": SimpleNamespace(async_scheduling=False),
        "speculative_config": None,
        "lora_config": None,
        "additional_config": {
            "tensor_transport": {
                "backend": "jaccl",
                "device_matrix": [
                    [None, ["rdma_en1", "rdma_en2"]],
                    [["rdma_en2", "rdma_en1"], None],
                ],
            }
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_only_explicit_gpt_oss_jaccl_tp2_is_admitted():
    validate_tensor_config(config())
    c = config()
    c.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(NotImplementedError):
        validate_tensor_config(c)
    c = config()
    c.parallel_config.data_parallel_size = 2
    with pytest.raises(NotImplementedError):
        validate_tensor_config(c)
    c = config()
    c.model_config.hf_config.model_type = "llama"
    with pytest.raises(NotImplementedError):
        validate_tensor_config(c)
    c = config()
    c.scheduler_config.async_scheduling = True
    with pytest.raises(NotImplementedError):
        validate_tensor_config(c)
    c = config()
    c.additional_config = {}
    with pytest.raises(ValueError, match="JACCL"):
        validate_tensor_config(c)


@pytest.mark.parametrize("rank", [0, 1])
def test_native_shard_retains_all_layers_and_halves_heads_and_cache(rank):
    with mx.stream(mx.cpu):
        model = _model(quantized=True)
        tp = TensorGroup(_Group(rank, 2))
        apply_tensor_shard(model, tp)
        assert len(model.layers) == 4
        for layer in model.layers:
            assert layer.self_attn.num_attention_heads == 4
            assert layer.self_attn.num_key_value_heads == 2
            assert layer.self_attn.sinks.shape == (4,)
            assert layer.mlp.sharding_group is tp.group
        assert model.layers[0].self_attn.q_proj.weight.shape[0] == 32
        assert model.layers[0].mlp.experts.gate_proj.weight.shape[-2] == 32


def test_token_sync_uses_rank_zero_choices_on_all_ranks(monkeypatch):
    calls = []

    def reduce(value, *, group, stream):
        calls.append((value.tolist(), stream))
        return mx.array([17, 29, 31], dtype=mx.int32)

    monkeypatch.setattr(mx.distributed, "all_sum", reduce)
    assert TensorGroup(_Group(0, 2)).synchronize_tokens([17, 29, 31]) == [17, 29, 31]
    assert TensorGroup(_Group(1, 2)).synchronize_tokens([2, 3, 4]) == [17, 29, 31]
    assert calls[0][0] == [17, 29, 31]
    assert calls[1][0] == [0, 0, 0]
    assert all(stream == mx.cpu for _, stream in calls)


def test_tensor_forward_finishes_collectives_before_next_prefill_chunk(monkeypatch):
    from vllm_metal.v1.model_runner import MetalModelRunner

    calls = []
    runner = SimpleNamespace(tp=object(), _paged_attention_runtime=None)
    monkeypatch.setattr(mx, "eval", lambda *values: calls.append(values))
    monkeypatch.setattr(
        mx, "async_eval", lambda *_: pytest.fail("TP must finish before next chunk")
    )
    output = mx.array([1, 2])
    MetalModelRunner._submit_paged_forward_outputs(runner, output)
    assert calls == [(output,)]
