# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel sharding: partition parsing, weight slicing, masked routing."""

import mlx.core as mx
import pytest

from vllm_metal.distributed.experts import expert_partition


def test_partition_defaults_to_even_and_validates_overrides(monkeypatch):
    assert expert_partition(2, 128) == [64, 64]
    monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", "56,72")
    assert expert_partition(2, 128) == [56, 72]
    for bad in ("64", "63,64", "0,128", "64,64,0", "a,b", ""):
        monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", bad)
        with pytest.raises(ValueError, match="VLLM_METAL_EXPERT_PARTITION"):
            expert_partition(2, 128)


def test_partition_rejects_uneven_default(monkeypatch):
    monkeypatch.delenv("VLLM_METAL_EXPERT_PARTITION", raising=False)
    with pytest.raises(ValueError, match="divide evenly"):
        expert_partition(2, 127)


def _config(**overrides):
    from types import SimpleNamespace

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


def test_expert_parallel_is_admitted_on_the_tp2_whitelist():
    from vllm_metal.distributed.tensor import validate_tensor_config

    cfg = _config()
    cfg.parallel_config.enable_expert_parallel = True
    validate_tensor_config(cfg)  # admitted
    cfg = _config()
    cfg.parallel_config.enable_expert_parallel = True
    cfg.scheduler_config.async_scheduling = True
    with pytest.raises(NotImplementedError):
        validate_tensor_config(cfg)


class _Group:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _ep_model(*, quantized: bool = False):
    from mlx_lm.models import gpt_oss

    model = gpt_oss.Model(
        gpt_oss.ModelArgs(
            num_hidden_layers=4,
            num_local_experts=4,
            num_experts_per_tok=2,
            vocab_size=64,
            hidden_size=64,
            intermediate_size=64,
            head_dim=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"] * 2,
        )
    )
    for layer in model.layers:
        layer.self_attn.sinks = mx.linspace(-1, 1, 8)
    if quantized:
        import mlx.nn as nn

        model.set_dtype(mx.float16)

        def quantization(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if ".experts." in path:
                return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            return {"group_size": 32, "bits": 8, "mode": "affine"}

        nn.quantize(model, class_predicate=quantization)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize(
    "rank,partition,expected", [(0, "2,2", 2), (1, "2,2", 2), (1, "1,3", 3)]
)
@pytest.mark.parametrize("quantized", [False, True], ids=["float32", "mxfp4-q8"])
def test_expert_shard_slices_experts_by_count_not_width(
    monkeypatch, rank, partition, expected, quantized
):
    from vllm_metal.distributed.experts import apply_expert_shard

    monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", partition)
    with mx.stream(mx.cpu):
        model = _ep_model(quantized=quantized)
        apply_expert_shard(model, _Group(rank, 2))
        start = 0 if rank == 0 else int(partition.split(",")[0])
        for layer in model.layers:
            attn = layer.self_attn
            assert attn.num_attention_heads == 4
            assert attn.num_key_value_heads == 2
            assert attn.sinks.shape == (4,)
            assert attn.q_proj.weight.shape[0] == 32
            experts = layer.mlp.experts
            for proj in (experts.gate_proj, experts.up_proj, experts.down_proj):
                assert proj.weight.shape[0] == expected
                if quantized:
                    assert proj.scales.shape[0] == expected
                assert proj.weight.shape[1] == 64  # width NOT halved
            assert layer.mlp.router.weight.shape[0] == 4  # router stays full
            assert layer.mlp.sharding_group is not None
            assert layer.mlp.expert_partition == (start, start + expected)


def test_expert_shard_rejects_cross_rank_partition_disagreement(monkeypatch):
    """The all_gather guard must fail loudly when ranks disagree, instead of
    silently serving experts owned by neither rank."""
    from vllm_metal.distributed.experts import apply_expert_shard

    monkeypatch.setattr(mx.distributed, "Group", _Group)
    monkeypatch.setattr(
        mx.distributed,
        "all_gather",
        lambda value, *, group=None, stream=None: mx.array([0, 2, 0, 2]),
    )
    monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", "2,2")
    with mx.stream(mx.cpu):
        model = _ep_model()
        with pytest.raises(ValueError, match="VLLM_METAL_EXPERT_PARTITION"):
            apply_expert_shard(model, _Group(1, 2))


def _run_ep_forward(monkeypatch, shard0, shard1, tokens):
    """One forward per rank, driven in lockstep on a shared activation.

    Real ranks hold identical activations because every collective (attention
    o_proj and masked MoE partial) all_sums both ranks' values. The fake
    pairs each consecutive rank-0/rank-1 collective the same way, so rank 1's
    every collective returns partial0 + partial1 and its logits are the
    combined output. Sequential per-model forwards cannot emulate this:
    collectives fire per layer, so the first rank's later layers would run on
    unsynchronized activations."""
    pending = []

    def fake_all_sum(value, *, group=None, stream=None):
        pending.append(value)
        if len(pending) < 2:
            return value
        total = pending[0] + pending[1]
        pending.clear()
        mx.eval(total)
        return total

    monkeypatch.setattr(mx.distributed, "all_sum", fake_all_sum)
    from mlx_lm.models.base import create_attention_mask

    ids = mx.array(tokens, dtype=mx.int32)
    x = shard0.model.embed_tokens(ids)
    full_mask = create_attention_mask(x, None)
    swa_mask = create_attention_mask(x, None, window_size=shard0.model.window_size)
    for layer0, layer1, layer_type in zip(
        shard0.model.layers, shard1.model.layers, shard0.model.layer_types, strict=True
    ):
        mask = full_mask if layer_type == "full_attention" else swa_mask
        residual = x
        h = layer0.input_layernorm(x)
        layer0.self_attn(h, mask)  # rank 0 partial, superseded by rank 1's combine
        x = residual + layer1.self_attn(h, mask)
        residual = x
        h = layer0.post_attention_layernorm(x)
        layer0.mlp(h)  # rank 0 partial, superseded by rank 1's combine
        x = residual + layer1.mlp(h)
    logits = shard0.lm_head(shard0.model.norm(x))
    mx.eval(logits)
    return logits


@pytest.mark.parametrize(
    "tokens",
    [
        [1, 7, 11, 23, 42, 3, 9, 17, 5, 8, 2, 13],  # 12 tokens x top-2 = 24 pairs
        list(range(1, 33)),  # 32 tokens x top-2 = 64 pairs -> SwitchGLU sorts
    ],
    ids=["unsorted24", "sorted64"],
)
@pytest.mark.parametrize("quantized", [False, True], ids=["float32", "mxfp4-q8"])
def test_expert_forward_matches_unsplit_reference(monkeypatch, tokens, quantized):
    """Both ranks' masked partials, summed by all_sum, equal the unsplit MoE.

    24 pairs stay below SwitchGLU's 64-slot sort threshold; 64 pairs take the
    _gather_sort/sorted-gather path with remapped local indices, dummy-0
    slots, and masking active on both ranks — production GPT-OSS (top-4)
    sorts from 16 tokens up."""
    from vllm_metal.distributed.experts import apply_expert_shard

    tokens = [tokens]
    with mx.stream(mx.cpu):
        reference = _ep_model(quantized=quantized)
        expected = reference(mx.array(tokens, dtype=mx.int32))
        mx.eval(expected)
        from copy import deepcopy

        shard0 = deepcopy(reference)
        shard1 = deepcopy(reference)
    monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", "2,2")
    apply_expert_shard(shard0, _Group(0, 2))
    apply_expert_shard(shard1, _Group(1, 2))
    combined = _run_ep_forward(monkeypatch, shard0, shard1, tokens)
    tolerance = 1e-4 if not quantized else 5e-2
    assert float(mx.max(mx.abs(combined - expected)).item()) <= tolerance


def test_expert_routing_sorted_path_and_full_coverage(monkeypatch):
    """Pin the sorted machinery under full coverage: 64 index pairs fire
    _gather_sort, but partition (0, 4) leaves the weights unsliced — no
    masking is active, so the output must match the reference exactly.
    Masking under the sorted path is covered by the sorted64 case above."""
    from vllm_metal.distributed import experts as experts_mod
    from vllm_metal.distributed.experts import apply_expert_shard

    with mx.stream(mx.cpu):
        reference = _ep_model()
        from copy import deepcopy

        full = deepcopy(reference)
    with monkeypatch.context() as m:
        m.setattr(experts_mod, "expert_partition", lambda ws, ne: [4, 0])
        apply_expert_shard(full, _Group(0, 2))
    assert full.layers[0].mlp.expert_partition == (0, 4)
    monkeypatch.setattr(
        mx.distributed, "all_sum", lambda value, *, group=None, stream=None: value
    )
    x = mx.random.normal((1, 32, 64))  # 32 tokens x top-2 = 64 pairs -> do_sort fires
    with mx.stream(mx.cpu):
        got = full.layers[0].mlp(x)
        expected = reference.layers[0].mlp(x)
        mx.eval(got, expected)
    assert float(mx.max(mx.abs(got - expected)).item()) <= 1e-4


def test_bootstrap_records_expert_parallel(monkeypatch):
    from vllm_metal.distributed.tensor import TensorGroup
    from vllm_metal.distributed.transport import PipelineTransportConfig

    monkeypatch.setattr(
        PipelineTransportConfig,
        "bootstrap_jaccl",
        classmethod(lambda cls, rank, peer_ips: _Group(rank, 2)),
    )
    cfg = _config()
    assert TensorGroup.bootstrap(0, ["a", "b"], cfg).expert_parallel is False
    cfg = _config()
    cfg.parallel_config.enable_expert_parallel = True
    assert TensorGroup.bootstrap(0, ["a", "b"], cfg).expert_parallel is True


def test_runner_applies_expert_shard_when_flagged(monkeypatch):
    import vllm_metal.v1.model_runner as runner_mod

    calls = []

    class _FakeExperts:
        def apply_expert_shard(self, model, tp):
            calls.append("ep")

    class _FakeTensor:
        def apply_tensor_shard(self, model, tp):
            calls.append("tp")

    import sys
    import types

    fake_experts = _FakeExperts()
    fake_tensor = _FakeTensor()
    runner_mod_experts = types.ModuleType("vllm_metal.distributed.experts")
    runner_mod_experts.apply_expert_shard = fake_experts.apply_expert_shard
    runner_mod_tensor = types.ModuleType("vllm_metal.distributed.tensor")
    runner_mod_tensor.apply_tensor_shard = fake_tensor.apply_tensor_shard
    monkeypatch.setitem(
        sys.modules, "vllm_metal.distributed.experts", runner_mod_experts
    )
    monkeypatch.setitem(sys.modules, "vllm_metal.distributed.tensor", runner_mod_tensor)

    tp = types.SimpleNamespace(rank=0, size=2, expert_parallel=True)
    runner = types.SimpleNamespace(
        tp=tp,
        model=object(),
        num_kv_heads=8,
        num_layers=4,
        kv_heads_per_layer=[8, 8],
    )
    # Extracted branch under test (see Step 3): a module-level helper keeps
    # this testable without building a full MetalModelRunner.
    runner_mod._apply_tensor_parallel_shards(runner)
    assert calls == ["ep"]
    runner.tp.expert_parallel = False
    runner_mod._apply_tensor_parallel_shards(runner)
    assert calls == ["ep", "tp"]
    assert runner.num_kv_heads == 2 and runner.kv_heads_per_layer == [2, 2]
