# SPDX-License-Identifier: Apache-2.0
"""Tests for MoE expert offloading."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import olmoe

from vllm_metal.config import MetalConfig
from vllm_metal.moe_offload import ExpertStore, OffloadedSwitchGLU

NUM_EXPERTS = 8
TOP_K = 2


def _tiny_olmoe(seed: int) -> nn.Module:
    mx.random.seed(seed)
    args = olmoe.ModelArgs.from_dict(
        {
            "model_type": "olmoe",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-5,
            "vocab_size": 128,
            "num_experts": NUM_EXPERTS,
            "num_experts_per_tok": TOP_K,
            "tie_word_embeddings": False,
        }
    )
    model = olmoe.Model(args)
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    return model


def _save_checkpoint(
    model: nn.Module, model_dir: Path, *, per_expert: bool, drop: str = ""
) -> None:
    weights = {}
    for name, array in tree_flatten(model.parameters()):
        if drop and ".switch_mlp." in name and name.endswith(drop):
            continue
        if per_expert and ".switch_mlp." in name:
            block, rest = name.split(".switch_mlp.", 1)
            for expert in range(array.shape[0]):
                weights[f"{block}.experts.{expert}.{rest}"] = array[expert]
        else:
            weights[name] = array
    model_dir.mkdir()
    mx.save_safetensors(str(model_dir / "model.safetensors"), weights)


def _expert_bytes(model: nn.Module) -> int:
    switch = model.model.layers[0].mlp.switch_mlp
    return sum(
        array.nbytes // NUM_EXPERTS for _, array in tree_flatten(switch.parameters())
    )


@pytest.fixture(
    params=[
        ("stacked", mx.float32),
        ("per_expert", mx.float32),
        ("stacked", mx.bfloat16),
        ("per_expert", mx.bfloat16),
    ],
    ids=["stacked-fp32", "per-expert-fp32", "stacked-bf16", "per-expert-bf16"],
)
def checkpoint(request, tmp_path) -> tuple[nn.Module, Path]:
    layout, dtype = request.param
    model = _tiny_olmoe(seed=1)
    model.set_dtype(dtype)
    model_dir = tmp_path / layout
    _save_checkpoint(model, model_dir, per_expert=layout == "per_expert")
    return model, model_dir


def _offloaded_twin(reference: nn.Module, model_dir: Path, cache_bytes: int):
    twin = _tiny_olmoe(seed=2)
    twin.load_weights(tree_flatten(reference.parameters()))
    store = ExpertStore.install(twin, str(model_dir), None, cache_bytes)
    return twin, store


@pytest.mark.parametrize("num_tokens", [1, 4, 32, 128])
@pytest.mark.parametrize(
    "dtype", [mx.float32, mx.float16, mx.bfloat16], ids=["fp32", "fp16", "bf16"]
)
def test_offloaded_forward_matches_resident_model(
    checkpoint, num_tokens, dtype
) -> None:
    """Both SwitchGLU paths (sorted gather from 64 routed slots) stay bit-exact."""
    reference, model_dir = checkpoint
    reference.set_dtype(dtype)
    # Room for two experts: every layer call evicts, so misses are exercised.
    twin, store = _offloaded_twin(reference, model_dir, 2 * _expert_bytes(reference))
    tokens = mx.random.randint(0, 128, (1, num_tokens))

    expected = reference(tokens)
    actual = twin(tokens)

    assert mx.array_equal(expected, actual)
    assert store.misses > 0
    switch = twin.model.layers[0].mlp.switch_mlp
    assert isinstance(switch, OffloadedSwitchGLU)
    assert switch.gate_proj.weight.shape[0] == 0


@pytest.mark.parametrize(
    "dtype", [mx.float32, mx.float16, mx.bfloat16], ids=["fp32", "fp16", "bf16"]
)
def test_cache_honours_its_byte_budget_and_counts_hits(checkpoint, dtype) -> None:
    reference, model_dir = checkpoint
    reference.set_dtype(dtype)
    _, store = _offloaded_twin(reference, model_dir, 3 * _expert_bytes(reference))
    path = "model.layers.0.mlp.switch_mlp"

    store.get(path, [0, 1, 2])
    store.get(path, [2, 3])
    assert store.bytes <= 3 * _expert_bytes(reference)
    assert (store.hits, store.misses) == (1, 4)

    # Expert 0 was the least recently used, so only it needs reading again.
    store.get(path, [1, 3])
    store.get(path, [0])
    assert (store.hits, store.misses) == (3, 5)


@pytest.mark.parametrize(
    "dtype", [mx.float32, mx.float16, mx.bfloat16], ids=["fp32", "fp16", "bf16"]
)
def test_reserve_covers_the_cache_and_one_stacked_layer(checkpoint, dtype) -> None:
    reference, model_dir = checkpoint
    reference.set_dtype(dtype)
    cache_bytes = 2 * _expert_bytes(reference)

    _, store = _offloaded_twin(reference, model_dir, cache_bytes)

    assert store.reserved_bytes == cache_bytes + NUM_EXPERTS * _expert_bytes(reference)


@pytest.mark.parametrize(
    ("drop", "match"),
    [
        ("weight", "layers.0.mlp.switch_mlp.down_proj.weight expert 0"),
        ("scales", "layers.0.mlp.switch_mlp.down_proj.scales expert 0"),
    ],
)
def test_missing_expert_tensors_fail_at_install(tmp_path, drop, match) -> None:
    reference = _tiny_olmoe(seed=1)
    broken = tmp_path / "broken"
    _save_checkpoint(reference, broken, per_expert=False, drop=f"down_proj.{drop}")

    with pytest.raises(KeyError, match=match):
        ExpertStore.install(_tiny_olmoe(seed=3), str(broken), None, 1 << 20)


def test_dense_model_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="SwitchGLU"):
        ExpertStore.install(nn.Linear(4, 4), str(tmp_path), None, 1 << 20)


class TestConfig:
    def test_cache_gb_sets_the_byte_budget(self) -> None:
        config = MetalConfig(mlx_device="gpu")
        config.apply_moe_expert_offload({"moe_expert_offload": {"cache_gb": 1.5}})
        assert config.moe_expert_cache_bytes == int(1.5 * 2**30)

    def test_absent_key_keeps_offloading_off(self) -> None:
        config = MetalConfig(mlx_device="gpu")
        config.apply_moe_expert_offload({"turboquant": True})
        assert config.moe_expert_cache_bytes == 0

    @pytest.mark.parametrize(
        "options",
        [True, {}, {"cache_gb": 0}, {"cache_gb": "2"}, {"cache_gb": 2, "x": 1}],
    )
    def test_invalid_options_are_rejected(self, options) -> None:
        config = MetalConfig(mlx_device="gpu")
        with pytest.raises(ValueError, match="moe_expert_offload"):
            config.apply_moe_expert_offload({"moe_expert_offload": options})


class TestPlatformGuards:
    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("pipeline_parallel_size", 2, "pipeline parallelism"),
            ("lora_config", SimpleNamespace(), "LoRA"),
            ("speculative_config", SimpleNamespace(method="ngram"), "speculative"),
        ],
    )
    def test_unsupported_features_are_rejected(self, field, value, match) -> None:
        from vllm_metal.platform import MetalPlatform

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(runner_type="generate"),
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            lora_config=None,
            speculative_config=None,
        )
        if field == "pipeline_parallel_size":
            vllm_config.parallel_config.pipeline_parallel_size = value
        else:
            setattr(vllm_config, field, value)

        with pytest.raises(NotImplementedError, match=match):
            MetalPlatform._check_moe_expert_offload_config(vllm_config, "mlx")

    def test_mps_backend_is_rejected(self) -> None:
        from vllm_metal.platform import MetalPlatform

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(runner_type="generate"),
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            lora_config=None,
            speculative_config=None,
        )

        with pytest.raises(NotImplementedError, match="MLX backend only"):
            MetalPlatform._check_moe_expert_offload_config(vllm_config, "mps")
