# SPDX-License-Identifier: Apache-2.0
"""Read-byte accounting through vLLM's public model metrics interface."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from itertools import count
from pathlib import Path

import mlx.core as mx
import msgspec
import pytest
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.qwen3 import Model, ModelArgs
from mlx_lm.utils import quantize_model
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.perf import (
    AttentionMetrics,
    ExecutionContext,
    FfnMetrics,
    ModelMetrics,
    PerfStats,
    UnembedMetrics,
)

from vllm_metal import register_metrics

_TINY_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "vocab_size": 256,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
}


@pytest.fixture
def checkpoint(tmp_path, inspect_vllm_models_in_process):
    ids = count()

    def make(
        *, dtype=mx.bfloat16, bits=4, group_size=64, tied=True, predicate=None, shards=1
    ):
        path = tmp_path / str(next(ids))
        path.mkdir()
        raw = {**_TINY_CONFIG, "tie_word_embeddings": tied}
        model = Model(ModelArgs.from_dict(raw))
        model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
        model, raw = quantize_model(
            model, raw, group_size=group_size, bits=bits, quant_predicate=predicate
        )
        weights = dict(tree_flatten(model.parameters()))
        items = list(weights.items())
        for shard in range(shards):
            mx.save_safetensors(
                str(path / f"model-{shard}.safetensors"), dict(items[shard::shards])
            )
        (path / "config.json").write_text(json.dumps(raw))
        config = EngineArgs(
            model=str(path),
            skip_tokenizer_init=True,
            max_model_len=256,
            enable_mfu_metrics=True,
        ).create_engine_config()
        register_metrics()
        return config, weights

    return make


def _upstream_breakdown(config, ctx, method):
    return {
        f"{metric.component_type()}.{key}": value
        for cls in (AttentionMetrics, FfnMetrics, UnembedMetrics)
        for metric in [cls.from_vllm_config(config)]
        for key, value in getattr(metric, method)(ctx).items()
    }


def _stored_projection_bytes(weights, tied):
    """Count actual tensors, excluding input lookup and normalization weights."""
    reads = dict.fromkeys(
        (
            "attn.qkv_weight",
            "attn.out_weight",
            "ffn.dense_up_gate_weights",
            "ffn.dense_down_weights",
            "unembed.weight",
        ),
        0,
    )
    head = "model.embed_tokens." if tied else "lm_head."
    for name, tensor in weights.items():
        if name.startswith(head):
            key = "unembed.weight"
        elif ".self_attn.o_proj." in name:
            key = "attn.out_weight"
        elif ".self_attn." in name and "_proj." in name:
            key = "attn.qkv_weight"
        elif ".mlp.down_proj." in name:
            key = "ffn.dense_down_weights"
        elif ".mlp." in name and "_proj." in name:
            key = "ffn.dense_up_gate_weights"
        else:
            continue
        reads[key] += tensor.nbytes
    return reads


def _check_reads(config, weights, *, prefill=False):
    ctx = ExecutionContext()
    ctx.add(22 if prefill else 1, 128, is_prefill=prefill)
    metrics = ModelMetrics(config)
    before = _upstream_breakdown(config, ctx, "get_read_bytes_breakdown")
    expected = {
        **before,
        **_stored_projection_bytes(
            weights, config.model_config.hf_text_config.tie_word_embeddings
        ),
    }
    assert metrics.get_read_bytes_breakdown(ctx) == expected
    assert metrics.get_read_bytes_breakdown(ctx, per_gpu=False) == expected
    for method in ("get_num_flops_breakdown", "get_write_bytes_breakdown"):
        assert getattr(metrics, method)(ctx) == _upstream_breakdown(config, ctx, method)
    return expected


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize(
    "bits,group_size,dtype,byte_size",
    [
        (4, 64, mx.bfloat16, 0.5625),
        (8, 32, mx.float16, 1.125),
        (4, 64, mx.float32, 0.625),
    ],
)
def test_reads_match_checkpoint_even_when_config_dtype_differs(
    checkpoint, prefill, bits, group_size, dtype, byte_size
):
    config, weights = checkpoint(bits=bits, group_size=group_size, dtype=dtype)
    reads = _check_reads(config, weights, prefill=prefill)
    assert config.quant_config is None
    assert config.model_config.quantization is None
    assert sum(v for k, v in reads.items() if "weight" in k) == 327680 * byte_size


@pytest.mark.parametrize(
    "skipped",
    ["lm_head", "model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"],
)
def test_partial_quantization_with_global_only_config(checkpoint, skipped):
    config, weights = checkpoint(tied=False, predicate=lambda path, _: path != skipped)
    assert set(config.model_config.hf_config.quantization) == {
        "bits",
        "group_size",
        "mode",
    }
    assert f"{skipped}.scales" not in weights
    _check_reads(config, weights)


def test_affine_per_layer_overrides_use_actual_storage(checkpoint):
    config, weights = checkpoint(
        predicate=lambda path, _: (
            {"bits": 8, "group_size": 32, "mode": "affine"}
            if path == "model.layers.0.mlp.up_proj"
            else True
        )
    )
    _check_reads(config, weights)


def test_checkpoint_headers_can_span_multiple_files(checkpoint):
    config, weights = checkpoint(shards=2)
    _check_reads(config, weights)


def test_metadata_cache_refreshes_after_local_checkpoint_replacement(checkpoint):
    config, weights = checkpoint()
    _check_reads(config, weights)
    weights = {
        name: tensor.astype(mx.float32)
        if name.endswith((".scales", ".biases"))
        else tensor
        for name, tensor in weights.items()
    }
    mx.save_safetensors(
        str(Path(config.model_config.model) / "model-0.safetensors"), weights
    )
    _check_reads(config, weights)


def test_remote_metadata_is_revision_aware_and_reused(checkpoint, monkeypatch):
    from vllm.transformers_utils.config import get_safetensors_params_metadata

    from vllm_metal.v1 import metrics

    config, weights = checkpoint()
    metadata = get_safetensors_params_metadata(config.model_config.model)
    calls = []

    def read(model, *, revision):
        calls.append((model, revision))
        return metadata

    monkeypatch.setattr(metrics, "get_safetensors_params_metadata", read)
    config.model_config.model = "test/affine-mfu"
    for revision in ("first-revision", "second-revision"):
        config.model_config.revision = revision
        _check_reads(config, weights)
    assert calls == [
        ("test/affine-mfu", "first-revision"),
        ("test/affine-mfu", "second-revision"),
    ]


def test_quantized_step_stats_survive_engine_ipc(checkpoint):
    config, _ = checkpoint()
    output = SchedulerOutput.make_empty()
    output.scheduled_cached_reqs.req_ids = ["decode"]
    output.scheduled_cached_reqs.num_computed_tokens = [127]
    output.num_scheduled_tokens = {"decode": 1}
    output.total_num_scheduled_tokens = 1
    stats = ModelMetrics(config).get_step_perf_stats_per_gpu(output)
    restored = msgspec.msgpack.decode(msgspec.msgpack.encode(stats), type=PerfStats)
    assert restored == stats
    assert restored.num_read_bytes_per_gpu > 327680 * 0.5625


def test_engine_process_discovers_metrics_without_manual_registration(
    checkpoint, tmp_path
):
    """EngineCore receives a pickled config and loads general plugins itself."""
    config, _ = checkpoint()
    config_path = tmp_path / "config.pkl"
    config_path.write_bytes(pickle.dumps(config))
    script = """
import pickle
import sys
from vllm.plugins import load_general_plugins
from vllm.v1.metrics.perf import ExecutionContext, ModelMetrics

with open(sys.argv[1], "rb") as f:
    config = pickle.load(f)
ctx = ExecutionContext()
ctx.add(1, 128, is_prefill=False)
before = ModelMetrics(config).get_read_bytes_breakdown(ctx)
load_general_plugins()
after = ModelMetrics(config).get_read_bytes_breakdown(ctx)
assert before["unembed.weight"] == 128 * 256 * 2
assert after["unembed.weight"] == 128 * 256 * 0.5625
load_general_plugins()
assert ModelMetrics(config).get_read_bytes_breakdown(ctx) == after
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path)],
        env={**os.environ, "VLLM_PLUGINS": "metal,metrics_metal"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "control", ["unquantized", "mps", "fp8", "non_affine", "tp", "pp", "missing"]
)
def test_other_or_unsupported_paths_keep_upstream_estimates(
    checkpoint, monkeypatch, control
):
    from vllm_metal.v1 import metrics

    config, _ = checkpoint()
    if control == "unquantized":
        del config.model_config.hf_config.quantization
    elif control == "mps":
        monkeypatch.setenv("VLLM_METAL_BACKEND", "mps")
    elif control == "fp8":
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        config.model_config.quantization = "fp8"
        config.quant_config = Fp8Config()
    elif control == "non_affine":
        config.model_config.hf_config.quantization["mode"] = "mxfp4"
    elif control == "tp":
        config.parallel_config.tensor_parallel_size = 2
    elif control == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    else:
        monkeypatch.setattr(
            metrics, "get_safetensors_params_metadata", lambda *a, **kw: {}
        )
    ctx = ExecutionContext()
    ctx.add(1, 128, is_prefill=False)
    for method in ("get_read_bytes_breakdown", "get_num_flops_breakdown"):
        assert getattr(ModelMetrics(config), method)(ctx) == _upstream_breakdown(
            config, ctx, method
        )


def test_unrecognized_projection_falls_back_for_only_its_component(checkpoint):
    config, weights = checkpoint()
    weights["model.layers.0.mlp.unknown.weight"] = weights.pop(
        "model.layers.0.mlp.up_proj.weight"
    )
    mx.save_safetensors(
        str(Path(config.model_config.model) / "model-0.safetensors"), weights
    )
    ctx = ExecutionContext()
    ctx.add(1, 128, is_prefill=False)
    reads = ModelMetrics(config).get_read_bytes_breakdown(ctx)
    upstream = _upstream_breakdown(config, ctx, "get_read_bytes_breakdown")
    assert {k: v for k, v in reads.items() if k.startswith("ffn.")} == {
        k: v for k, v in upstream.items() if k.startswith("ffn.")
    }
    assert reads["unembed.weight"] == 128 * 256 * 0.5625
    assert reads["attn.qkv_weight"] == upstream["attn.qkv_weight"] * 0.5625 / 2


@pytest.mark.parametrize(
    "field,value", [("bits", 0), ("group_size", -1), ("bits", True)]
)
def test_invalid_quantization_is_not_silently_estimated(checkpoint, field, value):
    config, _ = checkpoint()
    config.model_config.hf_config.quantization[field] = value
    with pytest.raises(ValueError, match="positive integer"):
        ModelMetrics(config)
