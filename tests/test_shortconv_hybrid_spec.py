# SPDX-License-Identifier: Apache-2.0
"""Conv hybrid (LFM2 ShortConv) detection, dims derivation, and KV specs."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from tests.stub_runner import make_stub_runner
from vllm_metal.attention.impls.linear import is_linear_attention
from vllm_metal.attention.impls.sdpa import is_sdpa
from vllm_metal.attention.impls.shortconv import is_shortconv
from vllm_metal.attention.patching import find_attn_attr
from vllm_metal.attention.runtime.shortconv_hybrid import (
    ShortConvHybridPagedAttentionRuntime,
)
from vllm_metal.v1.model_lifecycle import ModelLifecycle


def make_lfm2_args(**overrides):
    """Minimal mlx_lm LFM2 ``ModelArgs`` for tests.

    Shared with ``tests/attention/test_shortconv_wrapper.py`` so the kwarg
    blob lives once; pass overrides for dims a test pins explicitly.
    """
    from mlx_lm.models.lfm2 import ModelArgs

    kwargs = {
        "model_type": "lfm2",
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "norm_eps": 1e-5,
        "conv_bias": True,
        "conv_L_cache": 3,
        "block_dim": 32,
        "block_ff_dim": 64,
        "block_multiple_of": 8,
        "block_ffn_dim_multiplier": 1.0,
        "block_auto_adjust_ff_dim": False,
        "layer_types": ["conv", "full_attention"],
    }
    kwargs.update(overrides)
    return ModelArgs(**kwargs)


LAYER_TYPES = ["conv", "conv", "full_attention", "conv", "full_attention"]
NUM_LAYERS = len(LAYER_TYPES)
CONV_LAYERS = (0, 1, 3)
SDPA_LAYERS = frozenset({2, 4})
CONV_L_CACHE = 3
HIDDEN = 64
MAX_MODEL_LEN = 2048


# ---------------------------------------------------------------------------
# Module-level detection against real mlx_lm LFM2 modules
# ---------------------------------------------------------------------------


def test_lfm2_conv_layer_detected():
    """LFM2 conv layer should expose ``conv`` detected as ShortConv."""
    from mlx_lm.models.lfm2 import Lfm2DecoderLayer

    layer = Lfm2DecoderLayer(make_lfm2_args(), layer_idx=0)

    assert find_attn_attr(layer) == "conv"
    assert is_shortconv(layer.conv)
    assert not is_sdpa(layer.conv)
    assert not is_linear_attention(layer.conv)


def test_lfm2_attention_layer_detected_as_sdpa():
    """LFM2 full-attention layer (out_proj naming) should match SDPA."""
    from mlx_lm.models.lfm2 import Lfm2DecoderLayer

    layer = Lfm2DecoderLayer(make_lfm2_args(), layer_idx=1)

    assert find_attn_attr(layer) == "self_attn"
    assert is_sdpa(layer.self_attn)
    assert not is_shortconv(layer.self_attn)
    assert not is_linear_attention(layer.self_attn)


# ---------------------------------------------------------------------------
# Runner-level classification from model_args
# ---------------------------------------------------------------------------


def test_lfm2_layer_types_classify_as_conv_hybrid():
    runner = make_stub_runner(model_args={"layer_types": list(LAYER_TYPES)})
    assert runner.is_conv_hybrid
    assert not runner.is_hybrid


def test_gemma_layer_types_do_not_classify_as_conv_hybrid():
    runner = make_stub_runner(
        model_args={
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention"]
        }
    )
    assert not runner.is_conv_hybrid
    assert not runner.is_hybrid


def test_gdn_hybrid_does_not_classify_as_conv_hybrid():
    runner = make_stub_runner(model_args={"full_attention_interval": 4})
    assert runner.is_hybrid
    assert not runner.is_conv_hybrid


# ---------------------------------------------------------------------------
# Dims derivation + KV cache spec
# ---------------------------------------------------------------------------


def _conv_runner():
    runner = make_stub_runner(
        model_args={
            "layer_types": list(LAYER_TYPES),
            "conv_L_cache": CONV_L_CACHE,
            "hidden_size": HIDDEN,
        },
        num_layers=NUM_LAYERS,
        num_kv_cache_layers=NUM_LAYERS,
        num_kv_heads=2,
        head_dim=32,
        kv_cache_dtype=mx.float16,
        cache_config=SimpleNamespace(
            block_size=16,
            mamba_page_size_padded=None,
            mamba_block_size=MAX_MODEL_LEN,
            mamba_cache_mode="none",
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        model_config=SimpleNamespace(
            runner_type="generate",
            get_head_size=lambda: 32,
            max_model_len=MAX_MODEL_LEN,
        ),
    )
    lifecycle = ModelLifecycle(runner, runner._model_adapter)
    lifecycle._install_conv_hybrid_dims(runner.model_args)
    return runner


def test_conv_hybrid_dims_derived_from_layer_types():
    runner = _conv_runner()

    assert runner.sdpa_layer_indices == SDPA_LAYERS
    assert runner.num_sdpa_layers == len(SDPA_LAYERS)
    assert runner.conv_layer_indices == CONV_LAYERS
    assert runner.num_conv_layers == len(CONV_LAYERS)
    assert runner.conv_kernel_dim == CONV_L_CACHE
    assert runner.conv_hidden_size == HIDDEN


def test_conv_hybrid_kv_cache_spec_layers_and_fields():
    runner = _conv_runner()

    specs = runner._cache_policy.get_kv_cache_spec()

    assert set(specs) == {
        *(f"layers.{i}.self_attn" for i in sorted(SDPA_LAYERS)),
        *(f"layers.{i}.conv" for i in CONV_LAYERS),
    }
    for i in sorted(SDPA_LAYERS):
        spec = specs[f"layers.{i}.self_attn"]
        assert isinstance(spec, FullAttentionSpec)
        assert spec.block_size == 16
    for i in CONV_LAYERS:
        spec = specs[f"layers.{i}.conv"]
        assert isinstance(spec, MambaSpec)
        assert spec.shapes == ((CONV_L_CACHE - 1, HIDDEN),)
        assert spec.dtypes == (torch.float16,)
        # The config hook resolves mamba_block_size to max_model_len while
        # prefix caching is off; the spec must carry it verbatim.
        assert spec.block_size == MAX_MODEL_LEN
        assert spec.mamba_type == MambaAttentionBackendEnum.SHORT_CONV
        assert spec.mamba_cache_mode == "none"


def test_conv_hybrid_backend_routing():
    runner = _conv_runner()

    runtime = runner._cache_policy.build_paged_attention_runtime(block_size=16)

    assert isinstance(runtime, ShortConvHybridPagedAttentionRuntime)
    assert runtime._sdpa_indices == sorted(SDPA_LAYERS)
    assert runtime._conv_indices == list(CONV_LAYERS)


# ---------------------------------------------------------------------------
# Runtime layer classification guards
# ---------------------------------------------------------------------------


def test_runtime_rejects_unknown_layer_types():
    with pytest.raises(NotImplementedError, match="unsupported layer_types"):
        ShortConvHybridPagedAttentionRuntime(
            layer_types=["conv", "sliding_attention"],
            max_num_seqs=2,
            num_kv_heads=2,
            head_dim=32,
            conv_kernel_dim=CONV_L_CACHE,
            conv_dim=HIDDEN,
            block_size=16,
            dtype=mx.float16,
        )


@pytest.mark.parametrize("mode", ["align", "all"])
def test_runtime_rejects_state_caching_modes(mode: str):
    """Conv hybrids run none mode; block-keyed conv state is not implemented."""
    with pytest.raises(NotImplementedError, match="only 'none'"):
        ShortConvHybridPagedAttentionRuntime(
            layer_types=list(LAYER_TYPES),
            max_num_seqs=2,
            num_kv_heads=2,
            head_dim=32,
            conv_kernel_dim=CONV_L_CACHE,
            conv_dim=HIDDEN,
            block_size=16,
            dtype=mx.float16,
            mamba_cache_mode=mode,
        )


def test_platform_rejects_prefix_caching_for_conv_hybrids(monkeypatch):
    """Conv state is not block-keyed yet, so APC would reuse stale state."""
    from vllm_metal.config import reset_config
    from vllm_metal.platform import MetalPlatform

    monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
    reset_config()
    try:
        vllm_config = SimpleNamespace(
            speculative_config=None,
            lora_config=None,
            additional_config={},
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                data_parallel_size=1,
                world_size=1,
                disable_custom_all_reduce=False,
            ),
            cache_config=SimpleNamespace(
                block_size=None,
                kv_cache_dtype_skip_layers=[],
                enable_prefix_caching=True,
                mamba_cache_mode="none",
                mamba_ssm_cache_dtype="float32",
            ),
            model_config=SimpleNamespace(
                model="test-lfm2",
                disable_cascade_attn=False,
                tokenizer=None,
                max_model_len=2048,
                multimodal_config=None,
                hf_config=SimpleNamespace(
                    model_type="lfm2", layer_types=list(LAYER_TYPES)
                ),
                is_hybrid=True,
            ),
            scheduler_config=SimpleNamespace(
                async_scheduling=True,
                enable_chunked_prefill=True,
                max_num_batched_tokens=2048,
                max_num_scheduled_tokens=None,
                long_prefill_token_threshold=0,
            ),
        )
        with pytest.raises(NotImplementedError, match="conv hybrid"):
            MetalPlatform.check_and_update_config(vllm_config)
    finally:
        reset_config()
