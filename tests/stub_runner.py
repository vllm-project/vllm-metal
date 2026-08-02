# SPDX-License-Identifier: Apache-2.0
"""Shared test helper for constructing ``MetalModelRunner`` stubs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import torch

import vllm_metal.v1.model_runner as mr
from vllm_metal.v1.cache_policy import ModelCachePolicy
from vllm_metal.v1.lora import MetalLoRARuntime
from vllm_metal.v1.model_adapter import DefaultModelAdapter
from vllm_metal.v1.spec_decode import SpeculativeDecodeController
from vllm_metal.v1.structured_output import MetalStructuredOutputApplier


def make_stub_runner(
    *,
    model_args: dict[str, Any] | None = None,
    **attrs: Any,
) -> mr.MetalModelRunner:
    """Create a ``MetalModelRunner`` stub without running ``__init__``.

    Sets sensible defaults for all internal attributes.  ``_vocab_size``
    is derived from ``model_args["vocab_size"]`` automatically — never
    set it separately.  Pass keyword arguments to override any attribute.
    """
    runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)

    _model_args = model_args or {}

    defaults: dict[str, Any] = {
        "vllm_config": SimpleNamespace(speculative_config=None),
        "cache_config": SimpleNamespace(mamba_page_size_padded=None),
        "model_config": SimpleNamespace(
            runner_type="generate", get_head_size=lambda: 128
        ),
        "model": object(),
        "_is_vlm": False,
        "_multimodal_adapter": None,
        "_gemma4_mtp_assistant": None,
        "_drafter": None,
        "encoder_cache": None,
        "_paged_attention_runtime": None,
        "_paged_block_size": 0,
        "_paged_scheduler_group_indices": (),
        "_paged_group_block_sizes": (),
        "_request_states": {},
        "_paged_request_seq_lens": {},
        "_pending_output": None,
        "_draft_token_ids": None,
        "_execute_model_state": None,
        "pp": None,
        "_pp_model": None,
        "_model_adapter": DefaultModelAdapter(),
        "_spec_decode_controller": SpeculativeDecodeController(),
        "kv_heads_per_layer": None,
        "head_dim_per_layer": None,
        "sliding_window_per_layer": None,
        "use_async_scheduling": True,
        "device": torch.device("cpu"),
        "_sampler": None,
        "_logitsprocs": None,
        "_structured_output_applier": MetalStructuredOutputApplier(),
        "_lora": MetalLoRARuntime(),
        "_yoco_cache_mapping": None,
        "model_args": _model_args,
    }

    for k, v in defaults.items():
        setattr(runner, k, v)
    for k, v in attrs.items():
        setattr(runner, k, v)
    if "_paged_block_size" in attrs and "_paged_group_block_sizes" not in attrs:
        runner._paged_scheduler_group_indices = (0,)
        runner._paged_group_block_sizes = (attrs["_paged_block_size"],)
    if "_is_pooling" not in attrs:
        runner._is_pooling = (
            getattr(runner.model_config, "runner_type", None) == "pooling"
        )

    runner._cache_policy = ModelCachePolicy(runner, runner._model_adapter)

    # Derive _vocab_size from model_args — single source of truth.
    if "vocab_size" in _model_args:
        runner._vocab_size = _model_args["vocab_size"]

    return runner


def make_gemma4_mixed_mha_runner(
    num_layers: int,
    sliding_kv_heads: int,
    full_kv_heads: int,
    max_model_len: int = 16,
    original_max_model_len: int | None = 16,
    max_in_flight_tokens: int = 16,
    disable_hybrid_manager: bool = False,
    num_gpu_blocks_override: int | None = None,
) -> mr.MetalModelRunner:
    """Create a Gemma4 mixed sliding/full MHA runner stub."""
    layer_types = [
        "full_attention" if (index + 1) % 6 == 0 else "sliding_attention"
        for index in range(num_layers)
    ]
    model_args = {
        "global_head_dim": 512,
        "num_global_key_value_heads": full_kv_heads,
        "sliding_window": 1024,
        "layer_types": layer_types,
    }
    adapter = DefaultModelAdapter()
    per_layer = adapter.build_per_layer_kv_shapes(
        model_args,
        num_layers=num_layers,
        num_kv_heads=sliding_kv_heads,
        head_dim=256,
    )
    sliding_windows = adapter.build_sliding_window_per_layer(
        model_args,
        num_layers=num_layers,
    )
    assert per_layer is not None
    assert sliding_windows is not None
    kv_heads_per_layer, head_dim_per_layer = per_layer
    cache_config = SimpleNamespace(
        block_size=16,
        gpu_memory_utilization=1.0,
        num_gpu_blocks_override=num_gpu_blocks_override,
        kv_cache_memory_bytes=None,
        enable_prefix_caching=False,
        prefix_match_unit=None,
    )
    scheduler_config = SimpleNamespace(
        max_num_batched_tokens=max_in_flight_tokens,
        disable_hybrid_kv_cache_manager=disable_hybrid_manager,
    )
    vllm_config = SimpleNamespace(
        speculative_config=None,
        max_in_flight_tokens=max_in_flight_tokens,
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            original_max_model_len=original_max_model_len,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        kv_transfer_config=None,
    )
    return make_stub_runner(
        model_args=model_args,
        num_layers=num_layers,
        num_kv_cache_layers=num_layers,
        num_kv_heads=sliding_kv_heads,
        head_dim=512,
        kv_cache_dtype=mx.bfloat16,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        vllm_config=vllm_config,
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=object()) for _ in range(num_layers)]
        ),
        _model_adapter=adapter,
        kv_heads_per_layer=kv_heads_per_layer,
        head_dim_per_layer=head_dim_per_layer,
        sliding_window_per_layer=sliding_windows,
    )
