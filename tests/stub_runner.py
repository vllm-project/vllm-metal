# SPDX-License-Identifier: Apache-2.0
"""Shared test helper for constructing ``MetalModelRunner`` stubs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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
        "model_config": SimpleNamespace(runner_type="generate"),
        "model": object(),
        "_is_vlm": False,
        "_multimodal_adapter": None,
        "_gemma4_mtp_assistant": None,
        "_drafter": None,
        "encoder_cache": None,
        "_paged_attention_runtime": None,
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
    if "_is_pooling" not in attrs:
        runner._is_pooling = (
            getattr(runner.model_config, "runner_type", None) == "pooling"
        )

    runner._cache_policy = ModelCachePolicy(runner, runner._model_adapter)

    # Derive _vocab_size from model_args — single source of truth.
    if "vocab_size" in _model_args:
        runner._vocab_size = _model_args["vocab_size"]

    return runner
