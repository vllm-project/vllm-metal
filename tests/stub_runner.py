# SPDX-License-Identifier: Apache-2.0
"""Shared test helper for constructing ``MetalModelRunner`` stubs."""

from __future__ import annotations

from typing import Any

import torch

import vllm_metal.v1.model_runner as mr


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
        "model": object(),
        "_is_stt": False,
        "_is_vlm": False,
        "_paged_attention_backend": None,
        "_request_states": {},
        "_paged_request_seq_lens": {},
        "_finished_request_count": 0,
        "_prefix_cache": None,
        "_pending_output": None,
        "_execute_model_state": None,
        "use_async_scheduling": True,
        "device": torch.device("cpu"),
        "_sampler": None,
        "_logitsprocs": None,
        "model_args": _model_args,
    }

    for k, v in defaults.items():
        setattr(runner, k, v)
    for k, v in attrs.items():
        setattr(runner, k, v)

    # Derive _vocab_size from model_args — single source of truth.
    if "vocab_size" in _model_args:
        runner._vocab_size = _model_args["vocab_size"]

    return runner
