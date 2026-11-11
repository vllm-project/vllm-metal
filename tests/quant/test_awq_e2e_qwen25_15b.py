# SPDX-License-Identifier: Apache-2.0
"""Dtype alignment for AWQ loading on Qwen2.5-1.5B-Instruct-AWQ."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
import torch
from mlx.utils import tree_flatten

from tests.stub_runner import make_stub_runner
from vllm_metal.v1.model_lifecycle import ModelLifecycle

_AWQ_REPO = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"


def _runner_model_config(*, dtype):
    return SimpleNamespace(
        model=_AWQ_REPO,
        revision=None,
        hf_config=None,
        is_multimodal_model=False,
        trust_remote_code=False,
        dtype=dtype,
        quantization=None,
    )


@pytest.mark.slow
def test_awq_load_aligns_non_quant_dtypes_to_runner_target():
    """After AWQ load, non-quantized floating params must match the runner's
    target dtype, and quantized layers' scales/biases must NOT have been
    touched by the alignment step (their dtype is owned by the transform).
    """
    runner = make_stub_runner(model_config=_runner_model_config(dtype=torch.bfloat16))
    lifecycle = ModelLifecycle(runner, runner._model_adapter)

    model = lifecycle._load_generation_model(_AWQ_REPO, is_vlm=False)

    saw_quant_layer = False
    saw_non_quant_floating = False
    quant_dtype_pins = set()
    for _path, module in tree_flatten(
        model.leaf_modules(), is_leaf=nn.Module.is_module
    ):
        if isinstance(module, nn.QuantizedLinear):
            saw_quant_layer = True
            for name, value in module.parameters().items():
                dtype = getattr(value, "dtype", None)
                if dtype is not None and mx.issubdtype(dtype, mx.floating):
                    quant_dtype_pins.add((name, dtype))
            continue
        for name, value in module.parameters().items():
            dtype = getattr(value, "dtype", None)
            if dtype is None or not mx.issubdtype(dtype, mx.floating):
                continue
            saw_non_quant_floating = True
            assert dtype == mx.bfloat16, (
                f"non-quant floating param {name!r} on "
                f"{type(module).__name__} is {dtype}, expected bfloat16"
            )

    assert saw_quant_layer, "expected at least one QuantizedLinear leaf in 1.5B-AWQ"
    assert saw_non_quant_floating, (
        "expected at least one non-quant floating param (embed/layernorm/bias) "
        "in 1.5B-AWQ"
    )
    assert quant_dtype_pins, "no quantized floating params observed"
    # ``scales`` / ``biases`` are the AWQ-transform output and stay at
    # the transform's dtype (fp16 for this checkpoint). The alignment
    # step is intentionally selective; if it ever started casting these,
    # this assertion would catch it.
    quant_buffer_pins = {p for p in quant_dtype_pins if p[0] in ("scales", "biases")}
    assert quant_buffer_pins, "no QuantizedLinear scales/biases observed"
    assert all(p[1] == mx.float16 for p in quant_buffer_pins), (
        "AWQ-transform quant buffers must remain at the transform's dtype "
        f"(fp16 for this checkpoint); found: {quant_buffer_pins}"
    )
    # The regular linear ``bias`` on a QuantizedLinear (Qwen2 q/k/v
    # projections) is a normal floating param and MUST be aligned with
    # the engine runtime dtype, otherwise the projection emits
    # mixed-dtype activations into a bf16 KV cache / sampler.
    regular_bias_pins = {p for p in quant_dtype_pins if p[0] == "bias"}
    assert regular_bias_pins, (
        "expected at least one QuantizedLinear with a regular bias on "
        "Qwen2.5-AWQ (q/k/v projections)"
    )
    assert all(p[1] == mx.bfloat16 for p in regular_bias_pins), (
        "QuantizedLinear regular bias must be cast to the runtime target "
        f"dtype (bfloat16); found: {regular_bias_pins}"
    )
