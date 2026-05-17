# SPDX-License-Identifier: Apache-2.0
"""Slow dtype-alignment coverage for the AWQ load path on the dense matrix.

Three additional dense AWQ checkpoints beyond `test_awq_e2e_qwen25_15b.py`:
Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3.
Parametrized so future dense AWQ rows are one tuple line rather than a
new file.

* `test_awq_load_aligns_non_quant_dtypes_to_runner_target` -- dtype
  alignment after load. Architecture-aware: only Qwen2 has biases on
  q/k/v projections (it sets `bias=True` on each), so `regular_bias_pins`
  is asserted non-empty only for rows that opt in via
  `has_regular_quant_bias`. Llama-family architectures (Llama-3, Mistral)
  set `attention_bias=False` and their QuantizedLinear leaves carry only
  the AWQ-transform output buffers, no regular `bias` parameter.

No parametrized vLLM `LLM.generate` smoke test in this file, by design:
the 1.5B file already exercises the AWQ + Metal paged runner combination
end-to-end on `Qwen/Qwen2.5-1.5B-Instruct-AWQ`. Adding the same smoke
test for three more 7-8B models in the same pytest process leaks wired
Metal allocations between LLM instances (mx.clear_cache releases the
cache but not wired memory), so the second model load SIGABRTs in the
MLX allocator. The 1.5B smoke is sufficient as a paged-runner regression
guard; quantitative parity for these 3 models (cos/KL/top-1 vs bf16
reference, plus AWQ-repack vs naive MLX 4bit) is in
`tools/awq_parity.py`, run standalone -- numbers in the PR description.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
import torch
from mlx.utils import tree_flatten

from tests.stub_runner import make_stub_runner
from vllm_metal.v1 import model_lifecycle
from vllm_metal.v1.model_lifecycle import ModelLifecycle

# (awq_repo, has_regular_quant_bias)
#
# ``has_regular_quant_bias``: Qwen2 sets ``bias=True`` on q/k/v
# projections (a normal floating param living alongside scales/biases on
# the same QuantizedLinear leaf); Llama-3 and Mistral use the llama
# architecture path with ``attention_bias=False``, so their
# QuantizedLinear leaves carry only AWQ-transform buffers, no separate
# regular bias.
_DENSE_AWQ_MATRIX = [
    pytest.param(
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
        True,
        id="qwen2.5-7b",
    ),
    pytest.param(
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        False,
        id="llama-3.1-8b",
    ),
    pytest.param(
        "solidrust/Mistral-7B-Instruct-v0.3-AWQ",
        False,
        id="mistral-7b-v0.3",
    ),
]


def _runner_model_config(repo: str, *, dtype):
    return SimpleNamespace(
        model=repo,
        hf_config=None,
        is_multimodal_model=False,
        trust_remote_code=False,
        dtype=dtype,
    )


def _release_metal_state() -> None:
    """Drop the process-level model cache and hand Metal allocations back
    to the pool. Required between parametrized rows so the next 7-8B AWQ
    load gets the full budget on a single pytest process.
    """
    model_lifecycle.reset_model_cache()
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def _collect_awq_dtype_pins(model, *, expected_non_quant_dtype):
    """Walk the model leaves once. For each non-quant floating param,
    assert it matches ``expected_non_quant_dtype`` with a leaf-specific
    error message. For each ``nn.QuantizedLinear``, record
    ``(param_name, dtype)`` pairs of its floating params for the caller
    to assert on. Returns ``(saw_quant_layer, saw_non_quant_floating,
    quant_dtype_pins)``.

    Defined as a helper for memory-cleanup reasons: the loop locals
    (``module``, ``name``, ``value``) stay bound to the last iteration's
    leaf (often a large embedding or ``lm_head`` buffer) until the
    enclosing frame returns. Putting the traversal in this helper lets
    those references go out of scope on ``return``, so the caller's
    ``_release_metal_state`` can actually reclaim the buffers between
    parametrized rows.
    """
    saw_quant_layer = False
    saw_non_quant_floating = False
    quant_dtype_pins: set[tuple[str, mx.Dtype]] = set()
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
            assert dtype == expected_non_quant_dtype, (
                f"non-quant floating param {name!r} on "
                f"{type(module).__name__} is {dtype}, "
                f"expected {expected_non_quant_dtype}"
            )
    return saw_quant_layer, saw_non_quant_floating, quant_dtype_pins


@pytest.mark.slow
@pytest.mark.parametrize("awq_repo,has_regular_quant_bias", _DENSE_AWQ_MATRIX)
def test_awq_load_aligns_non_quant_dtypes_to_runner_target(
    monkeypatch,
    awq_repo,
    has_regular_quant_bias,
):
    """After AWQ load, non-quantized floating params must match the
    runner's target dtype, and quantized layers' scales/biases must NOT
    have been touched by the alignment step (their dtype is owned by the
    AWQ transform).
    """
    monkeypatch.setattr(model_lifecycle, "_MODEL_CACHE", {})
    # Pre-bind references so the ``finally`` ``del`` works even if the
    # load throws before they would otherwise be assigned.
    runner = None
    lifecycle = None
    model = None
    tokenizer = None
    try:
        runner = make_stub_runner(
            model_config=_runner_model_config(awq_repo, dtype=torch.bfloat16)
        )
        lifecycle = ModelLifecycle(runner, runner._model_adapter)

        model, tokenizer = lifecycle._load_generation_model(awq_repo, is_vlm=False)

        saw_quant_layer, saw_non_quant_floating, quant_dtype_pins = (
            _collect_awq_dtype_pins(model, expected_non_quant_dtype=mx.bfloat16)
        )

        assert saw_quant_layer, (
            f"expected at least one QuantizedLinear leaf in {awq_repo}"
        )
        assert saw_non_quant_floating, (
            f"expected at least one non-quant floating param "
            f"(embed/layernorm/bias) in {awq_repo}"
        )
        assert quant_dtype_pins, "no quantized floating params observed"

        # AWQ-transform output buffers (scales/biases) stay at the
        # transform's dtype (fp16 for these checkpoints). Alignment must
        # not touch them.
        quant_buffer_pins = {
            p for p in quant_dtype_pins if p[0] in ("scales", "biases")
        }
        assert quant_buffer_pins, "no QuantizedLinear scales/biases observed"
        assert all(p[1] == mx.float16 for p in quant_buffer_pins), (
            f"AWQ-transform quant buffers must stay at the transform's dtype "
            f"(fp16 for these checkpoints); found: {quant_buffer_pins}"
        )

        # Regular ``bias`` parameter is architecture-dependent.
        regular_bias_pins = {p for p in quant_dtype_pins if p[0] == "bias"}
        if has_regular_quant_bias:
            assert regular_bias_pins, (
                f"expected at least one QuantizedLinear with a regular bias "
                f"on {awq_repo} (Qwen2 q/k/v projections)"
            )
            assert all(p[1] == mx.bfloat16 for p in regular_bias_pins), (
                f"QuantizedLinear regular bias must be cast to the runtime "
                f"target dtype (bfloat16); found: {regular_bias_pins}"
            )
        else:
            assert not regular_bias_pins, (
                f"{awq_repo} uses attention_bias=False; QuantizedLinear "
                f"leaves should not carry a regular bias parameter, but "
                f"found: {regular_bias_pins}"
            )
    finally:
        # Drop references to the loaded model and its construction
        # scaffolding BEFORE ``_release_metal_state`` runs. Python keeps
        # locals alive until the frame returns, so ``mx.clear_cache()``
        # inside the helper would otherwise see the model's MLX buffers
        # as still-referenced and leave their wired allocations in place
        # for the next parametrized row. ``del`` here forces the
        # refcount to drop so ``gc.collect()`` + ``mx.clear_cache()``
        # actually reclaim the memory.
        del model, tokenizer, lifecycle, runner
        _release_metal_state()
