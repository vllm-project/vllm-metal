# SPDX-License-Identifier: Apache-2.0
"""Slow end-to-end coverage for the AWQ load path on Qwen2.5-1.5B-Instruct-AWQ.

Two tests, both `slow`-marked (opt out with `pytest -m "not slow"`):

* `test_awq_load_aligns_non_quant_dtypes_to_runner_target`
  Loads via `_load_generation_model` with a stub runner whose target
  dtype is bf16. Asserts every non-`QuantizedLinear` floating param
  matches bf16 and that quantized scales/biases stay at the dtype the
  upstream transform produced (we explicitly do NOT touch those).

* `test_awq_e2e_paged_runner_smoke`
  Full vLLM `LLM.generate` through the Metal paged runner. Asserts the
  greedy continuation includes "Paris" — guards against any AWQ load
  regression that produces token-level garbage (mixed dtype mismatch,
  layernorm absorption broken, lm_head triple foot-gun, etc.).

Quantitative parity (cos > 0.97 vs bf16 reference, TF top-1 >= 90%) is
covered by the standalone parity scripts referenced in the RFC; we do
not re-run that comparison in CI because it requires loading three 1.5B
models in one process. The smoke + dtype tests are the regression
guards.
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

_AWQ_REPO = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"


def _runner_model_config(*, dtype):
    return SimpleNamespace(
        model=_AWQ_REPO,
        hf_config=None,
        is_multimodal_model=False,
        trust_remote_code=False,
        dtype=dtype,
    )


@pytest.mark.slow
def test_awq_load_aligns_non_quant_dtypes_to_runner_target(monkeypatch):
    """After AWQ load, non-quantized floating params must match the runner's
    target dtype, and quantized layers' scales/biases must NOT have been
    touched by the alignment step (their dtype is owned by the transform).
    """
    # Ensure no other test left a cached entry around for this model.
    monkeypatch.setattr(model_lifecycle, "_MODEL_CACHE", {})

    runner = make_stub_runner(model_config=_runner_model_config(dtype=torch.bfloat16))
    lifecycle = ModelLifecycle(runner, runner._model_adapter)

    model, _tokenizer = lifecycle._load_generation_model(_AWQ_REPO, is_vlm=False)

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


@pytest.mark.slow
def test_awq_e2e_paged_runner_smoke():
    """Full vLLM stack: load 1.5B-AWQ, greedy 16 tokens, check 'Paris'
    appears. Guards against AWQ load regressions that produce token
    garbage even when individual unit tests pass.
    """
    from vllm import LLM, SamplingParams

    llm = None
    monkeypatch_ctx = pytest.MonkeyPatch.context()
    try:
        with monkeypatch_ctx as mp:
            mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
            mp.setenv("VLLM_METAL_MEMORY_FRACTION", "0.3")

            llm = LLM(
                model=_AWQ_REPO,
                max_model_len=512,
                max_num_seqs=1,
            )
            sp = SamplingParams(temperature=0, max_tokens=16)
            outputs = llm.generate(["The capital of France is"], sp)

            assert len(outputs) == 1
            text = outputs[0].outputs[0].text
            assert text, "AWQ load produced empty output"
            # Greedy first-token should be " Paris" (verified by the standalone
            # parity scripts; reproduced here as a regression guard).
            assert "Paris" in text, (
                f"expected 'Paris' in AWQ greedy continuation, got: {text!r}"
            )
    finally:
        # Order matters: ``ModelLifecycle`` stores the AWQ model in the
        # process-level ``_MODEL_CACHE``, so ``del llm`` alone cannot
        # release the weights — the cache holds an independent strong
        # reference. Drop the cache entry first so the subsequent
        # ``del`` + ``gc.collect`` actually reclaim the model; otherwise
        # later slow tests in the same pytest process can OOM on 16 GB
        # Metal machines.
        model_lifecycle.reset_model_cache()
        del llm
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
