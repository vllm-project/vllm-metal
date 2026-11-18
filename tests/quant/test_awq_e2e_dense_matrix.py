# SPDX-License-Identifier: Apache-2.0
"""Slow runtime-dtype coverage for the AWQ load path on the dense matrix.

Three additional dense AWQ checkpoints beyond `test_awq_e2e_qwen25_15b.py`:
Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3.
Parametrized so future dense AWQ rows are one tuple line rather than a
new file.

* `test_awq_load_keeps_checkpoint_dtype` -- the model runs in the
  checkpoint's fp16 when the runner resolves bfloat16. The rows cover
  both projection shapes: Qwen2 has a regular `bias` on q/k/v, the
  Llama-family architectures (Llama-3, Mistral) do not.

No parametrized vLLM `LLM.generate` smoke test in this file, by design:
`tests/e2e/test_awq_e2e_qwen25_15b.py` exercises the AWQ + Metal paged
runner combination end-to-end on `Qwen/Qwen2.5-1.5B-Instruct-AWQ`. Adding the same smoke
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
import pytest
import torch

from tests.stub_runner import make_stub_runner
from vllm_metal.v1.model_lifecycle import ModelLifecycle

_DENSE_AWQ_MATRIX = [
    pytest.param("Qwen/Qwen2.5-7B-Instruct-AWQ", id="qwen2.5-7b"),
    pytest.param(
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", id="llama-3.1-8b"
    ),
    pytest.param("solidrust/Mistral-7B-Instruct-v0.3-AWQ", id="mistral-7b-v0.3"),
]


def _runner_model_config(repo: str, *, dtype):
    return SimpleNamespace(
        model=repo,
        revision=None,
        hf_config=None,
        is_multimodal_model=False,
        trust_remote_code=False,
        dtype=dtype,
        quantization=None,
    )


def _release_metal_state() -> None:
    """Hand Metal allocations back to the pool between parametrized rows."""
    gc.collect()
    mx.clear_cache()


@pytest.mark.slow
@pytest.mark.parametrize("awq_repo", _DENSE_AWQ_MATRIX)
def test_awq_load_keeps_checkpoint_dtype(awq_repo):
    """The model keeps the checkpoint's fp16 when the runner resolves
    bfloat16; a partial cast would promote every activation to float32.
    """
    # Pre-bind references so the ``finally`` ``del`` works even if the
    # load throws before they would otherwise be assigned.
    runner = None
    lifecycle = None
    model = None
    try:
        runner = make_stub_runner(
            model_config=_runner_model_config(awq_repo, dtype=torch.bfloat16)
        )
        lifecycle = ModelLifecycle(runner, runner._model_adapter)

        model = lifecycle._load_generation_model(awq_repo, is_vlm=False)

        assert model(mx.array([[0]])).dtype == mx.float16
    finally:
        # Drop references to the loaded model and its construction
        # scaffolding BEFORE ``_release_metal_state`` runs. Python keeps
        # locals alive until the frame returns, so ``mx.clear_cache()``
        # inside the helper would otherwise see the model's MLX buffers
        # as still-referenced and leave their wired allocations in place
        # for the next parametrized row. ``del`` here forces the
        # refcount to drop so ``gc.collect()`` + ``mx.clear_cache()``
        # actually reclaim the memory.
        del model, lifecycle, runner
        _release_metal_state()
