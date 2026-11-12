# SPDX-License-Identifier: Apache-2.0
"""Runtime dtype of an AWQ load on Qwen2.5-1.5B-Instruct-AWQ."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
import torch

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
def test_awq_load_keeps_checkpoint_dtype():
    """The model keeps the checkpoint's fp16 when the runner resolves
    bfloat16; a partial cast would promote every activation to float32.
    """
    runner = make_stub_runner(model_config=_runner_model_config(dtype=torch.bfloat16))
    lifecycle = ModelLifecycle(runner, runner._model_adapter)

    model = lifecycle._load_generation_model(_AWQ_REPO, is_vlm=False)

    assert model(mx.array([[0]])).dtype == mx.float16
