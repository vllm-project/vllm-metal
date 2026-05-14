# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for multimodal runtime tests."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from vllm_metal.multimodal.qwen3_vl import Qwen3VLVisionEncodeResult


@pytest.fixture
def fake_encode_result():
    """Return a builder for ``Qwen3VLVisionEncodeResult`` cache entries."""

    def _make(
        hidden_states: mx.array,
        *,
        deepstack: Any | None = None,
    ) -> Qwen3VLVisionEncodeResult:
        return Qwen3VLVisionEncodeResult(
            hidden_states=hidden_states,
            deepstack_visual_embeds=deepstack,
        )

    return _make
