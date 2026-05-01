# SPDX-License-Identifier: Apache-2.0
"""M-RoPE position helpers for Qwen3.5-4B multimodal prompts."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from vllm_metal.v1.multimodal.feature_spec import MultiModalFeatureSpec


def _qwen2_5_vl_cls():
    from vllm.model_executor.models.qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )

    return Qwen2_5_VLForConditionalGeneration


class _MropeShim:
    """Duck-typed object for upstream vLLM's mm_features M-RoPE helper."""

    def __init__(self, *, spatial_merge_size: int, tokens_per_second: float) -> None:
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(
                spatial_merge_size=spatial_merge_size,
                tokens_per_second=tokens_per_second,
            )
        )

    def iter_mm_grid_thw(self, mm_features: list[MultiModalFeatureSpec]):
        yield from _qwen2_5_vl_cls().iter_mm_grid_thw(self, mm_features)


# === Public Helpers ===


def get_mrope_input_positions(
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
    *,
    spatial_merge_size: int,
    # Image prompts use the upstream default. Video support must pass the real
    # value from vision_config.tokens_per_second when that adapter is added.
    tokens_per_second: float = 1.0,
) -> tuple[mx.array, int]:
    """Return ``((3, seq_len) int32 positions, mrope_position_delta)``.

    Calls upstream vLLM's mm_features-driven Qwen2.5-VL M-RoPE helper through a
    minimal duck-typed shim, then converts the returned torch tensor to an MLX
    array.  This keeps the position-building policy upstream-owned while the
    vllm-metal runner can consume MLX arrays.

    Image prompts usually produce a negative ``mrope_position_delta`` because
    multiple placeholder tokens share compact 3D position coordinates.
    """
    if not input_tokens:
        return mx.zeros((3, 0), dtype=mx.int32), 0

    shim = _MropeShim(
        spatial_merge_size=spatial_merge_size,
        tokens_per_second=tokens_per_second,
    )
    torch_positions, mrope_position_delta = _qwen2_5_vl_cls().get_mrope_input_positions(
        shim,
        input_tokens,
        mm_features,
    )
    llm_positions = np.asarray(torch_positions.cpu().numpy())
    return mx.array(llm_positions, dtype=mx.int32), int(mrope_position_delta)
