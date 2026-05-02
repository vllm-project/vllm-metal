# SPDX-License-Identifier: Apache-2.0
"""M-RoPE position helpers for Qwen3.5-4B multimodal prompts."""

from __future__ import annotations

from functools import cache
from types import SimpleNamespace

import mlx.core as mx

from vllm_metal.v1.multimodal.feature_spec import (
    MultiModalFeatureSpec,
    iter_image_grid_thw,
)

# === Internal Shim ===


@cache
def _qwen3_vl_cls():
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

    return Qwen3VLForConditionalGeneration


def _qwen3_vl_image_config(spatial_merge_size: int) -> SimpleNamespace:
    """Build the minimal config read by Qwen3-VL's image-only M-RoPE path."""
    # The token-id fields are only read for video; keep sentinels here so the
    # static upstream helper can be reused without constructing a full model.
    return SimpleNamespace(
        video_token_id=-1,
        vision_start_token_id=-1,
        vision_end_token_id=-1,
        vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size),
    )


def _validate_image_features(
    mm_features: list[MultiModalFeatureSpec],
    *,
    spatial_merge_size: int,
) -> None:
    for _ in iter_image_grid_thw(
        mm_features,
        spatial_merge_size=spatial_merge_size,
    ):
        pass


# === Public Helpers ===


def get_mrope_input_positions(
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
    *,
    spatial_merge_size: int,
) -> tuple[mx.array, int]:
    """Return ``((3, seq_len) int32 positions, mrope_position_delta)``.

    Calls upstream vLLM's mm_features-driven Qwen3-VL M-RoPE helper with a
    minimal image-only config shim, then converts the returned torch tensor to
    an MLX array.  This keeps the position-building policy upstream-owned while
    the vllm-metal runner can consume MLX arrays.

    Image prompts usually produce a negative ``mrope_position_delta`` because
    multiple placeholder tokens share compact 3D position coordinates.
    """
    if spatial_merge_size <= 0:
        raise ValueError(
            f"spatial_merge_size must be positive, got {spatial_merge_size}"
        )

    if not input_tokens:
        return mx.zeros((3, 0), dtype=mx.int32), 0

    _validate_image_features(
        mm_features,
        spatial_merge_size=spatial_merge_size,
    )

    torch_positions, mrope_position_delta = _qwen3_vl_cls()._get_mrope_input_positions(
        input_tokens=input_tokens,
        mm_features=mm_features,
        config=_qwen3_vl_image_config(spatial_merge_size),
    )
    llm_positions = torch_positions.cpu().numpy()
    return mx.array(llm_positions, dtype=mx.int32), int(mrope_position_delta)
