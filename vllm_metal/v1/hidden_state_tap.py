# SPDX-License-Identifier: Apache-2.0
"""Capture qualified Qwen3 residuals while executing the native MLX body.

The temporary body shares weights and layers with the target, but owns its
observer list. The live target is never patched, even if a forward raises.
Native embedding, attention masks, positions and final normalization therefore
remain the responsibility of mlx-lm's Qwen3 implementation.
"""

from __future__ import annotations

from copy import copy
from typing import Any

import mlx.core as mx
from mlx_lm.models.qwen3 import Qwen3Model


class _LayerCapture:
    def __init__(self, layer: Any, outputs: list[mx.array]) -> None:
        self.layer = layer
        self.outputs = outputs

    def __call__(self, *args: Any, **kwargs: Any) -> mx.array:
        hidden = self.layer(*args, **kwargs)
        self.outputs.append(hidden)
        return hidden


def run_backbone_with_capture(
    backbone: Any,
    input_ids: mx.array,
    *,
    cache: Any,
    layer_ids: list[int],
) -> tuple[mx.array, mx.array]:
    """Return native final hidden and fused post-layer residuals from one pass.

    Feature IDs are physical decoder indices, in strictly increasing order.
    This helper captures pre-final-norm residuals; the serving config rejects
    final-layer and embedding taps whose training convention is unqualified.
    Only Qwen3 is supported. Other families need their own capture qualification.
    """
    if not isinstance(backbone, Qwen3Model):
        raise NotImplementedError("DSpark target capture currently requires Qwen3Model")
    if not layer_ids:
        raise ValueError("run_backbone_with_capture requires at least one layer_id")
    layers = backbone.layers
    if any(type(i) is not int or i < 0 or i >= len(layers) for i in layer_ids):
        raise IndexError(
            f"layer_ids {layer_ids} out of range for backbone with {len(layers)} layers"
        )
    if any(a >= b for a, b in zip(layer_ids, layer_ids[1:], strict=False)):
        raise ValueError("layer_ids must be strictly increasing")
    if cache is not None and len(cache) != len(layers):
        raise ValueError("target cache must have one entry per decoder layer")

    captured: list[mx.array] = []
    taps = set(layer_ids)
    # nn.Module is a dict: a shallow copy keeps all parameters and native layer
    # objects shared. Replacing the copy's list does not mutate the live model,
    # its parameter tree, or the paged-attention wrappers installed on its layers.
    body = copy(backbone)
    body.layers = [
        _LayerCapture(layer, captured) if i in taps else layer
        for i, layer in enumerate(layers)
    ]
    final = body(input_ids, cache=cache)
    if len(captured) != len(layer_ids):
        raise RuntimeError("native Qwen3 forward did not visit every requested layer")
    return final, mx.concatenate(captured, axis=-1)
