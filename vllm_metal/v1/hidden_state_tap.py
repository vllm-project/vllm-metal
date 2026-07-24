# SPDX-License-Identifier: Apache-2.0
"""Reusable hidden-state tap: capture a target backbone's residual stream after
an arbitrary selection of layers and fuse them.

Both the existing Gemma4 MTP path (last layer) and DSpark (several specific
intermediate layers) become users of this one helper. Mirrors the explicit
transformer-layer traversal used by the reference implementation: embed,
mask, loop over layers calling ``layer(h, mask, c)``, apply the final norm,
and capture the residual after each requested index (concatenated along the
feature axis).

``run_backbone_with_capture`` returns BOTH the post-norm final hidden (so the
target's own logits can be computed from the same forward — the integrated path
cannot afford a second pass) and the fused intermediate captures (for the
drafter). ``capture_layer_hidden_states`` is a fused-only view over it.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask


def run_backbone_with_capture(
    backbone: Any,
    input_ids: mx.array,
    *,
    cache: Any,
    layer_ids: list[int],
) -> tuple[mx.array, mx.array]:
    """Run ``backbone``'s full layer loop, returning final + fused residuals.

    Args:
        backbone: an MLX transformer body exposing ``embed_tokens``, ``layers``,
            and ``norm`` — i.e. what ``_target_backbone`` returns
            (``text_model(model).model``).
        input_ids: ``[batch, tokens]``.
        cache: per-layer cache list, passed straight through to each layer. The
            paged KV routing for patched models happens inside the attention via
            the active step context, not via this arg.
        layer_ids: 0-indexed layer positions to capture; residuals are fused in
            this order along the feature axis.

    Returns:
        ``(final, fused)`` where ``final`` is the post-norm hidden after the last
        layer (suitable for the target lm_head) and ``fused`` is
        ``[batch, tokens, len(layer_ids) * hidden]``.
    """
    if not layer_ids:
        raise ValueError("run_backbone_with_capture requires at least one layer_id")
    layers = backbone.layers
    if any(i < 0 or i >= len(layers) for i in layer_ids):
        raise IndexError(
            f"layer_ids {layer_ids} out of range for backbone with {len(layers)} layers"
        )

    tapset = set(layer_ids)
    h = backbone.embed_tokens(input_ids)
    # Match the body's own mask creation; patched attention reads paged KV from
    # the active step context regardless of this arg.
    mask = create_attention_mask(h, cache[0] if cache else None)
    captured: list[mx.array] = []
    for i, (layer, c) in enumerate(zip(layers, cache, strict=True)):
        h = layer(h, mask, c)
        if i in tapset:
            captured.append(h)
    final = backbone.norm(h)
    fused = mx.concatenate(captured, axis=-1)
    return final, fused


def capture_layer_hidden_states(
    backbone: Any,
    input_ids: mx.array,
    *,
    cache: Any,
    layer_ids: list[int],
) -> mx.array:
    """Fused-only view over :func:`run_backbone_with_capture`.

    Returns ``[batch, tokens, len(layer_ids) * hidden]`` — just the fused
    intermediate residuals, for callers (e.g. the M0 sanity harness) that feed
    a drafter without needing the target's own logits.
    """
    _, fused = run_backbone_with_capture(
        backbone, input_ids, cache=cache, layer_ids=layer_ids
    )
    return fused
