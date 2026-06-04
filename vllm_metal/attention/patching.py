# SPDX-License-Identifier: Apache-2.0
"""Model-graph walking and the shared paged-attention patch loop.

``find_layers`` / ``find_attn_attr`` locate the attention submodule inside an
mlx_lm (or mlx-vlm) model; ``walk_and_wrap`` is the single loop every paged
attention runtime uses to install its wrappers — fail-loud, no silent skip.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------


def find_layers(model: Any) -> list[Any]:
    """Find transformer layers in an mlx_lm / mlx-vlm model.

    Supports model structures like:
        model.language_model.model.layers   (VLMs)
        model.model.layers
        model.layers
    """
    # Unwrap VLM wrapper (e.g. LLaVA, Pixtral via mlx-vlm)
    root = getattr(model, "language_model", model)
    # Try root.model.layers (Qwen3 Model wrapper)
    layers_container = getattr(root, "model", root)
    if hasattr(layers_container, "layers"):
        return layers_container.layers
    elif hasattr(root, "layers"):
        return root.layers
    else:
        raise ValueError(
            f"Cannot find transformer layers in model of type {type(model)}"
        )


# Attribute names to probe on each layer, in priority order.
_ATTN_ATTR_NAMES = ("self_attn", "linear_attn", "attention")


def find_attn_attr(layer: Any) -> str | None:
    """Return the attention attribute name for a single layer, or None."""
    for name in _ATTN_ATTR_NAMES:
        if hasattr(layer, name):
            return name
    return None


# ---------------------------------------------------------------------------
# Shared patch loop
# ---------------------------------------------------------------------------


def walk_and_wrap(
    model: Any,
    wrap_layer: Any,
    *,
    only_layers: list[int] | None = None,
) -> int:
    """Walk a model's attention layers and install paged-attention wrappers.

    This is the single patch loop shared by every paged attention runtime.  For
    each transformer layer that has an attention submodule, it calls
    ``wrap_layer(layer_idx, attn)`` and installs the returned wrapper.

    ``wrap_layer`` must return the wrapper to install — either a freshly built
    one, or the *existing* wrapper after rebinding its cache refs in place (in
    which case it returns the same object and no re-assignment happens).  It
    must **raise** if it cannot handle a layer: there is no silent skip.  That
    is what closes the gap where a misclassified hybrid layer used to keep its
    original, unpaged attention.

    Args:
        wrap_layer: callable ``(layer_idx, attn) -> wrapper``.
        only_layers: if given, only patch these layer indices; others are left
            untouched (used by KV-sharing models that patch a subset).

    Returns the number of patched layers.
    """
    only_set = set(only_layers) if only_layers is not None else None
    patched = 0
    for layer_idx, layer in enumerate(find_layers(model)):
        if only_set is not None and layer_idx not in only_set:
            continue
        attn_attr = find_attn_attr(layer)
        if attn_attr is None:
            continue
        attn = getattr(layer, attn_attr)
        wrapper = wrap_layer(layer_idx, attn)
        if wrapper is None:
            raise RuntimeError(
                f"walk_and_wrap: wrap_layer returned None for layer {layer_idx} "
                f"({type(attn).__name__}); it must return a wrapper or raise."
            )
        if wrapper is not attn:
            setattr(layer, attn_attr, wrapper)
        patched += 1
    return patched
