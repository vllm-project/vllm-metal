# SPDX-License-Identifier: Apache-2.0
"""Model-specific compatibility helpers for MetalModelRunner.

Houses per-model workarounds and dim-resolution logic so that
``model_runner.py`` stays model-agnostic (per its coding style guide).
Any Gemma4/Qwen/etc. specific behavior belongs here, not inline in
the shared runner.
"""

from typing import Any

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# Workaround: models that vLLM flags as multimodal but must be loaded
# via mlx_lm instead of mlx_vlm.  Reasons are model-specific.
#
# - "gemma4": mlx_vlm's Gemma4 forward pass emits garbled output on the
#   same checkpoint that mlx_lm decodes correctly.  This override can be
#   removed once the mlx_vlm Gemma4 implementation matches mlx_lm's.
_TEXT_BACKBONE_OVERRIDE_TYPES: frozenset[str] = frozenset({"gemma4"})


def should_force_text_backbone(hf_config: Any) -> bool:
    """Return True if the model should be loaded via mlx_lm despite
    vLLM flagging it as multimodal.

    Args:
        hf_config: HuggingFace config object (with ``model_type`` attr).

    Returns:
        True if this model_type needs the mlx_lm text-backbone override.
    """
    model_type = getattr(hf_config, "model_type", "")
    return model_type in _TEXT_BACKBONE_OVERRIDE_TYPES


# ---------------------------------------------------------------------------
# Variable head dimension (Gemma4-style)
# ---------------------------------------------------------------------------


def resolve_max_head_dim(
    args: dict[str, Any],
    head_dim: int | None,
) -> int | None:
    """Return the max head_dim across all attention layer types.

    Gemma4-style models use different ``head_dim`` for sliding attention
    (256) versus full attention (512) layers.  The paged KV cache is
    sized for a single head_dim, so we use the larger value to ensure
    every layer's K/V fits in a cache block.

    Args:
        args: Model config dict (expects optional ``global_head_dim``).
        head_dim: The base head_dim from config.

    Returns:
        ``max(head_dim, global_head_dim)`` if both are set, else
        ``head_dim``.
    """
    global_head_dim = args.get("global_head_dim")
    if global_head_dim and head_dim:
        return max(int(head_dim), int(global_head_dim))
    return head_dim


# ---------------------------------------------------------------------------
# Variable KV head count (Gemma4 26B/31B)
# ---------------------------------------------------------------------------


def require_uniform_kv_heads(
    args: dict[str, Any],
    num_kv_heads: int | None,
) -> None:
    """Reject models that vary KV head count across attention layer types.

    Gemma4 26B/31B set ``num_global_key_value_heads`` different from
    ``num_key_value_heads`` for full vs. sliding attention layers.  The
    paged KV cache currently assumes a single ``num_kv_heads`` — these
    models need per-layer KV head allocation which is not yet supported.

    Args:
        args: Model config dict.
        num_kv_heads: Resolved base ``num_key_value_heads``.

    Raises:
        ValueError: If ``num_global_key_value_heads`` differs from
            ``num_kv_heads`` (paged path cannot handle it).
    """
    global_kv_heads = args.get("num_global_key_value_heads")
    if global_kv_heads and num_kv_heads and int(global_kv_heads) != int(num_kv_heads):
        raise ValueError(
            f"Paged attention does not support variable KV head count: "
            f"num_key_value_heads={num_kv_heads}, "
            f"num_global_key_value_heads={global_kv_heads}. "
            f"Use VLLM_METAL_USE_PAGED_ATTENTION=0 to fall back to the "
            f"non-paged path."
        )
