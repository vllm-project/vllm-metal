# SPDX-License-Identifier: Apache-2.0
"""KV cache dtype inference and policy.

The Metal paged-attention backend stores *activation* K/V tensors in an
MLX-native cache. Those tensors must be floating point. Some models may have
quantized *weights* (e.g. int8), so we must not derive the KV cache dtype from
weights without enforcing a float-only policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from vllm_metal.paged_attention_common import find_layers_and_attr

DEFAULT_KV_CACHE_DTYPE: mx.Dtype = mx.float16
ALLOWED_KV_CACHE_DTYPES: frozenset[mx.Dtype] = frozenset(
    {
        mx.float16,
        mx.bfloat16,
        mx.float32,
    }
)


@dataclass(frozen=True)
class KvCacheDtypeInference:
    """Result of inferring the KV cache dtype from a model."""

    dtype: mx.Dtype
    warning: str | None = None


def infer_kv_cache_dtype_from_model(
    model: Any, *, default: mx.Dtype = DEFAULT_KV_CACHE_DTYPE
) -> KvCacheDtypeInference:
    """Infer a float KV-cache dtype from an MLX(-LM/-VLM) model.

    Policy:
    - If the model's attention weight dtype is a supported float dtype, use it.
    - Otherwise, fall back to *default* and provide a warning string the caller
      may log.
    """
    try:
        layers, attn_attr = find_layers_and_attr(model)
        if not layers:
            raise ValueError("model has no transformer layers")

        attn = getattr(layers[0], attn_attr)
        # If the model is already patched, unwrap to the real attention module.
        attn = getattr(attn, "_inner", attn)

        mlx_dtype = attn.q_proj.weight.dtype
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        return KvCacheDtypeInference(
            dtype=default,
            warning=f"Cannot infer KV cache dtype from model ({exc}); using {default}",
        )

    if mlx_dtype not in ALLOWED_KV_CACHE_DTYPES:
        return KvCacheDtypeInference(
            dtype=default,
            warning=(
                f"Model weight dtype {mlx_dtype!r} is not a supported float dtype; "
                f"using {default} for KV cache instead"
            ),
        )

    return KvCacheDtypeInference(dtype=mlx_dtype)
