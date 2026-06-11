# SPDX-License-Identifier: Apache-2.0
"""Q8_0 GGUF Metal primitive wrappers."""

from __future__ import annotations

import mlx.core as mx

from vllm_metal.gguf.tensor import (
    GGUF_QTYPE_Q8_0,
    GGUFQuantizedTensor,
)


def q8_0_matmul(
    x: mx.array,
    weight: GGUFQuantizedTensor,
    *,
    output_dtype: mx.Dtype | None = None,
) -> mx.array:
    """Multiply activations by a raw GGUF Q8_0 weight.

    ``weight`` stores the logical matrix as ``(output_dims, input_dims)`` and
    this primitive computes ``x @ weight.T``. Inputs must be fp16, bf16, or
    fp32; the output dtype is the input dtype unless ``output_dtype`` requests a
    cast before dispatch.
    """
    if weight.qtype_id != GGUF_QTYPE_Q8_0:
        raise ValueError(f"q8_0_matmul requires Q8_0, got {weight.qtype_name}")
    if len(x.shape) == 0:
        raise ValueError("q8_0_matmul input must have at least one dimension")
    if x.shape[-1] != weight.input_dims:
        raise ValueError(
            f"Input dimension {x.shape[-1]} does not match GGUF weight "
            f"input dimension {weight.input_dims}"
        )

    x_2d = x.reshape(-1, weight.input_dims)
    if output_dtype is not None:
        if not mx.issubdtype(output_dtype, mx.floating):
            raise ValueError(f"output_dtype must be floating, got {output_dtype}")
        x_2d = x_2d.astype(output_dtype)
    elif not mx.issubdtype(x_2d.dtype, mx.floating):
        raise ValueError(f"q8_0_matmul input must be floating, got {x_2d.dtype}")

    x_2d = mx.contiguous(x_2d)
    qweight = mx.contiguous(weight.qweight)
    out = mx.zeros((x_2d.shape[0], weight.output_dims), dtype=x_2d.dtype)

    from vllm_metal.metal import ensure_gguf_ops

    ensure_gguf_ops().gguf_q8_0_mul_mat(
        x_2d,
        qweight,
        weight.qtype_id,
        out,
    )
    return out.reshape(*x.shape[:-1], weight.output_dims)
