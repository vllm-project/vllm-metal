# SPDX-License-Identifier: Apache-2.0
"""Gather and dequantize TQ pages directly into FP16/BF16 attention inputs."""

from functools import lru_cache
from typing import Any

import mlx.core as mx

from vllm_metal.attention.caches.turboquant import (
    FWHT_SUPPORTED_HEAD_DIMS,
    QUANT_PARAMS,
    get_fwht_signs,
)

# One SIMD group owns one (token, KV head) vector. The inverse FWHT follows
# kernels_v2/turboquant.metal: shuffle stages, then snapshot/commit in registers.
# FP32 never becomes a context-sized device array, and the original cache views
# are read with their real strides rather than copied to a contiguous pool.
_SOURCE = """
    const uint lane = thread_position_in_grid.x % 32;
    const size_t vector = thread_position_in_grid.x / 32;
    const uint head = vector % KV_HEADS;
    const size_t token = vector / KV_HEADS;
    const size_t page = pages[token * pages_strides[0]];
    const size_t offset = offsets[token * offsets_strides[0]];
    const size_t kb = page * k_data_strides[0] + offset * k_data_strides[1]
                    + head * k_data_strides[2];
    const size_t vb = page * v_data_strides[0] + offset * v_data_strides[1]
                    + head * v_data_strides[2];
    const size_t ksb = page * k_scale_strides[0] + offset * k_scale_strides[1]
                     + head * k_scale_strides[2];
    const size_t kzb = page * k_zero_strides[0] + offset * k_zero_strides[1]
                     + head * k_zero_strides[2];
    const size_t vsb = page * v_scale_strides[0] + offset * v_scale_strides[1]
                     + head * v_scale_strides[2];
    constexpr uint E = HEAD_DIM / 32;
    float values[E];
    #pragma unroll
    for (uint e = 0; e < E; ++e) {
        const uint d = lane + e * 32;
        float key;
        if (K_BITS == 8) {
            // The input's generated Metal type preserves signed int8 keys.
            key = float(k_data[kb + d * k_data_strides[3]]);
        } else {
            const uint bit = d * K_BITS;
            const uint byte = bit / 8;
            const uint shift = bit % 8;
            uint raw = uint(k_data[kb + byte * k_data_strides[3]]) >> shift;
            if (shift + K_BITS > 8)
                raw |= uint(k_data[kb + (byte + 1) * k_data_strides[3]]) << (8 - shift);
            key = float(raw & ((1u << K_BITS) - 1));
        }
        key = (key + float(k_zero[kzb + e * k_zero_strides[3]]))
                   * float(k_scale[ksb + e * k_scale_strides[3]]);
        k_out[vector * HEAD_DIM + d] = T(key);

        const uint bit = d * V_BITS;
        const uint byte = bit / 8;
        const uint shift = bit % 8;
        uint raw = uint(v_data[vb + byte * v_data_strides[3]]) >> shift;
        if (shift + V_BITS > 8)
            raw |= uint(v_data[vb + (byte + 1) * v_data_strides[3]]) << (8 - shift);
        values[e] = centroids[raw & ((1u << V_BITS) - 1)]
                  * float(v_scale[vsb + e * v_scale_strides[3]]);
    }
    #pragma unroll
    for (uint mask = 1; mask < 32; mask <<= 1) {
        #pragma unroll
        for (uint e = 0; e < E; ++e) {
            const float other = simd_shuffle_xor(values[e], mask);
            values[e] = (lane & mask) ? other - values[e] : values[e] + other;
        }
    }
    #pragma unroll
    for (uint mask = 1; mask < E; mask <<= 1) {
        float next[E];
        #pragma unroll
        for (uint e = 0; e < E; ++e)
            next[e] = (e & mask) ? values[e ^ mask] - values[e]
                                 : values[e] + values[e ^ mask];
        #pragma unroll
        for (uint e = 0; e < E; ++e) values[e] = next[e];
    }
    constexpr float inv_sqrt = HEAD_DIM == 64 ? 0.125f
                            : HEAD_DIM == 128 ? 0.08838834764831843f
                            : HEAD_DIM == 256 ? 0.0625f
                            : 0.04419417382415922f;
    #pragma unroll
    for (uint e = 0; e < E; ++e) {
        const uint d = lane + e * 32;
        v_out[vector * HEAD_DIM + d] = T(values[e] * inv_sqrt * signs[d]);
    }
"""


@lru_cache(maxsize=1)
def _kernel() -> Any:
    return mx.fast.metal_kernel(
        name="tq_materialize_pages",
        input_names=[
            "k_data",
            "v_data",
            "k_scale",
            "k_zero",
            "v_scale",
            "pages",
            "offsets",
            "centroids",
            "signs",
        ],
        output_names=["k_out", "v_out"],
        source=_SOURCE,
        ensure_row_contiguous=False,
    )


def materialize_turboquant_pages(
    k_data: mx.array,
    v_data: mx.array,
    k_scale: mx.array,
    k_zero: mx.array,
    v_scale: mx.array,
    pages: mx.array,
    offsets: mx.array,
    centroids: mx.array,
    *,
    head_dim: int,
    key_quant_type: str,
    value_bits: int,
    output_dtype: mx.Dtype,
) -> tuple[mx.array, mx.array]:
    """Read fresh writer handles with original page/token strides; write K/V once."""
    if head_dim not in FWHT_SUPPORTED_HEAD_DIMS or output_dtype not in (
        mx.float16,
        mx.bfloat16,
    ):
        raise ValueError(
            "TQ materialization requires FP16/BF16 and head_dim 64/128/256/512; "
            f"got head_dim={head_dim}, output_dtype={output_dtype}"
        )
    shape = (pages.size, k_data.shape[2], head_dim)
    k, v = _kernel()(
        inputs=[
            k_data,
            v_data,
            k_scale,
            k_zero,
            v_scale,
            pages,
            offsets,
            centroids,
            get_fwht_signs(head_dim),
        ],
        template=[
            ("T", output_dtype),
            ("HEAD_DIM", head_dim),
            ("KV_HEADS", shape[1]),
            ("K_BITS", QUANT_PARAMS[key_quant_type]["bits"]),
            ("V_BITS", value_bits),
        ],
        grid=(pages.size * shape[1] * 32, 1, 1),
        threadgroup=(min(128, pages.size * shape[1] * 32), 1, 1),
        output_shapes=[shape, shape],
        output_dtypes=[output_dtype, output_dtype],
    )
    return k, v
