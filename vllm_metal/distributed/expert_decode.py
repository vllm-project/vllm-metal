# SPDX-License-Identifier: Apache-2.0
"""Single-token MXFP4 expert projections that skip unowned slots on the GPU."""

from functools import cache

import mlx.core as mx


def supports_experts(experts):
    """Keep other quantizers and layouts on the existing MLX path."""
    for proj in (experts.gate_proj, experts.up_proj, experts.down_proj):
        if (
            getattr(proj, "mode", None) != "mxfp4"
            or proj.group_size != 32
            or proj.bits != 4
            or proj.weight.dtype != mx.uint32
            or proj.scales.dtype != mx.uint8
            or proj.weight.ndim != 3
            or proj.weight.shape[-1] % 4
        ):
            return False
    return True


@cache
def _projection_kernel():
    return mx.fast.metal_kernel(
        name="ep_masked_mxfp4_qmv",
        input_names=["x", "w", "scales", "indices", "bias"],
        output_names=["out"],
        header="constant float ep_fp4[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};",
        source=r"""
        uint slot = threadgroup_position_in_grid.z;
        uint row = threadgroup_position_in_grid.y * 4 + simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        if (row >= N) return;
        uint global_expert = indices[slot];
        if (global_expert < START || global_expert >= END) {
            if (lane == 0) out[slot*N+row] = T(0);
            return;
        }
        uint expert = global_expert - START;
        uint offset = (expert*N + row) * (K/8);
        uint soffset = (expert*N + row) * (K/32);
        uint xoffset = PER_SLOT ? slot*K : 0;
        float acc = 0;
        for (uint p = lane; p < K/8; p += 32) {
            uint packed = w[offset+p];
            uint exponent = scales[soffset+p/4];
            float scale = exponent == 0 ? as_type<float>(uint(0x00400000))
                          : exponent == 255 ? NAN : as_type<float>(exponent << 23);
            float dot = 0;
            for (uint j=0; j<8; ++j) {
                uint v = (packed >> (4*j)) & 15;
                float weight = ep_fp4[v & 7] * ((v & 8) ? -1.0f : 1.0f);
                dot += float(x[xoffset+p*8+j]) * weight;
            }
            acc += dot * scale;
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            T value = T(acc);
            if (HAS_BIAS) value = T(value + T(bias[expert*N+row]));
            out[slot*N+row] = value;
        }
        """,
        ensure_row_contiguous=True,
    )


def masked_expert_projection(proj, x, indices, start, end, per_slot=False):
    n = proj.weight.shape[1]
    k = proj.weight.shape[2] * 8
    out = _projection_kernel()(
        inputs=[
            x,
            proj.weight,
            proj.scales,
            indices,
            proj.get("bias", mx.zeros((1,), dtype=x.dtype)),
        ],
        template=[
            ("T", x.dtype),
            ("K", k),
            ("N", n),
            ("START", start),
            ("END", end),
            ("PER_SLOT", per_slot),
            ("HAS_BIAS", "bias" in proj),
        ],
        grid=(128, (n + 3) // 4, indices.size),
        threadgroup=(128, 1, 1),
        output_shapes=[(*indices.shape, n)],
        output_dtypes=[x.dtype],
    )[0]
    return out


def decode_local_experts(e, x, indices, start, end):
    up = masked_expert_projection(e.up_proj, x, indices, start, end)
    gate = masked_expert_projection(e.gate_proj, x, indices, start, end)
    return masked_expert_projection(
        e.down_proj, e.activation(up, gate), indices, start, end, True
    )
