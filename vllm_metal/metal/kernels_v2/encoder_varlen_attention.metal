// SPDX-License-Identifier: Apache-2.0
//
// Dense non-causal varlen encoder attention kernel.
//
// Grid:           (total_tokens, num_q_heads, 1)
// Threadgroup:    (32, 1, 1) — exactly one SIMD-group on Apple Silicon.
//
// One threadgroup computes one (query_token, q_head) output vector via a
// two-pass safe softmax (max → renormalize; distinct from the running
// (m, l, O) online softmax in sibling kernels_v2/pagedattention.metal):
//   pass 1 — segment-wide max via lane-strided keys + simd_max.
//   pass 2 — denom + per-dim numerator partials, simd_sum reductions, lane-
//            strided writeback (lane d % 32 owns dim d).
//
// Per-lane register state across both passes is q_reg[HEAD_DIM] (loaded
// once and reused) plus num_partial[HEAD_DIM] (live in pass 2), both fp32.
// At HEAD_DIM=128 that is ~1 KB/lane plus a handful of scalars, which must
// fit in the per-lane register file on Apple Silicon.  The perfgate
// (compile-smoke + perf-ratio) bounds register-spill regressions against
// this ceiling.
//
// Softmax runs in log2 space: scale is folded as `scale * log2(e)` at the
// top of the kernel so the pre-max score is already log2-scaled, and the
// pass-2 weight is `exp2(score - seg_max)` (1-instruction on Apple GPUs)
// rather than base-e `exp`.  Math identity: exp(qk*scale - m) ==
// exp2(qk*scale*log2e - m) when m has been accumulated in log2-scaled
// space (which it has — pass 1 takes the max of qk*scale_log2).  Matches
// the convention in sibling kernels_v2/pagedattention.metal.
//
// utils.metal is concatenated ahead of this file by the source builder in
// vllm_metal.metal.__init__, so bfloat16_t is available.

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Hand-mirrored on the C++ side at paged_ops.cpp.  Field order is part of
// the ABI; do not reorder without updating the C++ struct in lockstep.
struct EncoderVarlenParams {
    int   num_q_heads;
    int   num_kv_heads;
    int   total_tokens;
    int   num_segments;   // == cu_seqlens.shape[0] - 1; binary-search bound.
    int   max_seqlen;     // Reserved ABI / launch hint only; kernel must not
                          // read this unless host-side is_equivalent is updated.
    float softmax_scale;
};

// Metal-side ABI tripwire mirroring the C++ static_asserts in paged_ops.cpp.
// needs_rebuild() in build.py only stats the
// .cpp, so a .metal-only struct edit would JIT-compile cleanly and
// silently misdecode the C++ bytes; this size check fires at JIT time
// when the .metal struct drifts from the C++ side.  sizeof catches add /
// remove / type-widen drift; same-size reorders still need the C++-side
// per-field offsetof checks (asymmetric on purpose — Metal does not
// expose the standard `offsetof` macro through metal_stdlib).
static_assert(sizeof(EncoderVarlenParams) == 24,
              "EncoderVarlenParams ABI drift: size diverges from "
              "paged_ops.cpp; update both files in lockstep.");

// Binary-search cu_seqlens to find the segment containing token_idx.
// cu_seqlens is sorted ascending: [0, len_0, len_0+len_1, ..., total_tokens].
// Returns seg such that cu_seqlens[seg] <= token_idx < cu_seqlens[seg + 1].
inline int find_segment(const device int* cu_seqlens,
                        int token_idx, int num_segments) {
    int lo = 0, hi = num_segments;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (cu_seqlens[mid] <= token_idx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

template <typename T, uint HEAD_DIM>
[[kernel]] void encoder_varlen_attention_kernel(
    device const T*       q           [[buffer(0)]],
    device const T*       k           [[buffer(1)]],
    device const T*       v           [[buffer(2)]],
    device const int*     cu_seqlens  [[buffer(3)]],
    device       T*       out         [[buffer(4)]],
    constant     EncoderVarlenParams& params [[buffer(5)]],
    uint3 tg_id     [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]])
{
    const int token_idx = int(tg_id.x);
    const int head_idx  = int(tg_id.y);
    const int num_q     = params.num_q_heads;
    const int num_kv    = params.num_kv_heads;
    // Fold log2(e) into the scale so the pre-max score is already in
    // log2 space and pass-2 can use exp2 (see top-of-file comment).
    const float scale   = params.softmax_scale * M_LOG2E_F;

    const int seg     = find_segment(cu_seqlens, token_idx, params.num_segments);
    const int seg_lo  = cu_seqlens[seg];
    const int seg_hi  = cu_seqlens[seg + 1];

    // Q pointer for this (token, head).  v1 requires num_q_heads == num_kv_heads
    // (validated host-side).  K/V indexing already uses the num_kv stride, but
    // the GQA follow-up still needs a kernel edit to map head_idx -> kv_head_idx.
    const device T* q_ptr = q + (token_idx * num_q + head_idx) * HEAD_DIM;

    // v1 cost (deferred to the GQA-mapping follow-up):
    //   1. Every lane in the SIMD-group loads the full Q vector below — 32×
    //      redundant device traffic.  Successor should load HEAD_DIM/32 dims
    //      per lane and broadcast the rest via simd_shuffle.
    //   2. Pass 2 below recomputes Q·K rather than caching pass-1 dot
    //      products.  A threadgroup-scratch (or shuffle-based) cache would
    //      halve the pass-2 K-stream traffic.
    //   3. The pass-2 numerator reduction loop runs one full-warp simd_sum
    //      per output dim — at HEAD_DIM=128 that is 128 reductions where
    //      only one lane consumes each result (the writer selected by
    //      `simd_lane == d % 32`).  A reduce-scatter or shuffle-rotated
    //      layout where each lane owns disjoint dim subsets would cut the
    //      reduction count; the exact ratio depends on the chosen design.
    // Load Q into per-lane fp32 register array.
    float q_reg[HEAD_DIM];
    for (uint d = 0; d < HEAD_DIM; ++d) {
        q_reg[d] = float(q_ptr[d]);
    }

    // ---------------------------------------------------------------
    // Pass 1: segment-wide max.
    // ---------------------------------------------------------------
    float lane_max = -INFINITY;
    for (int j = seg_lo + int(simd_lane); j < seg_hi; j += 32) {
        const device T* k_ptr = k + (j * num_kv + head_idx) * HEAD_DIM;
        float qk = 0.0f;
        for (uint d = 0; d < HEAD_DIM; ++d) {
            qk += q_reg[d] * float(k_ptr[d]);
        }
        qk *= scale;
        lane_max = max(lane_max, qk);
    }
    const float seg_max = simd_max(lane_max);

    // ---------------------------------------------------------------
    // Pass 2: denom + per-dim numerator partials.
    //
    // Each lane re-streams *its assigned keys only* (lane-strided by 32) and
    // updates a HEAD_DIM-wide per-lane numerator vector.  Every lane updates
    // every dim because every output dim needs every assigned key's
    // contribution.
    // ---------------------------------------------------------------
    float denom_partial = 0.0f;
    float num_partial[HEAD_DIM];
    for (uint d = 0; d < HEAD_DIM; ++d) {
        num_partial[d] = 0.0f;
    }

    for (int j = seg_lo + int(simd_lane); j < seg_hi; j += 32) {
        const device T* k_ptr = k + (j * num_kv + head_idx) * HEAD_DIM;
        const device T* v_ptr = v + (j * num_kv + head_idx) * HEAD_DIM;
        float qk = 0.0f;
        for (uint d = 0; d < HEAD_DIM; ++d) {
            qk += q_reg[d] * float(k_ptr[d]);
        }
        qk = qk * scale - seg_max;
        const float w = exp2(qk);
        denom_partial += w;
        for (uint d = 0; d < HEAD_DIM; ++d) {
            num_partial[d] += w * float(v_ptr[d]);
        }
    }

    const float denom = simd_sum(denom_partial);
    const float inv_denom = 1.0f / denom;

    device T* out_ptr = out + (token_idx * num_q + head_idx) * HEAD_DIM;

    // Lane-strided writeback: lane (d % 32) owns dim d.  For HEAD_DIM=80 the
    // count is uneven (lanes 0..15 own 3 dims, lanes 16..31 own 2), but the
    // (simd_lane == d % 32) form handles that without a per-head-dim branch.
    for (uint d = 0; d < HEAD_DIM; ++d) {
        const float full_num_d = simd_sum(num_partial[d]);
        if (simd_lane == (d % 32)) {
            out_ptr[d] = T(full_num_d * inv_denom);
        }
    }
}

// ---------------------------------------------------------------------------
// Specialization macros — generates 12 [[host_name]] instantiations:
//   {f16, bf16, f32} × {64, 80, 96, 128}
//
// host_name spelling is part of the ABI; the C++ dispatch helper builds the
// same kname string from (dtype_tag, head_dim).  An off-by-one between this
// macro and the C++ name builder silently mis-binds an argument with no
// Metal-side error.
// ---------------------------------------------------------------------------

#define instantiate_encoder_varlen_attention_inner(type, type_tag, head_dim) \
    template [[host_name("encoder_varlen_attention_" #type_tag "_" #head_dim)]] \
    [[kernel]] void encoder_varlen_attention_kernel<type, head_dim>( \
        device const type*       q           [[buffer(0)]], \
        device const type*       k           [[buffer(1)]], \
        device const type*       v           [[buffer(2)]], \
        device const int*        cu_seqlens  [[buffer(3)]], \
        device       type*       out         [[buffer(4)]], \
        constant     EncoderVarlenParams& params [[buffer(5)]], \
        uint3 tg_id     [[threadgroup_position_in_grid]], \
        uint  simd_lane [[thread_index_in_simdgroup]]);

#define instantiate_encoder_varlen_attention(type, type_tag) \
    instantiate_encoder_varlen_attention_inner(type, type_tag, 64)  \
    instantiate_encoder_varlen_attention_inner(type, type_tag, 80)  \
    instantiate_encoder_varlen_attention_inner(type, type_tag, 96)  \
    instantiate_encoder_varlen_attention_inner(type, type_tag, 128)

instantiate_encoder_varlen_attention(half,       f16)
instantiate_encoder_varlen_attention(bfloat16_t, bf16)
instantiate_encoder_varlen_attention(float,      f32)
