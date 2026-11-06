// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

#define ADD_RMS_NORM_ARGS(T, W) \
    const device T* x [[buffer(0)]], \
    const device T* residual [[buffer(1)]], \
    const device W* weight [[buffer(2)]], \
    device T* output [[buffer(3)]], \
    device T* out_residual [[buffer(4)]], \
    constant long& width [[buffer(5)]], \
    constant float& epsilon [[buffer(6)]], \
    uint2 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]]

// Each row uses 256 threads (eight SIMD groups), including partial rows.
// vLLM rounds normalized values to the weight dtype before multiplication:
// FP32 weights (e.g. GemmaRMSNorm) keep the normalization and product in FP32.
template <typename T, typename W, bool AddResidual>
inline void normalize_row(
    const device T* x, const device T* residual, const device W* weight,
    device T* output, device T* out_residual, long width, float epsilon,
    uint2 group, uint tid, uint lane, uint simd_group,
    threadgroup float* partial) {
    const long row = long(group.y) * width;
    float squares = 0.0f;
    for (long col = tid; col < width; col += 256) {
        float value = float(x[row + col]);
        if constexpr (AddResidual) value += float(residual[row + col]);
        squares += value * value;
        if constexpr (AddResidual) out_residual[row + col] = T(value);
    }

    const float group_squares = simd_sum(squares);
    if (lane == 0) partial[simd_group] = group_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const float total = simd_sum(lane < 8 ? partial[lane] : 0.0f);
        if (lane == 0) partial[0] = rsqrt(total / float(width) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (long col = tid; col < width; col += 256) {
        // The stored residual is rounded; variance and normalization must use
        // the original FP32 sum, then round before multiplying the weight.
        float value = float(x[row + col]);
        if constexpr (AddResidual) value += float(residual[row + col]);
        const W normalized = W(value * partial[0]);
        output[row + col] = T(float(normalized) * float(weight[col]));
    }
}

template <typename T, typename W>
kernel void add_rms_norm(ADD_RMS_NORM_ARGS(T, W)) {
    threadgroup float partial[8];
    normalize_row<T, W, true>(x, residual, weight, output, out_residual,
        width, epsilon, group, tid, lane, simd_group, partial);
}

template [[host_name("add_rms_norm_fp16")]] kernel void add_rms_norm<half, half>(ADD_RMS_NORM_ARGS(half, half));
template [[host_name("add_rms_norm_bf16")]] kernel void add_rms_norm<bfloat, bfloat>(ADD_RMS_NORM_ARGS(bfloat, bfloat));
template [[host_name("add_rms_norm_fp16_weight_float")]] kernel void add_rms_norm<half, float>(ADD_RMS_NORM_ARGS(half, float));
template [[host_name("add_rms_norm_bf16_weight_float")]] kernel void add_rms_norm<bfloat, float>(ADD_RMS_NORM_ARGS(bfloat, float));
#undef ADD_RMS_NORM_ARGS

#define RMS_NORM_ARGS(T) \
    const device T* x [[buffer(0)]], \
    const device float* weight [[buffer(1)]], \
    device T* output [[buffer(2)]], \
    constant long& width [[buffer(3)]], \
    constant float& epsilon [[buffer(4)]], \
    uint2 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]]

template <typename T>
kernel void rms_norm_weight_float(RMS_NORM_ARGS(T)) {
    threadgroup float partial[8];
    normalize_row<T, float, false>(x, nullptr, weight, output, nullptr,
        width, epsilon, group, tid, lane, simd_group, partial);
}

template [[host_name("rms_norm_fp16_weight_float")]] kernel void rms_norm_weight_float<half>(RMS_NORM_ARGS(half));
template [[host_name("rms_norm_bf16_weight_float")]] kernel void rms_norm_weight_float<bfloat>(RMS_NORM_ARGS(bfloat));
#undef RMS_NORM_ARGS
