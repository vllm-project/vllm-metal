// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

#define ADD_RMS_NORM_ARGS(T) \
    const device T* x [[buffer(0)]], \
    const device T* residual [[buffer(1)]], \
    const device T* weight [[buffer(2)]], \
    device T* output [[buffer(3)]], \
    device T* out_residual [[buffer(4)]], \
    constant long& width [[buffer(5)]], \
    constant float& epsilon [[buffer(6)]], \
    uint2 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]]

// Each row uses 256 threads (eight SIMD groups), including partial rows.
template <typename T>
kernel void add_rms_norm(ADD_RMS_NORM_ARGS(T)) {
    const long row = long(group.y) * width;
    float squares = 0.0f;
    for (long col = tid; col < width; col += 256) {
        const float value = float(x[row + col]) + float(residual[row + col]);
        squares += value * value;
        out_residual[row + col] = T(value);
    }

    threadgroup float partial[8];
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
        const float value = float(x[row + col]) + float(residual[row + col]);
        const T normalized = T(value * partial[0]);
        output[row + col] = T(float(normalized) * float(weight[col]));
    }
}

template [[host_name("add_rms_norm_fp16")]] kernel void add_rms_norm<half>(ADD_RMS_NORM_ARGS(half));
template [[host_name("add_rms_norm_bf16")]] kernel void add_rms_norm<bfloat>(ADD_RMS_NORM_ARGS(bfloat));
#undef ADD_RMS_NORM_ARGS
