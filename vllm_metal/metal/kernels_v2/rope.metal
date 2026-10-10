// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

#define ROPE_ARGS(T) \
    const device T* q [[buffer(0)]], \
    const device T* k [[buffer(1)]], \
    const device T* cache [[buffer(2)]], \
    const device long* positions [[buffer(3)]], \
    device T* out_q [[buffer(4)]], \
    device T* out_k [[buffer(5)]], \
    constant long* shape [[buffer(6)]], \
    uint2 index [[thread_position_in_grid]]

// One thread rotates a pair; all packed Q/K rows share one dispatch.
// shape = {q_width, k_width, head_size, q_row_stride, k_row_stride}.
template <typename T>
kernel void rope_qk(ROPE_ARGS(T)) {
    const uint q_width = shape[0], k_width = shape[1], head = shape[2];
    const uint pair = index.x, row = index.y, half_head = head / 2;
    if (pair >= (q_width + k_width) / 2) return;

    const bool is_q = pair < q_width / 2;
    const uint local_pair = is_q ? pair : pair - q_width / 2;
    const uint lane = local_pair % half_head;
    const uint col = local_pair / half_head * head + lane;
    const device T* input = is_q ? q : k;
    device T* output = is_q ? out_q : out_k;
    const long stride = is_q ? shape[3] : shape[4];
    const uint width = is_q ? q_width : k_width;
    const long cache_row = positions[row] * head;
    const float c = float(cache[cache_row + lane]);
    const float s = float(cache[cache_row + lane + half_head]);
    const float x = float(input[long(row) * stride + col]);
    const float y = float(input[long(row) * stride + col + half_head]);
    // Match upstream's separate FP16/BF16 products before the add/subtract.
    output[long(row) * width + col] = T(float(T(x * c)) - float(T(y * s)));
    output[long(row) * width + col + half_head] = T(float(T(y * c)) + float(T(x * s)));
}

template [[host_name("rope_qk_fp16")]] kernel void rope_qk<half>(ROPE_ARGS(half));
template [[host_name("rope_qk_bf16")]] kernel void rope_qk<bfloat>(ROPE_ARGS(bfloat));
#undef ROPE_ARGS
