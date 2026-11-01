// SPDX-License-Identifier: Apache-2.0
// Matches vLLM MRv2's FP32 Murmur3 Gumbel-max stream (seed, position, token).
// https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/sample/gumbel.py#L74-L206
#include <metal_stdlib>
using namespace metal;

inline uint rotate32(uint x, uint n) { return (x << n) | (x >> (32 - n)); }
inline uint mix32(uint h, uint k) {
    k = rotate32(k * 0xCC9E2D51u, 15) * 0x1B873593u;
    return rotate32(h ^ k, 13) * 5u + 0xE6546B64u;
}
inline float uniform32(long seed, long position, uint token) {
    uint h = mix32(0, uint(seed));
    h = mix32(h, uint(ulong(seed) >> 32));
    h = mix32(h, uint(position));
    h = mix32(h, token) ^ 16u;
    h = (h ^ (h >> 16)) * 0x85EBCA6Bu;
    h = (h ^ (h >> 13)) * 0xC2B2AE35u;
    h ^= h >> 16;
    return float(h >> 16) * 1.52587890625e-05f
         + (float(h & 0xFFFF) + 0.5f) * 2.3283064365386963e-10f;
}
inline float log1p_neg(float u) {
    float p = 1.0f / 8;
    for (int i = 7; i >= 1; --i) p = 1.0f / i + u * p;
    return u < 0.25f ? -u * p : log(max(1.0f - u, 5.960464477539063e-08f));
}

kernel void gumbel_sample(
    const device float* logits [[buffer(0)]],
    const device long* mapping [[buffer(1)]],
    const device long* seeds [[buffer(2)]],
    const device long* positions [[buffer(3)]],
    const device float* temperatures [[buffer(4)]],
    device long* argmax [[buffer(5)]],
    device float* maximum [[buffer(6)]],
    constant long& vocab [[buffer(7)]],
    constant long& block_size [[buffer(8)]],
    constant long& blocks [[buffer(9)]],
    constant long& apply_temperature [[buffer(10)]],
    const device uint* token_ids [[buffer(11)]],
    constant long& indexed [[buffer(12)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    long slot = mapping[group.y];
    float temp = slot < 0 ? 0.0f : temperatures[slot];
    long seed = slot < 0 ? 0 : seeds[slot];
    long position = positions[group.y];
    float best = -INFINITY;
    uint index = UINT_MAX;
    for (long col = long(group.x) * block_size + tid;
         col < min(vocab, (long(group.x) + 1) * block_size); col += 256) {
        float value = logits[long(group.y) * vocab + col];
        uint token = indexed ? token_ids[long(group.y) * vocab + col] : uint(col);
        if (temp != 0.0f && value != -INFINITY) {
            if (apply_temperature && temp != 1.0f) value /= temp;
            value -= log(-log1p_neg(uniform32(seed, position, token)));
        }
        // Mask NaNs so an all-invalid row still selects its lowest candidate ID.
        if (isnan(value)) value = -INFINITY;
        if (value > best || (value == best && token < index)) {
            best = value;
            index = token;
        }
    }
    threadgroup float maxima[8];
    threadgroup uint indices[8];
    float simd_best = simd_max(best);
    uint simd_index = simd_min(best == simd_best ? index : UINT_MAX);
    if (lane == 0) {
        maxima[simd_group] = simd_best;
        indices[simd_group] = simd_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        best = lane < 8 ? maxima[lane] : -INFINITY;
        index = lane < 8 ? indices[lane] : UINT_MAX;
        simd_best = simd_max(best);
        simd_index = simd_min(best == simd_best ? index : UINT_MAX);
        if (lane == 0) {
            long out = long(group.y) * blocks + group.x;
            maximum[out] = simd_best;
            argmax[out] = long(simd_index);
        }
    }
}
