// TurboQuant KV cache compression helpers for Metal paged attention.
// This file is included by pagedattention.metal after Vec<> is declared.
//
// TurboQuant uses:
// - K: Asymmetric uniform quantization (int8/uint8 or sub-8-bit packed)
// - V: 3-bit Lloyd-Max quantization with FWHT rotation
//
// Copyright contributors to the vLLM project
// Licensed under the Apache License 2.0

#pragma once

// NOTE: This header is included by pagedattention.metal which already has:
//   #include <metal_stdlib>
//   #include <metal_simdgroup>
//   using namespace metal;
//   template <typename T, int VEC_SIZE> struct Vec {};

// ========================================== int8 (char) vector data types
// Used by TurboQuant int8 K cache (K_CACHE_T = char).

struct Char8_ {
    char4 x;
    char4 y;
};

template <> struct Vec<char, 1> { using Type = char; };
template <> struct Vec<char, 2> { using Type = char2; };
template <> struct Vec<char, 4> { using Type = char4; };
template <> struct Vec<char, 8> { using Type = Char8_; };

// ========================================== Type trait for signed char

template <typename T> inline constexpr bool is_char() { return false; }
template <> inline constexpr bool is_char<char>() { return true; }

// ========================================== Sub-8-bit unpacking

// Generic sub-8-bit unpack from a packed byte stream.
// Layout: element i occupies bits [i*bits, i*bits + bits) in the packed
// byte stream (little-endian within each byte). A value spans at most
// two consecutive bytes for any bits <= 8.
inline uint unpack_k_bits(const device uchar* bytes, int elem_idx, int bits) {
    int bit_pos = elem_idx * bits;
    int byte_idx = bit_pos >> 3;        // bit_pos / 8
    int bit_offset = bit_pos & 7;       // bit_pos % 8
    uint raw = uint(bytes[byte_idx]);
    if (bit_offset + bits > 8) {
        raw |= uint(bytes[byte_idx + 1]) << 8;
    }
    return (raw >> bit_offset) & ((1u << bits) - 1u);
}

// Unpack a single 3-bit value from packed bytes (8 values per 3 bytes).
// Used for V cache.
inline uchar unpack_3bit(const device uchar* packed, int elem_idx) {
    int group = elem_idx / 8;
    int pos = elem_idx % 8;
    int byte_base = group * 3;
    uint b0 = packed[byte_base];
    uint b1 = packed[byte_base + 1];
    uint b2 = packed[byte_base + 2];
    uint combined = b0 | (b1 << 8) | (b2 << 16);
    return uchar((combined >> (pos * 3)) & 0x7);
}

// ========================================== FWHT random sign tables
// Deterministic random signs — matches Python: key=mx.random.key(42)
// Generated via: signs = 1 - 2 * mx.random.randint(0, 2, shape=(N,), key=mx.random.key(42))

constant float FWHT_SIGNS_64[64] = {
     1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f,
    -1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f,
     1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f,
     1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f
};

constant float FWHT_SIGNS_128[128] = {
    -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f,
    -1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,
     1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f,
     1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,
    -1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,
    -1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f,
     1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f, -1.f,  1.f,
     1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f
};

constant float FWHT_SIGNS_256[256] = {
     1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,
     1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,
    -1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f,
    -1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f,
    -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f,
    -1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f,
    -1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,
     1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,
    -1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f,
    -1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f,
    -1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,
    -1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f, -1.f,
    -1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f,
     1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f,
    -1.f, -1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,
    -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f
};

// ========================================== K dequantization

// TurboQuant K dequant: asymmetric uniform quantization — (idx + zp) * scale
// Uchar overload for uint8 and sub-8-bit raw values (unsigned).
inline float tq_dequant_k(uchar index, float scale, float zero_point) {
    return (float(index) + zero_point) * scale;
}

// Char overload for int8 / q8_0 (signed).
inline float tq_dequant_k(char index, float scale, float zero_point) {
    return (float(index) + zero_point) * scale;
}

// Explicit uint overload for sub-8-bit raw values returned by unpack_k_bits.
inline float tq_dequant_k_raw(uint index, float scale, float zero_point) {
    return (float(index) + zero_point) * scale;
}

// ========================================== V dequantization

// TurboQuant V dequant: Lloyd-Max centroid lookup (arbitrary bit width)
// centroids: pointer to 2^bits centroid values (passed from Python)
// v_bits: quantization bit width (used to mask index)
inline float tq_dequant_v_centroid(uchar index, float scale, const device float* centroids, int v_bits) {
    uint mask = (1u << v_bits) - 1u;
    return centroids[index & mask] * scale;
}

// ========================================== FWHT sign lookup

// FWHT sign lookup by HEAD_SIZE — supported for 64, 128, and 256.
// The primary template returns 1.f and is never called for TQ-enabled kernels
// (runtime guard in MetalPagedKVCache enforces valid sizes), but must exist
// to satisfy the compiler for non-TQ specializations of other head sizes.
template<int HEAD_SIZE> inline float get_fwht_sign(uint idx) { return 1.f; }
template<> inline float get_fwht_sign<64>(uint idx)  { return FWHT_SIGNS_64[idx]; }
template<> inline float get_fwht_sign<128>(uint idx) { return FWHT_SIGNS_128[idx]; }
template<> inline float get_fwht_sign<256>(uint idx) { return FWHT_SIGNS_256[idx]; }

// ========================================== Inverse FWHT

// In-place inverse FWHT for HEAD_SIZE elements using threadgroup memory.
// Supports HEAD_SIZE = 64 (6 stages), 128 (7 stages), or 256 (8 stages).
// Each SIMD lane owns HEAD_SIZE/32 elements (lane i → indices i, i+32, i+64, ...).
template<int HEAD_SIZE>
inline void threadgroup_inverse_fwht(threadgroup float* fwht_buf, uint lane) {
    constexpr int NUM_STAGES = (HEAD_SIZE == 64) ? 6 : (HEAD_SIZE == 128) ? 7 : 8;
    constexpr int ELEMS_PER_LANE = HEAD_SIZE / 32;
    float vals[ELEMS_PER_LANE];

    #pragma unroll
    for (int stage = 0; stage < NUM_STAGES; stage++) {
        uint mask = 1u << stage;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            vals[e] = fwht_buf[lane + e * 32];
        }
        if (mask < 32) {
            // Partners in different lanes — safe to read from threadgroup memory
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_LANE; e++) {
                uint idx = lane + e * 32;
                float partner_val = fwht_buf[idx ^ mask];
                fwht_buf[idx] = (idx & mask) ? (partner_val - vals[e]) : (vals[e] + partner_val);
            }
        } else {
            // Partner owned by this thread at a different e offset
            float results[ELEMS_PER_LANE];
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_LANE; e++) {
                uint idx = lane + e * 32;
                uint partner_idx = idx ^ mask;
                int partner_e = static_cast<int>((partner_idx - lane) / 32);
                float partner_val = vals[partner_e];
                results[e] = (idx & mask) ? (partner_val - vals[e]) : (vals[e] + partner_val);
            }
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_LANE; e++) {
                fwht_buf[lane + e * 32] = results[e];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Normalisation: 1/sqrt(HEAD_SIZE) + random sign flip
    constexpr float INV_SQRT_N = (HEAD_SIZE == 64) ? 0.125f : (HEAD_SIZE == 128) ? 0.08838834764831843f : 0.0625f;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++) {
        uint idx = lane + e * 32;
        fwht_buf[idx] *= INV_SQRT_N * get_fwht_sign<HEAD_SIZE>(idx);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// ========================================== High-level K/V load helpers

// Load and dequantize a K vector for TurboQuant.
// Handles both 8-bit (char/uchar) and sub-8-bit packed formats.
template <typename T, typename K_CACHE_T, int VEC_SIZE>
inline void tq_load_k_vec(
    thread typename Vec<T, VEC_SIZE>::Type& k_vec_out,
    const device K_CACHE_T* k_ptr,
    const device half* key_scale_cache,
    const device half* key_zero_cache,
    int64_t k_scale_base_offset,
    int vec_idx,
    int k_bits
) {
    constexpr int SCALE_GROUP_SIZE = 32;
    using K_vec = typename Vec<T, VEC_SIZE>::Type;
    K_vec k_vec_result;
    thread T* result_ptr = reinterpret_cast<thread T*>(&k_vec_result);

    if constexpr (is_char<K_CACHE_T>()) {
        // int8 K path (signed)
        const device K_CACHE_T* k_elem_ptr = k_ptr + vec_idx * VEC_SIZE;
        #pragma unroll
        for (int e = 0; e < VEC_SIZE; e++) {
            int elem_idx = vec_idx * VEC_SIZE + e;
            int group_idx = elem_idx / SCALE_GROUP_SIZE;
            float s = key_scale_cache[k_scale_base_offset + group_idx];
            float z = key_zero_cache[k_scale_base_offset + group_idx];
            result_ptr[e] = T(tq_dequant_k(k_elem_ptr[e], s, z));
        }
    } else {
        // uchar K path (uint8 or sub-8-bit packed)
        if (k_bits >= 8) {
            // 8-bit unsigned: one byte per element
            const device uchar* k_elem_ptr = reinterpret_cast<const device uchar*>(k_ptr) + vec_idx * VEC_SIZE;
            #pragma unroll
            for (int e = 0; e < VEC_SIZE; e++) {
                int elem_idx = vec_idx * VEC_SIZE + e;
                int group_idx = elem_idx / SCALE_GROUP_SIZE;
                float s = key_scale_cache[k_scale_base_offset + group_idx];
                float z = key_zero_cache[k_scale_base_offset + group_idx];
                result_ptr[e] = T(tq_dequant_k(k_elem_ptr[e], s, z));
            }
        } else {
            // Sub-8-bit packed: unpack k_bits bits per logical element
            const device uchar* k_bytes = reinterpret_cast<const device uchar*>(k_ptr);
            #pragma unroll
            for (int e = 0; e < VEC_SIZE; e++) {
                int elem_idx = vec_idx * VEC_SIZE + e;
                int group_idx = elem_idx / SCALE_GROUP_SIZE;
                float s = key_scale_cache[k_scale_base_offset + group_idx];
                float z = key_zero_cache[k_scale_base_offset + group_idx];
                uint raw = unpack_k_bits(k_bytes, elem_idx, k_bits);
                result_ptr[e] = T(tq_dequant_k_raw(raw, s, z));
            }
        }
    }
    k_vec_out = k_vec_result;
}

// Generic sub-v_bits unpack from a packed byte stream (mirrors unpack_k_bits).
inline uint unpack_v_bits(const device uchar* bytes, int elem_idx, int bits) {
    int bit_pos = elem_idx * bits;
    int byte_idx = bit_pos >> 3;        // bit_pos / 8
    int bit_offset = bit_pos & 7;       // bit_pos % 8
    uint raw = uint(bytes[byte_idx]);
    if (bit_offset + bits > 8) {
        raw |= uint(bytes[byte_idx + 1]) << 8;
    }
    return (raw >> bit_offset) & ((1u << bits) - 1u);
}

// Load, dequantize, inverse-FWHT, and accumulate a V vector for TurboQuant.
// This handles the full V pipeline: unpack v_bits → centroid lookup → FWHT → accumulate.
// v_bits: quantization bit width (passed as function constant from kernel)
// v_centroids: pointer to 2^v_bits centroid values
template <int HEAD_SIZE, int NUM_SIMD_LANES>
inline void tq_load_and_accumulate_v(
    thread float* v_accs,
    threadgroup float* fwht_buf,
    const device uchar* v_ptr,
    const device half* value_scale_cache,
    int64_t v_scale_base_offset,
    float weight,
    uint lane,
    const device float* v_centroids,
    int v_bits
) {
    constexpr int SCALE_GROUP_SIZE = 32;
    constexpr int V_ELEMS_PER_THREAD = (HEAD_SIZE + NUM_SIMD_LANES - 1) / NUM_SIMD_LANES;

    // Load packed v_bits V, dequantize via centroid lookup, write to fwht_buf
    #pragma unroll
    for (int i = 0; i < V_ELEMS_PER_THREAD; i++) {
        const int d = lane + i * NUM_SIMD_LANES;
        if (d < HEAD_SIZE) {
            int group_idx = d / SCALE_GROUP_SIZE;
            float vs = value_scale_cache[v_scale_base_offset + group_idx];
            uchar v_idx = (v_bits == 3) ? unpack_3bit(v_ptr, d) : uchar(unpack_v_bits(v_ptr, d, v_bits));
            fwht_buf[d] = tq_dequant_v_centroid(v_idx, vs, v_centroids, v_bits);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Apply inverse FWHT to reconstruct the original V vector
    threadgroup_inverse_fwht<HEAD_SIZE>(fwht_buf, lane);

    // Accumulate: O += weight * V_reconstructed
    #pragma unroll
    for (int i = 0; i < V_ELEMS_PER_THREAD; i++) {
        const int d = lane + i * NUM_SIMD_LANES;
        if (d < HEAD_SIZE) {
            v_accs[i] += weight * fwht_buf[d];
        }
    }
}
