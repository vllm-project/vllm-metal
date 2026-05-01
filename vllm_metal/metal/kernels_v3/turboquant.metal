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

constant float FWHT_SIGNS_512[512] = {
    -1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f,
     1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f,
    -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f,
    -1.f, -1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,
    -1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f,
    -1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f,
    -1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,
     1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,
    -1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f,
    -1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f,
    -1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f,
    -1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f, -1.f,
     1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f,
     1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f,
     1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,
     1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f,
     1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f,  1.f,
    -1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f,
     1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f,
     1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f,
    -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,
     1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,
    -1.f, -1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f,
    -1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,
     1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f,
     1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f,
     1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f,
    -1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,
    -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f,
     1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,
    -1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f,
    -1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f
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

// FWHT sign lookup by HEAD_SIZE — supported for 64, 128, 256, and 512.
// The primary template returns 1.f and is never called for TQ-enabled kernels
// (runtime guard in MetalPagedKVCache enforces valid sizes), but must exist
// to satisfy the compiler for non-TQ specializations of other head sizes.
template<int HEAD_SIZE> inline float get_fwht_sign(uint idx) { return 1.f; }
template<> inline float get_fwht_sign<64>(uint idx)  { return FWHT_SIGNS_64[idx]; }
template<> inline float get_fwht_sign<128>(uint idx) { return FWHT_SIGNS_128[idx]; }
template<> inline float get_fwht_sign<256>(uint idx) { return FWHT_SIGNS_256[idx]; }
template<> inline float get_fwht_sign<512>(uint idx) { return FWHT_SIGNS_512[idx]; }

template<int HEAD_SIZE> inline constexpr int fwht_num_stages() { return 0; }
template<> inline constexpr int fwht_num_stages<64>() { return 6; }
template<> inline constexpr int fwht_num_stages<128>() { return 7; }
template<> inline constexpr int fwht_num_stages<256>() { return 8; }
template<> inline constexpr int fwht_num_stages<512>() { return 9; }

template<int HEAD_SIZE> inline constexpr float fwht_inv_sqrt_n() { return 1.f; }
template<> inline constexpr float fwht_inv_sqrt_n<64>() { return 0.125f; }
template<> inline constexpr float fwht_inv_sqrt_n<128>() { return 0.08838834764831843f; }
template<> inline constexpr float fwht_inv_sqrt_n<256>() { return 0.0625f; }
template<> inline constexpr float fwht_inv_sqrt_n<512>() { return 0.04419417382415922f; }

// ========================================== Inverse FWHT

// In-place inverse FWHT on a thread-local register array `vals` of size
// ELEMS_PER_LANE.  Each SIMD lane owns the slice
// { lane, lane+32, lane+64, ... } of the logical HEAD_SIZE-vector.
// The caller must supply `ELEMS_PER_LANE` explicitly; a `static_assert`
// enforces the only valid coupling (one element per 32 lanes) so any future
// mis-sized register array is a compile error, not silent OOB register access.
//
// Entirely register-resident:
//   * Stages where mask < 32 exchange values across lanes of the same simd-
//     group via `simd_shuffle_xor` — no threadgroup memory, no barriers.
//   * Stages where mask >= 32 are intra-thread: the partner lives in the
//     same lane at a different `e` offset.  RAW hazard across `e` iterations
//     is handled by the snapshot-then-commit pattern (memory 86c1ae49).
//
// Called from tq_load_and_accumulate_v which dequantises V straight into
// registers, runs this FWHT, and accumulates into v_accs — there is no
// threadgroup-memory round-trip on the V read path.  Supports HEAD_SIZE
// = 64 (6 stages), 128 (7 stages), 256 (8 stages), or 512 (9 stages).
template<int HEAD_SIZE, int ELEMS_PER_LANE>
inline void inverse_fwht_in_place(thread float* vals, uint lane) {
    static_assert(ELEMS_PER_LANE == HEAD_SIZE / 32,
        "inverse_fwht_in_place: ELEMS_PER_LANE must equal HEAD_SIZE/32; "
        "callers allocate `vals[V_ELEMS_PER_THREAD]` at NUM_SIMD_LANES=32 and "
        "passing a mismatched size would walk off the register array silently.");
    constexpr int NUM_STAGES = fwht_num_stages<HEAD_SIZE>();

    // Stages 0-4: intra-simdgroup butterflies via register shuffle.  Zero TG
    // memory, zero barriers.  On Apple GPUs simd_shuffle_xor is a single-
    // cycle warp-level primitive; compare to the old path which paid 5
    // sequential TG-memory-plus-barrier round-trips.
    #pragma unroll
    for (int stage = 0; stage < 5; stage++) {
        const uint mask = 1u << stage;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            const float partner = simd_shuffle_xor(vals[e], mask);
            const uint  idx     = lane + uint(e) * 32u;
            vals[e] = (idx & mask) ? (partner - vals[e]) : (vals[e] + partner);
        }
    }
    // Stages 5..NUM_STAGES-1: partner is on the same thread at a different
    // `e` offset.  Snapshot-then-commit to avoid reading a slot after an
    // earlier `e` iteration in the same stage wrote it (RAW across e).
    #pragma unroll
    for (int stage = 5; stage < NUM_STAGES; stage++) {
        const uint mask = 1u << stage;
        float results[ELEMS_PER_LANE];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            const uint idx         = lane + uint(e) * 32u;
            const uint partner_idx = idx ^ mask;
            const int  partner_e   = int((partner_idx - lane) / 32u);
            const float partner_val = vals[partner_e];
            results[e] = (idx & mask) ? (partner_val - vals[e]) : (vals[e] + partner_val);
        }
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            vals[e] = results[e];
        }
    }

    // Fused normalisation + random-sign flip in registers.  The sign table
    // is `constant` memory (broadcast-cached), so these reads are free.
    constexpr float INV_SQRT_N = fwht_inv_sqrt_n<HEAD_SIZE>();
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++) {
        const uint idx = lane + uint(e) * 32u;
        vals[e] *= INV_SQRT_N * get_fwht_sign<HEAD_SIZE>(idx);
    }
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

    // All VEC_SIZE elements share the same scale group.
    //
    // Proof: vec covers elements [vec_idx*VEC_SIZE, vec_idx*VEC_SIZE + VEC_SIZE).
    // VEC_SIZE is derived as MAX(16/(THREAD_GROUP_SIZE*sizeof(T)), 1) in the
    // paged-attention kernel, so VEC_SIZE ∈ {1, 2, 4, 8, 16} — all divisors of
    // SCALE_GROUP_SIZE=32.  Therefore (vec_idx*VEC_SIZE) % 32 + VEC_SIZE ≤ 32
    // whenever vec_idx*VEC_SIZE is aligned to VEC_SIZE (it is, by construction),
    // so no vec ever straddles a scale-group boundary.  One scale/zero load
    // per call instead of VEC_SIZE loads.
    const int group_idx = (vec_idx * VEC_SIZE) / SCALE_GROUP_SIZE;
    const float s = key_scale_cache[k_scale_base_offset + group_idx];
    const float z = key_zero_cache [k_scale_base_offset + group_idx];

    if constexpr (is_char<K_CACHE_T>()) {
        // int8 K path (signed)
        const device K_CACHE_T* k_elem_ptr = k_ptr + vec_idx * VEC_SIZE;
        #pragma unroll
        for (int e = 0; e < VEC_SIZE; e++) {
            result_ptr[e] = T(tq_dequant_k(k_elem_ptr[e], s, z));
        }
    } else {
        // uchar K path (uint8 or sub-8-bit packed)
        if (k_bits >= 8) {
            // 8-bit unsigned: one byte per element
            const device uchar* k_elem_ptr =
                reinterpret_cast<const device uchar*>(k_ptr) + vec_idx * VEC_SIZE;
            #pragma unroll
            for (int e = 0; e < VEC_SIZE; e++) {
                result_ptr[e] = T(tq_dequant_k(k_elem_ptr[e], s, z));
            }
        } else {
            // Sub-8-bit packed: unpack k_bits bits per logical element
            const device uchar* k_bytes = reinterpret_cast<const device uchar*>(k_ptr);
            #pragma unroll
            for (int e = 0; e < VEC_SIZE; e++) {
                const int  elem_idx = vec_idx * VEC_SIZE + e;
                const uint raw      = unpack_k_bits(k_bytes, elem_idx, k_bits);
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

    // Dequantise V directly into registers.  Centroid lookup + scale, no
    // threadgroup memory traffic on the V hot path.
    //
    // Deferred V FWHT
    //
    // Inverse FWHT is a *linear* transform.  By linearity:
    //
    //     Σᵢ wᵢ · InverseFWHT(dequant(Vᵢ))
    //   = InverseFWHT(Σᵢ wᵢ · dequant(Vᵢ))
    //
    // So we accumulate in the rotated-and-dequantised domain and apply
    // InverseFWHT *once* at the end of the attention kernel — not per V
    // token.  That replaces O(ctx × num_kv_heads) FWHTs per decode step
    // with O(num_kv_heads) FWHTs, a ~4 orders-of-magnitude reduction at
    // long context.  The paged_attention kernel calls
    // `inverse_fwht_in_place` once in warp 0 after cross-warp merge, just
    // before writing the final output.  `v_accs` therefore holds *rotated*
    // values throughout the block loop — correct by linearity, same final
    // numerics once the end-of-kernel FWHT fires.
    float vals[V_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < V_ELEMS_PER_THREAD; i++) {
        const int d = lane + i * NUM_SIMD_LANES;
        if (d < HEAD_SIZE) {
            const int group_idx = d / SCALE_GROUP_SIZE;
            const float vs = value_scale_cache[v_scale_base_offset + group_idx];
            const uchar v_idx = (v_bits == 3)
                ? unpack_3bit(v_ptr, d)
                : uchar(unpack_v_bits(v_ptr, d, v_bits));
            vals[i] = tq_dequant_v_centroid(v_idx, vs, v_centroids, v_bits);
        } else {
            vals[i] = 0.f;
        }
    }

    // Accumulate in rotated domain: O_rot += weight * dequant(V_rot).
    #pragma unroll
    for (int i = 0; i < V_ELEMS_PER_THREAD; i++) {
        const int d = lane + i * NUM_SIMD_LANES;
        if (d < HEAD_SIZE) {
            v_accs[i] += weight * vals[i];
        }
    }
}

// ===========================================================================
// Encode path: forward FWHT (one element per thread, TG size == HEAD_SIZE).
// Mirrors Python fwht(..., encode=True):
//   x_signed = x * random_signs;  x_rot = hadamard(x_signed);  x_rot /= sqrt(N)
// The random_signs table is the same deterministic constant consumed by the
// decode path (see FWHT_SIGNS_*).
//
// Stages 0..4 (butterfly mask < 32) are entirely within a simdgroup, so we
// use simd_shuffle_xor (register <-> register) instead of threadgroup memory
// + barriers — 5 stages with zero shared-memory traffic and zero barriers.
// Stages 5..NUM_STAGES-1 cross simdgroup boundaries and fall back to
// threadgroup memory with barriers.  Takes the input value `x` in a register
// (no pre-call TG write or barrier needed) and returns the transformed value
// in a register (no post-call barrier needed since every caller reads buf[t]
// only from its own thread, i.e. register-equivalent).
// ===========================================================================

template <int HEAD_SIZE>
inline float tg_forward_fwht_scalar(float x, threadgroup float* buf, uint t) {
    constexpr int   NUM_STAGES = fwht_num_stages<HEAD_SIZE>();
    constexpr float INV_SQRT_N = fwht_inv_sqrt_n<HEAD_SIZE>();

    x *= get_fwht_sign<HEAD_SIZE>(t);

    // Intra-simdgroup butterflies: purely register, no TG memory, no barriers.
    #pragma unroll
    for (int stage = 0; stage < 5; stage++) {
        const uint  mask    = 1u << stage;
        const float partner = simd_shuffle_xor(x, mask);
        x = (t & mask) ? (partner - x) : (x + partner);
    }

    // Cross-simdgroup butterflies: threadgroup memory + barriers.
    // NUM_STAGES is 6..8, so 1..3 iterations below.
    if (NUM_STAGES > 5) {
        buf[t] = x;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int stage = 5; stage < NUM_STAGES; stage++) {
            const uint  mask    = 1u << stage;
            const float me      = buf[t];
            const float partner = buf[t ^ mask];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf[t] = (t & mask) ? (partner - me) : (me + partner);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        x = buf[t];
    }

    return x * INV_SQRT_N;
}

// ===========================================================================
// Fused encode-and-cache kernel
//
// Replaces Python `turbo_quant_encode()` + 5 paged scatters with a single
// dispatch.  Writes K (any bit width in QUANT_PARAMS), V (packed v_bits),
// and the three scale/zero caches directly at slot_mapping[token] offsets.
//
// Grid:        (num_tokens, num_kv_heads, 1)
// Threadgroup: (HEAD_SIZE, 1, 1)
// One threadgroup encodes one (token, kv_head) pair.  Each simdgroup
// (32 lanes) corresponds to exactly one 32-element scale group, so
// min/max/RMS reductions are free via simd_min / simd_max / simd_sum.
//
// K spec (must match `quantize()` in turboquant.py):
//   signed (q8_0/int8, bits=8):
//     max_val = (1 << (bits-1)) - 1 = 127
//     scale   = fp16((x_max - x_min) / (2 * max_val))
//     zp      = round(fp16((x_max + x_min) / (2 * scale)))
//     idx     = clip(round(fp16(x / scale) - zp), -max_val, max_val)
//   unsigned (uint8 + sub-8-bit: q5_0/q4_0/int4/uint4/int2/uint2):
//     max_val = (1 << bits) - 1
//     scale   = fp16((x_max - x_min) / max_val)
//     zp      = round(fp16(x_min / scale))
//     idx     = clip(round(fp16(x / scale) - zp), 0, max_val)
//
// The `+ 1e-8` epsilon used in Python's scale guard underflows to 0 in fp16
// (min subnormal ~6e-8), so it is a no-op and we drop it — matching Python's
// actual behaviour bit-for-bit.
//
// Bit packing: 8-bit K writes one byte per element directly; sub-8-bit K
// stages unsigned indices in `k_idx_buf` then the first `k_packed` threads
// assemble bytes in parallel (identical to V packing).  The key_cache
// buffer is bound as uchar* universally — int8 values use two's-complement
// bit patterns which read back correctly as `int8_t` in decode kernels.
// ===========================================================================

constant int  tq_enc_k_bits   [[function_constant(80)]];
constant bool tq_enc_k_signed [[function_constant(81)]];
constant int  tq_enc_v_bits   [[function_constant(90)]];

// Generic byte-packer: assembles one output byte from the staged uchar index
// buffer `idx_buf`.  Each byte covers 8 contiguous bit positions which may
// straddle up to two logical elements when `bits` does not divide 8 (e.g.
// 3-bit: 8 values -> 3 bytes).  `bits` must be in [1, 8].
inline uint tq_pack_byte(threadgroup const uchar* idx_buf,
                         int bit_start, int bits) {
    const int first_e = bit_start / bits;
    const int last_e  = (bit_start + 7) / bits;
    const uint mask   = (1u << bits) - 1u;
    uint byte = 0;
    for (int e = first_e; e <= last_e; e++) {
        const int  e_bit  = e * bits;
        const int  shift  = e_bit - bit_start;
        const uint idx_v  = uint(idx_buf[e]) & mask;
        if (shift >= 0) {
            byte |= idx_v << shift;
        } else {
            byte |= idx_v >> (-shift);
        }
    }
    return byte & 0xFFu;
}

template <typename T, int HEAD_SIZE>
[[kernel]] void tq_encode(
    const device T*       __restrict__ key                [[buffer(0)]],
    const device T*       __restrict__ value              [[buffer(1)]],
    device uchar*         __restrict__ key_cache          [[buffer(2)]],
    device uchar*         __restrict__ value_cache        [[buffer(3)]],
    device half*          __restrict__ key_scale_cache    [[buffer(4)]],
    device half*          __restrict__ value_scale_cache  [[buffer(5)]],
    device half*          __restrict__ key_zero_cache     [[buffer(6)]],
    const device int64_t* __restrict__ slot_mapping       [[buffer(7)]],
    const device float*   __restrict__ v_centroids        [[buffer(8)]],
    const constant int&   num_kv_heads                    [[buffer(9)]],
    const constant int&   block_size                      [[buffer(10)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid3  [[thread_position_in_threadgroup]],
    uint  sid   [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]]
) {
    constexpr int SG_SIZE = 32;

    const uint t     = tid3.x;
    const int  token = int(tgid.x);
    const int  kvh   = int(tgid.y);

    const int64_t slot = slot_mapping[token];
    if (slot < 0) {
        return;
    }
    const int block_idx = int(slot / block_size);
    const int block_off = int(slot % block_size);

    constexpr int head_dim     = HEAD_SIZE;
    constexpr int scale_groups = HEAD_SIZE / SG_SIZE;
    const int     k_packed     = (head_dim * tq_enc_k_bits + 7) / 8;
    const int     v_packed     = (head_dim * tq_enc_v_bits + 7) / 8;

    // -------- Source element (one per thread) --------
    const int64_t src_base =
        (int64_t(token) * num_kv_heads + kvh) * head_dim;
    const float k_val = float(key[src_base + t]);
    const float v_val = float(value[src_base + t]);

    // -------- Threadgroup staging buffers --------
    threadgroup float fwht_buf[HEAD_SIZE];    // V FWHT rotation scratch
    threadgroup uchar k_idx_buf[HEAD_SIZE];   // K indices (sub-8-bit path)
    threadgroup uchar v_idx_buf[HEAD_SIZE];   // V indices (always staged)

    // Precomputed V-centroid midpoints (searchsorted boundaries).  Sized for
    // the worst case (v_bits = 8 → 255 boundaries; ~1 KB of TG memory).  For
    // smaller v_bits only a prefix is used.  Strided fill covers arbitrary
    // num_centroids up to the max without requiring HEAD_SIZE >= num_centroids.
    //
    // Why: without this, every one of the HEAD_SIZE threads per TG would
    // independently read v_centroids[i] and v_centroids[i+1] from device
    // memory inside the searchsorted loop — at large batch sizes that adds
    // up to hundreds of megabytes of redundant device reads across the full
    // grid.  Preloading once per TG reduces that to num_centroids reads per
    // TG and folds the per-thread midpoint compute into a single pass.
    threadgroup float v_boundaries[255];

    // Strided preload.  Fully async with the K-encode work that follows —
    // the explicit barrier just before the searchsorted consumer (below) is
    // the sole synchronisation point.  We deliberately do NOT lean on the
    // forward FWHT's internal barriers or the sub-8-bit K-pack barrier to
    // publish these writes: doing so silently couples this searchsorted path
    // to implementation details of unrelated helpers, which would break the
    // preload the next time those helpers are rearranged.
    const int num_centroids = 1 << tq_enc_v_bits;
    #pragma unroll 1
    for (int i = int(t); i < num_centroids - 1; i += HEAD_SIZE) {
        v_boundaries[i] = 0.5f * (v_centroids[i] + v_centroids[i + 1]);
    }

    // -------- Cache destination bases --------
    const int64_t token_base =
        (int64_t(block_idx) * block_size + block_off) * num_kv_heads;
    const int64_t kc_base     = (token_base + kvh) * k_packed;
    const int64_t vc_base     = (token_base + kvh) * v_packed;
    const int64_t scale_base  = (token_base + kvh) * scale_groups;

    // ======================================================================
    // K encode: asymmetric uniform, signed or unsigned.  All arithmetic
    // routed through fp16 (matching Python's fp16 scale/zp storage).
    // ======================================================================
    const float k_min_f = simd_min(k_val);
    const float k_max_f = simd_max(k_val);

    half  k_scale_h;
    float k_zp_f;
    int   k_idx_i;

    if (tq_enc_k_signed) {
        // q8_0 / int8  (bits == 8)
        const int max_val = (1 << (tq_enc_k_bits - 1)) - 1;
        k_scale_h = half(half(k_max_f - k_min_f) / half(2.0f * float(max_val)));
        const half k_sum_h = half(k_max_f + k_min_f);
        k_zp_f    = rint(float(k_sum_h / (half(2.0f) * k_scale_h)));
        k_idx_i   = int(rint(float(half(k_val) / k_scale_h) - k_zp_f));
        k_idx_i   = clamp(k_idx_i, -max_val, max_val);
    } else {
        // uint8 / q5_0 / q4_0 / int4 / uint4 / int2 / uint2
        const int max_val = (1 << tq_enc_k_bits) - 1;
        k_scale_h = half(half(k_max_f - k_min_f) / half(float(max_val)));
        k_zp_f    = rint(float(half(k_min_f) / k_scale_h));
        k_idx_i   = int(rint(float(half(k_val) / k_scale_h) - k_zp_f));
        k_idx_i   = clamp(k_idx_i, 0, max_val);
    }

    // Lane 0 of each simdgroup publishes the per-group scale + zero_point.
    if (lane == 0) {
        key_scale_cache[scale_base + int(sid)] = k_scale_h;
        key_zero_cache [scale_base + int(sid)] = half(k_zp_f);
    }

    // -------- Write K indices --------
    if (tq_enc_k_bits == 8) {
        // 8-bit (signed or unsigned): one byte per element, direct write.
        // Signed values are stored as two's-complement bit patterns via the
        // uchar cast, so decode kernels reading `int8_t*` see the right sign.
        key_cache[kc_base + t] = uchar(uint(k_idx_i) & 0xFFu);
    } else {
        // Sub-8-bit unsigned: stage indices then pack bytes in parallel.
        k_idx_buf[t] = uchar(uint(k_idx_i) & ((1u << tq_enc_k_bits) - 1u));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(t) < k_packed) {
            const uint byte = tq_pack_byte(k_idx_buf, int(t) * 8, tq_enc_k_bits);
            key_cache[kc_base + t] = uchar(byte);
        }
    }

    // ======================================================================
    // V encode: FWHT rotation + Lloyd-Max quantization.
    // Python flow (mirrored exactly):
    //   x_rot   = fwht(v, encode=True)           # fp16
    //   scale   = sqrt(mean(x_rot^2))            # fp16, per 32-elem block
    //   x_norm  = x_rot / scale                  # fp16 (1e-8 guard dropped)
    //   idx     = searchsorted(boundaries, x_norm)
    // Boundaries are midpoints of ascending fp32 centroids.
    // ======================================================================
    // FWHT takes the input value in-register and returns the rotated value
    // in-register; threadgroup memory is used only for cross-simdgroup
    // butterfly stages.  No pre- or post-call barrier needed.
    const float v_rot_f   = tg_forward_fwht_scalar<HEAD_SIZE>(v_val, fwht_buf, t);
    const half  v_rot_h   = half(v_rot_f);
    const float v_sqsum   = simd_sum(float(v_rot_h) * float(v_rot_h));
    const half  v_scale_h = half(sqrt(v_sqsum * (1.0f / float(SG_SIZE))));
    const float v_norm    = float(v_rot_h / v_scale_h);

    // Searchsorted against the preloaded midpoint boundaries.  The explicit
    // barrier below is the sync point that publishes the strided preload of
    // `v_boundaries` to all threads in the TG.  We don't rely on any barrier
    // baked into tg_forward_fwht_scalar or the K-pack path — those are
    // implementation details of *other* code paths and MUST NOT be load-
    // bearing for this one.  The barrier sits on the V-encode hot path but
    // adds at most one cycle above the preload cost since the writes have
    // been inflight since kernel entry.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int v_idx = 0;
    for (int i = 0; i < num_centroids - 1; i++) {
        if (v_norm > v_boundaries[i]) v_idx++;
    }
    v_idx_buf[t] = uchar(v_idx & ((1 << tq_enc_v_bits) - 1));

    if (lane == 0) {
        value_scale_cache[scale_base + int(sid)] = v_scale_h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------- Pack V indices into bytes --------
    if (int(t) < v_packed) {
        const uint byte = tq_pack_byte(v_idx_buf, int(t) * 8, tq_enc_v_bits);
        value_cache[vc_base + t] = uchar(byte);
    }
}

#define instantiate_tq_encode(T, HS)                                           \
  template [[host_name("tq_encode_" #T "_hs" #HS)]] [[kernel]] void          \
  tq_encode<T, HS>(                                                            \
      const device T*       __restrict__ key                [[buffer(0)]],     \
      const device T*       __restrict__ value              [[buffer(1)]],     \
      device uchar*         __restrict__ key_cache          [[buffer(2)]],     \
      device uchar*         __restrict__ value_cache        [[buffer(3)]],     \
      device half*          __restrict__ key_scale_cache    [[buffer(4)]],     \
      device half*          __restrict__ value_scale_cache  [[buffer(5)]],     \
      device half*          __restrict__ key_zero_cache     [[buffer(6)]],     \
      const device int64_t* __restrict__ slot_mapping       [[buffer(7)]],     \
      const device float*   __restrict__ v_centroids        [[buffer(8)]],     \
      const constant int&   num_kv_heads                    [[buffer(9)]],     \
      const constant int&   block_size                      [[buffer(10)]],    \
      uint3 tgid  [[threadgroup_position_in_grid]],                            \
      uint3 tid3  [[thread_position_in_threadgroup]],                          \
      uint  sid   [[simdgroup_index_in_threadgroup]],                          \
      uint  lane  [[thread_index_in_simdgroup]]);

instantiate_tq_encode(half, 64)
instantiate_tq_encode(half, 128)
instantiate_tq_encode(half, 256)
instantiate_tq_encode(half, 512)
instantiate_tq_encode(bfloat16_t, 64)
instantiate_tq_encode(bfloat16_t, 128)
instantiate_tq_encode(bfloat16_t, 256)
instantiate_tq_encode(bfloat16_t, 512)
instantiate_tq_encode(float, 64)
instantiate_tq_encode(float, 128)
instantiate_tq_encode(float, 256)
instantiate_tq_encode(float, 512)
