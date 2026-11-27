// SPDX-License-Identifier: Apache-2.0

constexpr uint BLOCK_BYTES = (QTYPE == 2) ? 84u : ((QTYPE == 3) ? 110u : 210u);
static_assert(QTYPE == 2 || QTYPE == 3 || QTYPE == 6,
              "Unsupported GGUF raw-block QTYPE");
static_assert(alignof(packed_ushort4) == 2,
              "packed_ushort4 must be 2-byte aligned for 110B/210B superblocks");

// 110B (Q3_K) and 210B (Q6_K) superblocks sit at 2-byte-aligned addresses on
// odd indices, where 4-byte-aligned packed_uint2 loads truncate the pointer.
auto load_u32x2 = [](device const uint8_t* p) -> uint2 {
    return as_type<uint2>(ushort4(*(device const packed_ushort4*)p));
};

auto unpack_q2k = [&](device const uint8_t* b, uint lane,
                      thread uchar4& c0, thread uchar4& c1,
                      thread float& s, thread float& m) {
    uint g = lane >> 1;
    uint half_idx = lane >> 4;
    uint shift = 2u * ((lane >> 2) & 3u);
    uint scm = b[g];
    float d = float(*(device const half*)(b + 80));
    float dmin = float(*(device const half*)(b + 82));
    s = d * float(scm & 0x0Fu);
    m = dmin * float(scm >> 4);
    uint2 qs8 = load_u32x2(b + 16u + half_idx * 32u + (lane & 3u) * 8u);
    c0 = as_type<uchar4>((qs8.x >> shift) & 0x03030303u);
    c1 = as_type<uchar4>((qs8.y >> shift) & 0x03030303u);
};

auto unpack_q3k = [&](device const uint8_t* b, uint lane,
                      thread char4& q0, thread char4& q1,
                      thread float& s) {
    uint g = lane >> 1;
    uint half_idx = lane >> 4;
    uint shift = 2u * ((lane >> 2) & 3u);
    uint hbit = lane >> 2;
    uint byte_off = (lane & 3u) * 8u;
    float d = float(*(device const half*)(b + 108));
    uint lo = (g < 8u) ? (b[96u + g] & 0x0Fu) : (b[96u + g - 8u] >> 4);
    uint hi = (b[104u + (g & 3u)] >> (2u * (g >> 2))) & 3u;
    s = d * float(int(lo | (hi << 4)) - 32);
    uint2 qs8 = load_u32x2(b + 32u + half_idx * 32u + byte_off);
    uint2 hm8 = load_u32x2(b + byte_off);
    uint u0 = ((qs8.x >> shift) & 0x03030303u) |
              (((hm8.x >> hbit) & 0x01010101u) << 2);
    uint u1 = ((qs8.y >> shift) & 0x03030303u) |
              (((hm8.y >> hbit) & 0x01010101u) << 2);
    q0 = as_type<char4>(u0) - char4(4);
    q1 = as_type<char4>(u1) - char4(4);
};

auto unpack_q6k = [&](device const uint8_t* b, uint lane,
                      thread char4& q0, thread char4& q1,
                      thread float& s) {
    uint g = lane >> 1;
    uint half_idx = lane >> 4;
    uint ql_shift = (lane & 8u) >> 1;
    uint qh_shift = 2u * ((lane >> 2) & 3u);
    float d = float(*(device const half*)(b + 208));
    s = d * float(((device const int8_t*)(b + 192))[g]);
    uint2 ql8 = load_u32x2(b + half_idx * 64u + (lane & 7u) * 8u);
    uint2 qh8 = load_u32x2(b + 128u + half_idx * 32u + (lane & 3u) * 8u);
    uint u0 = ((ql8.x >> ql_shift) & 0x0F0F0F0Fu) |
              (((qh8.x >> qh_shift) & 0x03030303u) << 4);
    uint u1 = ((ql8.y >> ql_shift) & 0x0F0F0F0Fu) |
              (((qh8.y >> qh_shift) & 0x03030303u) << 4);
    q0 = as_type<char4>(u0) - char4(32);
    q1 = as_type<char4>(u1) - char4(32);
};

#if GGUF_KERNEL_MODE == 0
    uint tid = thread_position_in_grid.x;
    uint blocks_per_row = uint(blocks_shape[1]) / BLOCK_BYTES;
    uint total_chunks = uint(blocks_shape[0]) * blocks_per_row * 32u;
    if (tid >= total_chunks) return;

    uint block = tid >> 5;
    uint lane = tid & 31u;
    device const uint8_t* b = blocks + (size_t)block * BLOCK_BYTES;
    device T* dst = out + (size_t)tid * 8u;

    if constexpr (QTYPE == 2) {
        uchar4 c0, c1;
        float s, m;
        unpack_q2k(b, lane, c0, c1, s, m);
        dst[0] = static_cast<T>(s * float(c0.x) - m);
        dst[1] = static_cast<T>(s * float(c0.y) - m);
        dst[2] = static_cast<T>(s * float(c0.z) - m);
        dst[3] = static_cast<T>(s * float(c0.w) - m);
        dst[4] = static_cast<T>(s * float(c1.x) - m);
        dst[5] = static_cast<T>(s * float(c1.y) - m);
        dst[6] = static_cast<T>(s * float(c1.z) - m);
        dst[7] = static_cast<T>(s * float(c1.w) - m);
    } else {
        char4 q0, q1;
        float s;
        if constexpr (QTYPE == 3) {
            unpack_q3k(b, lane, q0, q1, s);
        } else {
            unpack_q6k(b, lane, q0, q1, s);
        }
        dst[0] = static_cast<T>(s * float(q0.x));
        dst[1] = static_cast<T>(s * float(q0.y));
        dst[2] = static_cast<T>(s * float(q0.z));
        dst[3] = static_cast<T>(s * float(q0.w));
        dst[4] = static_cast<T>(s * float(q1.x));
        dst[5] = static_cast<T>(s * float(q1.y));
        dst[6] = static_cast<T>(s * float(q1.z));
        dst[7] = static_cast<T>(s * float(q1.w));
    }

#elif GGUF_KERNEL_MODE == 1 || GGUF_KERNEL_MODE == 2
    uint lane = thread_index_in_simdgroup;
    uint blocks_per_row = uint(blocks_shape[1]) / BLOCK_BYTES;
    uint in_features = blocks_per_row * 256u;

#if GGUF_KERNEL_MODE == 1
    uint out_rows = uint(blocks_shape[0]);
    uint local_base = thread_position_in_grid.y * NR;
    if (local_base >= out_rows) return;
    uint global_base = local_base;
    device const T* x_base = x;
#else
    uint out_rows = uint(blocks_shape[0]) / uint(NUM_EXPERTS);
    uint local_base = thread_position_in_grid.y * NR;
    uint assignment = thread_position_in_grid.z;
    if (local_base >= out_rows || assignment >= uint(idx_shape[0])) return;
    uint expert = uint(idx[assignment]);
    uint global_base = expert * out_rows + local_base;
    uint x_repeat = uint(idx_shape[0]) / uint(x_shape[0]);
    uint x_row = assignment / x_repeat;
    device const T* x_base = x + (size_t)x_row * in_features;
#endif

    float acc[NR][NB];
    #pragma unroll
    for (uint r = 0; r < NR; ++r) {
        #pragma unroll
        for (uint batch = 0; batch < NB; ++batch) {
            acc[r][batch] = 0.0f;
        }
    }

    auto accumulate_block = [&](device const uint8_t* b,
                                thread const float xv[NB][8],
                                thread const float xsum[NB],
                                thread float row_acc[NB]) {
        if constexpr (QTYPE == 2) {
            uchar4 c0, c1;
            float s, m;
            unpack_q2k(b, lane, c0, c1, s, m);
            float w0 = float(c0.x), w1 = float(c0.y), w2 = float(c0.z), w3 = float(c0.w);
            float w4 = float(c1.x), w5 = float(c1.y), w6 = float(c1.z), w7 = float(c1.w);
            #pragma unroll
            for (uint batch = 0; batch < NB; ++batch) {
                float dot = fma(xv[batch][0], w0, xv[batch][1] * w1)
                          + fma(xv[batch][2], w2, xv[batch][3] * w3)
                          + fma(xv[batch][4], w4, xv[batch][5] * w5)
                          + fma(xv[batch][6], w6, xv[batch][7] * w7);
                row_acc[batch] += s * dot - m * xsum[batch];
            }
        } else {
            (void)xsum;
            char4 q0, q1;
            float s;
            if constexpr (QTYPE == 3) {
                unpack_q3k(b, lane, q0, q1, s);
            } else {
                unpack_q6k(b, lane, q0, q1, s);
            }
            float w0 = float(q0.x), w1 = float(q0.y), w2 = float(q0.z), w3 = float(q0.w);
            float w4 = float(q1.x), w5 = float(q1.y), w6 = float(q1.z), w7 = float(q1.w);
            #pragma unroll
            for (uint batch = 0; batch < NB; ++batch) {
                float dot = fma(xv[batch][0], w0, xv[batch][1] * w1)
                          + fma(xv[batch][2], w2, xv[batch][3] * w3)
                          + fma(xv[batch][4], w4, xv[batch][5] * w5)
                          + fma(xv[batch][6], w6, xv[batch][7] * w7);
                row_acc[batch] += s * dot;
            }
        }
    };

    const bool full_rows = (local_base + NR <= out_rows);
    const size_t row_byte_stride = (size_t)blocks_per_row * BLOCK_BYTES;
    device const uint8_t* row_base = blocks + (size_t)global_base * row_byte_stride;

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        float xv[NB][8];
        float xsum[NB];
        uint x_col = blk * 256u + lane * 8u;
        #pragma unroll
        for (uint batch = 0; batch < NB; ++batch) {
            device const T* xp = x_base + (size_t)batch * in_features + x_col;
            #pragma unroll
            for (uint i = 0; i < 8; ++i) {
                xv[batch][i] = float(xp[i]);
            }
            if constexpr (QTYPE == 2) {
                xsum[batch] = ((xv[batch][0] + xv[batch][1]) + (xv[batch][2] + xv[batch][3]))
                            + ((xv[batch][4] + xv[batch][5]) + (xv[batch][6] + xv[batch][7]));
            } else {
                xsum[batch] = 0.0f;
            }
        }

        device const uint8_t* blk_ptr = row_base + (size_t)blk * BLOCK_BYTES;
        if (full_rows) {
            #pragma unroll
            for (uint r = 0; r < NR; ++r) {
                accumulate_block(blk_ptr + r * row_byte_stride, xv, xsum, acc[r]);
            }
        } else {
            #pragma unroll
            for (uint r = 0; r < NR; ++r) {
                if (local_base + r < out_rows) {
                    accumulate_block(blk_ptr + r * row_byte_stride, xv, xsum, acc[r]);
                }
            }
        }
    }

    #pragma unroll
    for (uint r = 0; r < NR; ++r) {
        uint local_row = local_base + r;
        if (local_row < out_rows) {
            #pragma unroll
            for (uint batch = 0; batch < NB; ++batch) {
                float total = simd_sum(acc[r][batch]);
                if (lane == 0) {
#if GGUF_KERNEL_MODE == 1
                    y[(size_t)batch * out_rows + local_row] = static_cast<T>(total);
#else
                    y[(size_t)assignment * out_rows + local_row] = static_cast<T>(total);
#endif
                }
            }
        }
    }
#endif
