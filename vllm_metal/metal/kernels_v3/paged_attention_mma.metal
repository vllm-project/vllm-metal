// SPDX-License-Identifier: Apache-2.0
//
// v3 paged-attention kernel (MMA path).
//
// Status: C3 — multi-block iteration with online softmax.
//   For each KV block:
//     m_new   = max(m_old, m_block)
//     alpha   = exp(m_old - m_new)            (0 when m_old == -INF)
//     P       = exp(S - m_new)
//     l_block = rowsum(P)
//     l_new   = alpha * l_old + l_block
//     O_new   = alpha * O_old + (P @ V_block)
//   Final epilogue: O / l
//
// Eligibility filter and production wiring still come in C4.
//
// Design doc: research/mma_kernel_design.md
//
// MVP scope (gated by host-side eligibility filter, added in C4):
//   - Q dtype: bfloat16
//   - K/V dtype: bfloat16
//   - HEAD_SIZE: 128
//   - BLOCK_SIZE: 16
//   - BLOCK_M:   8
//   - Causal mask, varlen + paged blocks
//
// Type machinery, find_seq_idx, resolve_seq_and_q_block in attn_common.h.
//
// Portions of this file are adapted from the vLLM project
// (https://github.com/vllm-project/vllm)
// Licensed under the Apache License 2.0
// Copyright contributors to the vLLM project

#include "attn_common.h"
#include <metal_simdgroup_matrix>

// ========================================== MMA paged-attention kernel
//
// Threadgroup memory layout (BLOCK_M=8, BLOCK_SIZE=16, head_size=128, bf16):
//   Q_tile     : T[8][128]      = 2048 B
//   K_stage    : T[16][128]     = 4096 B
//   V_stage    : T[16][128]     = 4096 B
//   S_buf      : float[8][16]   =  512 B   (Q@K^T scores; reused as P_buf bf16)
//   O_stage    : float[8][128]  = 4096 B   (running fp32 accumulator)
//   m_buf      : float[8]       =   32 B
//   l_buf      : float[8]       =   32 B
//   alpha_buf  : float[8]       =   32 B
//   Total: 14944 B (limit: 32768 B per threadgroup)
//
// Grid: (num_kv_heads, total_q_blocks, 1)
// Threadgroup: (32, 1, 1) — one simdgroup per program

template <typename T, int HEAD_SIZE, int BLOCK_SIZE, int BLOCK_M>
[[kernel]] void paged_attention_mma(
    device T *out [[buffer(0)]],                           // [total_q, n_q_heads, head_size]
    device const T *q [[buffer(1)]],                       // [total_q, n_q_heads, head_size]
    device const T *k_cache [[buffer(2)]],                 // [num_blocks, block_size, n_kv_heads, head_size]
    device const T *v_cache [[buffer(3)]],                 // [num_blocks, block_size, n_kv_heads, head_size]
    const constant int &num_kv_heads [[buffer(4)]],
    const constant float &scale [[buffer(5)]],
    device const uint32_t *block_tables [[buffer(6)]],     // [num_seqs, max_blocks_per_seq]
    device const uint32_t *seq_lens [[buffer(7)]],         // [num_seqs]
    const constant int &max_num_blocks_per_seq [[buffer(8)]],
    const constant int &q_stride [[buffer(9)]],
    const constant int &kv_block_stride [[buffer(10)]],
    const constant int &kv_head_stride [[buffer(11)]],
    device const int32_t *cu_seqlens_q [[buffer(12)]],
    const constant int &num_seqs [[buffer(13)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tgs [[threadgroups_per_grid]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    // --- Per-program identification ---
    const int kv_head_idx = (int)tg_pos.x;
    const int q_block_global = (int)tg_pos.y;
    const uint thread_idx = thread_pos.x;
    const int num_q_heads = q_stride / HEAD_SIZE;
    const int num_queries_per_kv = num_q_heads / num_kv_heads;
    const int block_q = BLOCK_M / num_queries_per_kv;

    // --- Resolve (seq_idx, q_block_local_idx, ...) ---
    int seq_idx, q_block_local_idx, q_seq_start, q_len, kv_seq_len;
    resolve_seq_and_q_block(cu_seqlens_q, seq_lens, q_block_global,
                            num_seqs, block_q,
                            seq_idx, q_block_local_idx,
                            q_seq_start, q_len, kv_seq_len);

    if (seq_idx >= num_seqs) return;
    if (q_block_local_idx * block_q >= q_len) return;

    // --- Threadgroup memory layout ---
    constexpr int Q_TILE_BYTES   = BLOCK_M    * HEAD_SIZE  * sizeof(T);
    constexpr int K_STAGE_BYTES  = BLOCK_SIZE * HEAD_SIZE  * sizeof(T);
    constexpr int V_STAGE_BYTES  = BLOCK_SIZE * HEAD_SIZE  * sizeof(T);
    constexpr int S_BUF_BYTES    = BLOCK_M    * BLOCK_SIZE * sizeof(float);
    constexpr int O_STAGE_BYTES  = BLOCK_M    * HEAD_SIZE  * sizeof(float);
    constexpr int M_BUF_BYTES    = BLOCK_M                 * sizeof(float);
    constexpr int L_BUF_BYTES    = BLOCK_M                 * sizeof(float);
    // alpha_buf follows — no offset needed beyond the running total.

    threadgroup T     *Q_tile    = (threadgroup T*)(shared_mem);
    threadgroup T     *K_stage   = (threadgroup T*)(shared_mem + Q_TILE_BYTES);
    threadgroup T     *V_stage   = (threadgroup T*)(shared_mem + Q_TILE_BYTES + K_STAGE_BYTES);
    threadgroup float *S_buf     = (threadgroup float*)(shared_mem + Q_TILE_BYTES + K_STAGE_BYTES + V_STAGE_BYTES);
    threadgroup T     *P_buf     = (threadgroup T*)S_buf;  // alias — overwrites S after softmax
    threadgroup float *O_stage   = (threadgroup float*)(shared_mem + Q_TILE_BYTES + K_STAGE_BYTES + V_STAGE_BYTES + S_BUF_BYTES);
    threadgroup float *m_buf     = (threadgroup float*)(shared_mem + Q_TILE_BYTES + K_STAGE_BYTES + V_STAGE_BYTES + S_BUF_BYTES + O_STAGE_BYTES);
    threadgroup float *l_buf     = (threadgroup float*)(shared_mem + Q_TILE_BYTES + K_STAGE_BYTES + V_STAGE_BYTES + S_BUF_BYTES + O_STAGE_BYTES + M_BUF_BYTES);
    threadgroup float *alpha_buf = (threadgroup float*)(shared_mem + Q_TILE_BYTES + K_STAGE_BYTES + V_STAGE_BYTES + S_BUF_BYTES + O_STAGE_BYTES + M_BUF_BYTES + L_BUF_BYTES);

    // --- Phase 1: cooperative Q load (with pre-applied softmax scale) ---
    constexpr int Q_TILE_ELEMS = BLOCK_M * HEAD_SIZE;
    constexpr int O_STAGE_ELEMS = BLOCK_M * HEAD_SIZE;
    constexpr int ELEMS_PER_THREAD_Q = (Q_TILE_ELEMS + 31) / 32;
    for (int i = 0; i < ELEMS_PER_THREAD_Q; ++i) {
        int flat = (int)thread_idx + i * 32;
        if (flat >= Q_TILE_ELEMS) break;
        int row = flat / HEAD_SIZE;
        int col = flat % HEAD_SIZE;
        int q_token_offset = row / num_queries_per_kv;
        int q_head_offset  = row - q_token_offset * num_queries_per_kv;
        int global_q_token = q_seq_start + q_block_local_idx * block_q + q_token_offset;
        int global_q_head  = kv_head_idx * num_queries_per_kv + q_head_offset;

        T value = T(0);
        if (q_token_offset < block_q && global_q_token < q_seq_start + q_len) {
            T raw = q[(int64_t)global_q_token * q_stride + global_q_head * HEAD_SIZE + col];
            value = T((float)raw * scale);
        }
        Q_tile[row * HEAD_SIZE + col] = value;
    }

    // --- Phase 1.5: initialize online softmax state ---
    // m, l, O_stage all initialized in parallel with Q load above (they touch
    // disjoint regions of threadgroup memory).
    if (thread_idx < BLOCK_M) {
        m_buf[thread_idx] = -INFINITY;
        l_buf[thread_idx] = 0.0f;
    }
    for (int i = (int)thread_idx; i < O_STAGE_ELEMS; i += 32) {
        O_stage[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ============================================================
    // Multi-block iteration with online softmax.
    // ============================================================
    const int num_blocks = (kv_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    constexpr int N_FRAGS_QK = BLOCK_SIZE / 8;  // 2
    constexpr int K_FRAGS_QK = HEAD_SIZE / 8;   // 16
    constexpr int N_FRAGS_PV = HEAD_SIZE / 8;   // 16
    constexpr int K_FRAGS_PV = BLOCK_SIZE / 8;  // 2
    constexpr int KV_STAGE_ELEMS = BLOCK_SIZE * HEAD_SIZE;
    constexpr int ELEMS_PER_THREAD_KV = (KV_STAGE_ELEMS + 31) / 32;

    for (int kv_block_idx = 0; kv_block_idx < num_blocks; ++kv_block_idx) {
        const int physical_block = (int)block_tables[seq_idx * max_num_blocks_per_seq + kv_block_idx];

        // --- Phase 2a: cooperative K stage ---
        {
            int slot_stride_elem = num_kv_heads * HEAD_SIZE;
            int64_t kv_block_base = (int64_t)physical_block * kv_block_stride
                                  + (int64_t)kv_head_idx * kv_head_stride;
            for (int i = 0; i < ELEMS_PER_THREAD_KV; ++i) {
                int flat = (int)thread_idx + i * 32;
                if (flat >= KV_STAGE_ELEMS) break;
                int slot = flat / HEAD_SIZE;
                int dim  = flat % HEAD_SIZE;
                int kv_token = kv_block_idx * BLOCK_SIZE + slot;
                T value = T(0);
                if (kv_token < kv_seq_len) {
                    value = k_cache[kv_block_base + (int64_t)slot * slot_stride_elem + dim];
                }
                K_stage[slot * HEAD_SIZE + dim] = value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 2b: Q @ K^T MMA → S_buf ---
        for (int n_frag = 0; n_frag < N_FRAGS_QK; ++n_frag) {
            simdgroup_matrix<float, 8, 8> S_frag = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int k_frag = 0; k_frag < K_FRAGS_QK; ++k_frag) {
                simdgroup_matrix<bfloat16_t, 8, 8> Q_frag, K_frag_T;
                simdgroup_load(Q_frag, Q_tile + k_frag * 8, /*elements_per_row=*/HEAD_SIZE);
                simdgroup_load(K_frag_T,
                               K_stage + n_frag * 8 * HEAD_SIZE + k_frag * 8,
                               /*elements_per_row=*/HEAD_SIZE,
                               /*matrix_origin=*/ulong2(0, 0),
                               /*transpose_matrix=*/true);
                simdgroup_multiply_accumulate(S_frag, Q_frag, K_frag_T, S_frag);
            }
            simdgroup_store(S_frag, S_buf + n_frag * 8, /*elements_per_row=*/BLOCK_SIZE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 2c: online softmax update ---
        // 32 threads cooperate on (BLOCK_M=8, BLOCK_SIZE=16) = 128 elements,
        // 4 threads per row, each handles cols [(thread_idx % 4)*4, (thread_idx % 4)*4+4).
        {
            int row = (int)thread_idx / 4;
            int sub_col = (int)thread_idx % 4;

            int q_token_offset = row / num_queries_per_kv;
            int global_q_token_in_seq = q_block_local_idx * block_q + q_token_offset;
            int effective_context = (kv_seq_len - q_len) + global_q_token_in_seq + 1;
            bool row_valid = (q_token_offset < block_q) && (global_q_token_in_seq < q_len);

            // Read S values for this thread's 4 columns, applying mask.
            float vals[4];
            for (int i = 0; i < 4; ++i) {
                int col = sub_col * 4 + i;
                int kv_token = kv_block_idx * BLOCK_SIZE + col;
                float s = S_buf[row * BLOCK_SIZE + col];
                bool valid = row_valid
                             && (kv_token < effective_context)
                             && (kv_token < kv_seq_len);
                vals[i] = valid ? s : -INFINITY;
            }

            // Block-local row-max.
            float m_block = max(max(vals[0], vals[1]), max(vals[2], vals[3]));
            m_block = max(m_block, simd_shuffle_xor(m_block, 1));
            m_block = max(m_block, simd_shuffle_xor(m_block, 2));

            // Combine with running max.
            float m_old = m_buf[row];
            float m_new = max(m_old, m_block);
            // alpha rescales the previous accumulator. When m_old is -INF
            // (no valid contribution yet), there's nothing to rescale, so
            // alpha = 0.  This is critical for numerical safety — exp(-INF -
            // m_new) would be 0 anyway, but explicit gating avoids NaN if
            // m_new is also -INF.
            float alpha = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);

            // Compute exp(s - m_new).  When m_new is -INF (entire row masked
            // forever), keep exp_vals at 0 to avoid NaN.
            float exp_vals[4];
            float l_block = 0.0f;
            for (int i = 0; i < 4; ++i) {
                if (vals[i] == -INFINITY || m_new == -INFINITY) {
                    exp_vals[i] = 0.0f;
                } else {
                    exp_vals[i] = exp(vals[i] - m_new);
                    l_block += exp_vals[i];
                }
            }
            l_block += simd_shuffle_xor(l_block, 1);
            l_block += simd_shuffle_xor(l_block, 2);

            float l_old = l_buf[row];
            float l_new = l_old * alpha + l_block;

            // Write P (unnormalized; final / l happens in epilogue).
            for (int i = 0; i < 4; ++i) {
                int col = sub_col * 4 + i;
                P_buf[row * BLOCK_SIZE + col] = T(exp_vals[i]);
            }

            // One thread per row updates the per-row state.
            if (sub_col == 0) {
                m_buf[row]     = m_new;
                l_buf[row]     = l_new;
                alpha_buf[row] = alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 2d: rescale O_stage by per-row alpha ---
        // Cooperative: each thread handles a few elements of O_stage and
        // looks up its row's alpha from alpha_buf.
        for (int i = (int)thread_idx; i < O_STAGE_ELEMS; i += 32) {
            int row = i / HEAD_SIZE;
            float a = alpha_buf[row];
            O_stage[i] *= a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 2e: cooperative V stage ---
        {
            int slot_stride_elem = num_kv_heads * HEAD_SIZE;
            int64_t kv_block_base = (int64_t)physical_block * kv_block_stride
                                  + (int64_t)kv_head_idx * kv_head_stride;
            for (int i = 0; i < ELEMS_PER_THREAD_KV; ++i) {
                int flat = (int)thread_idx + i * 32;
                if (flat >= KV_STAGE_ELEMS) break;
                int slot = flat / HEAD_SIZE;
                int dim  = flat % HEAD_SIZE;
                int kv_token = kv_block_idx * BLOCK_SIZE + slot;
                T value = T(0);
                if (kv_token < kv_seq_len) {
                    value = v_cache[kv_block_base + (int64_t)slot * slot_stride_elem + dim];
                }
                V_stage[slot * HEAD_SIZE + dim] = value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 2f: P @ V MMA → accumulate into O_stage ---
        // Unlike C2, O_frag is loaded from O_stage (which holds alpha *
        // O_old after the rescale), accumulated, and stored back.
        for (int n_frag = 0; n_frag < N_FRAGS_PV; ++n_frag) {
            simdgroup_matrix<float, 8, 8> O_frag;
            simdgroup_load(O_frag, O_stage + n_frag * 8, /*elements_per_row=*/HEAD_SIZE);
            for (int k_frag = 0; k_frag < K_FRAGS_PV; ++k_frag) {
                simdgroup_matrix<bfloat16_t, 8, 8> P_frag, V_frag;
                simdgroup_load(P_frag, P_buf + k_frag * 8, /*elements_per_row=*/BLOCK_SIZE);
                simdgroup_load(V_frag,
                               V_stage + k_frag * 8 * HEAD_SIZE + n_frag * 8,
                               /*elements_per_row=*/HEAD_SIZE);
                simdgroup_multiply_accumulate(O_frag, P_frag, V_frag, O_frag);
            }
            simdgroup_store(O_frag, O_stage + n_frag * 8, /*elements_per_row=*/HEAD_SIZE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }  // end kv_block loop

    // --- Phase 3: epilogue — normalize O by l, write to output ---
    for (int i = 0; i < ELEMS_PER_THREAD_Q; ++i) {
        int flat = (int)thread_idx + i * 32;
        if (flat >= Q_TILE_ELEMS) break;
        int row = flat / HEAD_SIZE;
        int col = flat % HEAD_SIZE;
        int q_token_offset = row / num_queries_per_kv;
        int q_head_offset  = row - q_token_offset * num_queries_per_kv;
        int global_q_token = q_seq_start + q_block_local_idx * block_q + q_token_offset;
        int global_q_head  = kv_head_idx * num_queries_per_kv + q_head_offset;
        if (q_token_offset >= block_q) continue;
        if (global_q_token >= q_seq_start + q_len) continue;
        float l = l_buf[row];
        float inv_l = 1.0f / (l + 1e-6f);
        out[(int64_t)global_q_token * q_stride + global_q_head * HEAD_SIZE + col]
            = T(O_stage[row * HEAD_SIZE + col] * inv_l);
    }

    (void)simd_tid;
    (void)simd_lid;
}

// ========================================== Template instantiations
//
// MVP: bfloat16, head_size=128, block_size=16, BLOCK_M=8.

#define instantiate_paged_attention_mma(type, head_size, block_size, block_m)  \
    template [[host_name("paged_attention_mma_" #type                          \
                         "_hs" #head_size                                      \
                         "_bs" #block_size                                     \
                         "_bm" #block_m)]] [[kernel]] void                     \
    paged_attention_mma<type, head_size, block_size, block_m>(                 \
        device type *out [[buffer(0)]],                                        \
        device const type *q [[buffer(1)]],                                    \
        device const type *k_cache [[buffer(2)]],                              \
        device const type *v_cache [[buffer(3)]],                              \
        const constant int &num_kv_heads [[buffer(4)]],                        \
        const constant float &scale [[buffer(5)]],                             \
        device const uint32_t *block_tables [[buffer(6)]],                     \
        device const uint32_t *seq_lens [[buffer(7)]],                         \
        const constant int &max_num_blocks_per_seq [[buffer(8)]],              \
        const constant int &q_stride [[buffer(9)]],                            \
        const constant int &kv_block_stride [[buffer(10)]],                    \
        const constant int &kv_head_stride [[buffer(11)]],                     \
        device const int32_t *cu_seqlens_q [[buffer(12)]],                     \
        const constant int &num_seqs [[buffer(13)]],                           \
        threadgroup char *shared_mem [[threadgroup(0)]],                       \
        uint3 tg_pos [[threadgroup_position_in_grid]],                         \
        uint3 tgs [[threadgroups_per_grid]],                                   \
        uint3 thread_pos [[thread_position_in_threadgroup]],                   \
        uint simd_tid [[simdgroup_index_in_threadgroup]],                      \
        uint simd_lid [[thread_index_in_simdgroup]])

instantiate_paged_attention_mma(bfloat16_t, 128, 16, 8);
