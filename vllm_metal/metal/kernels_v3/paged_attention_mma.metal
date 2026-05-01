// SPDX-License-Identifier: Apache-2.0
//
// v3 paged-attention kernel (MMA path).
//
// Status: C2 — single-block Q@K → naive softmax → P@V via simdgroup_matrix MMAs.
// C3 will replace single-block with multi-block + online softmax.
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
//   - C2 also requires: kv_seq_len <= BLOCK_SIZE (single-block case)
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
//   Q_tile  : T[8][128]      = 2048 B
//   K_stage : T[16][128]     = 4096 B
//   V_stage : T[16][128]     = 4096 B
//   S_buf   : float[8][16]   =  512 B   (Q@K^T scores; reused as P_buf bf16)
//   O_stage : float[8][128]  = 4096 B   (P@V output, then cast to T on write)
//   Total: 14848 B (limit: 32768 B per threadgroup)
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
    constexpr int Q_TILE_BYTES = BLOCK_M * HEAD_SIZE * sizeof(T);
    constexpr int K_STAGE_BYTES = BLOCK_SIZE * HEAD_SIZE * sizeof(T);
    constexpr int V_STAGE_BYTES = BLOCK_SIZE * HEAD_SIZE * sizeof(T);
    constexpr int S_BUF_BYTES = BLOCK_M * BLOCK_SIZE * sizeof(float);

    threadgroup T *Q_tile = (threadgroup T*)(shared_mem);
    threadgroup T *K_stage = (threadgroup T*)(shared_mem + Q_TILE_BYTES);
    threadgroup T *V_stage = (threadgroup T*)(shared_mem + Q_TILE_BYTES + K_STAGE_BYTES);
    threadgroup float *S_buf = (threadgroup float*)(shared_mem + Q_TILE_BYTES + K_STAGE_BYTES + V_STAGE_BYTES);
    threadgroup T *P_buf = (threadgroup T*)S_buf;  // alias — overwrites S after softmax cast
    threadgroup float *O_stage = (threadgroup float*)(shared_mem + Q_TILE_BYTES + K_STAGE_BYTES + V_STAGE_BYTES + S_BUF_BYTES);

    // --- Phase 1: cooperative Q load (with pre-applied softmax scale) ---
    constexpr int Q_TILE_ELEMS = BLOCK_M * HEAD_SIZE;
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
            // Pre-multiply by softmax scale so the QK MMA's accumulator is already scaled.
            value = T((float)raw * scale);
        }
        Q_tile[row * HEAD_SIZE + col] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ============================================================
    // C2: process the FIRST KV block only (kv_block_idx = 0).
    // C3 will wrap this in an outer loop with online softmax.
    // ============================================================
    const int kv_block_idx = 0;
    const int physical_block = (int)block_tables[seq_idx * max_num_blocks_per_seq + kv_block_idx];

    // --- Phase 2a: cooperative K stage ---
    // Cache layout: k_cache[block, slot, kv_head, dim] with strides
    //   block_stride = BLOCK_SIZE * num_kv_heads * head_size
    //   slot_stride  = num_kv_heads * head_size
    //   kv_head_stride = head_size
    //   dim_stride   = 1
    constexpr int KV_STAGE_ELEMS = BLOCK_SIZE * HEAD_SIZE;
    constexpr int ELEMS_PER_THREAD_KV = (KV_STAGE_ELEMS + 31) / 32;
    {
        int slot_stride_elem = num_kv_heads * HEAD_SIZE;
        int64_t kv_block_base = (int64_t)physical_block * kv_block_stride
                              + (int64_t)kv_head_idx * kv_head_stride;
        for (int i = 0; i < ELEMS_PER_THREAD_KV; ++i) {
            int flat = (int)thread_idx + i * 32;
            if (flat >= KV_STAGE_ELEMS) break;
            int slot = flat / HEAD_SIZE;
            int dim = flat % HEAD_SIZE;
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
    // S = Q @ K^T  with shapes:
    //   Q          : (BLOCK_M=8, HEAD_SIZE=128)
    //   K_stage    : (BLOCK_SIZE=16, HEAD_SIZE=128) row-major
    //   S          : (BLOCK_M=8, BLOCK_SIZE=16)
    // Fragment plan: M_FRAGS=1, N_FRAGS=2, K_FRAGS=16. 32 MMAs total.
    constexpr int N_FRAGS_QK = BLOCK_SIZE / 8;  // 2
    constexpr int K_FRAGS_QK = HEAD_SIZE / 8;   // 16

    for (int n_frag = 0; n_frag < N_FRAGS_QK; ++n_frag) {
        simdgroup_matrix<float, 8, 8> S_frag = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int k_frag = 0; k_frag < K_FRAGS_QK; ++k_frag) {
            simdgroup_matrix<bfloat16_t, 8, 8> Q_frag, K_frag_T;
            // Q sub-tile: rows [0,8), cols [k_frag*8, k_frag*8+8)
            simdgroup_load(Q_frag, Q_tile + k_frag * 8, /*elements_per_row=*/HEAD_SIZE);
            // K sub-tile: rows [n_frag*8, n_frag*8+8), cols [k_frag*8, k_frag*8+8)
            // Loaded with transpose=true → MMA receives K^T sub-tile.
            simdgroup_load(K_frag_T,
                           K_stage + n_frag * 8 * HEAD_SIZE + k_frag * 8,
                           /*elements_per_row=*/HEAD_SIZE,
                           /*matrix_origin=*/ulong2(0, 0),
                           /*transpose_matrix=*/true);
            simdgroup_multiply_accumulate(S_frag, Q_frag, K_frag_T, S_frag);
        }
        // Store this 8x8 S sub-tile to columns [n_frag*8, n_frag*8+8) of S_buf.
        simdgroup_store(S_frag, S_buf + n_frag * 8, /*elements_per_row=*/BLOCK_SIZE);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 2c: causal mask + naive single-block softmax ---
    // 32 threads cooperate on 8 rows × 16 cols = 128 elements (4 per thread).
    // Each row is owned by 4 threads (thread_idx / 4 == row), and each thread
    // covers cols [(thread_idx % 4) * 4, (thread_idx % 4) * 4 + 4).
    {
        int row = (int)thread_idx / 4;
        int sub_col = (int)thread_idx % 4;

        int q_token_offset = row / num_queries_per_kv;
        int q_head_offset  = row - q_token_offset * num_queries_per_kv;
        (void)q_head_offset;
        int global_q_token_in_seq = q_block_local_idx * block_q + q_token_offset;
        int effective_context = (kv_seq_len - q_len) + global_q_token_in_seq + 1;
        bool row_valid = (q_token_offset < block_q) && (global_q_token_in_seq < q_len);

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

        // Row-max via XOR reductions across the 4 threads owning this row.
        float row_max = max(max(vals[0], vals[1]), max(vals[2], vals[3]));
        row_max = max(row_max, simd_shuffle_xor(row_max, 1));
        row_max = max(row_max, simd_shuffle_xor(row_max, 2));
        // NaN-safe: if entire row was masked, use 0 to avoid -inf - -inf = NaN.
        if (row_max == -INFINITY) row_max = 0.0f;

        // Compute exp(s - max) and partial row sum.
        float exp_vals[4];
        float row_sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            exp_vals[i] = exp(vals[i] - row_max);
            row_sum += exp_vals[i];
        }
        row_sum += simd_shuffle_xor(row_sum, 1);
        row_sum += simd_shuffle_xor(row_sum, 2);

        // Normalize and cast to bf16 P. Aliased over S_buf bytes — safe because
        // each thread reads its 4 elements of S into registers above before any
        // P writes happen, and because P uses the first half of S_buf's bytes.
        float inv_sum = 1.0f / (row_sum + 1e-6f);
        for (int i = 0; i < 4; ++i) {
            int col = sub_col * 4 + i;
            P_buf[row * BLOCK_SIZE + col] = T(exp_vals[i] * inv_sum);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 2d: cooperative V stage ---
    {
        int slot_stride_elem = num_kv_heads * HEAD_SIZE;
        int64_t kv_block_base = (int64_t)physical_block * kv_block_stride
                              + (int64_t)kv_head_idx * kv_head_stride;
        for (int i = 0; i < ELEMS_PER_THREAD_KV; ++i) {
            int flat = (int)thread_idx + i * 32;
            if (flat >= KV_STAGE_ELEMS) break;
            int slot = flat / HEAD_SIZE;
            int dim = flat % HEAD_SIZE;
            int kv_token = kv_block_idx * BLOCK_SIZE + slot;
            T value = T(0);
            if (kv_token < kv_seq_len) {
                value = v_cache[kv_block_base + (int64_t)slot * slot_stride_elem + dim];
            }
            V_stage[slot * HEAD_SIZE + dim] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 2e: P @ V MMA → O_stage ---
    // O = P @ V  with shapes:
    //   P_buf      : (BLOCK_M=8, BLOCK_SIZE=16)
    //   V_stage    : (BLOCK_SIZE=16, HEAD_SIZE=128) row-major
    //   O_stage    : (BLOCK_M=8, HEAD_SIZE=128)  in float
    // Fragment plan: M_FRAGS=1, N_FRAGS=16, K_FRAGS=2. 32 MMAs.
    constexpr int N_FRAGS_PV = HEAD_SIZE / 8;   // 16
    constexpr int K_FRAGS_PV = BLOCK_SIZE / 8;  // 2

    for (int n_frag = 0; n_frag < N_FRAGS_PV; ++n_frag) {
        simdgroup_matrix<float, 8, 8> O_frag = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int k_frag = 0; k_frag < K_FRAGS_PV; ++k_frag) {
            simdgroup_matrix<bfloat16_t, 8, 8> P_frag, V_frag;
            // P sub-tile: rows [0,8), cols [k_frag*8, k_frag*8+8)
            simdgroup_load(P_frag, P_buf + k_frag * 8, /*elements_per_row=*/BLOCK_SIZE);
            // V sub-tile: rows [k_frag*8, k_frag*8+8), cols [n_frag*8, n_frag*8+8)
            simdgroup_load(V_frag,
                           V_stage + k_frag * 8 * HEAD_SIZE + n_frag * 8,
                           /*elements_per_row=*/HEAD_SIZE);
            simdgroup_multiply_accumulate(O_frag, P_frag, V_frag, O_frag);
        }
        // Store this 8x8 O sub-tile to columns [n_frag*8, n_frag*8+8) of O_stage.
        simdgroup_store(O_frag, O_stage + n_frag * 8, /*elements_per_row=*/HEAD_SIZE);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 3: cooperative output write (float → T cast) ---
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
        out[(int64_t)global_q_token * q_stride + global_q_head * HEAD_SIZE + col]
            = T(O_stage[row * HEAD_SIZE + col]);
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
