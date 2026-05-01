// SPDX-License-Identifier: Apache-2.0
//
// v3 paged-attention kernel (MMA path).
//
// Status: C1 skeleton — Q load + zero output write only.
// Future commits will add Q@K MMA, online softmax, P@V MMA.
//
// Design doc: research/mma_kernel_design.md
//
// MVP scope (gated by host-side eligibility filter):
//   - Q dtype: bfloat16
//   - K/V dtype: bfloat16 (no FP8, no TurboQuant)
//   - HEAD_SIZE: 128
//   - BLOCK_SIZE: 16
//   - BLOCK_M:   8 (one MMA tile in M direction)
//   - Causal mask only (no SWA, no sinks, no softcap, no ALiBi)
//   - Varlen + paged blocks: yes
//
// Type machinery, find_seq_idx, resolve_seq_and_q_block, and shared function
// constants are in attn_common.h, concatenated before this file by the build
// system.
//
// Portions of this file are adapted from the vLLM project
// (https://github.com/vllm-project/vllm)
// Licensed under the Apache License 2.0
// Copyright contributors to the vLLM project

#include "attn_common.h"

// ========================================== MMA paged-attention kernel
//
// Grid: (num_kv_heads, total_q_blocks, 1)
//   total_q_blocks = q.shape[0] / BLOCK_Q + num_seqs   (upper bound)
// Threadgroup: (NUM_THREADS=32, 1, 1)  — one simdgroup per program
//
// Per-program work: produce BLOCK_M output rows (= BLOCK_Q query tokens
// × num_queries_per_kv q-heads) for one kv_head_idx.
//
// BLOCK_Q is computed at runtime as BLOCK_M / num_queries_per_kv.  The host
// dispatcher's eligibility filter requires num_queries_per_kv ∈ {1,2,4,8}
// so that this division is exact and BLOCK_Q ∈ {1,2,4,8}.

template <typename T, int HEAD_SIZE, int BLOCK_SIZE, int BLOCK_M>
[[kernel]] void paged_attention_mma(
    device T *out [[buffer(0)]],                           // [total_q, n_q_heads, head_size]
    device const T *q [[buffer(1)]],                       // [total_q, n_q_heads, head_size]
    device const T *k_cache [[buffer(2)]],                 // [num_blocks, block_size, n_kv_heads, head_size]
    device const T *v_cache [[buffer(3)]],                 // [num_blocks, block_size, n_kv_heads, head_size]
    const constant int &num_kv_heads [[buffer(4)]],
    const constant float &scale [[buffer(5)]],
    device const uint32_t *block_tables [[buffer(6)]],     // [num_seqs, max_blocks_per_seq]
    device const uint32_t *seq_lens [[buffer(7)]],         // [num_seqs]  total kv length
    const constant int &max_num_blocks_per_seq [[buffer(8)]],
    const constant int &q_stride [[buffer(9)]],            // = n_q_heads * head_size
    const constant int &kv_block_stride [[buffer(10)]],
    const constant int &kv_head_stride [[buffer(11)]],
    device const int32_t *cu_seqlens_q [[buffer(12)]],     // [num_seqs+1]
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

    // Upper-bound padding programs and out-of-range q-blocks early-return.
    if (seq_idx >= num_seqs) return;
    if (q_block_local_idx * block_q >= q_len) return;

    // --- Threadgroup memory layout ---
    //   [0 ..  Q_TILE_BYTES)    Q_tile : T[BLOCK_M][HEAD_SIZE]
    // (KV staging buffers come in C2.)
    constexpr int Q_TILE_ELEMS = BLOCK_M * HEAD_SIZE;
    threadgroup T *Q_tile = reinterpret_cast<threadgroup T *>(shared_mem);

    // --- Phase 1: cooperative Q load from device → threadgroup memory ---
    //
    // 32 threads × ELEMS_PER_THREAD elements each = Q_TILE_ELEMS total.
    // For BLOCK_M=8, HEAD_SIZE=128: 32 × 32 = 1024 elements (perfect cover).
    constexpr int ELEMS_PER_THREAD = (Q_TILE_ELEMS + 31) / 32;
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int flat = (int)thread_idx + i * 32;
        if (flat >= Q_TILE_ELEMS) break;

        int row = flat / HEAD_SIZE;            // 0..BLOCK_M-1
        int col = flat % HEAD_SIZE;            // 0..HEAD_SIZE-1

        // row = q_token_offset * num_queries_per_kv + q_head_offset
        int q_token_offset = row / num_queries_per_kv;
        int q_head_offset  = row - q_token_offset * num_queries_per_kv;

        int global_q_token = q_seq_start + q_block_local_idx * block_q + q_token_offset;
        int global_q_head  = kv_head_idx * num_queries_per_kv + q_head_offset;

        T value = T(0);
        // Guard against the last partial Q-block in a sequence.
        if (q_token_offset < block_q && global_q_token < q_seq_start + q_len) {
            value = q[(int64_t)global_q_token * q_stride
                      + global_q_head * HEAD_SIZE
                      + col];
        }
        Q_tile[row * HEAD_SIZE + col] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- C1 stub: skip Q@K MMA, online softmax, V load, P@V MMA ---
    // (Will be added in C2 and C3.)

    // --- Phase 3: cooperative output write ---
    //
    // For C1: write zeros to the output at the same (q_token, q_head)
    // positions that Q was loaded from.  This validates the output address
    // arithmetic; a future commit replaces the zeros with the real attention
    // output computed from the MMA path.
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int flat = (int)thread_idx + i * 32;
        if (flat >= Q_TILE_ELEMS) break;

        int row = flat / HEAD_SIZE;
        int col = flat % HEAD_SIZE;

        int q_token_offset = row / num_queries_per_kv;
        int q_head_offset  = row - q_token_offset * num_queries_per_kv;

        int global_q_token = q_seq_start + q_block_local_idx * block_q + q_token_offset;
        int global_q_head  = kv_head_idx * num_queries_per_kv + q_head_offset;

        // Skip the partial-block padding rows.
        if (q_token_offset >= block_q) continue;
        if (global_q_token >= q_seq_start + q_len) continue;

        out[(int64_t)global_q_token * q_stride
            + global_q_head * HEAD_SIZE
            + col] = T(0);
    }

    // Suppress unused-parameter warnings for buffers reserved for C2/C3.
    (void)k_cache;
    (void)v_cache;
    (void)scale;
    (void)block_tables;
    (void)max_num_blocks_per_seq;
    (void)kv_block_stride;
    (void)kv_head_stride;
    (void)kv_seq_len;
    (void)simd_tid;
    (void)simd_lid;
}

// ========================================== Template instantiations
//
// MVP: bfloat16, head_size=128, block_size=16, BLOCK_M=8.
// Future commits will expand this matrix.

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
