// SPDX-License-Identifier: Apache-2.0
//
// v3 paged-attention kernels (vector path).
//
// This file contains:
//   - paged_attention<...>           — the main attention compute kernel
//   - paged_attention_v2_reduce<...> — split-K partition reduce kernel
//   - host_name template instantiations for all (T, K_CACHE_T, V_CACHE_T,
//     HEAD_SIZE, BLOCK_SIZE) combinations.
//
// Type machinery, FP8 helpers, Qk_dot, block_sum, find_seq_idx, and shared
// function constants (use_partitioning, use_alibi, etc.) are in attn_common.h,
// concatenated before this file by the build system.
//
// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.
//
// Portions of this file are adapted from the vLLM project
// (https://github.com/vllm-project/vllm)
// Licensed under the Apache License 2.0
// Copyright contributors to the vLLM project

#include "attn_common.h"

template <typename T, typename K_CACHE_T, typename V_CACHE_T, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int NUM_SIMD_LANES, int PARTITION_SIZE = 0>
[[kernel]] void paged_attention(
    device float *exp_sums
    [[buffer(0), function_constant(use_partitioning)]], // [num_seqs, num_heads,
                                                        // max_num_partitions]
    device float *max_logits
    [[buffer(1), function_constant(use_partitioning)]], // [num_seqs, num_heads,
                                                        // max_num_partitions]
    device T *out
    [[buffer(2)]], // [num_seqs, num_heads, max_num_partitions, head_size]
    device const T *q [[buffer(3)]], // [num_seqs, num_heads, head_size]
    device const K_CACHE_T *k_cache
    [[buffer(4)]], // [num_blocks, block_size, num_kv_heads, head_size]
    device const V_CACHE_T *v_cache
    [[buffer(5)]], // [num_blocks, block_size, num_kv_heads, head_size]
    const device float *__restrict__ k_scale
    [[buffer(6), function_constant(use_fp8_scales)]], // [1]
    const device float *__restrict__ v_scale
    [[buffer(7), function_constant(use_fp8_scales)]], // [1]
    const constant int &num_kv_heads [[buffer(8)]],   // [num_heads]
    const constant float &scale [[buffer(9)]],
    const constant float &softcapping [[buffer(10)]],
    device const uint32_t *block_tables
    [[buffer(11)]], // [num_seqs, max_num_blocks_per_seq]
    device const uint32_t *context_lens [[buffer(12)]], // [num_seqs]
    const constant int &max_num_blocks_per_seq [[buffer(13)]],
    device const float *alibi_slopes
    [[buffer(14), function_constant(use_alibi)]], // [num_heads]
    const constant int &q_stride [[buffer(15)]],
    const constant int &kv_block_stride [[buffer(16)]],
    const constant int &kv_head_stride [[buffer(17)]],
    const device float *sinks
    [[buffer(18), function_constant(use_sinks)]], // [num_heads]
    device const int32_t *cu_seqlens_q [[buffer(19)]],  // [num_seqs + 1]
    const constant int &num_seqs [[buffer(20)]],
    const constant int &sliding_window [[buffer(21)]],  // -1 = disabled
    const device half *key_scale_cache
    [[buffer(22), function_constant(use_turboquant)]], // [num_blocks, block_size, num_kv_heads, head_size/32]
    const device half *value_scale_cache
    [[buffer(23), function_constant(use_turboquant)]], // [num_blocks, block_size, num_kv_heads, head_size/32]
    const constant int &v_block_stride
    [[buffer(24), function_constant(use_turboquant)]],
    const constant int &v_head_stride
    [[buffer(25), function_constant(use_turboquant)]],
    const device half *key_zero_cache
    [[buffer(26), function_constant(use_turboquant)]], // [num_blocks, block_size, num_kv_heads, head_size/32]
    const device float *v_centroids
    [[buffer(27), function_constant(use_turboquant)]], // [2^v_bits] Lloyd-Max centroids for V
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // Varlen: each threadgroup handles one query token.
  // Use binary search on cu_seqlens_q to find which sequence it belongs to.
  const int q_token_idx = threadgroup_position_in_grid.y;
  const int seq_idx = find_seq_idx(cu_seqlens_q, q_token_idx, num_seqs);
  const int q_seq_start = cu_seqlens_q[seq_idx];
  const int q_len = cu_seqlens_q[seq_idx + 1] - q_seq_start;
  const int q_pos_in_seq = q_token_idx - q_seq_start;
  const int partition_idx = threadgroup_position_in_grid.z;
  const int max_num_partitions = threadgroups_per_grid.z;
  const int thread_idx = thread_position_in_threadgroup.x;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const uint32_t context_len = context_lens[seq_idx];  // total KV length for this seq

  // Causal: this query token can attend to KV positions [0, effective_context_len).
  const int effective_context_len = (int)context_len - q_len + q_pos_in_seq + 1;
  if (effective_context_len <= 0) {
    // No KV tokens to attend to. Caller guarantees out is zero-initialized.
    return;
  }

  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= effective_context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(effective_context_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(NUM_SIMD_LANES / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE
                                       // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, NUM_SIMD_LANES);
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  const int head_idx = threadgroup_position_in_grid.x;
  const int num_heads = threadgroups_per_grid.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = !use_alibi ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);
  using K_vec = typename Vec<T, VEC_SIZE>::Type;
  using Q_vec = typename Vec<T, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<K_CACHE_T, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the thread group size is 4, then the first thread in the
  // group has 0, 4, 8, ... th vectors of the query, and the second thread has
  // 1, 5, 9, ... th vectors of the query, and so on.
  const device T *q_ptr = q + q_token_idx * q_stride + head_idx * HEAD_SIZE;
  threadgroup Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const device Q_vec *>(q_ptr + vec_idx * VEC_SIZE);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Workspace for cross-warp reduction of online softmax state.
  // Layout: [NUM_WARPS] for m, [NUM_WARPS] for l, then HEAD_SIZE floats
  // for the tree reduction of O accumulators.
  threadgroup float red_smem[2 * NUM_WARPS];

  // Token stride within a block: num_kv_heads * (stride between adjacent KV heads).
  // For non-TQ / 8-bit K, kv_head_stride == HEAD_SIZE.
  // For sub-8-bit TQ K, kv_head_stride == k_packed_dim (head_size * k_bits / 8),
  // so using kv_head_stride here keeps K pointer arithmetic correct in both cases.
  const int kv_token_stride = num_kv_heads * kv_head_stride;

  // ========== Online softmax: per-warp state ==========
  // Each warp maintains its own running (m, l, O) across its KV blocks.
  // m = running max of QK scores (for numerical stability)
  // l = running sum of exp(score - m) (softmax denominator)
  // O = running unnormalized output accumulator
  float warp_m = -FLT_MAX;  // running max
  float warp_l = 0.f;       // running exp sum

  constexpr int V_ELEMS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_SIMD_LANES);
  float v_accs[V_ELEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < V_ELEMS_PER_THREAD; i++) {
    v_accs[i] = 0.f;
  }

  // ========== Fused QK + online softmax + V accumulation ==========
  // Each warp processes a subset of KV blocks. Within each block, we:
  //   1. Compute QK dot products (same thread-group mechanism as v1)
  //   2. Collect scores for this block into a per-warp buffer
  //   3. Update online softmax state (m, l) and rescale O
  //   4. Accumulate weighted V into O
  //
  // The scores for one KV block (up to BLOCK_SIZE floats per warp) are
  // stored in threadgroup memory. Each warp uses its own slice so there
  // are no conflicts between warps.
  threadgroup float *warp_scores =
      reinterpret_cast<threadgroup float *>(shared_mem) +
      warp_idx * BLOCK_SIZE;
  // NOTE: the previous fwht_buf per-warp workspace (NUM_WARPS * HEAD_SIZE
  // floats) has been removed — the TurboQuant V dequant path now runs
  // register-only via `tq_load_and_accumulate_v` + `inverse_fwht_in_place`,
  // so no threadgroup memory is needed for V reconstruction.  The host-side
  // shmem-size calculation in paged_ops.cpp no longer adds the TQ bonus.

  const device uint32_t *block_table =
      block_tables + seq_idx * max_num_blocks_per_seq;

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // --- Step 1: Compute QK scores for this block ---
    // The QK dot product uses "thread groups" (sub-warp groups) where
    // each thread group cooperatively computes the dot product for one
    // KV token. Only thread_group_offset==0 holds the final score.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * NUM_SIMD_LANES) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const device K_CACHE_T *k_ptr =
            k_cache + physical_block_number * kv_block_stride +
            physical_block_offset * kv_token_stride +
            kv_head_idx * kv_head_stride;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;

        if constexpr (is_uchar<K_CACHE_T>()) {
          // uchar K: FP8 (non-TQ) or TurboQuant uint8 / sub-8-bit K path.
          if (use_turboquant) {
            constexpr int SCALE_GROUP_SIZE = 32;
            constexpr int SCALE_GROUPS = HEAD_SIZE / SCALE_GROUP_SIZE;
            const int64_t k_scale_base_offset =
                physical_block_number * (int64_t)(BLOCK_SIZE * num_kv_heads * SCALE_GROUPS) +
                physical_block_offset * (num_kv_heads * SCALE_GROUPS) +
                kv_head_idx * SCALE_GROUPS;
            tq_load_k_vec<T, K_CACHE_T, VEC_SIZE>(
                k_vecs[j], k_ptr, key_scale_cache, key_zero_cache,
                k_scale_base_offset, vec_idx, k_bits);
          } else {
            // FP8 path
            Quant_vec k_vec_quant = *reinterpret_cast<const device Quant_vec *>(
                k_ptr + vec_idx * VEC_SIZE);
            k_vecs[j] = fp8_convert<K_vec, Quant_vec>(k_vec_quant, *k_scale);
          }
        } else if constexpr (is_char<K_CACHE_T>()) {
          // char K: TQ int8 K — always asymmetric dequant (no FP8 for char)
          constexpr int SCALE_GROUP_SIZE = 32;
          constexpr int SCALE_GROUPS = HEAD_SIZE / SCALE_GROUP_SIZE;
          const int64_t k_scale_base_offset =
              physical_block_number * (int64_t)(BLOCK_SIZE * num_kv_heads * SCALE_GROUPS) +
              physical_block_offset * (num_kv_heads * SCALE_GROUPS) +
              kv_head_idx * SCALE_GROUPS;
          tq_load_k_vec<T, K_CACHE_T, VEC_SIZE>(
              k_vecs[j], k_ptr, key_scale_cache, key_zero_cache,
              k_scale_base_offset, vec_idx, k_bits);
        } else {
          k_vecs[j] = *reinterpret_cast<const device K_vec *>(
              k_ptr + vec_idx * VEC_SIZE);
        }
      }

      float qk = scale * Qk_dot<T, THREAD_GROUP_SIZE>::dot(
                             q_vecs[thread_group_offset], k_vecs);

      if (softcapping > 0.0f) {
        qk = tanh(qk / softcapping) * softcapping;
      }

      qk +=
          (alibi_slope != 0) ? alibi_slope * (token_idx - effective_context_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Causal mask: only attend to KV positions < effective_context_len.
        bool mask = token_idx >= effective_context_len;
        // Sliding window mask: skip positions too far in the past.
        if (sliding_window >= 0) {
          mask = mask || (token_idx < effective_context_len - sliding_window);
        }
        warp_scores[physical_block_offset] = mask ? -FLT_MAX : qk;
      }
    }

    // Ensure all scores for this block are written before reading them.
    // Each warp writes/reads its own warp_scores slice, so we only need
    // intra-SIMD-group visibility. Using simdgroup_barrier (not
    // threadgroup_barrier) because warps may have different iteration
    // counts in the warp-strided loop — threadgroup_barrier would be UB.
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 2: Online softmax update ---
    // Find block-local max (only lane 0 of each thread group has scores,
    // but for V accumulation ALL lanes need the weights, so we broadcast).
    // Valid tokens in this block:
    const int block_start_token = block_idx * BLOCK_SIZE;
    const int block_valid_tokens =
        MIN(BLOCK_SIZE, effective_context_len - block_start_token);

    // Find max score in this block (all lanes participate for speed).
    float block_max = -FLT_MAX;
    for (int t = lane; t < block_valid_tokens; t += NUM_SIMD_LANES) {
      block_max = max(block_max, warp_scores[t]);
    }
    // Reduce max within the warp.
#pragma unroll
    for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
      block_max = max(block_max, simd_shuffle_xor(block_max, mask));
    }

    // Compute correction factor to rescale previous state.
    float new_m = max(warp_m, block_max);
    // NaN-safe: if new_m is still -inf (all masked), clamp to 0.
    if (new_m == -FLT_MAX) new_m = 0.f;

    float old_correction = exp(warp_m - new_m);
    // If warp_m was -FLT_MAX (first iteration), correction = 0, which
    // correctly zeroes out the (already zero) previous O and l.
    if (warp_m == -FLT_MAX) old_correction = 0.f;

    // Rescale running state.
#pragma unroll
    for (int i = 0; i < V_ELEMS_PER_THREAD; i++) {
      v_accs[i] *= old_correction;
    }
    warp_l *= old_correction;
    warp_m = new_m;

    // --- Step 3: Compute exp weights and accumulate V ---
    // For each valid token in this block, compute its softmax weight
    // (unnormalized) and accumulate into O.
    for (int tok = 0; tok < block_valid_tokens; tok++) {
      const float score = warp_scores[tok];
      const float w = exp(score - warp_m);
      warp_l += w;

      // Load V and accumulate: O += w * V
      if (use_turboquant) {
        // TurboQuant V: 3-bit Lloyd-Max + FWHT dequantization.
        const device uchar *v_ptr =
            reinterpret_cast<const device uchar *>(v_cache) +
            (int64_t)physical_block_number * v_block_stride +
            tok * (num_kv_heads * v_head_stride) +
            kv_head_idx * v_head_stride;
        constexpr int SCALE_GROUP_SIZE = 32;
        constexpr int SCALE_GROUPS = HEAD_SIZE / SCALE_GROUP_SIZE;
        const int64_t v_scale_base_offset =
            physical_block_number * (int64_t)(BLOCK_SIZE * num_kv_heads * SCALE_GROUPS) +
            tok * (num_kv_heads * SCALE_GROUPS) +
            kv_head_idx * SCALE_GROUPS;
        tq_load_and_accumulate_v<HEAD_SIZE, NUM_SIMD_LANES>(
            v_accs, v_ptr, value_scale_cache, v_scale_base_offset, w, lane,
            v_centroids, v_bits);
      } else {
        const device V_CACHE_T *v_ptr =
            v_cache + physical_block_number * kv_block_stride +
            tok * kv_token_stride +
            kv_head_idx * kv_head_stride;
#pragma unroll
        for (int i = 0; i < V_ELEMS_PER_THREAD; i++) {
          const int d = lane + i * NUM_SIMD_LANES;
          if (d < HEAD_SIZE) {
            float v_val;
            if constexpr (is_uchar<V_CACHE_T>()) {
              v_val = fp8_e4m3_to_float(v_ptr[d]) * (*v_scale);
            } else {
              v_val = float(v_ptr[d]);
            }
            v_accs[i] += w * v_val;
          }
        }
      }
    }

    // Barrier before next iteration reuses warp_scores.
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Ensure all warps have finished the KV loop before reusing shared_mem
  // for the merge. Without this barrier, early-exiting warps could write
  // merge state into shared_mem regions still used as warp_scores by
  // slower warps (the memory aliases).
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ========== Cross-warp merge of online softmax state ==========
  // Each warp has its own (warp_m, warp_l, v_accs[]). We merge by having
  // all warps write their state to shared memory, then warp 0 reads and
  // merges sequentially. Simple and barrier-safe (all barriers are
  // reached by all threads in the threadgroup).

  // For non-partitioned mode, include the sink in each warp's state.
  if (!USE_PARTITIONING && use_sinks) {
    float sink_val = sinks[head_idx];
    float new_m = max(warp_m, sink_val);
    float old_corr = (warp_m == -FLT_MAX) ? 0.f : exp(warp_m - new_m);
#pragma unroll
    for (int i = 0; i < V_ELEMS_PER_THREAD; i++) {
      v_accs[i] *= old_corr;
    }
    warp_l = warp_l * old_corr + exp(sink_val - new_m);
    warp_m = new_m;
  }

  // Shared memory layout for merge:
  //   merge_m[NUM_WARPS]: per-warp max values
  //   merge_l[NUM_WARPS]: per-warp exp sums
  //   merge_O[NUM_WARPS * HEAD_SIZE]: per-warp O accumulators
  threadgroup float *merge_m =
      reinterpret_cast<threadgroup float *>(shared_mem);
  threadgroup float *merge_l = merge_m + NUM_WARPS;
  threadgroup float *merge_O =
      reinterpret_cast<threadgroup float *>(shared_mem) + 2 * NUM_WARPS;

  // All warps write their state to shared memory.
  if (lane == 0) {
    merge_m[warp_idx] = warp_m;
    merge_l[warp_idx] = warp_l;
  }
  // Each warp writes its O accumulator.
  {
    threadgroup float *my_O = &merge_O[warp_idx * HEAD_SIZE];
#pragma unroll
    for (int j = 0; j < V_ELEMS_PER_THREAD; j++) {
      const int d = lane + j * NUM_SIMD_LANES;
      if (d < HEAD_SIZE) {
        my_O[d] = v_accs[j];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Warp 0 reads all warp states and merges sequentially.
  if (warp_idx == 0) {
    // Start with warp 0's state (already in v_accs, warp_m, warp_l).
    // Merge warps 1..NUM_WARPS-1 into it.
    for (int w = 1; w < NUM_WARPS; w++) {
      float other_m = merge_m[w];
      float other_l = merge_l[w];

      // Skip warps that processed no blocks (m == -FLT_MAX).
      if (other_m == -FLT_MAX && other_l == 0.f) continue;

      float new_m = max(warp_m, other_m);
      if (new_m == -FLT_MAX) new_m = 0.f;

      float my_corr = (warp_m == -FLT_MAX) ? 0.f : exp(warp_m - new_m);
      float other_corr = (other_m == -FLT_MAX) ? 0.f : exp(other_m - new_m);

      const threadgroup float *other_O = &merge_O[w * HEAD_SIZE];
#pragma unroll
      for (int j = 0; j < V_ELEMS_PER_THREAD; j++) {
        const int d = lane + j * NUM_SIMD_LANES;
        if (d < HEAD_SIZE) {
          v_accs[j] = v_accs[j] * my_corr + other_O[d] * other_corr;
        }
      }
      warp_m = new_m;
      warp_l = warp_l * my_corr + other_l * other_corr;
    }

    // For partitioned mode, persist the merged partition statistics for the
    // reduce kernel. These must match the normalized tmp_out written below.
    if (USE_PARTITIONING && thread_idx == 0 && use_partitioning) {
      device float *max_logits_ptr =
          max_logits + q_token_idx * num_heads * max_num_partitions +
          head_idx * max_num_partitions + partition_idx;
      *max_logits_ptr = warp_m;
      device float *exp_sums_ptr = exp_sums +
                                   q_token_idx * num_heads * max_num_partitions +
                                   head_idx * max_num_partitions + partition_idx;
      *exp_sums_ptr = warp_l;
    }

    // TurboQuant V: we've been accumulating in the rotated (FWHT) domain
    // the whole block loop.
    //
    // Non-partitioned path (USE_PARTITIONING == false): apply inverse FWHT
    // ONCE here to the merged per-head output, before the final normalise +
    // write to `out`.  Replaces O(ctx) per-token FWHTs with exactly one FWHT
    // per head per kernel dispatch.
    //
    // Partitioned path (USE_PARTITIONING == true): SKIP the FWHT here and
    // write the rotated, un-FWHT'd partial sum to `tmp_out`.  The reduce
    // kernel then combines partitions in fp32 and applies a single inverse
    // FWHT on the merged result (see `paged_attention_v2_reduce`).  This is
    // exact by linearity of FWHT —
    //   InverseFWHT(Σ_j w_j · V_rot_j) = Σ_j w_j · InverseFWHT(V_rot_j)
    // — but numerically superior: the weighted sum across partitions is
    // accumulated in fp32 *before* any FWHT rounding, and the single FWHT
    // at the end absorbs all fp16 casting once.  The old code applied the
    // FWHT per partition and stored the fp16-cast post-FWHT result in
    // tmp_out, compounding rounding error across partitions.  At long
    // contexts with many partitions the drift was measurable.
    //
    // Both branches are constexpr on USE_PARTITIONING so exactly one survives
    // per instantiation; the non-TQ specialisation dead-strips the entire
    // block via the `use_turboquant` function constant.
    //
    // Compile-time guard: paged_attention is templated for non-power-of-2
    // head sizes too (80, 112, 192, ...) for non-TQ workloads.  For those,
    // V_ELEMS_PER_THREAD = ceil(HEAD_SIZE/32) > HEAD_SIZE/32, which would
    // trip inverse_fwht_in_place's `static_assert(ELEMS_PER_LANE == HEAD_SIZE/32)`
    // at template instantiation time even though the runtime branch is dead
    // (use_turboquant=false for those sizes).  `if constexpr` blocks the
    // template instantiation entirely.  TQ requires HEAD_SIZE ∈ {64,128,256,512},
    // all multiples of 32, so this never blocks a real TQ dispatch.
    if constexpr (HEAD_SIZE % 32 == 0) {
      if (use_turboquant && !USE_PARTITIONING) {
        inverse_fwht_in_place<HEAD_SIZE, V_ELEMS_PER_THREAD>(v_accs, lane);
      }
    }

    // Final normalization: O = O / l
    const float inv_l = 1.f / (warp_l + 1e-6f);

    device T *out_ptr =
        out + q_token_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int j = 0; j < V_ELEMS_PER_THREAD; j++) {
      const int d = lane + j * NUM_SIMD_LANES;
      if (d < HEAD_SIZE) {
        *(out_ptr + d) = T(v_accs[j] * inv_l);
      }
    }
  }
}

template <typename T, int HEAD_SIZE, int NUM_THREADS, int NUM_SIMD_LANES,
          int PARTITION_SIZE = 0>
[[kernel]] void paged_attention_v2_reduce(
    device T *out [[buffer(0)]], const device float *exp_sums [[buffer(1)]],
    const device float *max_logits [[buffer(2)]],
    const device T *tmp_out [[buffer(3)]],
    device uint32_t *context_lens [[buffer(4)]],
    const constant int &max_num_partitions [[buffer(5)]],
    const device float *sinks
    [[buffer(6), function_constant(use_sinks)]], // [num_heads]
    device const int32_t *cu_seqlens_q [[buffer(7)]],  // [num_seqs + 1]
    const constant int &num_seqs [[buffer(8)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int num_heads = threadgroups_per_grid.x;
  const int head_idx = threadgroup_position_in_grid.x;
  // Varlen: grid.y is q_token_idx (one per query token), not seq_idx.
  const int q_token_idx = threadgroup_position_in_grid.y;
  const int seq_idx = find_seq_idx(cu_seqlens_q, q_token_idx, num_seqs);
  const int q_seq_start = cu_seqlens_q[seq_idx];
  const int q_len = cu_seqlens_q[seq_idx + 1] - q_seq_start;
  const int q_pos_in_seq = q_token_idx - q_seq_start;
  const uint32_t context_len = context_lens[seq_idx];
  const int effective_context_len = (int)context_len - q_len + q_pos_in_seq + 1;
  const int num_partitions = DIVIDE_ROUND_UP(effective_context_len, PARTITION_SIZE);

  // ========================================================================
  // Workspace declarations (function scope — Metal requires threadgroup
  // allocations at function scope, not inside conditional branches).  These
  // are hoisted above the early-out so the TQ deferred-FWHT path can share
  // the same buffers whether we early-out or run the full reduce.
  // ========================================================================
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  // Reduction workspace (main path only).
  threadgroup float red_smem[2 * NUM_WARPS];

  // TQ deferred-FWHT staging: warp 0 applies a single inverse FWHT to the
  // cross-partition weighted sum, which lives here in fp32 before the final
  // fp16 cast to `out`.  Declared unconditionally; dead in non-TQ kernels.
  // Size: HEAD_SIZE * 4 bytes = at most 2 KB at HEAD_SIZE=512.
  threadgroup float combined[HEAD_SIZE];

  // ========================================================================
  // Early-out: only one partition actually contributed.  We still need to
  // apply the deferred inverse FWHT on the TQ path because paged_attention
  // writes tmp_out in the rotated (un-FWHT'd) domain for PARTITION_SIZE>0.
  // ========================================================================
  if (num_partitions == 1 && !use_sinks) {
    device T *out_ptr =
        out + q_token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const device T *tmp_out_ptr =
        tmp_out + q_token_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE;

    if (use_turboquant) {
      // Warp 0 owns the entire HEAD_SIZE vector as a lane-strided register
      // slice: lane `l` holds indices { l, l+32, l+64, ... }.  Load,
      // InverseFWHT once, write — no cross-warp sync needed since only
      // warp 0 participates.
      if (warp_idx == 0) {
        constexpr int ELEMS_PER_LANE = HEAD_SIZE / 32;
        float v_accs[ELEMS_PER_LANE];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
          const int d = lane + e * 32;
          v_accs[e] = (d < HEAD_SIZE) ? float(tmp_out_ptr[d]) : 0.0f;
        }
        inverse_fwht_in_place<HEAD_SIZE, ELEMS_PER_LANE>(v_accs, lane);
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
          const int d = lane + e * 32;
          if (d < HEAD_SIZE) {
            out_ptr[d] = T(v_accs[e]);
          }
        }
      }
    } else {
      // Non-TQ: plain copy from partition 0.
      for (int i = thread_position_in_threadgroup.x; i < HEAD_SIZE;
           i += threads_per_threadgroup.x) {
        out_ptr[i] = tmp_out_ptr[i];
      }
    }
    return;
  }

  // Load max logits to shared memory.
  threadgroup float *shared_max_logits =
      reinterpret_cast<threadgroup float *>(shared_mem);
  const device float *max_logits_ptr =
      max_logits + q_token_idx * num_heads * max_num_partitions +
      head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = thread_position_in_threadgroup.x; i < num_partitions;
       i += threads_per_threadgroup.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = max(max_logit, l);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    max_logit = max(max_logit, simd_shuffle_xor(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = max(max_logit, simd_shuffle_xor(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = simd_shuffle(max_logit, 0);

  // Include the sink in the global max before rescaling.
  if (use_sinks) {
    max_logit = max(max_logit, sinks[head_idx]);
  }

  // Load rescaled exp sums to shared memory.
  threadgroup float *shared_exp_sums = reinterpret_cast<threadgroup float *>(
      shared_mem + sizeof(float) * num_partitions);
  const device float *exp_sums_ptr = exp_sums +
                                     q_token_idx * num_heads * max_num_partitions +
                                     head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = thread_position_in_threadgroup.x; i < num_partitions;
       i += threads_per_threadgroup.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * exp(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  global_exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(
      &red_smem[NUM_WARPS], global_exp_sum, simd_tid, simd_lid);

  // Include the sink in the global exp sum.
  if (use_sinks) {
    global_exp_sum += exp(sinks[head_idx] - max_logit);
  }

  const float inv_global_exp_sum = 1.0f / (global_exp_sum + 1e-6f);

  // ========================================================================
  // Aggregate tmp_out to out.
  //
  // Non-TQ path: weighted sum of per-partition outputs, cast to T, done.
  //
  // TQ path: per-partition tmp_out entries are in the ROTATED (un-FWHT'd)
  // domain (paged_attention skips the FWHT when PARTITION_SIZE>0).  We
  // accumulate the fp32 weighted sum into shared `combined[HEAD_SIZE]`, then
  // warp 0 applies the deferred inverse FWHT once on the merged vector and
  // writes the final fp16 output.  This is exact by linearity of FWHT —
  //   InverseFWHT(Σ_j w_j · V_rot_j) = Σ_j w_j · InverseFWHT(V_rot_j)
  // — and numerically strictly better than applying the FWHT per partition
  // because (a) cross-partition sums happen in fp32 with no intermediate
  // fp16 round-trip, (b) the 1/sqrt(N) normalisation that squeezes values
  // near fp16's ULP boundary is applied exactly once on the final merged
  // vector, and (c) the number of FWHTs per kernel drops from
  // O(num_partitions) to 1.
  // ========================================================================
  const device T *tmp_out_ptr =
      tmp_out + q_token_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE;
  device T *out_ptr =
      out + q_token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;

  if (use_turboquant) {
    // Stage weighted partial sums into `combined[]` in fp32.
    for (int i = thread_position_in_threadgroup.x; i < HEAD_SIZE;
         i += NUM_THREADS) {
      float acc = 0.0f;
      for (int j = 0; j < num_partitions; ++j) {
        acc += float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] *
               inv_global_exp_sum;
      }
      combined[i] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Warp 0 applies the single inverse FWHT and writes out.
    if (warp_idx == 0) {
      constexpr int ELEMS_PER_LANE = HEAD_SIZE / 32;
      float v_accs[ELEMS_PER_LANE];
      #pragma unroll
      for (int e = 0; e < ELEMS_PER_LANE; e++) {
        const int d = lane + e * 32;
        v_accs[e] = (d < HEAD_SIZE) ? combined[d] : 0.0f;
      }
      inverse_fwht_in_place<HEAD_SIZE, ELEMS_PER_LANE>(v_accs, lane);
      #pragma unroll
      for (int e = 0; e < ELEMS_PER_LANE; e++) {
        const int d = lane + e * 32;
        if (d < HEAD_SIZE) {
          out_ptr[d] = T(v_accs[e]);
        }
      }
    }
  } else {
    // Non-TQ: direct weighted sum + write.
#pragma unroll
    for (int i = thread_position_in_threadgroup.x; i < HEAD_SIZE;
         i += NUM_THREADS) {
      float acc = 0.0f;
      for (int j = 0; j < num_partitions; ++j) {
        acc += float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] *
               inv_global_exp_sum;
      }
      out_ptr[i] = T(acc);
    }
  }
}

#define instantiate_paged_attention_inner(type, k_cache_type, v_cache_type,    \
                                          head_size, block_size, num_threads,  \
                                          num_simd_lanes, partition_size)      \
  template [[host_name("paged_attention_" #type "_cache_" #k_cache_type        \
                       "_" #v_cache_type                                       \
                       "_hs" #head_size "_bs" #block_size "_nt" #num_threads   \
                       "_nsl" #num_simd_lanes                                  \
                       "_ps" #partition_size)]] [[kernel]] void                \
  paged_attention<type, k_cache_type, v_cache_type, head_size, block_size,     \
                  num_threads, num_simd_lanes, partition_size>(                \
      device float *exp_sums                                                   \
      [[buffer(0), function_constant(use_partitioning)]],                      \
      device float *max_logits                                                 \
      [[buffer(1), function_constant(use_partitioning)]],                      \
      device type *out [[buffer(2)]], device const type *q [[buffer(3)]],      \
      device const k_cache_type *k_cache [[buffer(4)]],                        \
      device const v_cache_type *v_cache [[buffer(5)]],                        \
      device const float *k_scale                                              \
      [[buffer(6), function_constant(use_fp8_scales)]],                        \
      device const float *v_scale                                              \
      [[buffer(7), function_constant(use_fp8_scales)]],                        \
      const constant int &num_kv_heads [[buffer(8)]],                          \
      const constant float &scale [[buffer(9)]],                               \
      const constant float &softcapping [[buffer(10)]],                        \
      device const uint32_t *block_tables [[buffer(11)]],                      \
      device const uint32_t *context_lens [[buffer(12)]],                      \
      const constant int &max_num_blocks_per_seq [[buffer(13)]],               \
      device const float *alibi_slopes                                         \
      [[buffer(14), function_constant(use_alibi)]],                            \
      const constant int &q_stride [[buffer(15)]],                             \
      const constant int &kv_block_stride [[buffer(16)]],                      \
      const constant int &kv_head_stride [[buffer(17)]],                       \
      const device float *sinks [[buffer(18), function_constant(use_sinks)]],  \
      device const int32_t *cu_seqlens_q [[buffer(19)]],                       \
      const constant int &num_seqs [[buffer(20)]],                             \
      const constant int &sliding_window [[buffer(21)]],                       \
      const device half *key_scale_cache                                       \
      [[buffer(22), function_constant(use_turboquant)]],                       \
      const device half *value_scale_cache                                     \
      [[buffer(23), function_constant(use_turboquant)]],                       \
      const constant int &v_block_stride                                       \
      [[buffer(24), function_constant(use_turboquant)]],                       \
      const constant int &v_head_stride                                        \
      [[buffer(25), function_constant(use_turboquant)]],                       \
      const device half *key_zero_cache                                        \
      [[buffer(26), function_constant(use_turboquant)]],                       \
      const device float *v_centroids                                          \
      [[buffer(27), function_constant(use_turboquant)]],                       \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_paged_attention_v2_reduce_inner(                           \
    type, head_size, num_threads, num_simd_lanes, partition_size)              \
  template [[host_name("paged_attention_v2_reduce_" #type "_hs" #head_size     \
                       "_nt" #num_threads "_nsl" #num_simd_lanes               \
                       "_ps" #partition_size)]] [[kernel]] void                \
  paged_attention_v2_reduce<type, head_size, num_threads, num_simd_lanes,      \
                            partition_size>(                                   \
      device type * out [[buffer(0)]],                                         \
      const device float *exp_sums [[buffer(1)]],                              \
      const device float *max_logits [[buffer(2)]],                            \
      const device type *tmp_out [[buffer(3)]],                                \
      device uint32_t *context_lens [[buffer(4)]],                             \
      const constant int &max_num_partitions [[buffer(5)]],                    \
      const device float *sinks [[buffer(6), function_constant(use_sinks)]],   \
      device const int32_t *cu_seqlens_q [[buffer(7)]],                        \
      const constant int &num_seqs [[buffer(8)]],                              \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint3 threads_per_threadgroup [[threads_per_threadgroup]],               \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_paged_attention_heads(                                     \
    type, k_cache_type, v_cache_type, block_size, num_threads,                 \
    num_simd_lanes, partition_size)                                            \
  instantiate_paged_attention_inner(type, k_cache_type, v_cache_type, 64,      \
                                    block_size, num_threads, num_simd_lanes,   \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, k_cache_type, v_cache_type, 80,      \
                                    block_size, num_threads, num_simd_lanes,   \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, k_cache_type, v_cache_type, 96,      \
                                    block_size, num_threads, num_simd_lanes,   \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, k_cache_type, v_cache_type, 112,     \
                                    block_size, num_threads, num_simd_lanes,   \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, k_cache_type, v_cache_type, 128,     \
                                    block_size, num_threads, num_simd_lanes,   \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, k_cache_type, v_cache_type, 192,     \
                                    block_size, num_threads, num_simd_lanes,   \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, k_cache_type, v_cache_type, 256,     \
                                    block_size, num_threads, num_simd_lanes,   \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, k_cache_type, v_cache_type, 512,     \
                                    block_size, num_threads, num_simd_lanes,   \
                                    partition_size);

#define instantiate_paged_attention_v2_reduce_heads(                           \
    type, num_threads, num_simd_lanes, partition_size)                         \
  instantiate_paged_attention_v2_reduce_inner(type, 64, num_threads,           \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 80, num_threads,           \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 96, num_threads,           \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 112, num_threads,          \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 128, num_threads,          \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 192, num_threads,          \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 256, num_threads,          \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 512, num_threads,          \
                                              num_simd_lanes, partition_size);

#define instantiate_paged_attention_block_size(type, k_cache_type,             \
                                               v_cache_type, num_threads,      \
                                               num_simd_lanes, partition_size) \
  instantiate_paged_attention_heads(type, k_cache_type, v_cache_type, 8,       \
                                    num_threads, num_simd_lanes,               \
                                    partition_size);                           \
  instantiate_paged_attention_heads(type, k_cache_type, v_cache_type, 16,      \
                                    num_threads, num_simd_lanes,               \
                                    partition_size);                           \
  instantiate_paged_attention_heads(type, k_cache_type, v_cache_type, 32,      \
                                    num_threads, num_simd_lanes,               \
                                    partition_size);

// TODO: tune num_threads = 256
// NOTE: partition_size = 0
#define instantiate_paged_attention_v1(type, k_cache_type, v_cache_type,       \
                                       num_simd_lanes)                         \
  instantiate_paged_attention_block_size(type, k_cache_type, v_cache_type,     \
                                         256, num_simd_lanes, 0);

// TODO: tune num_threads = 256
// NOTE: partition_size = VLLM_METAL_PARTITION_SIZE
#define instantiate_paged_attention_v2(type, k_cache_type, v_cache_type,       \
                                       num_simd_lanes)                         \
  instantiate_paged_attention_block_size(type, k_cache_type, v_cache_type,     \
                                         256, num_simd_lanes,                  \
                                         VLLM_METAL_PARTITION_SIZE);

// TODO: tune num_threads = 256
// NOTE: partition_size = VLLM_METAL_PARTITION_SIZE
#define instantiate_paged_attention_v2_reduce(type, num_simd_lanes)            \
  instantiate_paged_attention_v2_reduce_heads(                                  \
      type, 256, num_simd_lanes, VLLM_METAL_PARTITION_SIZE);

// Non-TQ: same K/V cache type (kernel name: paged_attention_<T>_cache_<CT>_<CT>_hs...)
instantiate_paged_attention_v1(float, float, float, 32);
instantiate_paged_attention_v1(bfloat16_t, bfloat16_t, bfloat16_t, 32);
instantiate_paged_attention_v1(half, half, half, 32);

// FP8 non-TQ (uchar K + uchar V)
instantiate_paged_attention_v1(float, uchar, uchar, 32);
instantiate_paged_attention_v1(bfloat16_t, uchar, uchar, 32);
instantiate_paged_attention_v1(half, uchar, uchar, 32);

instantiate_paged_attention_v2_reduce(float, 32);
instantiate_paged_attention_v2_reduce(bfloat16_t, 32);
instantiate_paged_attention_v2_reduce(half, 32);

instantiate_paged_attention_v2(float, float, float, 32);
instantiate_paged_attention_v2(bfloat16_t, bfloat16_t, bfloat16_t, 32);
instantiate_paged_attention_v2(half, half, half, 32);

// FP8 non-TQ (uchar K + uchar V) — also used for TQ uint8 K (differentiated by use_turboquant FC)
instantiate_paged_attention_v2(float, uchar, uchar, 32);
instantiate_paged_attention_v2(bfloat16_t, uchar, uchar, 32);
instantiate_paged_attention_v2(half, uchar, uchar, 32);

// TurboQuant: int8 K (char) + 3-bit V (uchar) — both non-partitioned (ps0) and partitioned
instantiate_paged_attention_v1(float, char, uchar, 32);
instantiate_paged_attention_v1(bfloat16_t, char, uchar, 32);
instantiate_paged_attention_v1(half, char, uchar, 32);

instantiate_paged_attention_v2(float, char, uchar, 32);
instantiate_paged_attention_v2(bfloat16_t, char, uchar, 32);
instantiate_paged_attention_v2(half, char, uchar, 32);
