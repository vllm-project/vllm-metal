// SPDX-License-Identifier: Apache-2.0
//
// Paged Multi-head Latent Attention (MLA) kernel.
//
// Tracks RFC https://github.com/vllm-project/vllm-metal/issues/360.
//
// paged_mla_attention — fused score+softmax+V over a single threadgroup
// per (head_group, q_token). The `use_partitioning` function constant and
// PARTITION_SIZE template parameter are scaffolded for a future 2-pass
// mode (each partition writes scratch max/lse/output, a separate reduce
// kernel merges across partitions); only the single-pass mode
// (use_partitioning=false, PARTITION_SIZE=0) is wired up today.
//
// Decode kernel parallelism scheme mirrors MLX's sdpa_vector
// (mlx/backend/metal/kernels/sdpa_vector.h, ml-explore/mlx@v0.31.2):
//   - Threadgroup: BN simdgroups × BD lanes = NUM_THREADS threads.
//   - simd_lid (lane) holds a contiguous dim-slice of Q (KV_LORA_RANK / BD
//     elements) and the V output accumulator (same slice).
//   - simd_gid (simdgroup) strides over KV tokens — each simdgroup processes
//     one token at a time, all BD lanes cooperate on the dot product via
//     simd_sum. BN simdgroups process BN tokens in parallel per "wave".
//   - Online softmax state lives in registers per simdgroup; cross-simdgroup
//     merge at the end uses a transpose-via-shmem trick + simd_sum.
//
// Cross-head amortization (HEADS_PER_TG > 1):
//   One threadgroup processes HEADS_PER_TG query heads sharing the same
//   latent KV. K is loaded once per simdgroup wave and reused for G dot
//   products. Per-thread state grows G× (Q_local, V_local hold G heads
//   worth of dim slice), so NUM_THREADS is reduced in lockstep so the
//   register budget per TG stays roughly constant. Total launches drop
//   from B×H to B×ceil(H/G), so KV bandwidth is amortized G×.
//
// This is structurally different from kernels_v2/pagedattention.metal which
// puts each lane on a single token's full dot product. For MLA the dim
// (KV_LORA_RANK=512) is too large for that to be efficient — distributing
// across lanes is the only way to match MLX SDPA's per-token throughput.

#include "utils.metal"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

// ========================================== Function constants

constant bool mla_use_partitioning [[function_constant(10)]];

// ========================================== Main kernel
//
// Buffer layout:
//
//   0: exp_sums    [num_seqs, num_heads, max_num_partitions]  fp32
//                  (only when mla_use_partitioning)
//   1: max_logits  [num_seqs, num_heads, max_num_partitions]  fp32
//                  (only when mla_use_partitioning)
//   2: out / tmp_out
//        non-partitioned: [total_q_tokens, num_heads, KV_LORA_RANK] T
//        partitioned    : [num_seqs, num_heads, max_num_partitions, KV_LORA_RANK] T
//   3: q_nope       [total_q_tokens, num_heads, KV_LORA_RANK] T  (post-embed_q)
//   4: q_pe         [total_q_tokens, num_heads, QK_ROPE_HEAD_DIM] T  (post-RoPE)
//   5: latent_cache [num_blocks, BLOCK_SIZE, KV_LORA_RANK + QK_ROPE_HEAD_DIM] T
//   6: block_tables [num_seqs, max_num_blocks_per_seq] uint32
//   7: context_lens [num_seqs] uint32
//   8: cu_seqlens_q [num_seqs + 1] int32  (P1 trivial: [0,1,...,num_seqs])
//   9: num_seqs (constant int)
//  10: max_num_blocks_per_seq (constant int)
//  11: scale (constant float)
//
// Grid:
//   non-partitioned: (num_heads / HEADS_PER_TG, total_q_tokens, 1)
//   partitioned    : (num_heads / HEADS_PER_TG, total_q_tokens,
//                     max_num_partitions)
//
// Per-thread register state (BD=NUM_SIMD_LANES=32 typical):
//   q_nope_local : HEADS_PER_TG * KV_LORA_RANK / BD       fp32
//   q_pe_local   : HEADS_PER_TG * QK_ROPE_HEAD_DIM / BD   fp32
//   v_local      : HEADS_PER_TG * KV_LORA_RANK / BD       fp32   (V acc)
// Threadgroup memory:
//   max_scores[HEADS_PER_TG * BN]      fp32
//   sum_exp_scores[HEADS_PER_TG * BN]  fp32
//   outputs[BD * BD]                   fp32  (transpose buffer for cross-
//                                             simdgroup O reduce, reused
//                                             across heads. Sized for the
//                                             write index `lane*BD + sg`,
//                                             which reaches (BD-1)*BD +
//                                             (BN-1) for any BN ≤ BD.)

template <typename T, int KV_LORA_RANK, int QK_ROPE_HEAD_DIM, int BLOCK_SIZE,
          int HEADS_PER_TG, int NUM_THREADS, int NUM_SIMD_LANES,
          int PARTITION_SIZE = 0>
[[kernel, max_total_threads_per_threadgroup(NUM_THREADS)]] void paged_mla_attention(
    device float *exp_sums
    [[buffer(0), function_constant(mla_use_partitioning)]],
    device float *max_logits
    [[buffer(1), function_constant(mla_use_partitioning)]],
    device T *out [[buffer(2)]],
    device const T *q_nope [[buffer(3)]],
    device const T *q_pe [[buffer(4)]],
    device const T *latent_cache [[buffer(5)]],
    device const uint32_t *block_tables [[buffer(6)]],
    device const uint32_t *context_lens [[buffer(7)]],
    device const int32_t *cu_seqlens_q [[buffer(8)]],
    const constant int &num_seqs [[buffer(9)]],
    const constant int &max_num_blocks_per_seq [[buffer(10)]],
    const constant float &scale [[buffer(11)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)cu_seqlens_q;
  (void)num_seqs;

  constexpr int LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
  constexpr int BD = NUM_SIMD_LANES;          // lanes per simdgroup
  constexpr int BN = NUM_THREADS / BD;        // simdgroups per threadgroup
  constexpr int QK_PER_THREAD = KV_LORA_RANK / BD;
  constexpr int QPE_PER_THREAD = QK_ROPE_HEAD_DIM / BD;
  constexpr int V_PER_THREAD = KV_LORA_RANK / BD;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  static_assert(KV_LORA_RANK % BD == 0,
                "KV_LORA_RANK must be divisible by NUM_SIMD_LANES");
  static_assert(QK_ROPE_HEAD_DIM % BD == 0,
                "QK_ROPE_HEAD_DIM must be divisible by NUM_SIMD_LANES");
  static_assert(BN <= BD,
                "Cross-simdgroup merge needs BN<=BD (lane indexes the BN axis)");
  static_assert(!USE_PARTITIONING || PARTITION_SIZE % BLOCK_SIZE == 0,
                "PARTITION_SIZE must be divisible by BLOCK_SIZE");

  const int q_token_idx = threadgroup_position_in_grid.y;
  const int head_group_idx = threadgroup_position_in_grid.x;
  const int num_head_groups = threadgroups_per_grid.x;
  const int num_heads = num_head_groups * HEADS_PER_TG;
  const int head_idx_base = head_group_idx * HEADS_PER_TG;
  const int partition_idx = threadgroup_position_in_grid.z;
  const int max_num_partitions = threadgroups_per_grid.z;
  const int lane = (int)simd_lid;
  const int sg = (int)simd_gid;

  // P1 decode: q_token_idx == seq_idx (wrapper-enforced).
  const int seq_idx = q_token_idx;
  const int ctx_len = (int)context_lens[seq_idx];

  const int token_start =
      USE_PARTITIONING ? partition_idx * PARTITION_SIZE : 0;
  const int token_end =
      USE_PARTITIONING ? min(token_start + PARTITION_SIZE, ctx_len) : ctx_len;

  // Partition has no work — early-out (caller pre-zeros partial buffers).
  if (USE_PARTITIONING && token_start >= ctx_len) {
    return;
  }

  // ---- Phase A: load Q for HEADS_PER_TG heads into per-lane registers ----
  // Layout: q_nope_local[h * QK_PER_THREAD + j] holds head h's dim slice
  // [lane * QK_PER_THREAD + j]. With G=1 this is identical to the previous
  // single-head kernel.
  float q_nope_local[HEADS_PER_TG * QK_PER_THREAD];
  float q_pe_local[HEADS_PER_TG * QPE_PER_THREAD];
  float v_local[HEADS_PER_TG * V_PER_THREAD];
  float max_score[HEADS_PER_TG];
  float sum_exp_score[HEADS_PER_TG];

#pragma unroll
  for (int h = 0; h < HEADS_PER_TG; h++) {
    const int head_idx = head_idx_base + h;
    const device T *q_nope_ptr =
        q_nope + (q_token_idx * num_heads + head_idx) * KV_LORA_RANK +
        lane * QK_PER_THREAD;
    const device T *q_pe_ptr =
        q_pe + (q_token_idx * num_heads + head_idx) * QK_ROPE_HEAD_DIM +
        lane * QPE_PER_THREAD;
#pragma unroll
    for (int j = 0; j < QK_PER_THREAD; j++) {
      q_nope_local[h * QK_PER_THREAD + j] = float(q_nope_ptr[j]);
    }
#pragma unroll
    for (int j = 0; j < QPE_PER_THREAD; j++) {
      q_pe_local[h * QPE_PER_THREAD + j] = float(q_pe_ptr[j]);
    }
#pragma unroll
    for (int j = 0; j < V_PER_THREAD; j++) {
      v_local[h * V_PER_THREAD + j] = 0.0f;
    }
    max_score[h] = -INFINITY;
    sum_exp_score[h] = 0.0f;
  }

  // ---- Phase B: simdgroup-strided token loop ----
  // Each simdgroup processes one token at a time. K is loaded ONCE per
  // simdgroup wave and reused for HEADS_PER_TG dot products — this is the
  // KV-bandwidth amortization.
  const device uint32_t *block_table_row =
      block_tables + (uint64_t)seq_idx * max_num_blocks_per_seq;

  for (int t = token_start + sg; t < token_end; t += BN) {
    const int block_idx = t / BLOCK_SIZE;
    const int block_offset = t % BLOCK_SIZE;
    const uint32_t physical_block = block_table_row[block_idx];
    const device T *token_ptr =
        latent_cache + (uint64_t)physical_block * BLOCK_SIZE * LATENT_DIM +
        block_offset * LATENT_DIM;

    // Load this lane's slice of K_norm and K_pe from cache (shared across
    // all HEADS_PER_TG heads).
    float k_norm_local[QK_PER_THREAD];
#pragma unroll
    for (int j = 0; j < QK_PER_THREAD; j++) {
      k_norm_local[j] = float(token_ptr[lane * QK_PER_THREAD + j]);
    }
    float k_pe_local[QPE_PER_THREAD];
#pragma unroll
    for (int j = 0; j < QPE_PER_THREAD; j++) {
      k_pe_local[j] =
          float(token_ptr[KV_LORA_RANK + lane * QPE_PER_THREAD + j]);
    }

    // Per-head: partial dot product → simd_sum → score → online softmax →
    // V accumulation. The simd_sum is once per head; can't fold across
    // heads since each head has its own score.
#pragma unroll
    for (int h = 0; h < HEADS_PER_TG; h++) {
      float partial = 0.0f;
#pragma unroll
      for (int j = 0; j < QK_PER_THREAD; j++) {
        partial += q_nope_local[h * QK_PER_THREAD + j] * k_norm_local[j];
      }
#pragma unroll
      for (int j = 0; j < QPE_PER_THREAD; j++) {
        partial += q_pe_local[h * QPE_PER_THREAD + j] * k_pe_local[j];
      }
      // Pre-multiply by log2(e) so the softmax runs in log2 space and the
      // subsequent fast::exp2 calls are 1-instruction Metal intrinsics
      // (identity: exp(x - y) = exp2((x - y) * log2(e))). Matches the
      // pagedattention.metal pattern.
      const float score = simd_sum(partial) * scale * M_LOG2E_F;

      const float new_max = max(max_score[h], score);
      const float factor = (max_score[h] == -INFINITY) ? 0.0f
                                                       : fast::exp2(max_score[h] - new_max);
      const float exp_score = fast::exp2(score - new_max);
      max_score[h] = new_max;
      sum_exp_score[h] = sum_exp_score[h] * factor + exp_score;

      // V == kv_norm in absorbed MLA, so reuse k_norm_local — saves a load.
#pragma unroll
      for (int j = 0; j < V_PER_THREAD; j++) {
        v_local[h * V_PER_THREAD + j] =
            v_local[h * V_PER_THREAD + j] * factor + exp_score * k_norm_local[j];
      }
    }
  }

  // ---- Phase C: cross-simdgroup merge ----
  // Per head: each simdgroup writes its (max, sum_exp) to shmem, lane `l`
  // reads simdgroup `l`'s value, computes the global rescale factor, and
  // uses a transpose-via-shmem trick to reduce each dim slice across
  // simdgroups via a single simd_sum.

  threadgroup float *max_scores = (threadgroup float *)shared_mem;
  threadgroup float *sum_exp_scores = max_scores + HEADS_PER_TG * BN;
  threadgroup float *outputs = sum_exp_scores + HEADS_PER_TG * BN;

  if (lane == 0) {
#pragma unroll
    for (int h = 0; h < HEADS_PER_TG; h++) {
      max_scores[h * BN + sg] = max_score[h];
      sum_exp_scores[h * BN + sg] = sum_exp_score[h];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Per-head merge. The transpose-via-shmem buffer (`outputs`) is reused
  // across heads; the trailing barrier in the inner loop synchronizes the
  // next head's first write.
  //
  // Coverage requires BN simdgroups to between them produce all KV_LORA_RANK
  // output dims. The "natural" trick (each simdgroup owns one V_PER_THREAD
  // = KV_LORA_RANK/BD slice) only covers BN*V_PER_THREAD = (BN/BD)*KV_LORA_RANK
  // dims. When BN < BD (e.g. NUM_THREADS=256 → BN=8, BD=32), each simdgroup
  // must cover NUM_PASSES = BD/BN slices to reach full coverage. NUM_PASSES=1
  // for the BN=BD=32 default.
  constexpr int NUM_PASSES = BD / BN;
  static_assert(BD % BN == 0, "BD must be divisible by BN");
#pragma unroll
  for (int h = 0; h < HEADS_PER_TG; h++) {
    const float my_simdgroup_max =
        (lane < BN) ? max_scores[h * BN + lane] : -INFINITY;
    const float global_max = simd_max(my_simdgroup_max);
    const float rescale =
        (my_simdgroup_max == -INFINITY) ? 0.0f
                                        : fast::exp2(my_simdgroup_max - global_max);
    const float global_sum =
        simd_sum((lane < BN) ? sum_exp_scores[h * BN + lane] * rescale : 0.0f);
    const float inv_global = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // Transpose-via-shmem merge across simdgroups.
    //   write: outputs[lane * BD + sg] = v_local[i]
    //          → entry at offset (lane * BD + sg) holds simdgroup-sg lane-lane's
    //            partial for dim (lane * V_PER_THREAD + i). Only sg in [0, BN)
    //            is written, so the BN-wide dim of `outputs` is sparse when
    //            BN<BD.
    //   read:  outputs[(p * BN + sg) * BD + lane]
    //          → simdgroup sg's lane `l` picks up source-lane (p*BN+sg)
    //            simdgroup-l's partial; simd_sum across lanes-of-reader sums
    //            across source-simdgroups for that fixed source-lane. Only
    //            lanes l < BN have valid data in shmem; lanes l >= BN must
    //            contribute 0 to the simd_sum.
    // After simd_sum, simdgroup `sg` holds final values for dim slice
    // [(p * BN + sg) * V_PER_THREAD, (p * BN + sg + 1) * V_PER_THREAD).
    // We write to device memory immediately rather than buffering
    // NUM_PASSES * V_PER_THREAD floats per thread — the v_final[] register
    // array would push larger HEADS_PER_TG variants over the per-thread
    // register budget.
    const int head_idx = head_idx_base + h;
    device T *out_base = USE_PARTITIONING
        ? (out +
           ((q_token_idx * num_heads + head_idx) * max_num_partitions +
            partition_idx) * KV_LORA_RANK)
        : (out + (q_token_idx * num_heads + head_idx) * KV_LORA_RANK);
#pragma unroll
    for (int p = 0; p < NUM_PASSES; p++) {
#pragma unroll
      for (int i = 0; i < V_PER_THREAD; i++) {
        outputs[lane * BD + sg] = v_local[h * V_PER_THREAD + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float other =
            (lane < BN) ? outputs[(p * BN + sg) * BD + lane] : 0.0f;
        const float final_val = simd_sum(other * rescale) * inv_global;
        if (lane == 0) {
          out_base[(p * BN + sg) * V_PER_THREAD + i] = T(final_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    // ---- Phase D: per-head LSE / max metadata for the partitioned reduce ----
    if (USE_PARTITIONING) {
      if (lane == 0 && sg == 0) {
        max_logits[(q_token_idx * num_heads + head_idx) * max_num_partitions +
                   partition_idx] = global_max;
        // exp_sums stores l (sum-of-exps at global_max) so reduce kernel can
        // weight partitions correctly.
        exp_sums[(q_token_idx * num_heads + head_idx) * max_num_partitions +
                 partition_idx] = global_sum;
      }
    }
  }
}


// ========================================== Instantiations
//
// Single-pass main kernel: dtype × block_size × heads_per_tg.
//   G=1 → NUM_THREADS=1024 (32 simdgroups, current sdpa_vector layout).
//   G=2 → NUM_THREADS=512  (16 simdgroups, 2× KV-bandwidth amortization).

#define instantiate_mla(type, kv_lora_rank, qk_rope_head_dim, block_size,      \
                        heads_per_tg, num_threads, partition_size)             \
  template [[host_name("paged_mla_attention_" #type "_kvr" #kv_lora_rank       \
                       "_pe" #qk_rope_head_dim "_bs" #block_size "_g"          \
                       #heads_per_tg "_nt" #num_threads "_nsl32_ps"            \
                       #partition_size)]] [[kernel]] void                      \
  paged_mla_attention<type, kv_lora_rank, qk_rope_head_dim, block_size,        \
                      heads_per_tg, num_threads, 32, partition_size>(          \
      device float * exp_sums                                                  \
      [[buffer(0), function_constant(mla_use_partitioning)]],                  \
      device float *max_logits                                                 \
      [[buffer(1), function_constant(mla_use_partitioning)]],                  \
      device type *out [[buffer(2)]],                                          \
      device const type *q_nope [[buffer(3)]],                                 \
      device const type *q_pe [[buffer(4)]],                                   \
      device const type *latent_cache [[buffer(5)]],                           \
      device const uint32_t *block_tables [[buffer(6)]],                       \
      device const uint32_t *context_lens [[buffer(7)]],                       \
      device const int32_t *cu_seqlens_q [[buffer(8)]],                        \
      const constant int &num_seqs [[buffer(9)]],                              \
      const constant int &max_num_blocks_per_seq [[buffer(10)]],               \
      const constant float &scale [[buffer(11)]],                              \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

// G=1 (single-head per TG, NUM_THREADS=1024).
instantiate_mla(half, 512, 64, 16, 1, 1024, 0);
instantiate_mla(half, 512, 64, 32, 1, 1024, 0);
instantiate_mla(bfloat16_t, 512, 64, 16, 1, 1024, 0);
instantiate_mla(bfloat16_t, 512, 64, 32, 1, 1024, 0);

// G=2 (2 heads per TG, NUM_THREADS=512). 2× KV-bandwidth amortization.
instantiate_mla(half, 512, 64, 16, 2, 512, 0);
instantiate_mla(half, 512, 64, 32, 2, 512, 0);
instantiate_mla(bfloat16_t, 512, 64, 16, 2, 512, 0);
instantiate_mla(bfloat16_t, 512, 64, 32, 2, 512, 0);
