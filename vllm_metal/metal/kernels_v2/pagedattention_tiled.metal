// SPDX-License-Identifier: Apache-2.0
// Tiled paged attention — Flash-Attention-style kernel using simdgroup 8×8 MMA.
//
// Separate from pagedattention.metal (the per-token kernel) to keep the two
// implementations independent.  Both are compiled into the same Metal library
// via the build system's source concatenation.
//
// Requires utils.metal (provides DIVIDE_ROUND_UP, MIN, MAX) and
// <metal_stdlib> / using namespace metal (from earlier in the concat).

// ========================================== Tiled Paged Attention
// Each threadgroup processes BQ query tokens against TILE_KV KV tokens
// per tile using simdgroup 8×8 matrix multiply-accumulate.
//
// Grid: (num_query_heads, total_q_blocks, 1)
// Threadgroup: (NUM_THREADS, 1, 1)
//
// Q-block indexing follows the same over-allocation scheme as vLLM's
// Triton unified attention: total_q_blocks = total_q_tokens / BQ + num_seqs.

template <typename T, int HEAD_SIZE, int BLOCK_SIZE,
          int BQ = 8, int TILE_KV = 32, int NUM_THREADS = 128>
[[kernel]] void paged_attention_tiled(
    device T *out [[buffer(2)]],
    device const T *q [[buffer(3)]],
    device const T *k_cache [[buffer(4)]],
    device const T *v_cache [[buffer(5)]],
    const constant int &num_kv_heads [[buffer(8)]],
    const constant float &scale [[buffer(9)]],
    const constant float &softcapping [[buffer(10)]],
    device const uint32_t *block_tables [[buffer(11)]],
    device const uint32_t *context_lens [[buffer(12)]],
    const constant int &max_num_blocks_per_seq [[buffer(13)]],
    const constant int &q_stride [[buffer(15)]],
    const constant int &kv_block_stride [[buffer(16)]],
    const constant int &kv_head_stride [[buffer(17)]],
    device const int32_t *cu_seqlens_q [[buffer(19)]],
    const constant int &num_seqs [[buffer(20)]],
    const constant int &sliding_window [[buffer(21)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tgpg [[threadgroups_per_grid]],
    uint3 tpt [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  constexpr int NUM_SIMD_LANES = 32;
  constexpr int NUM_SG = NUM_THREADS / NUM_SIMD_LANES;
  constexpr int HD_TILES = HEAD_SIZE / 8;
  constexpr int PV_TILES_PER_SG = HD_TILES / NUM_SG;

  static_assert(HEAD_SIZE % 8 == 0, "HEAD_SIZE must be a multiple of 8");
  static_assert(TILE_KV % 8 == 0, "TILE_KV must be a multiple of 8");
  static_assert(TILE_KV <= NUM_SIMD_LANES, "TILE_KV must be <= NUM_SIMD_LANES for softmax");
  static_assert(HD_TILES % NUM_SG == 0, "HD_TILES must be divisible by NUM_SG");

  const int thread_idx = tpt.x;
  const int head_idx = tgp.x;
  const int q_block_global_idx = tgp.y;
  const int num_heads = tgpg.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const int kv_token_stride = num_kv_heads * kv_head_stride;

  // === Varlen: resolve sequence and Q-block position ===
  // Uses block-space binary search (not token-space like find_seq_idx):
  // the predicate cu_seqlens_q[mid]/BQ + mid accounts for the per-sequence
  // over-allocation sentinel in the Q-block grid.
  int seq_idx;
  {
    int lo = 0, hi = num_seqs;
    while (lo < hi) {
      int mid = (lo + hi + 1) / 2;
      if (cu_seqlens_q[mid] / BQ + mid <= q_block_global_idx) lo = mid;
      else hi = mid - 1;
    }
    seq_idx = lo;
  }

  const int q_seq_start = cu_seqlens_q[seq_idx];
  const int cur_batch_query_len = cu_seqlens_q[seq_idx + 1] - q_seq_start;
  const int q_block_start = q_seq_start / BQ + seq_idx;
  const int q_block_local = q_block_global_idx - q_block_start;
  const int q_pos_start = q_block_local * BQ;

  if (q_pos_start >= cur_batch_query_len) return;

  const int seq_len = int(context_lens[seq_idx]);
  const int context_len = seq_len - cur_batch_query_len;
  const int valid_q = min(BQ, cur_batch_query_len - q_pos_start);

  constexpr int Q_ELEMS = BQ * HEAD_SIZE;
  constexpr int KV_ELEMS = TILE_KV * HEAD_SIZE;
  constexpr int S_ELEMS = BQ * TILE_KV;
  constexpr int O_ELEMS = BQ * HEAD_SIZE;

  threadgroup T *Q_smem = reinterpret_cast<threadgroup T *>(shared_mem);
  threadgroup T *KV_smem = Q_smem + Q_ELEMS;
  threadgroup float *S_smem = reinterpret_cast<threadgroup float *>(KV_smem + KV_ELEMS);
  threadgroup float *O_smem = S_smem + S_ELEMS;
  threadgroup float *M_smem = O_smem + O_ELEMS;
  threadgroup float *L_smem = M_smem + BQ;

  const float scale_log2 = scale * M_LOG2E_F;

  // === Load Q into Q_smem [BQ, HEAD_SIZE] ===
  const device T *q_base = q + (q_seq_start + q_pos_start) * q_stride
                             + head_idx * HEAD_SIZE;
  for (int i = thread_idx; i < Q_ELEMS; i += NUM_THREADS) {
    int r = i / HEAD_SIZE;
    int d = i % HEAD_SIZE;
    Q_smem[i] = (r < valid_q) ? q_base[r * q_stride + d] : T(0);
  }

  // === Initialize O, M, L ===
  for (int i = thread_idx; i < O_ELEMS; i += NUM_THREADS) {
    O_smem[i] = 0.f;
  }
  if (thread_idx < BQ) {
    M_smem[thread_idx] = -FLT_MAX;
    L_smem[thread_idx] = 0.f;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device uint32_t *block_table =
      block_tables + seq_idx * max_num_blocks_per_seq;
  const int num_kv_tiles = DIVIDE_ROUND_UP(seq_len, TILE_KV);

  // === Main KV tile loop (4 barriers per iteration) ===
  for (int tile_idx = 0; tile_idx < num_kv_tiles; tile_idx++) {
    const int tile_start = tile_idx * TILE_KV;

    // Causal skip
    if (tile_start > context_len + q_pos_start + valid_q - 1) break;

    // --- Load K: per-token block_table lookup, coalesced head reads ---
    for (int t = sg_idx; t < TILE_KV; t += NUM_SG) {
      int kv_pos = tile_start + t;
      if (kv_pos < seq_len) {
        int64_t pb = int64_t(block_table[kv_pos / BLOCK_SIZE]);
        const device T *k_ptr = k_cache + pb * kv_block_stride
            + (kv_pos % BLOCK_SIZE) * kv_token_stride
            + kv_head_idx * kv_head_stride;
        for (int d = lane; d < HEAD_SIZE; d += NUM_SIMD_LANES)
          KV_smem[t * HEAD_SIZE + d] = k_ptr[d];
      } else {
        for (int d = lane; d < HEAD_SIZE; d += NUM_SIMD_LANES)
          KV_smem[t * HEAD_SIZE + d] = T(0);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);  // B1: K visible

    // --- QK matmul: S[BQ, TILE_KV] = Q × K^T via 8×8 MMA ---
    {
      simdgroup_matrix<float, 8, 8> S_acc(0);

#pragma unroll
      for (int d = 0; d < HEAD_SIZE; d += 8) {
        simdgroup_matrix<T, 8, 8> q_frag;
        simdgroup_load(q_frag, Q_smem + d, HEAD_SIZE);

        simdgroup_matrix<T, 8, 8> k_frag;
        simdgroup_load(k_frag, KV_smem + sg_idx * 8 * HEAD_SIZE + d,
                       HEAD_SIZE, ulong2(0), true);

        simdgroup_multiply_accumulate(S_acc, q_frag, k_frag, S_acc);
      }

      simdgroup_store(S_acc, S_smem + sg_idx * 8, TILE_KV);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);  // B2: S visible

    // --- Softmax (inline scale+mask) + O rescale + V load ---
    {
      constexpr int ROWS_PER_SG = BQ / NUM_SG;
#pragma unroll
      for (int ri = 0; ri < ROWS_PER_SG; ri++) {
        const int row = sg_idx * ROWS_PER_SG + ri;
        if (row >= BQ) continue;

        const int q_abs_pos = context_len + q_pos_start + row;

        float s = S_smem[row * TILE_KV + lane] * scale_log2;
        if (softcapping > 0.0f) {
          float s_orig = s / M_LOG2E_F;
          s = softcapping * precise::tanh(s_orig / softcapping) * M_LOG2E_F;
        }

        int kv_pos = tile_start + int(lane);
        bool masked = (kv_pos > q_abs_pos) || (kv_pos >= seq_len)
                      || (row >= valid_q);
        if (sliding_window >= 0)
          masked = masked || (kv_pos < q_abs_pos + 1 - sliding_window);
        if (masked) s = -FLT_MAX;

        float block_max = simd_max(s);
        float old_m = M_smem[row];
        float new_m = max(old_m, block_max);
        if (new_m == -FLT_MAX) new_m = 0.f;

        float old_corr = (old_m == -FLT_MAX) ? 0.f : exp2(old_m - new_m);
        float e = (s == -FLT_MAX) ? 0.f : exp2(s - new_m);
        float row_sum = simd_sum(e);

        L_smem[row] = L_smem[row] * old_corr + row_sum;
        M_smem[row] = new_m;

        S_smem[row * TILE_KV + lane] = e;

        for (int d = lane; d < HEAD_SIZE; d += NUM_SIMD_LANES)
          O_smem[row * HEAD_SIZE + d] *= old_corr;
      }
    }

    // V load: writes KV_smem (separate from S_smem/O_smem, no conflict)
    for (int t = sg_idx; t < TILE_KV; t += NUM_SG) {
      int kv_pos = tile_start + t;
      if (kv_pos < seq_len) {
        int64_t pb = int64_t(block_table[kv_pos / BLOCK_SIZE]);
        const device T *v_ptr = v_cache + pb * kv_block_stride
            + (kv_pos % BLOCK_SIZE) * kv_token_stride
            + kv_head_idx * kv_head_stride;
        for (int d = lane; d < HEAD_SIZE; d += NUM_SIMD_LANES)
          KV_smem[t * HEAD_SIZE + d] = v_ptr[d];
      } else {
        for (int d = lane; d < HEAD_SIZE; d += NUM_SIMD_LANES)
          KV_smem[t * HEAD_SIZE + d] = T(0);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);  // B3: P, O rescaled, V visible

    // --- PV matmul: O += P × V via 8×8 MMA ---
    // P kept in float (S_smem); uses float×half→float mixed-precision MMA.
#pragma unroll
    for (int c = 0; c < PV_TILES_PER_SG; c++) {
      int hd_start = (sg_idx * PV_TILES_PER_SG + c) * 8;

      simdgroup_matrix<float, 8, 8> pv_acc;
      simdgroup_load(pv_acc, O_smem + hd_start, HEAD_SIZE);

#pragma unroll
      for (int t = 0; t < TILE_KV; t += 8) {
        simdgroup_matrix<float, 8, 8> p_frag;
        simdgroup_load(p_frag, S_smem + t, TILE_KV);

        simdgroup_matrix<T, 8, 8> v_frag;
        simdgroup_load(v_frag, KV_smem + t * HEAD_SIZE + hd_start, HEAD_SIZE);

        simdgroup_multiply_accumulate(pv_acc, p_frag, v_frag, pv_acc);
      }

      simdgroup_store(pv_acc, O_smem + hd_start, HEAD_SIZE);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);  // B4: O written
  } // end KV tile loop

  // === Final normalize + write output ===
  device T *out_base = out + (q_seq_start + q_pos_start) * q_stride
                           + head_idx * HEAD_SIZE;
  for (int i = thread_idx; i < valid_q * HEAD_SIZE; i += NUM_THREADS) {
    int r = i / HEAD_SIZE;
    int d = i % HEAD_SIZE;
    float val = O_smem[i] / (L_smem[r] + 1e-6f);
    out_base[r * q_stride + d] = T(val);
  }
}

// ---- Template instantiation ----

#define instantiate_paged_attention_tiled_inner(type, head_size, block_size)    \
  template [[host_name("paged_attention_tiled_" #type                          \
                       "_hs" #head_size "_bs" #block_size                      \
                       "_bq8_tk32_nt128")]]                                    \
  [[kernel]] void paged_attention_tiled<type, head_size, block_size,           \
                                        8, 32, 128>(                           \
      device type *out [[buffer(2)]],                                          \
      device const type *q [[buffer(3)]],                                      \
      device const type *k_cache [[buffer(4)]],                                \
      device const type *v_cache [[buffer(5)]],                                \
      const constant int &num_kv_heads [[buffer(8)]],                          \
      const constant float &scale [[buffer(9)]],                               \
      const constant float &softcapping [[buffer(10)]],                        \
      device const uint32_t *block_tables [[buffer(11)]],                      \
      device const uint32_t *context_lens [[buffer(12)]],                      \
      const constant int &max_num_blocks_per_seq [[buffer(13)]],               \
      const constant int &q_stride [[buffer(15)]],                             \
      const constant int &kv_block_stride [[buffer(16)]],                      \
      const constant int &kv_head_stride [[buffer(17)]],                       \
      device const int32_t *cu_seqlens_q [[buffer(19)]],                       \
      const constant int &num_seqs [[buffer(20)]],                             \
      const constant int &sliding_window [[buffer(21)]],                       \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 tgp [[threadgroup_position_in_grid]],                              \
      uint3 tgpg [[threadgroups_per_grid]],                                    \
      uint3 tpt [[thread_position_in_threadgroup]],                            \
      uint sg_idx [[simdgroup_index_in_threadgroup]],                          \
      uint lane [[thread_index_in_simdgroup]]);

#define instantiate_paged_attention_tiled_heads(type, block_size)               \
  instantiate_paged_attention_tiled_inner(type, 64, block_size);               \
  instantiate_paged_attention_tiled_inner(type, 96, block_size);               \
  instantiate_paged_attention_tiled_inner(type, 128, block_size);              \
  instantiate_paged_attention_tiled_inner(type, 192, block_size);              \
  instantiate_paged_attention_tiled_inner(type, 256, block_size);

#define instantiate_paged_attention_tiled_all(type)                             \
  instantiate_paged_attention_tiled_heads(type, 8);                            \
  instantiate_paged_attention_tiled_heads(type, 16);                           \
  instantiate_paged_attention_tiled_heads(type, 32);

instantiate_paged_attention_tiled_all(half);
instantiate_paged_attention_tiled_all(bfloat16_t);
