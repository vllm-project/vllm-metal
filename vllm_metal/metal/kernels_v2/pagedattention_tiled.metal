// SPDX-License-Identifier: Apache-2.0
// Flash-Attention-2 style paged attention kernel for Apple Silicon (BQ=32).
//
// Design points:
//
//   1. Q lives in registers for the entire KV loop (loaded ONCE).
//   2. O accumulator lives in registers (no per-tile O_smem round-trip).
//   3. Each simdgroup owns 8 complete Q rows (BQ=32 / NUM_SG=4 = 8).
//   4. Online softmax done entirely in registers via simd_shuffle_xor.
//   5. Mask applied to register S fragments (no S_smem round-trip).
//   6. 2 barriers per KV tile (K-load and V-load) instead of 4.
//
// The MMA fragment layout uses Apple's 8×8 simdgroup_matrix tile.  Each lane
// holds 2 elements of one fragment via thread_elements(); the (row, col) of
// those 2 elements is given by frag_coord() below — derived from MLX
// FlashAttention's MFAMMAFrag::get_coord (csrc/async_v2_kernel.metal:282).
//
// Paged-specific concerns vs. dense SDPA:
//   - K/V loaded with per-token block_tables[i/BLOCK_SIZE] lookup (cannot
//     use simdgroup_async_copy because each token may live in a different
//     physical block).  Cooperative threadgroup load instead.
//   - Varlen Q packing via cu_seqlens_q + binary search.
//   - Causal mask, sliding window, softcapping applied per-frag.
//
// Template: <T, HEAD_SIZE, BLOCK_SIZE, BQ=32, TILE_KV=32, NUM_THREADS=128>

// Requires utils.metal (DIVIDE_ROUND_UP, MIN, MAX), <metal_stdlib>, and
// `using namespace metal` from earlier in the source concat.

// ─────────────────────────────────────────────────────────────────────────
// Helper: thread → (row, col) inside an 8×8 simdgroup_matrix fragment.
// Element 0 of thread_elements() lives at (fm, fn).
// Element 1 of thread_elements() lives at (fm, fn+1).
// Verified against simdgroup_load+store round-trip.
//
// Per-lane ownership of the 8×8 fragment (Lk = lane k; each lane owns 2
// horizontally-adjacent cells of one row):
//
//          c0    c1    c2    c3    c4    c5    c6    c7
// row0 │   L0    L0    L1    L1    L8    L8    L9    L9
// row1 │   L2    L2    L3    L3    L10   L10   L11   L11
// row2 │   L4    L4    L5    L5    L12   L12   L13   L13
// row3 │   L6    L6    L7    L7    L14   L14   L15   L15
// row4 │   L16   L16   L17   L17   L24   L24   L25   L25
// row5 │   L18   L18   L19   L19   L26   L26   L27   L27
// row6 │   L20   L20   L21   L21   L28   L28   L29   L29
// row7 │   L22   L22   L23   L23   L30   L30   L31   L31
//
// Each row is split over exactly 4 lanes — e.g. row0 → {L0, L1, L8, L9}.
// Those 4 differ only in lane-id bits 0 and 3, which is exactly why the
// per-row softmax reduction below is simd_shuffle_xor with masks 1 then 8.
// ─────────────────────────────────────────────────────────────────────────
inline short2 frag_coord(ushort lane_id) {
  const short qid = short(lane_id) / 4;
  const short fm  = (qid & 4) + (short(lane_id) / 2) % 4;
  const short fn  = (qid & 2) * 2 + (short(lane_id) % 2) * 2;
  return short2{fn, fm};
}

// ─────────────────────────────────────────────────────────────────────────
// Reduce one row of the 8×8 fragment across the 4 lanes that share it.
//
// Masks {1, 8} are FORCED by the frag_coord layout above: the row index fm
// depends on lane bits {b4,b2,b1}, so the 4 lanes of a row vary only in
// bits {b0,b3} = XOR masks 1 and 8.  Two shuffles (log2(4)) reduce them,
// and because XOR-shuffle is a symmetric butterfly the result is replicated
// to all 4 lanes — the per-row online-softmax state below relies on that.
//
// Worked example — row 0 lives in lanes {0,1,8,9}; each lane starts with
// the max over its own 2 columns:
//
//             L0    L1    L8    L9
// start:       3     7     2     5            (true row max = 7; goal: all four = 7)
//
// step A:  x = max(x, simd_shuffle_xor(x, 1))      partner = lane ^ 1
//          L0↔L1   (0^1=1, 1^1=0)   L8↔L9  (8^1=9, 9^1=8)
//          L0 = max(3,7)=7   L1 = max(7,3)=7   L8 = max(2,5)=5   L9 = max(5,2)=5
//                  └── pair {0,1} now both 7 ──┘  └── pair {8,9} now both 5 ──┘
//
// step B:  x = max(x, simd_shuffle_xor(x, 8))      partner = lane ^ 8
//          L0↔L8   (0^8=8, 8^8=0)   L1↔L9  (1^8=9, 9^8=1)
//          L0 = max(7,5)=7   L8 = max(5,7)=7   L1 = max(7,5)=7   L9 = max(5,7)=7
//
// end:        7     7     7     7            ✓ every lane of the row holds the max
//
// Do NOT replace with simd_sum: all 8 rows of the fragment live in one
// 32-lane simdgroup, so a 32-lane reduction would fold the rows together.
//
// Op mirrors MLX Steel BaseMMAFrag<T,8,8>::row_reduce
// (mlx/backend/metal/kernels/steel/attn/mma.h) so this stays diff-able
// against upstream; vendored, not #included, to keep the Metal build off
// MLX's private steel header tree.
// ─────────────────────────────────────────────────────────────────────────
struct FragMax { static inline float apply(float a, float b) { return max(a, b); } };
struct FragSum { static inline float apply(float a, float b) { return a + b; } };

template <typename Op>
inline float frag_row_reduce(float v) {
  v = Op::apply(v, simd_shuffle_xor(v, ushort(1)));
  v = Op::apply(v, simd_shuffle_xor(v, ushort(8)));
  return v;
}

template <typename T, int HEAD_SIZE, int BLOCK_SIZE,
          int BQ = 32, int TILE_KV = 32, int NUM_THREADS = 128>
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
  constexpr int TD = HEAD_SIZE / 8;        // # of 8-wide D fragments
  constexpr int TK = TILE_KV / 8;          // # of 8-wide K fragments
  constexpr int ROWS_PER_SG = BQ / NUM_SG; // 8 rows per simdgroup

  static_assert(HEAD_SIZE % 8 == 0, "HEAD_SIZE must be a multiple of 8");
  static_assert(TILE_KV % 8 == 0, "TILE_KV must be a multiple of 8");
  static_assert(BQ % NUM_SG == 0, "BQ must be divisible by NUM_SG");
  static_assert(ROWS_PER_SG == 8, "ROWS_PER_SG must equal 8 (frag rows)");

  const int thread_idx = tpt.x;
  const int head_idx = tgp.x;
  const int q_block_global_idx = tgp.y;
  const int num_heads = tgpg.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const int kv_token_stride = num_kv_heads * kv_head_stride;

  // ─ Varlen: resolve sequence and Q-block position (binary search) ──────
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

  // ─ Threadgroup memory layout (A1: bank-conflict padding) ──────────────
  // Each row padded to LD = HEAD_SIZE + SMEM_PAD so the 8 columns of every
  // 8×8 simdgroup_load/store land on distinct threadgroup-memory banks
  // (a HEAD_SIZE power-of-two stride aliases them).  The fp32 O_smem that
  // aliases Q_smem at exit reuses the SAME LD.
  //   Q_smem: BQ × LD,   K_smem/V_smem: TILE_KV × LD.
  constexpr int SMEM_PAD = 16 / sizeof(T);   // 8 elems (16 B) for fp16/bf16
  constexpr int LD = HEAD_SIZE + SMEM_PAD;   // padded leading dim
  constexpr int Q_ELEMS = BQ * LD;
  constexpr int KV_ELEMS = TILE_KV * LD;

  threadgroup T *Q_smem = reinterpret_cast<threadgroup T *>(shared_mem);
  threadgroup T *K_smem = Q_smem + Q_ELEMS;
  threadgroup T *V_smem = K_smem + KV_ELEMS;

  const float scale_log2 = scale * M_LOG2E_F;

  // ─ Load Q into Q_smem [BQ, HEAD_SIZE] (cooperative) ───────────────────
  // Rows with row >= valid_q get zero-Q (their S values will be masked
  // anyway, but zeroing avoids garbage propagating through QK).
  const device T *q_base = q + (q_seq_start + q_pos_start) * q_stride
                             + head_idx * HEAD_SIZE;
  for (int i = thread_idx; i < BQ * HEAD_SIZE; i += NUM_THREADS) {
    int r = i / HEAD_SIZE;
    int d = i % HEAD_SIZE;
    Q_smem[r * LD + d] = (r < valid_q) ? q_base[r * q_stride + d] : T(0);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ─ Load Q into register fragments (once, reused across all KV tiles) ──
  // Each simdgroup owns 8 contiguous rows: [sg_idx*8, sg_idx*8 + 8).
  // For each fragment, we load Q_smem[sg_idx*8..sg_idx*8+8, d*8..d*8+8].
  //
  // We use vec<T,2> arrays (MLX MFAMMAFrag pattern) for register storage.
  // simdgroup_matrix instances are materialized only at the MMA call site —
  // this gives the compiler an unambiguous signal that the data lives in
  // per-thread registers, not in some implicit threadgroup memory.
  using vec2T = vec<T, 2>;
  using vec2F = vec<float, 2>;
  vec2T Qreg[TD];
  #pragma unroll
  for (int d = 0; d < TD; d++) {
    simdgroup_matrix<T, 8, 8> tmp;
    simdgroup_load(tmp, Q_smem + sg_idx * 8 * LD + d * 8, LD);
    Qreg[d] = reinterpret_cast<thread vec2T &>(tmp.thread_elements());
  }

  // ─ Initialize per-simdgroup-row online softmax state ──────────────────
  // Each thread tracks max/sum for the row it owns inside the 8×8 frag.
  // Lane → (fm, fn) — fm is the row this thread contributes to.
  const short2 fc = frag_coord(ushort(lane));
  const short fm = fc.y;   // row within the 8-row tile owned by this SG
  const short fn = fc.x;   // col within the 8-col fragment

  // Per-thread accumulator state — broadcast across lanes that share fm.
  float max_score = -INFINITY;
  float sum_score = 0.0f;

  // O accumulator: TD fragments of (8 × 8) per simdgroup, register-resident.
  // Each lane holds vec<float,2> per fragment.
  vec2F Oreg[TD];
  #pragma unroll
  for (int d = 0; d < TD; d++) {
    Oreg[d] = vec2F(0.0f);
  }

  const device uint32_t *block_table =
      block_tables + seq_idx * max_num_blocks_per_seq;
  const int num_kv_tiles = DIVIDE_ROUND_UP(seq_len, TILE_KV);

  // ─ MAIN KV TILE LOOP ──────────────────────────────────────────────────
  for (int tile_idx = 0; tile_idx < num_kv_tiles; tile_idx++) {
    const int tile_start = tile_idx * TILE_KV;

    // Causal skip: if the entire tile is beyond the maximum q_abs_pos this
    // threadgroup will produce, we can stop.
    if (tile_start > context_len + q_pos_start + valid_q - 1) break;

    // ─ Load K AND V cooperatively (paged, fused, vec2-vectorized) ─────
    // Both K_smem and V_smem are filled in the same loop:
    //   * block_table lookup is shared (same kv_pos for both K and V)
    //   * memory controller interleaves K and V reads from the same
    //     physical block, hiding K-load latency behind V-load and v.v.
    //   * saves a barrier vs loading them separately
    //
    // vec<T,2> loads halve the number of load/store instructions issued.
    // For HEAD_SIZE=64 with 32 lanes: each lane does 1 vec2 = 4 bytes per
    // token, covering all 64 channels.  HEAD_SIZE=128: 2 vec2 per lane.
    //
    // Alignment: every contributing stride (kv_block_stride, kv_token_stride,
    // kv_head_stride = HEAD_SIZE) is even because HEAD_SIZE is a multiple
    // of 8, so `off` and `lane*VEC` are both even → 4-byte aligned vec2.
    static_assert(HEAD_SIZE % 2 == 0, "HEAD_SIZE must be even for vec2 loads");
    constexpr int VEC = 2;
    using vec2T = vec<T, VEC>;
    #pragma unroll
    for (int t_iter = 0; t_iter < TILE_KV / NUM_SG; t_iter++) {
      int t = sg_idx * (TILE_KV / NUM_SG) + t_iter;
      int kv_pos = tile_start + t;
      if (kv_pos < seq_len) {
        int64_t pb = int64_t(block_table[kv_pos / BLOCK_SIZE]);
        const int64_t off = pb * kv_block_stride
            + (kv_pos % BLOCK_SIZE) * kv_token_stride
            + kv_head_idx * kv_head_stride;
        const device T *k_ptr = k_cache + off;
        const device T *v_ptr = v_cache + off;
        #pragma unroll
        for (int d = lane * VEC; d < HEAD_SIZE; d += NUM_SIMD_LANES * VEC) {
          *((threadgroup vec2T *)&K_smem[t * LD + d]) =
              *((const device vec2T *)(k_ptr + d));
          *((threadgroup vec2T *)&V_smem[t * LD + d]) =
              *((const device vec2T *)(v_ptr + d));
        }
      } else {
        const vec2T zero(T(0));
        #pragma unroll
        for (int d = lane * VEC; d < HEAD_SIZE; d += NUM_SIMD_LANES * VEC) {
          *((threadgroup vec2T *)&K_smem[t * LD + d]) = zero;
          *((threadgroup vec2T *)&V_smem[t * LD + d]) = zero;
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);  // K & V visible

    // ─ QK matmul: S[8, TILE_KV] = Q × K^T via 8×8 MMA ──────────────────
    // Q is in registers (Qreg); K is read from shared memory with transpose.
    // Each simdgroup computes the 8 rows it owns × all TILE_KV cols.
    // S accumulator (Sreg) is float-vec<2> per lane per K-frag.
    vec2F Sreg[TK];
    #pragma unroll
    for (int k = 0; k < TK; k++) {
      Sreg[k] = vec2F(0.0f);
    }

    #pragma unroll
    for (int d = 0; d < TD; d++) {
      // Materialize Q fragment from register vec → simdgroup_matrix
      simdgroup_matrix<T, 8, 8> q_frag;
      reinterpret_cast<thread vec2T &>(q_frag.thread_elements()) = Qreg[d];

      #pragma unroll
      for (int k = 0; k < TK; k++) {
        simdgroup_matrix<T, 8, 8> k_frag;
        simdgroup_load(k_frag,
                       K_smem + k * 8 * LD + d * 8,
                       LD, ulong2(0), /*transpose=*/true);

        // Materialize S accumulator from register vec → simdgroup_matrix.
        simdgroup_matrix<float, 8, 8> s_frag;
        reinterpret_cast<thread vec2F &>(s_frag.thread_elements()) = Sreg[k];

        simdgroup_multiply_accumulate(s_frag, q_frag, k_frag, s_frag);

        // Pull result back into register vec.
        Sreg[k] = reinterpret_cast<thread vec2F &>(s_frag.thread_elements());
      }
    }

    // ─ Scale + apply mask + softcap (all in registers) ─────────────────
    // Fast path: when the entire tile is "before" the causal frontier of
    // every Q row in this threadgroup, no causal/padding mask is needed.
    // This applies to most tiles in long-prefill (tile_start + TILE_KV - 1
    // < min_q_abs_pos in this threadgroup; min_q_abs_pos = context_len +
    // q_pos_start).  Saves ~16 ALU ops per element on tiles 0..N_safe-1.
    const int q_abs_pos = context_len + q_pos_start + sg_idx * 8 + fm;
    const bool row_masked = (sg_idx * 8 + fm) >= valid_q;
    const int min_q_abs_pos = context_len + q_pos_start;
    const bool tile_no_mask = (tile_start + TILE_KV - 1) < min_q_abs_pos
                              && (tile_start + TILE_KV) <= seq_len
                              && softcapping <= 0.0f
                              && sliding_window < 0;

    if (tile_no_mask && !row_masked) {
      #pragma unroll
      for (int k = 0; k < TK; k++) {
        Sreg[k][0] *= scale_log2;
        Sreg[k][1] *= scale_log2;
      }
    } else {
      #pragma unroll
      for (int k = 0; k < TK; k++) {
        #pragma unroll
        for (int jj = 0; jj < 2; jj++) {
          float s = Sreg[k][jj] * scale_log2;
          if (softcapping > 0.0f) {
            float s_orig = s / M_LOG2E_F;
            s = softcapping * precise::tanh(s_orig / softcapping) * M_LOG2E_F;
          }
          int kv_pos = tile_start + k * 8 + fn + jj;
          bool masked = row_masked
                        || (kv_pos > q_abs_pos)
                        || (kv_pos >= seq_len);
          if (sliding_window >= 0)
            masked = masked || (kv_pos < q_abs_pos + 1 - sliding_window);
          Sreg[k][jj] = masked ? -INFINITY : s;
        }
      }
    }

    // ─ Row max: local max across 2*TK elements, then XOR-reduce ─────────
    float local_max = -INFINITY;
    #pragma unroll
    for (int k = 0; k < TK; k++) {
      local_max = max(local_max, max(Sreg[k][0], Sreg[k][1]));
    }
    float row_max = frag_row_reduce<FragMax>(local_max);

    float new_max = max(max_score, row_max);
    float factor;
    if (new_max > max_score) {
      factor = (max_score == -INFINITY) ? 0.0f
                                        : fast::exp2(max_score - new_max);
    } else {
      factor = 1.0f;
      if (max_score == -INFINITY) new_max = 0.0f;
    }
    max_score = new_max;

    // ─ Exponentiate + row sum ───────────────────────────────────────────
    float local_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TK; k++) {
      #pragma unroll
      for (int jj = 0; jj < 2; jj++) {
        float p = (Sreg[k][jj] == -INFINITY)
                      ? 0.0f
                      : fast::exp2(Sreg[k][jj] - new_max);
        Sreg[k][jj] = p;
        local_sum += p;
      }
    }
    float row_sum = frag_row_reduce<FragSum>(local_sum);

    sum_score = sum_score * factor + row_sum;

    // ─ Rescale O in registers ───────────────────────────────────────────
    #pragma unroll
    for (int d = 0; d < TD; d++) {
      Oreg[d] *= factor;
    }

    // ─ PV matmul: O += P × V via 8×8 MMA ───────────────────────────────
    // V was loaded above together with K — no additional barrier needed.
    // (One barrier at the END of this iteration protects V_smem against
    //  the next iteration's K+V load.)
    // P is in registers (Sreg, fp32). V is in smem. O accumulator stays in
    // registers (Oreg).  P is kept fp32 into the MMA (mixed float×T→float),
    // matching the pre-PR kernel exactly — this PR is precision-neutral.
    // Sreg is already vec2F, so this is the same MFAMMAFrag reinterpret used
    // for Q/S/O (no downcast).  Narrowing P to T is a separate,
    // benchmark-gated follow-up, not bundled into this perf migration.
    #pragma unroll
    for (int k = 0; k < TK; k++) {
      simdgroup_matrix<float, 8, 8> p_frag;
      reinterpret_cast<thread vec2F &>(p_frag.thread_elements()) = Sreg[k];

      #pragma unroll
      for (int d = 0; d < TD; d++) {
        simdgroup_matrix<T, 8, 8> v_frag;
        simdgroup_load(v_frag,
                       V_smem + k * 8 * LD + d * 8,
                       LD);

        // Materialize O accumulator fragment from register vec.
        simdgroup_matrix<float, 8, 8> o_frag;
        reinterpret_cast<thread vec2F &>(o_frag.thread_elements()) = Oreg[d];

        simdgroup_multiply_accumulate(o_frag, p_frag, v_frag, o_frag);

        Oreg[d] = reinterpret_cast<thread vec2F &>(o_frag.thread_elements());
      }
    }

    // Protect V_smem against the next iteration's K+V load.  PV above
    // reads ALL rows of V_smem; the next iteration's load writes V_smem.
    // Without this barrier, fast simdgroups can corrupt slow ones' reads.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } // end KV tile loop

  // ─ Final normalize: O /= sum_score, then store ────────────────────────
  // sum_score is broadcast-replicated across lanes with same fm.
  float inv_sum = 1.0f / (sum_score + 1e-6f);
  #pragma unroll
  for (int d = 0; d < TD; d++) {
    Oreg[d] *= inv_sum;
  }

  // Reuse Q_smem as the output staging buffer (it's no longer needed).
  // Each simdgroup stores its 8 rows × HEAD_SIZE into Q_smem (as float).
  threadgroup float *O_smem = reinterpret_cast<threadgroup float *>(Q_smem);
  #pragma unroll
  for (int d = 0; d < TD; d++) {
    simdgroup_matrix<float, 8, 8> o_frag;
    reinterpret_cast<thread vec2F &>(o_frag.thread_elements()) = Oreg[d];
    simdgroup_store(o_frag,
                    O_smem + sg_idx * 8 * LD + d * 8,
                    LD);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write to global memory, respecting valid_q.
  device T *out_base = out + (q_seq_start + q_pos_start) * q_stride
                           + head_idx * HEAD_SIZE;
  const int total = valid_q * HEAD_SIZE;
  for (int i = thread_idx; i < total; i += NUM_THREADS) {
    int r = i / HEAD_SIZE;
    int d = i % HEAD_SIZE;
    out_base[r * q_stride + d] = T(O_smem[r * LD + d]);
  }
}

// ─── Template instantiation ──────────────────────────────────────────────

#define instantiate_paged_attention_tiled_inner(type, head_size, block_size)   \
  template [[host_name("paged_attention_tiled_" #type                          \
                       "_hs" #head_size "_bs" #block_size                      \
                       "_bq32_tk32_nt128")]]                                   \
  [[kernel]] void paged_attention_tiled<type, head_size, block_size,           \
                                        32, 32, 128>(                          \
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

#define instantiate_paged_attention_tiled_heads(type, block_size)              \
  instantiate_paged_attention_tiled_inner(type, 64, block_size);               \
  instantiate_paged_attention_tiled_inner(type, 96, block_size);               \
  instantiate_paged_attention_tiled_inner(type, 128, block_size);

#define instantiate_paged_attention_tiled_all(type)                            \
  instantiate_paged_attention_tiled_heads(type, 8);                            \
  instantiate_paged_attention_tiled_heads(type, 16);                           \
  instantiate_paged_attention_tiled_heads(type, 32);

instantiate_paged_attention_tiled_all(half);
instantiate_paged_attention_tiled_all(bfloat16_t);
