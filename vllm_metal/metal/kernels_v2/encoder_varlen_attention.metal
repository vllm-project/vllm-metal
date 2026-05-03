// SPDX-License-Identifier: Apache-2.0
//
// Dense non-causal varlen encoder attention kernel — flash-attention with
// register-resident MMA fragments and in-register online softmax, mirroring
// MLX's `steel_attention` kernel structure (BQ=32, BK=32, WM=4, WN=1).
//
// M3 redesign vs. M2
//   • All MMA tiles (Q, K, S, V, O) live as per-lane `vec<float,2>` fragments
//     using Apple's simdgroup_matrix<float,8,8> lane layout (the same one
//     MLX targets via BaseMMAFrag::get_coord).  Each lane holds 1 row × 2
//     contiguous cols of every 8x8 frag it participates in.
//   • Softmax row reductions run in registers via simd_shuffle_xor(1) +
//     simd_shuffle_xor(8) — the two-shuffle butterfly that traverses the
//     four lanes that share a row in the simdgroup_matrix layout.  No
//     more s_buf round-trip via simdgroup_store / simdgroup_load.
//   • α-rescale of the running O accumulator is now a per-lane scalar
//     `o_frags[d] *= factor` instead of a `diag(α) @ O` 8x8 MMA.  At
//     HEAD_DIM=128 that's TD=16 register-vec multiplies replacing 16 MMAs.
//   • Threadgroup memory uses pad = 16/sizeof(T) on the inner stride
//     (matches MLX's padQ / padK / padV) to avoid bank conflicts on
//     simdgroup-cooperative loads.
//   • V_smem is now row-major [K_TILE, HEAD_DIM]  (M2 had it transposed
//     as a hold-over from the iter1 manual kernel).
//
// Per-lane state
//   q_frag:       vec<float,2>      reloaded each `dd` (Q is dim-tiled)
//   s_frags[TK]:  vec<float,2>[TK]  S = Q @ K^T accumulated across dd
//   v_frag:       vec<float,2>      reloaded each (id, ik) MMA
//   o_frags[TD]:  vec<float,2>[TD]  running unnormalised output (fp32)
//   max_score:    float             online softmax row max
//   sum_score:    float             online softmax row sum (= l)
//
// MMA dispatch identity
//   simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat) requires
//   simdgroup_matrix operands.  We marshal `vec<float,2>` ↔
//   `simdgroup_float8x8::thread_elements()` via a reinterpret_cast.
//   Apple's simdgroup_matrix<float,8,8> stores exactly two fp32 elements
//   per lane in a `vec<float,2>` thread_elements slot, so the cast is a
//   bitwise no-op.
//
// Lane coord (MLX BaseMMAFrag::get_coord)
//   const ushort qid = simd_lane_id / 4;
//   const ushort fm  = (qid & 4) + ((simd_lane_id / 2) % 4);   // 0..7 row
//   const ushort fn  = (qid & 2) * 2 + (simd_lane_id % 2) * 2; // 0/2/4/6 col
//   Lane owns frag[fm, fn..fn+1].  Four lanes per row are { l, l^1, l^8,
//   l^9 } — confirming xor-1 + xor-8 as a 4-way row reduction butterfly.
//
// f32 path (`encoder_varlen_attention_manual_kernel`) is unchanged from
// M2 — fp32 elements double tg-memory pressure and the 32×32 MMA tile
// blows the 32 KB cap at HEAD_DIM=128, so f32 stays on the smaller-tile
// manual-SIMD fallback (Q=8, K=16).
//
// utils.metal is concatenated ahead of this file by the source builder in
// vllm_metal.metal.__init__, so bfloat16_t is available.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

// Hand-mirrored on the C++ side at paged_ops.cpp.  Field order is part of
// the ABI; do not reorder without updating the C++ struct in lockstep.
struct EncoderVarlenParams {
    int   num_q_heads;
    int   num_kv_heads;
    int   total_tokens;
    int   num_segments;   // == cu_seqlens.shape[0] - 1; binary-search bound.
    int   max_seqlen;     // Drives launch-grid pitch (tiles_per_segment).
    float softmax_scale;
};

// Metal-side ABI tripwire mirroring the C++ static_asserts in paged_ops.cpp.
// needs_rebuild() in build.py only stats the .cpp, so a .metal-only struct
// edit would JIT-compile cleanly and silently misdecode the C++ bytes;
// this size check fires at JIT time when the .metal struct drifts from
// the C++ side.
static_assert(sizeof(EncoderVarlenParams) == 24,
              "EncoderVarlenParams ABI drift: size diverges from "
              "paged_ops.cpp; update both files in lockstep.");

// ---------------------------------------------------------------------------
// MMA kernel — bf16 / f16
// ---------------------------------------------------------------------------

template <typename T, uint HEAD_DIM>
[[kernel]] void encoder_varlen_attention_mma_kernel(
    device const T*       q           [[buffer(0)]],
    device const T*       k           [[buffer(1)]],
    device const T*       v           [[buffer(2)]],
    device const int*     cu_seqlens  [[buffer(3)]],
    device       T*       out         [[buffer(4)]],
    constant     EncoderVarlenParams& params [[buffer(5)]],
    uint3 tg_id        [[threadgroup_position_in_grid]],
    uint  lid          [[thread_index_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_id      [[simdgroup_index_in_threadgroup]])
{
    constexpr uint THREADS    = 256;
    constexpr uint NUM_SIMDS  = 8;
    constexpr uint Q_TILE     = 64;
    constexpr uint K_TILE     = 32;
    constexpr uint kFragSize  = 8;
    constexpr uint TQ         = 1;                       // Each simdgroup owns 1×8=8 query rows
    constexpr uint TK         = K_TILE / kFragSize;       // 4
    constexpr uint TD         = HEAD_DIM / kFragSize;     // 8 / 10 / 12 / 16
    static_assert(HEAD_DIM % kFragSize == 0,
                  "HEAD_DIM must be a multiple of 8 for the 8x8 MMA path");

    // Padding in tg memory to avoid bank conflicts on Apple GPU.
    // pad = 16/sizeof(T) matches MLX's steel_attention padQ/padK/padV
    // and helps hd64/80/128 by 2–5%.  At HEAD_DIM=96 the padded inner
    // stride (104 elements / 208 bytes for bf16) is empirically *worse*
    // than the unpadded (96 / 192 bytes), losing ~8% on the bench —
    // 96 is exactly 1.5 bank cycles wide so the unpadded layout already
    // distributes lanes across banks evenly, and the +8 extra elements
    // both inflate tg-memory pressure and shift the bank pattern in a
    // way that re-introduces conflicts.  Disabling padding for hd96
    // recovers that loss.  Verified by ablation 2026-05-03.
    constexpr uint pad_default = 16 / sizeof(T);
    constexpr uint padQ    = (HEAD_DIM == 96) ? 0 : pad_default;
    constexpr uint padK    = pad_default;                  // K_TILE=32 always
    constexpr uint padV    = (HEAD_DIM == 96) ? 0 : pad_default;
    constexpr uint LDQ_tgp = HEAD_DIM + padQ;             // Q row stride
    constexpr uint LDK_tgp = K_TILE  + padK;              // K col stride (K is transposed)
    constexpr uint LDV_tgp = HEAD_DIM + padV;             // V row stride

    // ---- Threadgroup memory ------------------------------------------------
    // Q_smem  row-major  [Q_TILE, LDQ_tgp]   — Q[q, d]
    // KV_smem aliases    K_smem and V_smem (max of the two; non-overlapping
    //                    in time — V is loaded after the Q@K^T MMAs done).
    //   K_smem  transposed [HEAD_DIM, LDK_tgp]  — K[d, k] (logically K^T)
    //   V_smem  row-major  [K_TILE,   LDV_tgp]  — V[k, d]
    threadgroup T Q_smem[Q_TILE * LDQ_tgp];
    constexpr uint tgp_mem_K  = HEAD_DIM * LDK_tgp;
    constexpr uint tgp_mem_V  = K_TILE   * LDV_tgp;
    constexpr uint tgp_mem_KV = (tgp_mem_K > tgp_mem_V) ? tgp_mem_K : tgp_mem_V;
    threadgroup T KV_smem[tgp_mem_KV];
    threadgroup T* Ks = KV_smem;
    threadgroup T* Vs = KV_smem;

    const int   head_idx = int(tg_id.y);
    const int   num_q    = params.num_q_heads;
    const int   num_kv   = params.num_kv_heads;
    const float scale    = params.softmax_scale * M_LOG2E_F;

    // Decode (segment, q_base) from tg_id.x.  tiles_per_segment uses
    // params.max_seqlen as the launch-grid pitch (matches C++ launcher).
    const int tiles_per_segment =
        (params.max_seqlen + int(Q_TILE) - 1) / int(Q_TILE);
    const int seg         = int(tg_id.x) / tiles_per_segment;
    const int tile_in_seg = int(tg_id.x) % tiles_per_segment;
    if (seg >= params.num_segments) return;

    const int seg_lo = cu_seqlens[seg];
    const int seg_hi = cu_seqlens[seg + 1];
    const int q_base = seg_lo + tile_in_seg * int(Q_TILE);
    if (q_base >= seg_hi) return;
    const int q_end   = min(q_base + int(Q_TILE), seg_hi);
    const int q_count = q_end - q_base;

    // ---- Lane coords within an 8×8 simdgroup_matrix frag ------------------
    // (MLX BaseMMAFrag::get_coord, the canonical Apple simdgroup_matrix
    //  layout on Apple GPU 7+.)  Lane owns 1 row × 2 cols at (fm, fn..fn+1).
    const ushort qid = simd_lane_id / 4;
    const ushort fm  = (qid & 4) + ((simd_lane_id / 2) % 4);   // 0..7
    const ushort fn  = (qid & 2) * 2 + (simd_lane_id % 2) * 2; // 0,2,4,6

    // Per-simdgroup row offset.  Each simdgroup owns rows [tm, tm+8).
    const ushort tm = simd_id * kFragSize;

    // ---- Cooperative load of Q-tile (vectorized, 16-byte) ----------------
    // Q_smem[q, d] = q[q_base + q, head_idx, d].  HEAD_DIM is a multiple
    // of VEC_T (=16/sizeof(T)) for every supported head dim, so the
    // vec read+write are exact-aligned.  Padded rows zero-filled.
    constexpr uint VEC_T = 16 / sizeof(T);     // 8 (bf16/f16), 4 (f32)
    static_assert(HEAD_DIM % VEC_T == 0,
                  "HEAD_DIM must be a multiple of 16/sizeof(T) for vec loads");
    constexpr uint Q_VECS = (Q_TILE * HEAD_DIM) / VEC_T;
    for (uint vi = lid; vi < Q_VECS; vi += THREADS) {
        const uint base = vi * VEC_T;
        const uint qrow = base / HEAD_DIM;
        const uint d    = base % HEAD_DIM;
        threadgroup vec<T, VEC_T>* qdst =
            reinterpret_cast<threadgroup vec<T, VEC_T>*>(
                &Q_smem[qrow * LDQ_tgp + d]);
        if (qrow < uint(q_count)) {
            const int qi = q_base + int(qrow);
            *qdst = *reinterpret_cast<const device vec<T, VEC_T>*>(
                &q[(qi * num_q + head_idx) * HEAD_DIM + d]);
        } else {
            *qdst = vec<T, VEC_T>(T(0));
        }
    }

    // ---- Per-lane register state ------------------------------------------
    // Each lane "owns" query row (tm + fm) for the duration of the kernel.
    // The four lanes that share that row (l, l^1, l^8, l^9) all maintain
    // the same max_score / sum_score (after the 2-shuffle butterfly).
    using FragF = vec<float, 2>;
    float max_score = -INFINITY;
    float sum_score = 0.0f;
    FragF o_frags[TD];
    for (uint dt = 0; dt < TD; ++dt) o_frags[dt] = FragF(0.0f);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Tg-memory offsets for fragment loads -----------------------------
    // Q_smem[(tm + fm) * LDQ_tgp + fn + dd*kFragSize + {0,1}]
    const uint Qs_offset    = uint(tm + fm) * LDQ_tgp + uint(fn);
    constexpr uint Qs_d_stride = kFragSize;          // dd advances along d cols
    // K_smem[(fm + dd*kFragSize) * LDK_tgp + fn + kt*kFragSize + {0,1}]
    const uint Ks_base      = uint(fm) * LDK_tgp + uint(fn);
    constexpr uint Ks_d_stride = kFragSize * LDK_tgp; // dd advances along d rows
    constexpr uint Ks_k_stride = kFragSize;           // kt advances along k cols
    // V_smem[(fm + ik*kFragSize) * LDV_tgp + fn + id*kFragSize + {0,1}]
    const uint Vs_base      = uint(fm) * LDV_tgp + uint(fn);
    constexpr uint Vs_k_stride = kFragSize * LDV_tgp; // ik advances along k rows
    constexpr uint Vs_d_stride = kFragSize;           // id advances along d cols


    // =====================================================================
    // K-tile loop
    // =====================================================================
    for (int tile_lo = seg_lo; tile_lo < seg_hi; tile_lo += int(K_TILE)) {
        const int tile_hi = min(tile_lo + int(K_TILE), seg_hi);
        const int k_count = tile_hi - tile_lo;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Cooperative load K (transposed; vec read + scatter write) ---
        // K_smem[d, krow] = K[ki, d].  Reads are vectorized (8 contiguous
        // d's per device load); writes are scattered into the LDK_tgp-
        // stride layout and stay scalar.  Net effect: 1/VEC_T fewer
        // device-mem transactions.
        constexpr uint K_VECS = (K_TILE * HEAD_DIM) / VEC_T;
        for (uint vi = lid; vi < K_VECS; vi += THREADS) {
            const uint base    = vi * VEC_T;
            const uint krow    = base / HEAD_DIM;
            const uint d_start = base % HEAD_DIM;
            vec<T, VEC_T> k_vec;
            if (krow < uint(k_count)) {
                const int ki = tile_lo + int(krow);
                k_vec = *reinterpret_cast<const device vec<T, VEC_T>*>(
                    &k[(ki * num_kv + head_idx) * HEAD_DIM + d_start]);
            } else {
                k_vec = vec<T, VEC_T>(T(0));
            }
            for (uint i = 0; i < VEC_T; ++i) {
                Ks[(d_start + i) * LDK_tgp + krow] = k_vec[i];
            }
        }

        // S accumulator (one frag per K-axis 8-block; accumulated across dd)
        FragF s_frags[TK];
        for (uint kt = 0; kt < TK; ++kt) s_frags[kt] = FragF(0.0f);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- S = Q @ K^T via TD × TK MMAs --------------------------------
        // Outer: dd iterates the shared dim.  Q frag is reused across kt.
        // simdgroup_barrier(mem_none) bracket every dd for instruction-
        // ordering hint; inside the kt loop the loads are independent of
        // the previous MMA and don't need extra barriers.
        for (uint dd = 0; dd < TD; ++dd) {
            simdgroup_barrier(mem_flags::mem_none);

            // Load this lane's 1×2 slice of the Q frag at d offset dd*8.
            FragF q_frag;
            q_frag.x = float(Q_smem[Qs_offset + dd * Qs_d_stride + 0]);
            q_frag.y = float(Q_smem[Qs_offset + dd * Qs_d_stride + 1]);

            for (uint kt = 0; kt < TK; ++kt) {
                FragF k_frag;
                k_frag.x = float(Ks[Ks_base + dd * Ks_d_stride
                                            + kt * Ks_k_stride + 0]);
                k_frag.y = float(Ks[Ks_base + dd * Ks_d_stride
                                            + kt * Ks_k_stride + 1]);

                simdgroup_float8x8 D_mat, A_mat, B_mat, C_mat;
                reinterpret_cast<thread FragF&>(A_mat.thread_elements()) = q_frag;
                reinterpret_cast<thread FragF&>(B_mat.thread_elements()) = k_frag;
                reinterpret_cast<thread FragF&>(C_mat.thread_elements()) = s_frags[kt];
                simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
                s_frags[kt] = reinterpret_cast<thread FragF&>(D_mat.thread_elements());
            }
        }

        // ---- Apply softmax scale (fold-of-log2e already in `scale`) ------
        for (uint kt = 0; kt < TK; ++kt) {
            s_frags[kt] *= scale;
        }

        // ---- Mask out keys past tile_hi ----------------------------------
        // Lane owns cols (fn, fn+1) of frag kt at global K positions
        // (kt*8 + fn, kt*8 + fn + 1).  Set to -INF so exp2 → 0.
        if (k_count < int(K_TILE)) {
            for (uint kt = 0; kt < TK; ++kt) {
                const int col0 = int(kt * kFragSize) + int(fn);
                if (col0     >= k_count) s_frags[kt].x = -INFINITY;
                if (col0 + 1 >= k_count) s_frags[kt].y = -INFINITY;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Cooperative load V (row-major, vectorized 16-byte) ----------
        // V_smem[krow, d] = V[ki, d]; row-major writes are vectorizable.
        constexpr uint V_VECS = (K_TILE * HEAD_DIM) / VEC_T;
        for (uint vi = lid; vi < V_VECS; vi += THREADS) {
            const uint base = vi * VEC_T;
            const uint krow = base / HEAD_DIM;
            const uint d    = base % HEAD_DIM;
            threadgroup vec<T, VEC_T>* vdst =
                reinterpret_cast<threadgroup vec<T, VEC_T>*>(
                    &Vs[krow * LDV_tgp + d]);
            if (krow < uint(k_count)) {
                const int ki = tile_lo + int(krow);
                *vdst = *reinterpret_cast<const device vec<T, VEC_T>*>(
                    &v[(ki * num_kv + head_idx) * HEAD_DIM + d]);
            } else {
                *vdst = vec<T, VEC_T>(T(0));
            }
        }

        // ---- Online softmax (in registers) -------------------------------
        // Row max: per-frag (xor 1, xor 8) butterfly across the 4 lanes
        // that share this row, accumulated into `new_max` across all TK
        // frags.  Final per-row max is identical on all 4 lanes.
        float new_max = max_score;
        for (uint kt = 0; kt < TK; ++kt) {
            float v = max(s_frags[kt].x, s_frags[kt].y);
            v = max(v, simd_shuffle_xor(v, ushort(1)));
            v = max(v, simd_shuffle_xor(v, ushort(8)));
            new_max = max(new_max, v);
        }
        // NaN-safe: if every key in this and all prior tiles was masked,
        // pin new_max to 0 so exp2(s - 0) is well-defined (s == -INF → 0).
        if (new_max == -INFINITY) new_max = 0.0f;

        const float factor =
            (max_score == -INFINITY) ? 0.0f : exp2(max_score - new_max);
        max_score = new_max;

        // P = exp2(S - new_max), in place; lane-local row sum.
        float sum_score_tmp = 0.0f;
        for (uint kt = 0; kt < TK; ++kt) {
            s_frags[kt].x = exp2(s_frags[kt].x - new_max);
            s_frags[kt].y = exp2(s_frags[kt].y - new_max);
            sum_score_tmp += s_frags[kt].x + s_frags[kt].y;
        }
        sum_score_tmp += simd_shuffle_xor(sum_score_tmp, ushort(1));
        sum_score_tmp += simd_shuffle_xor(sum_score_tmp, ushort(8));

        sum_score = sum_score * factor + sum_score_tmp;

        // ---- Rescale O by factor (per-lane scalar; no diag-α MMA) -------
        for (uint dt = 0; dt < TD; ++dt) {
            o_frags[dt] *= factor;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O += P @ V via TD × TK MMAs ---------------------------------
        // O[iq][id] += S[iq][ik] @ V[ik][id], iq fixed at 0 (TQ=1).
        // ik OUTER / id INNER: within an ik iteration, all TD MMAs hit
        // distinct o_frags[id] outputs and are independent — gives the
        // compiler TD parallel MMAs to schedule before the next ik.
        for (uint ik = 0; ik < TK; ++ik) {
            for (uint id = 0; id < TD; ++id) {
                FragF v_frag;
                v_frag.x = float(Vs[Vs_base + ik * Vs_k_stride
                                            + id * Vs_d_stride + 0]);
                v_frag.y = float(Vs[Vs_base + ik * Vs_k_stride
                                            + id * Vs_d_stride + 1]);

                simdgroup_float8x8 D_mat, A_mat, B_mat, C_mat;
                reinterpret_cast<thread FragF&>(A_mat.thread_elements()) = s_frags[ik];
                reinterpret_cast<thread FragF&>(B_mat.thread_elements()) = v_frag;
                reinterpret_cast<thread FragF&>(C_mat.thread_elements()) = o_frags[id];
                simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
                o_frags[id] = reinterpret_cast<thread FragF&>(D_mat.thread_elements());
            }
        }
    }

    // =====================================================================
    // Final write: O / l → out (only valid query rows).
    // Lane (fm, fn) owns query row (tm + fm) and d cols
    // (id*8 + fn, id*8 + fn + 1) for each id frag.
    // =====================================================================
    const float inv_l = (sum_score > 0.0f) ? (1.0f / sum_score) : 0.0f;
    const int q_row_local = int(tm) + int(fm);
    if (q_row_local < q_count) {
        const int qi = q_base + q_row_local;
        device T* out_row = out + (qi * num_q + head_idx) * HEAD_DIM;
        for (uint id = 0; id < TD; ++id) {
            out_row[id * kFragSize + fn + 0] = T(o_frags[id].x * inv_l);
            out_row[id * kFragSize + fn + 1] = T(o_frags[id].y * inv_l);
        }
    }
}

// ---------------------------------------------------------------------------
// Manual SIMD kernel — f32 fallback (Q_TILE=8, K_TILE=16 to fit tg-mem)
// ---------------------------------------------------------------------------

template <typename T, uint HEAD_DIM, uint Q_TILE, uint K_TILE>
[[kernel]] void encoder_varlen_attention_manual_kernel(
    device const T*       q           [[buffer(0)]],
    device const T*       k           [[buffer(1)]],
    device const T*       v           [[buffer(2)]],
    device const int*     cu_seqlens  [[buffer(3)]],
    device       T*       out         [[buffer(4)]],
    constant     EncoderVarlenParams& params [[buffer(5)]],
    uint3 tg_id        [[threadgroup_position_in_grid]],
    uint  lid          [[thread_index_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_id      [[simdgroup_index_in_threadgroup]])
{
    constexpr uint THREADS    = 128;
    constexpr uint NUM_SIMDS  = 4;
    constexpr uint Q_PER_SIMD = Q_TILE / NUM_SIMDS;
    static_assert(Q_TILE % NUM_SIMDS == 0,
                  "Q_TILE must be a multiple of NUM_SIMDS (4)");
    static_assert(K_TILE <= 32,
                  "K_TILE > 32 unsupported in v1 (lane = 1 key)");

    threadgroup T     Q_buf[Q_TILE * HEAD_DIM];
    threadgroup T     K_buf[HEAD_DIM * K_TILE];
    threadgroup T     V_buf[HEAD_DIM * K_TILE];
    threadgroup float O_buf[Q_TILE * HEAD_DIM];
    threadgroup float m_buf[Q_TILE];
    threadgroup float l_buf[Q_TILE];

    const int head_idx = int(tg_id.y);
    const int num_q    = params.num_q_heads;
    const int num_kv   = params.num_kv_heads;
    const float scale  = params.softmax_scale * M_LOG2E_F;

    const int tiles_per_segment =
        (params.max_seqlen + int(Q_TILE) - 1) / int(Q_TILE);
    const int seg         = int(tg_id.x) / tiles_per_segment;
    const int tile_in_seg = int(tg_id.x) % tiles_per_segment;
    if (seg >= params.num_segments) return;

    const int seg_lo = cu_seqlens[seg];
    const int seg_hi = cu_seqlens[seg + 1];
    const int q_base = seg_lo + tile_in_seg * int(Q_TILE);
    if (q_base >= seg_hi) return;
    const int q_end   = min(q_base + int(Q_TILE), seg_hi);
    const int q_count = q_end - q_base;

    for (uint e = lid; e < Q_TILE * HEAD_DIM; e += THREADS) {
        const uint qrow = e / HEAD_DIM;
        const uint d    = e % HEAD_DIM;
        if (qrow < uint(q_count)) {
            const int qi = q_base + int(qrow);
            Q_buf[e] = q[(qi * num_q + head_idx) * HEAD_DIM + d];
        } else {
            Q_buf[e] = T(0);
        }
    }

    if (lid < Q_TILE) {
        m_buf[lid] = -INFINITY;
        l_buf[lid] = 0.0f;
    }
    for (uint e = lid; e < Q_TILE * HEAD_DIM; e += THREADS) {
        O_buf[e] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int tile_lo = seg_lo; tile_lo < seg_hi; tile_lo += int(K_TILE)) {
        const int tile_hi = min(tile_lo + int(K_TILE), seg_hi);
        const int k_count = tile_hi - tile_lo;

        for (uint e = lid; e < K_TILE * HEAD_DIM; e += THREADS) {
            const uint krow = e / HEAD_DIM;
            const uint d    = e % HEAD_DIM;
            const uint dst  = d * K_TILE + krow;
            if (krow < uint(k_count)) {
                const int ki = tile_lo + int(krow);
                K_buf[dst] = k[(ki * num_kv + head_idx) * HEAD_DIM + d];
                V_buf[dst] = v[(ki * num_kv + head_idx) * HEAD_DIM + d];
            } else {
                K_buf[dst] = T(0);
                V_buf[dst] = T(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint local_q = 0; local_q < Q_PER_SIMD; ++local_q) {
            const uint q_idx = simd_id * Q_PER_SIMD + local_q;

            float s_l;
            if (simd_lane_id < uint(K_TILE) && simd_lane_id < uint(k_count)) {
                float dot = 0.0f;
                for (uint d = 0; d < HEAD_DIM; ++d) {
                    dot += float(Q_buf[q_idx * HEAD_DIM + d]) *
                           float(K_buf[d * K_TILE + simd_lane_id]);
                }
                s_l = dot * scale;
            } else {
                s_l = -INFINITY;
            }

            const float tile_row_max = simd_max(s_l);
            const float m_old        = m_buf[q_idx];
            const float m_new        = max(m_old, tile_row_max);
            const float alpha        = exp2(m_old - m_new);

            const float p_l    = exp2(s_l - m_new);
            const float l_sum  = simd_sum(p_l);

            if (simd_lane_id == 0) {
                m_buf[q_idx] = m_new;
                l_buf[q_idx] = alpha * l_buf[q_idx] + l_sum;
            }

            const bool lane_has_key = simd_lane_id < uint(K_TILE);
            for (uint d = 0; d < HEAD_DIM; ++d) {
                const float v_ld =
                    lane_has_key
                        ? float(V_buf[d * K_TILE + simd_lane_id])
                        : 0.0f;
                const float pv_local = p_l * v_ld;
                const float pv_sum   = simd_sum(pv_local);
                if (simd_lane_id == 0) {
                    O_buf[q_idx * HEAD_DIM + d] =
                        alpha * O_buf[q_idx * HEAD_DIM + d] + pv_sum;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint e = lid; e < Q_TILE * HEAD_DIM; e += THREADS) {
        const uint qrow = e / HEAD_DIM;
        const uint d    = e % HEAD_DIM;
        if (qrow < uint(q_count)) {
            const int qi  = q_base + int(qrow);
            const float o = O_buf[e] / l_buf[qrow];
            out[(qi * num_q + head_idx) * HEAD_DIM + d] = T(o);
        }
    }
}

// ---------------------------------------------------------------------------
// Specialization macros — bf16/f16 dispatch to the MMA kernel; f32 dispatches
// to the manual kernel.  All twelve [[host_name]] strings stay in the
// `encoder_varlen_attention_<dtype>_<head_dim>` format that the C++
// dispatcher in paged_ops.cpp builds.  The C++ launcher chooses the
// appropriate (Q_TILE, threadgroup) per dtype:
//   bf16/f16 → Q_TILE=32, threadgroup=128
//   f32      → Q_TILE=8,  threadgroup=128
// ---------------------------------------------------------------------------

#define instantiate_encoder_varlen_mma(type, type_tag, head_dim) \
    template [[host_name("encoder_varlen_attention_" #type_tag "_" #head_dim)]] \
    [[kernel]] void encoder_varlen_attention_mma_kernel<type, head_dim>( \
        device const type*       q           [[buffer(0)]], \
        device const type*       k           [[buffer(1)]], \
        device const type*       v           [[buffer(2)]], \
        device const int*        cu_seqlens  [[buffer(3)]], \
        device       type*       out         [[buffer(4)]], \
        constant     EncoderVarlenParams& params [[buffer(5)]], \
        uint3 tg_id        [[threadgroup_position_in_grid]], \
        uint  lid          [[thread_index_in_threadgroup]], \
        uint  simd_lane_id [[thread_index_in_simdgroup]], \
        uint  simd_id      [[simdgroup_index_in_threadgroup]]);

#define instantiate_encoder_varlen_manual(type, type_tag, head_dim, q_tile, k_tile) \
    template [[host_name("encoder_varlen_attention_" #type_tag "_" #head_dim)]] \
    [[kernel]] void encoder_varlen_attention_manual_kernel<type, head_dim, q_tile, k_tile>( \
        device const type*       q           [[buffer(0)]], \
        device const type*       k           [[buffer(1)]], \
        device const type*       v           [[buffer(2)]], \
        device const int*        cu_seqlens  [[buffer(3)]], \
        device       type*       out         [[buffer(4)]], \
        constant     EncoderVarlenParams& params [[buffer(5)]], \
        uint3 tg_id        [[threadgroup_position_in_grid]], \
        uint  lid          [[thread_index_in_threadgroup]], \
        uint  simd_lane_id [[thread_index_in_simdgroup]], \
        uint  simd_id      [[simdgroup_index_in_threadgroup]]);

#define instantiate_encoder_varlen_mma_set(type, type_tag) \
    instantiate_encoder_varlen_mma(type, type_tag,  64) \
    instantiate_encoder_varlen_mma(type, type_tag,  80) \
    instantiate_encoder_varlen_mma(type, type_tag,  96) \
    instantiate_encoder_varlen_mma(type, type_tag, 128)

#define instantiate_encoder_varlen_manual_set(type, type_tag) \
    instantiate_encoder_varlen_manual(type, type_tag,  64, 8, 16) \
    instantiate_encoder_varlen_manual(type, type_tag,  80, 8, 16) \
    instantiate_encoder_varlen_manual(type, type_tag,  96, 8, 16) \
    instantiate_encoder_varlen_manual(type, type_tag, 128, 8, 16)

instantiate_encoder_varlen_mma_set(half,       f16)
instantiate_encoder_varlen_mma_set(bfloat16_t, bf16)
instantiate_encoder_varlen_manual_set(float,   f32)
