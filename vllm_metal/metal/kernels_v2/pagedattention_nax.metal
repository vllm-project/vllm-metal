// SPDX-License-Identifier: Apache-2.0
// NAX (M5 tensor-unit) paged prefill attention. Portions adapted from MLX
// steel_attention_nax (MIT, Copyright © 2025 Apple Inc.).
//
// QK^T and PV use MPP 16x32x16 matmul2d operations. Q/K/V move directly from
// device memory to register fragments without threadgroup staging.
// relaxed_precision=true is required by the cooperative-tensor register layout
// mirrored in nax_coord(); it also truncates the fp32 P operand of PV.
//
// Each 16-row K/V fragment is two 8-token page-local loads. Block sizes
// 8/16/32 therefore need four block-table lookups per 32-token KV tile.
// This optional metallib requires macOS 26.2.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

typedef bfloat bfloat16_t;

constant bool nax_use_sinks [[function_constant(40)]];

template <typename T>
using nax_frag8 = metal::vec<T, 8>;

#define NAX_FINITE_MIN (-3.402823466e+38f)

// Lane -> (col base fn, row fm) inside a 16x16 fragment: each lane owns rows
// {fm, fm+8} x cols {fn..fn+3}, element i*4+j = (fm + i*8, fn + j).  Mirrors
// mlx BaseNAXFrag::get_coord; see the relaxed_precision note above for why
// this map is descriptor-dependent.
METAL_FUNC short2 nax_coord() {
  const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  return short2{fn, fm};
}

// C[16x32] (two 16x16 frags) += A[16x16] @ B, one simdgroup.
//   TB=true : B frags are rows 0-15 / 16-31 of a (32 tokens x 16) tile (K^T)
//   TB=false: B frags are cols 0-15 / 16-31 of a (16 x 32) tile (V)
template <typename CT, typename AT, typename BT, bool TB>
METAL_FUNC void nax_mma(thread nax_frag8<CT>& Cn0, thread nax_frag8<CT>& Cn1,
                        const thread nax_frag8<AT>& A,
                        const thread nax_frag8<BT>& Bn0,
                        const thread nax_frag8<BT>& Bn1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16,
      /*transpose_left=*/false, /*transpose_right=*/TB,
      /*relaxed_precision=*/true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
  auto ct_b = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>, CT>();
#pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    ct_a[i] = A[i];
    ct_b[i] = Bn0[i];
    ct_b[8 + i] = Bn1[i];
    ct_c[i] = Cn0[i];
    ct_c[8 + i] = Cn1[i];
  }
  op.run(ct_a, ct_b, ct_c);
#pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    Cn0[i] = ct_c[i];
    Cn1[i] = ct_c[8 + i];
  }
}

struct NaxMaxOp {
  METAL_FUNC static float apply(float a, float b) { return metal::max(a, b); }
};
struct NaxSumOp {
  METAL_FUNC static float apply(float a, float b) { return a + b; }
};

// Reduce each of the lane's 2 rows across the 4 lanes sharing it.  The row
// index depends only on lane bits {4,2,1}, so the 4 lanes of a row differ in
// bits {0,3} -> xor masks 1 then 8 (same butterfly as the tiled kernel).
template <typename Op>
METAL_FUNC void nax_row_reduce(const thread nax_frag8<float>& f,
                               thread float* vals) {
#pragma clang loop unroll(full)
  for (short i = 0; i < 2; i++) {
    float r = Op::apply(Op::apply(f[i * 4 + 0], f[i * 4 + 1]),
                        Op::apply(f[i * 4 + 2], f[i * 4 + 3]));
    r = Op::apply(r, simd_shuffle_xor(r, ushort(1)));
    r = Op::apply(r, simd_shuffle_xor(r, ushort(8)));
    vals[i] = Op::apply(vals[i], r);
  }
}

template <typename T, int HEAD_SIZE, int BLOCK_SIZE>
[[kernel, max_total_threads_per_threadgroup(128)]] void paged_attention_nax(
    device T *out [[buffer(2)]],
    device const T *q [[buffer(3)]],
    device const T *k_cache [[buffer(4)]],
    device const T *v_cache [[buffer(5)]],
    const constant int &num_kv_heads [[buffer(8)]],
    const constant float &scale [[buffer(9)]],
    const constant float &softcapping [[buffer(10)]],
    device const int *block_tables [[buffer(11)]],
    device const int *context_lens [[buffer(12)]],
    const constant int &max_num_blocks_per_seq [[buffer(13)]],
    const constant int &q_stride [[buffer(15)]],
    const constant int &kv_block_stride [[buffer(16)]],
    const constant int &kv_head_stride [[buffer(17)]],
    device const float *sinks
    [[buffer(18), function_constant(nax_use_sinks)]],
    device const int *cu_seqlens_q [[buffer(19)]],
    const constant int &num_seqs [[buffer(20)]],
    const constant int &sliding_window [[buffer(21)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tgpg [[threadgroups_per_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]])
{
  constexpr int BQ = 64;       // 16 rows per simdgroup x 4 simdgroups
  constexpr int TILE_KV = 32;  // 2 K fragments of 16 tokens
  constexpr int TD = HEAD_SIZE / 16;
  static_assert(HEAD_SIZE % 16 == 0, "HEAD_SIZE must be a multiple of 16");
  static_assert(BLOCK_SIZE % 8 == 0,
                "an 8-token fragment half must never cross a block boundary");

  const int head_idx = int(tgp.x);
  const int q_block_global_idx = int(tgp.y);
  const int num_heads = int(tgpg.x);
  const int sg = int(sg_idx);

  const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
  const long kv_token_stride = long(num_kv_heads) * kv_head_stride;

  // Varlen: map global q-block -> (sequence, block-within-sequence).
  // Verbatim from pagedattention_tiled.metal with BQ=64.
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
  const int q_pos_start =
      (q_block_global_idx - (q_seq_start / BQ + seq_idx)) * BQ;
  if (q_pos_start >= cur_batch_query_len) return;

  const int seq_len = context_lens[seq_idx];
  const int context_len = seq_len - cur_batch_query_len;
  const int valid_q = min(BQ, cur_batch_query_len - q_pos_start);
  const int row_lim = valid_q - sg * 16;  // may be <= 0 (fully padded SG)

  const short2 sc = nax_coord();
  const short fm = sc.y;
  const short fn = sc.x;

  // All bases and lane offsets keep vec4 fragment loads aligned. Row-half
  // guards preserve those wide loads at partial-tile boundaries.
  using T4 = metal::vec<T, 4>;

  const device T *q_base = q
      + long(q_seq_start + q_pos_start + sg * 16) * q_stride
      + head_idx * HEAD_SIZE;
  const device T *q_row[2];
  bool q_ok[2];
#pragma clang loop unroll(full)
  for (short i = 0; i < 2; i++) {
    q_row[i] = q_base + (fm + i * 8) * q_stride + fn;
    q_ok[i] = (fm + i * 8) < row_lim;
  }
  const long bt_row = long(seq_idx) * max_num_blocks_per_seq;

  // Sinks enter the initial denominator-only softmax state.
  float max_score[2];
  float sum_score[2] = {0.0f, 0.0f};
  if (nax_use_sinks) {
    const float s = M_LOG2E_F * sinks[head_idx];
    max_score[0] = s; max_score[1] = s;
    sum_score[0] = 1.0f; sum_score[1] = 1.0f;
  } else {
    max_score[0] = NAX_FINITE_MIN; max_score[1] = NAX_FINITE_MIN;
  }

  nax_frag8<float> Otile[TD];
#pragma clang loop unroll(full)
  for (short d = 0; d < TD; d++) Otile[d] = nax_frag8<float>(0.0f);

  const float scale_log2 = scale * M_LOG2E_F;
  const int min_q_abs_sg = context_len + q_pos_start + sg * 16;

  // Sliding window: tiles before every row's window are fully masked -> skip.
  int kb_start = 0;
  if (sliding_window >= 0) {
    kb_start = max(0, (context_len + q_pos_start + 1 - sliding_window)
                          / TILE_KV);
  }
  const int num_kv_tiles = (seq_len + TILE_KV - 1) / TILE_KV;

  for (int kb = kb_start; kb < num_kv_tiles; kb++) {
    const int tile_start = kb * TILE_KV;
    if (tile_start > context_len + q_pos_start + valid_q - 1) break;

    // Resolve the tile's 4 half-bases (one block lookup each) and fold in
    // this lane's row, so the load loops are pure `ptr + constant` vec4s.
    const device T *k_row[4];
    const device T *v_row[4];
    bool kv_ok[4];
#pragma clang loop unroll(full)
    for (short h = 0; h < 4; h++) {
      const int tok = tile_start + h * 8;
      long off = 0;
      short rows = 0;
      if (tok < seq_len) {
        const long pb = long(block_tables[bt_row + tok / BLOCK_SIZE]);
        off = pb * long(kv_block_stride)
            + long(tok % BLOCK_SIZE) * kv_token_stride
            + kv_head_idx * long(kv_head_stride);
        rows = short(min(8, seq_len - tok));
      }
      const long lane_off = off + fm * kv_token_stride + fn;
      k_row[h] = k_cache + lane_off;
      v_row[h] = v_cache + lane_off;
      kv_ok[h] = fm < rows;
    }

    // Q is re-read per KV tile to reduce register pressure.
    nax_frag8<float> S0 = nax_frag8<float>(0.0f);
    nax_frag8<float> S1 = nax_frag8<float>(0.0f);
#pragma clang loop unroll_count(4)
    for (short id = 0; id < TD; id++) {
      nax_frag8<T> Qf, K0, K1;
#pragma clang loop unroll(full)
      for (short i = 0; i < 2; i++) {
        const T4 qv = q_ok[i]
            ? *(const device T4 *)(q_row[i] + id * 16) : T4(T(0));
        // K frag 0 rows 0-15 = halves {0,1}; frag 1 rows 16-31 = {2,3}.
        const T4 k0 = kv_ok[i]
            ? *(const device T4 *)(k_row[i] + id * 16) : T4(T(0));
        const T4 k1 = kv_ok[2 + i]
            ? *(const device T4 *)(k_row[2 + i] + id * 16) : T4(T(0));
#pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
          Qf[i * 4 + j] = qv[j];
          K0[i * 4 + j] = k0[j];
          K1[i * 4 + j] = k1[j];
        }
      }
      nax_mma<float, T, T, true>(S0, S1, Qf, K0, K1);
    }

    // Scale + softcap + mask.  Fast path when the whole tile is strictly
    // causal-past for every row this SG owns and needs no other masking.
    const bool tile_no_mask = (tile_start + TILE_KV - 1) < min_q_abs_sg
        && (tile_start + TILE_KV) <= seq_len
        && softcapping <= 0.0f
        && sliding_window < 0
        && row_lim >= 16;
    if (tile_no_mask) {
#pragma clang loop unroll(full)
      for (short e = 0; e < 8; e++) {
        S0[e] *= scale_log2;
        S1[e] *= scale_log2;
      }
    } else {
#pragma clang loop unroll(full)
      for (short i = 0; i < 2; i++) {
        const int q_abs = context_len + q_pos_start + sg * 16 + fm + i * 8;
        const bool row_masked = (fm + i * 8) >= row_lim;
#pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
#pragma clang loop unroll(full)
          for (short f = 0; f < 2; f++) {
            const int kv_pos = tile_start + f * 16 + fn + j;
            float s = (f == 0 ? S0[i * 4 + j] : S1[i * 4 + j]) * scale_log2;
            if (softcapping > 0.0f) {
              const float s_orig = s / M_LOG2E_F;
              s = softcapping * precise::tanh(s_orig / softcapping)
                  * M_LOG2E_F;
            }
            bool masked = row_masked
                || (kv_pos > q_abs)
                || (kv_pos >= seq_len);
            if (sliding_window >= 0)
              masked = masked || (kv_pos < q_abs + 1 - sliding_window);
            const float v = masked ? NAX_FINITE_MIN : s;
            if (f == 0) S0[i * 4 + j] = v; else S1[i * 4 + j] = v;
          }
        }
      }
    }

    // Online softmax, finite-sentinel form: masked logits are NAX_FINITE_MIN
    // and exp gets a zero select.  The select is what keeps rows whose
    // sliding window has not opened yet exact; their running max is still
    // the sentinel, and exp2(s - max) would otherwise be exp2(0) = 1.
    float new_max[2] = {max_score[0], max_score[1]};
    nax_row_reduce<NaxMaxOp>(S0, new_max);
    nax_row_reduce<NaxMaxOp>(S1, new_max);

    float factor[2];
#pragma clang loop unroll(full)
    for (short i = 0; i < 2; i++) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
#pragma clang loop unroll(full)
    for (short i = 0; i < 2; i++) {
#pragma clang loop unroll(full)
      for (short j = 0; j < 4; j++) {
        const float s0 = S0[i * 4 + j];
        const float s1 = S1[i * 4 + j];
        S0[i * 4 + j] =
            (s0 == NAX_FINITE_MIN) ? 0.0f : fast::exp2(s0 - new_max[i]);
        S1[i * 4 + j] =
            (s1 == NAX_FINITE_MIN) ? 0.0f : fast::exp2(s1 - new_max[i]);
      }
    }

    float row_sum[2] = {0.0f, 0.0f};
    nax_row_reduce<NaxSumOp>(S0, row_sum);
    nax_row_reduce<NaxSumOp>(S1, row_sum);
#pragma clang loop unroll(full)
    for (short i = 0; i < 2; i++) {
      sum_score[i] = sum_score[i] * factor[i] + row_sum[i];
    }
#pragma clang loop unroll(full)
    for (short d = 0; d < TD; d++) {
#pragma clang loop unroll(full)
      for (short i = 0; i < 2; i++) {
#pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
          Otile[d][i * 4 + j] *= factor[i];
        }
      }
    }

    simdgroup_barrier(mem_flags::mem_none);

    // O += P @ V.  P is fp32 in registers, but the matmul2d descriptor is
    // relaxed_precision (required by the register layout nax_coord() mirrors),
    // so the tensor unit truncates it -- unlike pagedattention_tiled.metal,
    // which keeps P fp32 through the MMA and is precision-neutral by contract.
    // That divergence is what tools/nax_prefill_parity.py bounds.
    // V is re-read per d-pair like MLX; the bases are already resolved.
#pragma clang loop unroll(full)
    for (short id = 0; id < TD; id += 2) {
      if (TD >= 8 && id == TD / 2) {
        // At TD=8 this is the MLX hd=128 scheduling barrier verbatim. The
        // vLLM hd=256/512 extensions use the same midpoint split at TD=16/32.
        threadgroup_barrier(mem_flags::mem_none);
      }
#pragma clang loop unroll(full)
      for (short f = 0; f < 2; f++) {
        nax_frag8<T> V0, V1;
#pragma clang loop unroll(full)
        for (short i = 0; i < 2; i++) {
          const short h = 2 * f + i;
          const T4 v0 = kv_ok[h]
              ? *(const device T4 *)(v_row[h] + id * 16) : T4(T(0));
          const T4 v1 = kv_ok[h]
              ? *(const device T4 *)(v_row[h] + (id + 1) * 16) : T4(T(0));
#pragma clang loop unroll(full)
          for (short j = 0; j < 4; j++) {
            V0[i * 4 + j] = v0[j];
            V1[i * 4 + j] = v1[j];
          }
        }
        nax_mma<float, float, T, false>(
            Otile[id], Otile[id + 1], (f == 0 ? S0 : S1), V0, V1);
      }
    }
  }

  // Normalize and store valid rows only.
  threadgroup_barrier(mem_flags::mem_none);
  device T *out_base = out
      + long(q_seq_start + q_pos_start + sg * 16) * q_stride
      + head_idx * HEAD_SIZE;
#pragma clang loop unroll(full)
  for (short i = 0; i < 2; i++) {
    if (q_ok[i]) {
      // Match tiled attention's zero-denominator guard for fully masked rows.
      const float rcp = 1.0f / (sum_score[i] + 1e-6f);
      device T *out_row = out_base + (fm + i * 8) * q_stride + fn;
#pragma clang loop unroll(full)
      for (short d = 0; d < TD; d++) {
        T4 ov;
#pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
          ov[j] = static_cast<T>(Otile[d][i * 4 + j] * rcp);
        }
        *(device T4 *)(out_row + d * 16) = ov;
      }
    }
  }
}

#define instantiate_paged_attention_nax(type, head_size, block_size)          \
  template [[host_name("paged_attention_nax_" #type                          \
                       "_hs" #head_size "_bs" #block_size)]]                 \
  [[kernel]] void paged_attention_nax<type, head_size, block_size>(          \
      device type *out [[buffer(2)]],                                        \
      device const type *q [[buffer(3)]],                                    \
      device const type *k_cache [[buffer(4)]],                              \
      device const type *v_cache [[buffer(5)]],                              \
      const constant int &num_kv_heads [[buffer(8)]],                        \
      const constant float &scale [[buffer(9)]],                             \
      const constant float &softcapping [[buffer(10)]],                      \
      device const int *block_tables [[buffer(11)]],                         \
      device const int *context_lens [[buffer(12)]],                         \
      const constant int &max_num_blocks_per_seq [[buffer(13)]],             \
      const constant int &q_stride [[buffer(15)]],                           \
      const constant int &kv_block_stride [[buffer(16)]],                    \
      const constant int &kv_head_stride [[buffer(17)]],                     \
      device const float *sinks                                              \
      [[buffer(18), function_constant(nax_use_sinks)]],                      \
      device const int *cu_seqlens_q [[buffer(19)]],                         \
      const constant int &num_seqs [[buffer(20)]],                           \
      const constant int &sliding_window [[buffer(21)]],                     \
      uint3 tgp [[threadgroup_position_in_grid]],                            \
      uint3 tgpg [[threadgroups_per_grid]],                                  \
      uint sg_idx [[simdgroup_index_in_threadgroup]]);

#define instantiate_paged_attention_nax_all(type)                            \
  instantiate_paged_attention_nax(type, 64, 8);                              \
  instantiate_paged_attention_nax(type, 64, 16);                             \
  instantiate_paged_attention_nax(type, 64, 32);                             \
  instantiate_paged_attention_nax(type, 128, 8);                             \
  instantiate_paged_attention_nax(type, 128, 16);                            \
  instantiate_paged_attention_nax(type, 128, 32);                            \
  instantiate_paged_attention_nax(type, 256, 8);                             \
  instantiate_paged_attention_nax(type, 256, 16);                            \
  instantiate_paged_attention_nax(type, 256, 32);                            \
  instantiate_paged_attention_nax(type, 96, 8);                              \
  instantiate_paged_attention_nax(type, 96, 16);                             \
  instantiate_paged_attention_nax(type, 96, 32);                             \
  instantiate_paged_attention_nax(type, 512, 8);                             \
  instantiate_paged_attention_nax(type, 512, 16);                            \
  instantiate_paged_attention_nax(type, 512, 32);

instantiate_paged_attention_nax_all(half);
instantiate_paged_attention_nax_all(bfloat16_t);
