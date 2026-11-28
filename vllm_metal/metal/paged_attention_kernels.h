// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "nax_prefill_work.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

// Host-side descriptions of the shared shaders. Stream ownership, feature
// eligibility and pipeline-cache keys remain in each framework's launcher.
namespace vllm_metal::kernels {

// Tiled kernel: Flash-Attention-style with simdgroup 8×8 MMA.
// One TileConfig per supported HEAD_SIZE. NUM_THREADS = NUM_SG * 32.
struct TileConfig {
  int BQ;
  int TILE_KV;
  int NUM_THREADS;
  // Simdgroups per 8-row group; > 1 splits the O columns between them
  // (see D_SPLIT in pagedattention_tiled.metal).
  int D_SPLIT = 1;
};

// ─ How to add a new HEAD_SIZE ────────────────────────────────────────────
// Budget: smem <= 32 KB (Apple Silicon per-threadgroup memory limit, M1-M4).
// Formula:
//   smem = (BQ + 2*TILE_KV) * (HEAD_SIZE + 8) * 2 bytes
//   where  BQ+2*TILE_KV = Q-rows + K-rows + V-rows
//          HEAD_SIZE+8  = row stride (+8 = SMEM_PAD for bank-conflict
//                                     avoidance; see pagedattention_tiled.metal)
//          *2           = sizeof(bf16/half); fp8 KV would change this
//
// Constraints on (BQ, TILE_KV):
//   BQ <= 2*TILE_KV       (O_smem fp32 reuses Q+K+V region at kernel exit)
//   BQ / NUM_SG == 8      (each simdgroup owns 8 Q rows; 8x8 MMA fragment)
//   HEAD_SIZE, TILE_KV multiples of 8
// NUM_THREADS = NUM_SG * 32 (one Apple simdgroup = 32 lanes).
//
// HEAD_SIZE -> (BQ, TILE_KV, NUM_THREADS, NUM_SG, smem):
//   64, 96, 128 -> (32, 32, 128, 4, 24-26 KB)
//   256         -> (16, 16,  64, 2,   25.3 KB)
//   512         -> ( 8,  8, 128, 4,   25.9 KB)  // D_SPLIT=4: four simdgroups share
//                                              // the 8 rows; each computes S over
//                                              // its 128 columns of Q/K (partials
//                                              // summed through a 1 KB exchange)
//                                              // and owns 128 O columns
// 80, 112 excluded by HD_TILES % NUM_SG(4) == 0.
// ─────────────────────────────────────────────────────────────────────────
inline std::optional<TileConfig> select_tile_config(int head_size) {
  switch (head_size) {
    case 64: case 96: case 128:
      return TileConfig{32, 32, 128};
    case 256:
      return TileConfig{16, 16, 64};
    case 512:
      return TileConfig{8, 8, 128, 4};
    default:
      return std::nullopt;
  }
}

inline constexpr int kNaxThreads = 128;

struct NaxConfig {
  int BQ;
  int HEADS_PER_TG;
  int NUM_THREADS;
  int GQA_RATIO = 1;
};

inline constexpr NaxConfig kNaxBaseline{kNaxBQ, 1, kNaxThreads};

// The caller has already selected ordinary causal NAX for one sequence.
inline NaxConfig select_nax_config(
    const NaxGeometry& geometry, NaxSequence sequence, bool upstream_split, int cores) {
  if (cores <= 0 || sequence.query_tokens <= 1 || sequence.kv_tokens < sequence.query_tokens ||
      geometry.qk_dim != 256 || geometry.value_dim != 256 || geometry.heads <= 0 ||
      geometry.kv_heads <= 0 || geometry.heads % geometry.kv_heads != 0) return kNaxBaseline;
  const int ratio = geometry.heads / geometry.kv_heads;
  if (!((ratio == 6 && geometry.kernel_page == 16) ||
        (ratio == 8 && geometry.kernel_page == 32))) return kNaxBaseline;
  if (geometry.physical_page < geometry.kernel_page || geometry.physical_page % geometry.kernel_page ||
      (geometry.key_head_stride != geometry.qk_dim && geometry.key_head_stride != 2 * geometry.qk_dim) ||
      geometry.key_token_stride != geometry.kv_heads * geometry.key_head_stride) return kNaxBaseline;

  const int width = nax_prefill_width(sequence, geometry.heads, ratio, upstream_split, cores);
  return width == 1 ? kNaxBaseline : NaxConfig{kNaxGroupedBQ, width, 32 * width, ratio};
}

inline constexpr NaxConfig nax_config_for_pipeline(
    NaxConfig cfg, size_t max_threads) {
  return cfg.HEADS_PER_TG > 1 && static_cast<size_t>(cfg.NUM_THREADS) > max_threads
      ? kNaxBaseline : cfg;
}

inline std::string tiled_name(
    const std::string& dtype, int head_size, int block_size, TileConfig cfg) {
  return "paged_attention_tiled_" + dtype +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_bq" + std::to_string(cfg.BQ) +
      "_tk" + std::to_string(cfg.TILE_KV) +
      "_nt" + std::to_string(cfg.NUM_THREADS) +
      "_ds" + std::to_string(cfg.D_SPLIT);
}

inline size_t tiled_shared_bytes(
    TileConfig cfg, int head_size, int t_size) {
  // S, O, m, l are register-resident, so no S/O/M/L threadgroup buffers.
  // Output staging reuses Q_smem as fp32 O_smem at exit; fits because
  // BQ*LD*4 <= (BQ+2*TILE_KV)*LD*2  <=>  BQ <= 2*TILE_KV.
  // A1: leading dim padded by 16 B for bank-conflict avoidance —
  // smem_pad/ld MUST match SMEM_PAD/LD in pagedattention_tiled.metal.
  const int smem_pad = 16 / t_size;
  const int ld       = head_size + smem_pad;
  size_t shmem = static_cast<size_t>(
      (cfg.BQ + 2 * cfg.TILE_KV) * ld * t_size);  // Q + K + V, padded
  if (cfg.D_SPLIT > 1) {
    // Partial-S exchange: [BQ/8 row groups][D_SPLIT][TILE_KV/8] fp32 8×8
    // fragments (mirrors S_xchg in pagedattention_tiled.metal).
    shmem += static_cast<size_t>(
        (cfg.BQ / 8) * cfg.D_SPLIT * (cfg.TILE_KV / 8) * 64 * sizeof(float));
  }
  return shmem;
}

inline std::string nax_name(
    const std::string& dtype, int head_size, int block_size,
    NaxConfig cfg = kNaxBaseline) {
  std::string name = "paged_attention_nax_" + dtype +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size);
  if (cfg.HEADS_PER_TG > 1) {
    name += "_gqa" + std::to_string(cfg.GQA_RATIO) +
        "_h" + std::to_string(cfg.HEADS_PER_TG) + "_full";
  }
  return name;
}

// Engage the split while the base decode grid (num_q_heads * num_seqs) stays
// below ~8 threadgroups per GPU core.  On a 14-core M1 Pro that is 112.  At 8K
// context with the fixed 512-token (16-way) split, this regime measured:
// conc=1 -33%, conc=2 -6.7%, conc=4 -5.3%, fading to ~-1.6% (noise) by conc=8 —
// so the split stays on through ~conc 6 and disengages beyond, where it stops
// paying.  Scales with core count.
// (An adaptive runtime split count was tried and reverted: this kernel is
// memory-bound, so it likes oversubscription — fewer splits hurt, and ~512-token
// partitions are the sweet spot, so a fixed size + a wider gate is simpler and
// just as fast.)
inline constexpr int kDecodeGroupsPerCore = 8;

inline bool should_split_decode(
    int heads, int query_tokens, int partitions, int gpu_cores) {
  return heads * query_tokens < gpu_cores * kDecodeGroupsPerCore
      && partitions >= 2;
}

inline size_t paged_threadgroup_bytes(
    int head_size, int block_size, int element_bytes,
    bool window, int window_rows) {
  constexpr int NUM_THREADS = 256;
  constexpr int NUM_SIMD_LANES = 32;
  constexpr int NUM_WARPS      = NUM_THREADS / NUM_SIMD_LANES;
  // Window mode widens the per-warp score slices to one BLOCK_SIZE slice
  // per row and stages the sub-window's query rows after them; the layout
  // must mirror the kernel's shared_mem carve exactly.
  const int rows_per_tg = window ? window_rows : 1;
  int warp_scores_bytes = NUM_WARPS * rows_per_tg * block_size
                          * static_cast<int>(sizeof(float));
  int q_window_bytes = window
      ? window_rows * head_size * element_bytes
      : 0;
  int merge_bytes = (2 * NUM_WARPS + NUM_WARPS * head_size)
                    * static_cast<int>(sizeof(float));
  size_t shmem = static_cast<size_t>(
      std::max(warp_scores_bytes + q_window_bytes, merge_bytes));
  shmem = (shmem + 15) & ~size_t(15);
  return shmem;
}

inline std::string paged_name(
    const std::string& dtype, const std::string& k_cache_dtype,
    const std::string& v_cache_dtype, int head_size, int block_size,
    int partition_size) {
  return "paged_attention_" + dtype + "_cache_" + k_cache_dtype + "_" + v_cache_dtype +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_nt256_nsl32_ps" + std::to_string(partition_size);
}

inline std::string reshape_and_cache_name(
    const std::string& dtype, const std::string& cache_dtype) {
  return "reshape_and_cache_kv_" + dtype + "_cache_" + cache_dtype;
}

inline std::string paged_reduce_kernel_name(
    const std::string& dtype, int head_size, int partition_size) {
  return "paged_attention_v2_reduce_" + dtype + "_hs" + std::to_string(head_size)
      + "_nt256_nsl32_ps" + std::to_string(partition_size);
}

inline size_t paged_reduce_threadgroup_bytes(int64_t num_partitions) {
  // Two FP32 statistics per partition; Metal requires 16-byte alignment.
  return (static_cast<size_t>(num_partitions) * 2 * sizeof(float) + 15) &
      ~size_t(15);
}

}  // namespace vllm_metal::kernels
