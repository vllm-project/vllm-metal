// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

// Host-side descriptions of the shared shaders. Stream ownership, feature
// selection and pipeline-cache keys remain in each framework's launcher.
namespace vllm_metal::kernels {

// Tiled kernel: Flash-Attention-style with simdgroup 8×8 MMA.
// One TileConfig per supported HEAD_SIZE. NUM_THREADS = NUM_SG * 32.
struct TileConfig {
  int BQ;
  int TILE_KV;
  int NUM_THREADS;
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
//   512         -> ( 8,  8,  32, 1,   24.9 KB)  // no in-threadgroup SG parallelism
// 80, 112 excluded by HD_TILES % NUM_SG(4) == 0.
// ─────────────────────────────────────────────────────────────────────────
inline std::optional<TileConfig> select_tile_config(int head_size) {
  switch (head_size) {
    case 64: case 96: case 128:
      return TileConfig{32, 32, 128};
    case 256:
      return TileConfig{16, 16, 64};
    case 512:
      return TileConfig{8, 8, 32};
    default:
      return std::nullopt;
  }
}

inline constexpr int kNaxBQ = 64;
inline constexpr int kNaxThreads = 128;

inline std::string tiled_name(
    const std::string& dtype, int head_size, int block_size, TileConfig cfg) {
  return "paged_attention_tiled_" + dtype +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_bq" + std::to_string(cfg.BQ) +
      "_tk" + std::to_string(cfg.TILE_KV) +
      "_nt" + std::to_string(cfg.NUM_THREADS);
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
  return static_cast<size_t>(
      (cfg.BQ + 2 * cfg.TILE_KV) * ld * t_size);  // Q + K + V, padded
}

inline std::string nax_name(
    const std::string& dtype, int head_size, int block_size) {
  return "paged_attention_nax_" + dtype +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size);
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
