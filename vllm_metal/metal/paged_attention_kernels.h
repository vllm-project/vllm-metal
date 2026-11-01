// SPDX-License-Identifier: Apache-2.0
#pragma once

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

struct NaxConfig {
  int BQ;
  int HEADS_PER_TG;
  int NUM_THREADS;
};

inline constexpr NaxConfig kNaxBaseline{kNaxBQ, 1, kNaxThreads};

// A shared source-work score for compiled grouped layouts. Complete SG work
// supplies a capacity budget; whole Q tails and isolated KV edges are separate.
// Coefficients are conservative empirical weights, not predicted milliseconds.
// Calibration terms, measured counterexamples and hardware scope are recorded
// in docs/gqa-prefill.md; keep kernel and serving evidence separate.
// In particular, idle cores cannot amplify the baseline tail-work credit.
struct NaxPrefillWork {
  double issued, active, guarded, guarded_q, kv_edge, fast;
  double stream, tail_stream;
  int64_t groups, full_blocks;
};
inline NaxPrefillWork nax_prefill_work(int q, int prefix, int heads, int kv_heads, int bq, bool split) {
  const int64_t full = q / bq, tail = q % bq;
  const int64_t prefix_tiles = (int64_t(prefix) + 31) / 32;
  const int64_t visits = bq == 64
      ? full * prefix_tiles + full * (full + 1)
      : full * prefix_tiles + full * full / 4
        + (prefix % 32 == 0 || prefix % 32 > 16) * ((full + 1) / 2);
  const int64_t end = tail ? (int64_t(prefix) + q + 31) / 32 : 0;
  const int64_t full_end = ((int64_t(prefix) + q) / 32) * 32;
  // C++ integer division truncates toward zero. Clamp before division so
  // this matches the nonnegative source-loop count even below one KV tile.
  const int64_t edge_blocks = full - std::min(full,
      std::max(int64_t(0), full_end - prefix) / bq);
  const int64_t fast_visits = split ? visits - edge_blocks : 0;
  const int slices = bq / 16;
  const double issued = double(heads) * slices * (visits + end);
  const double active = double(heads) * (slices * visits + ((tail + 15) / 16) * end);
  const double fast = double(heads) * slices * fast_visits;
  return {issued, active, issued - fast, double(heads) * slices * end,
          double(heads) * slices * edge_blocks, fast,
          double(visits + end) / (full + (tail != 0)), double(end),
          (bq == 64 ? heads : kv_heads) * (full + (tail != 0)), full};
}

struct NaxGqaScore {
  bool eligible;
  bool admit;
  double value;
  int heads_per_tg;
};

inline double nax_gqa_split_score(
    const NaxPrefillWork& baseline, const NaxPrefillWork& grouped,
    int heads_per_group, double head_growth, double padding_savings,
    double critical_savings) {
  constexpr double kPaddingBenefit = 13;
  constexpr double kCriticalPathBenefit = 8;
  constexpr double kFullTileBenefit = 2.5;
  constexpr double kGuardedQueryBenefit = 15;
  constexpr double kStreamCost = 10;
  constexpr double kExtraStateCost = 3.5;
  constexpr double kGuardedQueryCost = 16;

  const double extra_heads = std::max(0., head_growth);
  const double benefit = kPaddingBenefit * padding_savings
      + kCriticalPathBenefit * critical_savings + kFullTileBenefit * grouped.fast
      + kGuardedQueryBenefit * (baseline.guarded_q - grouped.guarded_q);
  const double cost = (kStreamCost / heads_per_group
      + kExtraStateCost * extra_heads * extra_heads) * grouped.issued
      + kGuardedQueryCost * grouped.guarded_q;
  return (benefit - cost) / baseline.issued;
}

inline double nax_gqa_guarded_score(
    const NaxPrefillWork& baseline, const NaxPrefillWork& grouped,
    int heads, int gpu_cores, double head_growth,
    double critical_savings, double memory_work_fraction) {
  constexpr double kPartialQueryBenefit = 4;
  constexpr double kCriticalPathBenefit = .875;
  constexpr double kReuseBenefit = 3.5;
  constexpr double kGuardedTileBenefit = 4;
  constexpr double kGridCost = 10.5;
  constexpr double kExtraGridCost = 32;
  constexpr double kStreamCost = 3;
  constexpr double kExtraStreamCost = 3.5;
  constexpr double kCoreCost = 11.5;
  constexpr double kKvEdgeCost = 2;
  constexpr double kExtraStateCost = .75;
  constexpr double kGuardedQueryCost = 5;

  const double extra_heads = std::max(0., head_growth);
  const double partial_savings = (baseline.guarded_q - grouped.guarded_q)
      * std::min(1., double(gpu_cores)
          / std::max(int64_t(1), heads * baseline.full_blocks));
  const double reuse = head_growth
      * std::min(1., double(gpu_cores) / baseline.groups) * grouped.issued;
  const double lost_grid_work = std::max(int64_t(0),
      std::min(int64_t(gpu_cores), baseline.groups)
      - std::min(int64_t(gpu_cores), grouped.groups)) * grouped.stream;
  const double benefit = kPartialQueryBenefit * partial_savings
      + kCriticalPathBenefit * critical_savings + kReuseBenefit * reuse
      + kGuardedTileBenefit * (baseline.guarded - baseline.guarded_q);
  double cost = (kGridCost + kExtraGridCost * extra_heads * extra_heads
      * std::min(1., std::max(0., double(baseline.groups) / gpu_cores - 1)))
      * lost_grid_work;
  cost += (kStreamCost + kExtraStreamCost * extra_heads
      * (1 - memory_work_fraction)) * grouped.issued
      + kCoreCost * gpu_cores + kKvEdgeCost * grouped.kv_edge;
  cost += kExtraStateCost * extra_heads * extra_heads * grouped.issued
      * std::min(1., double(grouped.groups) / gpu_cores)
      + kGuardedQueryCost * grouped.guarded_q;
  return (benefit - cost) / baseline.issued;
}

inline NaxGqaScore nax_gqa_work_score(
    int q, int prefix, int heads, int kv_heads, int page,
    int cores, bool upstream_split) {
  constexpr int kGroupedQueryTile = 16;
  constexpr int kSupportedRatio = 6;
  constexpr int kSupportedPageSize = 16;
  constexpr int kNarrowHeadLimit = 4;
  constexpr int kGuardedWideCores = 10;
  constexpr int kGuardedWideGroupsPerCore = 18;
  constexpr double kGuardedWideWorkPerCore = 16384;
  constexpr int kWideGroupsPerCore = 48;
  constexpr double kSaturatedWorkPerCore = 65536;
  constexpr int kDenseGroupsPerCore = 32;
  constexpr double kDenseFullFraction = .95;
  constexpr double kAdmissionMargin = .5;
  constexpr double kTailReserveWeight = 8;

  if (q <= 1 || prefix < 0 || heads <= 0 || kv_heads <= 0 ||
      heads % kv_heads || cores <= 0) return {false, false, 0, 0};
  const int ratio = heads / kv_heads;
  if (ratio != kSupportedRatio || page != kSupportedPageSize)
    return {false, false, 0, 0};
  const auto baseline = nax_prefill_work(
      q, prefix, heads, kv_heads, kNaxBQ, upstream_split);
  const auto full_group = nax_prefill_work(
      q, prefix, heads, kv_heads, kGroupedQueryTile, true);
  const double full_work = full_group.fast;
  const bool measured_stride = (kv_heads & (kv_heads - 1)) == 0;
  // The wider guarded layout regressed on larger GPUs; keep its 10-core gate.
  const bool guarded_wide = measured_stride &&
      cores == kGuardedWideCores && !upstream_split &&
      int64_t(heads) * (q / kGroupedQueryTile) >=
          int64_t(kGuardedWideGroupsPerCore) * cores &&
      full_work >= kGuardedWideWorkPerCore * cores;
  double memory_work_fraction =
      std::min(1., full_work / (kSaturatedWorkPerCore * cores));
  if (guarded_wide) memory_work_fraction = 1.;
  const int head_limit =
      int64_t(heads) * (q / kGroupedQueryTile) >=
          int64_t(kWideGroupsPerCore) * cores ||
      full_work >= kSaturatedWorkPerCore * cores || guarded_wide
          ? kSupportedRatio : kNarrowHeadLimit;
  const bool dense_full = measured_stride && upstream_split &&
      int64_t(kv_heads) * (q / kGroupedQueryTile) >=
          int64_t(kDenseGroupsPerCore) * cores &&
      full_group.fast >= kDenseFullFraction * full_group.issued;
  const double tail_depth =
      q % kGroupedQueryTile ? (int64_t(prefix) + q + 31) / 32 : 0;
  const double threshold = kAdmissionMargin
      + kTailReserveWeight * cores * (ratio / 2) * tail_depth / baseline.issued;
  NaxGqaScore best{true, false, -1e300, 0};
  for (const int heads_per_group : {ratio / 2, ratio}) {
    if (heads_per_group > head_limit ||
        (heads_per_group > kNarrowHeadLimit &&
         int64_t(heads / heads_per_group) * (q / kGroupedQueryTile) < cores))
      continue;
    const auto grouped = nax_prefill_work(
        q, prefix, heads, heads / heads_per_group, kGroupedQueryTile, true);
    const double head_growth = double(heads_per_group) / kNarrowHeadLimit - 1;
    const double padding_savings =
        baseline.issued - baseline.active - grouped.issued + grouped.active;
    const double critical_savings = std::min(int64_t(cores), int64_t(heads))
        * ((q % kNaxBQ + kGroupedQueryTile - 1) / kGroupedQueryTile)
        * baseline.tail_stream
        - std::min(int64_t(cores), baseline.groups)
        * heads_per_group * grouped.tail_stream;
    const double value = upstream_split
        ? nax_gqa_split_score(baseline, grouped, heads_per_group, head_growth,
                             padding_savings, critical_savings)
        : nax_gqa_guarded_score(baseline, grouped, heads, cores, head_growth,
                               critical_savings, memory_work_fraction);
    if (dense_full && heads_per_group == kSupportedRatio)
      return {true, true, value, heads_per_group};
    if (value > best.value)
      best = {true, value > threshold, value, heads_per_group};
  }
  return best;
}

inline NaxConfig select_nax_config(
    int num_heads, int num_kv_heads, int head_size, int block_size,
    int query_tokens, int num_seqs, int max_seq_len, int q_block_offset,
    bool grouped_enabled, bool bf16, bool unmasked, bool upstream_split,
    int gpu_cores) {
  // BQ64 mixed suffix offsets cannot address the BQ16 grouped shader.
  if (!grouped_enabled || !bf16 || !unmasked || head_size != 256 ||
      num_seqs != 1 || q_block_offset != 0 || query_tokens <= 1 ||
      max_seq_len < query_tokens) return kNaxBaseline;
  const auto choice = nax_gqa_work_score(
      query_tokens, max_seq_len - query_tokens, num_heads, num_kv_heads,
      block_size, gpu_cores, upstream_split);
  return choice.admit
      ? NaxConfig{16, choice.heads_per_tg, 32 * choice.heads_per_tg}
      : kNaxBaseline;
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
    const std::string& dtype, int head_size, int block_size,
    NaxConfig cfg = kNaxBaseline) {
  std::string name = "paged_attention_nax_" + dtype +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size);
  if (cfg.HEADS_PER_TG > 1)
    name += cfg.HEADS_PER_TG == 6 ? "_gqa6_full" : "_gqa6_sub_full";
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
