// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace vllm_metal::kernels {

inline constexpr int kNaxBQ = 64;
inline constexpr int kNaxGroupedBQ = 16;
inline constexpr int kNaxKvTile = 32;
inline constexpr double kNaxGuardOverhead = 0.2;

struct NaxSequence {
  int query_tokens;
  int kv_tokens;

  int64_t kv_tiles() const {
    return (int64_t(kv_tokens) + kNaxKvTile - 1) / kNaxKvTile;
  }
};

struct NaxGeometry {
  int heads;
  int kv_heads;
  int qk_dim;
  int value_dim;
  int kernel_page;
  int physical_page;
  int64_t key_head_stride;
  int64_t key_token_stride;
};

// One visit is one 16-query-row SIMD group processing one 32-token KV tile.
struct NaxPrefillWork {
  double visits;
  double guarded_visits;
  int64_t query_blocks;

  double weighted_visits() const {
    return visits + kNaxGuardOverhead * guarded_visits;
  }
};

// Sum ceil((prefix + i * query_tile) / kv_tile), i=1..blocks. The two
// compiled Q tiles need at most two residue classes, independent of Q length.
inline int64_t nax_full_tile_visits(
    int64_t blocks, int64_t prefix, int query_tile, int kv_tile) {
  const int divisor = std::gcd(query_tile, kv_tile);
  const int period = kv_tile / divisor;
  const int step = query_tile / divisor;
  int64_t visits = 0;
  for (int i = 1; i <= period && i <= blocks; ++i) {
    const int64_t count = (blocks - i) / period + 1;
    const int64_t first = (prefix + int64_t(i) * query_tile + kv_tile - 1) / kv_tile;
    visits += count * first + int64_t(step) * count * (count - 1) / 2;
  }
  return visits;
}

inline NaxPrefillWork nax_sequence_work(
    NaxSequence sequence, int heads, int bq, bool split) {
  const int q = sequence.query_tokens;
  const int prefix = sequence.kv_tokens - q;
  constexpr int bkv = kNaxKvTile;
  const int64_t full = q / bq, tail = q % bq;
  const int64_t full_visits = nax_full_tile_visits(full, prefix, bq, bkv);
  const int64_t tail_scan = tail ? sequence.kv_tiles() : 0;
  const int64_t full_end = int64_t(sequence.kv_tokens / bkv) * bkv;
  const int64_t edge_blocks = full - std::min(full,
      std::max(int64_t(0), full_end - prefix) / bq);
  const double simd_groups = double(heads) * (bq / 16);
  return {
      simd_groups * (full_visits + tail_scan),
      simd_groups * ((split ? edge_blocks : full_visits) + tail_scan),
      full + (tail != 0)};
}

// Calibrated for BF16 D256 with GQA6/GQA8. See vllm-project/vllm-metal#1300.
inline int nax_prefill_width(
    NaxSequence sequence, int heads, int ratio, bool upstream_split, int cores) {
  constexpr double kTilesPerCore = 8.;
  constexpr double kKvReadCost = 1. / 3.;
  constexpr double kWideGroupCost = 2.;
  constexpr double kMinimumGain = 0.10;
  const auto bulk = [&](double tile_work, int width, double penalty = 0.) {
    return tile_work * (1. + kKvReadCost / width) * (1. + penalty) / (kTilesPerCore * cores);
  };

  const double scan = double(sequence.kv_tiles());
  const double guarded_scan = scan * (1. + kNaxGuardOverhead);
  const double ordinary_scan = upstream_split && sequence.query_tokens % kNaxBQ == 0
      ? scan : guarded_scan;
  const double grouped_scan = sequence.query_tokens % kNaxGroupedBQ == 0 ? scan : guarded_scan;
  const auto ordinary = nax_sequence_work(sequence, heads, kNaxBQ, upstream_split);
  const double ordinary_cost = std::max(bulk(ordinary.weighted_visits(), 1), ordinary_scan);
  // Half-width bulk is at most 5/6 (GQA6) or 13/16 (GQA8) of ordinary bulk.
  // All grouped widths share this scan floor, so it alone can veto admission.
  if (grouped_scan >= (1. - kMinimumGain) * ordinary_cost) return 1;

  const auto grouped = nax_sequence_work(sequence, heads, kNaxGroupedBQ, true);
  const double tile_work = grouped.weighted_visits();
  // Half width has cheaper bulk than h2. At the scan floor they tie and h2 wins.
  if (bulk(tile_work, 2) <= grouped_scan) return 2;

  // Full width can beat half width only in the throughput-limited region.
  const int half_width = ratio / 2;
  const int64_t full_groups = grouped.query_blocks * (heads / ratio);
  const double extra_rows = double(kNaxGroupedBQ * ratio) / kNaxBQ - 1.;
  const double penalty = kWideGroupCost * extra_rows * extra_rows * cores / full_groups;
  return bulk(tile_work, ratio, penalty) < bulk(tile_work, half_width) ? ratio : half_width;
}

}  // namespace vllm_metal::kernels
