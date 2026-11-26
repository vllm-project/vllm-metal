// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace vllm_metal::kernels {

inline constexpr int kNaxBQ = 64;
inline constexpr int kNaxKvTile = 32;

struct NaxSequence {
  int query_tokens;
  int kv_tokens;
};

struct NaxLayout {
  int query_tile;
  int kv_tile;
  int heads_per_group;
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
  int64_t groups;
  int64_t longest_full_scan;
  int64_t tail_scan;
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
    NaxSequence sequence, int heads, NaxLayout layout, bool split) {
  const int q = sequence.query_tokens;
  const int prefix = sequence.kv_tokens - q;
  const int bq = layout.query_tile, bkv = layout.kv_tile;
  const int64_t full = q / bq, tail = q % bq;
  const int64_t full_visits = nax_full_tile_visits(full, prefix, bq, bkv);
  const int64_t tail_scan = tail ? (int64_t(sequence.kv_tokens) + bkv - 1) / bkv : 0;
  const int64_t full_end = int64_t(sequence.kv_tokens / bkv) * bkv;
  const int64_t edge_blocks = full - std::min(full,
      std::max(int64_t(0), full_end - prefix) / bq);
  const double simd_groups = double(heads) * (bq / 16);
  return {
      simd_groups * (full_visits + tail_scan),
      simd_groups * ((split ? edge_blocks : full_visits) + tail_scan),
      (heads / layout.heads_per_group) * (full + (tail != 0)),
      full ? (prefix + full * bq + bkv - 1) / bkv : 0,
      tail_scan};
}

inline double nax_prefill_cost(
    NaxSequence sequence, const NaxGeometry& geometry, NaxLayout layout,
    bool split, int cores) {
  const auto work = nax_sequence_work(sequence, geometry.heads, layout, split);
  // Fit to paired fixed-width D256 timings (https://github.com/vllm-project/vllm-metal/pull/1300).
  constexpr double kGuardOverhead = 0.2;
  constexpr double kTilesPerCore = 8.;
  constexpr double kKvReadCost = 1. / 3.;
  constexpr double kGroupsPerCore = 5.;
  constexpr double kWideTileCost = 0.4;
  constexpr double kWideScanCost = 0.1;

  // A wider group shares KV reads across heads but holds more live Q rows.
  // D256 is common to all candidates, so state scales with rows per group.
  const double width = layout.heads_per_group;
  const double extra_state = std::max(0., layout.query_tile * width / kNaxBQ - 1.);
  const double scarcity = std::min(1., kGroupsPerCore * cores / work.groups);
  const double tile_work = work.visits + kGuardOverhead * work.guarded_visits;
  const double bulk = tile_work * (1. + kKvReadCost / width)
      * (1. + kWideTileCost * extra_state * extra_state * scarcity)
      / (kTilesPerCore * cores);

  // More groups cannot shorten the longest serial KV scan of any one tile.
  const double full_scan = work.longest_full_scan * (split ? 1. : 1. + kGuardOverhead);
  const double tail_scan = work.tail_scan * (1. + kGuardOverhead);
  const double serial = std::max(full_scan, tail_scan) * (1. + kWideScanCost * extra_state);
  return std::max(bulk, serial);
}

}  // namespace vllm_metal::kernels
