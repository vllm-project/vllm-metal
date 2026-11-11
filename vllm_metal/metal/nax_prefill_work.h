// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

namespace vllm_metal::kernels {

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

// Counts of the shader's 16-row SIMD-group tile visits. These are source
// operations, not measured memory transactions or hardware register usage.
struct NaxPrefillWork {
  double issued = 0, active = 0, guarded = 0, guarded_q = 0, kv_edge = 0, fast = 0;
  double stream = 0, tail_stream = 0;
  int64_t groups = 0, full_blocks = 0;
};

struct NaxPrefillResources {
  NaxPrefillWork work;
  double stream_footprint_bytes, logical_kv_bytes, full_stream;
  int64_t launched_groups, logical_state_bytes;
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
  NaxPrefillWork work;
  const int q = sequence.query_tokens;
  if (q <= 0) return work;
  const int prefix = sequence.kv_tokens - q;
  const int bq = layout.query_tile, bkv = layout.kv_tile;
  const int64_t full = q / bq, tail = q % bq;
  const int64_t visits = nax_full_tile_visits(full, prefix, bq, bkv);
  const int64_t end = tail ? (int64_t(sequence.kv_tokens) + bkv - 1) / bkv : 0;
  const int64_t full_end = int64_t(sequence.kv_tokens / bkv) * bkv;
  const int64_t edge_blocks = full - std::min(full,
      std::max(int64_t(0), full_end - prefix) / bq);
  const int slices = bq / 16;
  const int64_t q_blocks = full + (tail != 0);
  work.issued = double(heads) * slices * (visits + end);
  work.active = double(heads) * (slices * visits + ((tail + 15) / 16) * end);
  work.fast = split ? double(heads) * slices * (visits - edge_blocks) : 0;
  work.guarded = work.issued - work.fast;
  work.guarded_q = double(heads) * slices * end;
  work.kv_edge = double(heads) * slices * edge_blocks;
  work.stream = double(visits + end) / q_blocks;
  work.tail_stream = double(end);
  work.groups = (heads / layout.heads_per_group) * q_blocks;
  work.full_blocks = full;
  return work;
}

// Admission is single-sequence D256. Count the actual launch grid, including
// its extra (empty) group when the query length is an exact tile multiple.
inline NaxPrefillResources nax_prefill_resources(
    NaxSequence sequence, NaxGeometry geometry, NaxLayout layout, bool split) {
  const auto work = nax_sequence_work(sequence, geometry.heads, layout, split);
  const int head_groups = geometry.heads / layout.heads_per_group;
  const int full_q = sequence.query_tokens / layout.query_tile * layout.query_tile;
  const int64_t full_end = int64_t(sequence.kv_tokens) - sequence.query_tokens + full_q;
  constexpr int kBfloat16Bytes = 2;
  constexpr int kFloatBytes = 4;
  const int kv_bytes = kBfloat16Bytes * (geometry.qk_dim + geometry.value_dim);
  const int64_t rows = int64_t(layout.query_tile) * layout.heads_per_group;
  // Accumulators, Q, scores, and two FP32 softmax statistics per row.
  const int64_t state = rows * (kFloatBytes * geometry.value_dim
      + kBfloat16Bytes * geometry.qk_dim + kFloatBytes * (layout.kv_tile + 2));
  return {work, work.stream * head_groups * layout.kv_tile * kv_bytes,
      double(sequence.kv_tokens) * geometry.kv_heads * kv_bytes,
      full_q ? double((full_end + layout.kv_tile - 1) / layout.kv_tile) : 0.,
      int64_t(sequence.query_tokens / layout.query_tile + 1) * head_groups, state};
}

// Fitted on paired 10/20/40-core kernel measurements, then frozen for an
// independent shape panel. Calibration and serving results:
// https://github.com/vllm-project/vllm-metal/pull/1300
struct NaxPageRates {
  double narrow, middle, wide;
  double guarded_tile, guarded_stream;
};
inline constexpr NaxPageRates kNaxPage16Rates{
    1.0169304348373918, .9432343252369108, .7268784060030199,
    1.622904174104084, 1.2258362190278242};
inline constexpr NaxPageRates kNaxPage32Rates{
    .9920785752048182, .9680837381817905, .4508176063839598,
    1.5306808620260304, 1.4003105500708977};

inline double nax_prefill_cost(
    const NaxPrefillResources& resources, NaxGeometry geometry,
    NaxLayout layout, int cores, bool upstream_split) {
  const auto& w = resources.work;
  const auto& rates = geometry.kernel_page == 16 ? kNaxPage16Rates : kNaxPage32Rates;
  const int ratio = geometry.heads / geometry.kv_heads;
  const int width = layout.heads_per_group;
  double rate = 1.;  // Upstream BQ64 reference rate.
  if (width == 2) rate = rates.narrow;
  else if (width == ratio / 2) rate = rates.middle;
  else if (width == ratio) rate = rates.wide;

  constexpr double kGroupedTail = 1.34722704155022;
  constexpr double kUpstreamTail = 1.2468577321483032;
  constexpr double kInactiveTile = 1.4542458472707103;
  constexpr double kTilesPerCore = 9.088170462137231;
  constexpr double kWideStatePenalty = 1.326972148183182;
  constexpr double kStateGroupsPerCore = 29.185172825937;
  constexpr double kStreamStatePenalty = .08081622214154811;
  constexpr double kReferenceDimensions = 512.;  // Dq + Dv for D256.
  constexpr double kReferenceState = 64. * 1672.;  // BQ64, one head.
  constexpr double kCriticalStateFloor = .5;
  const double tail_cost = width > 1 ? kGroupedTail : kUpstreamTail;
  const double inactive = w.issued - w.active;
  const double tail = w.guarded_q - inactive;
  const double guarded = w.active - w.fast - tail;
  const double dimension = (geometry.qk_dim + geometry.value_dim) / kReferenceDimensions;
  const double state = resources.logical_state_bytes / kReferenceState;
  const double wide = std::max(0., state - 1.);

  // Total matrix work and the deepest serial KV scan constrain different grids.
  double bulk = dimension * (w.fast + rates.guarded_tile * guarded
      + tail_cost * tail + kInactiveTile * inactive) / (kTilesPerCore * cores);
  // Extra Q blocks can hide a wide group's state cost only up to the
  // calibrated limit on concurrent streams per independent KV head.
  constexpr double kGroupsPerKvHead = 512.;
  const double state_groups = std::min(
      double(w.groups), kGroupsPerKvHead * geometry.kv_heads);
  rate *= 1 + kWideStatePenalty * wide * wide
      * std::min(1., kStateGroupsPerCore * cores / state_groups);
  bulk *= rate;
  const bool split = width > 1 || upstream_split;
  double critical = dimension * std::max(
      resources.full_stream * (split ? 1. : rates.guarded_stream), w.tail_stream * tail_cost);
  critical *= 1 + kStreamStatePenalty * std::max(0., state - kCriticalStateFloor);

  // Estimate KV working-set pressure and how many streams can run in parallel.
  constexpr double kKvUnitBytes = 32768.;
  constexpr double kFootprintKnee = .0015885238571067709;
  constexpr double kFootprintExponent = .25000080683650183;
  constexpr double kGroupsPerStream = 50.82434564183909;
  constexpr double kStreamWeight = .022333392978314857;
  const double footprint = resources.logical_kv_bytes / (cores * kKvUnitBytes);
  const double stream_fraction = std::pow(
      footprint / (footprint + kFootprintKnee), kFootprintExponent);
  const double parallelism = std::max(1.,
      std::min(double(cores), w.groups / kGroupsPerStream));
  const double stream = kStreamWeight * resources.stream_footprint_bytes
      / kKvUnitBytes * stream_fraction / parallelism;
  // Smooth maximum of overlapping costs, plus fixed and per-group launch costs.
  constexpr double kLaunch = 6.850719284203303;
  constexpr double kGroupLaunch = .00033546262790716194;
  const double roof = std::sqrt(std::sqrt(
      std::pow(bulk, 4) + std::pow(critical, 4) + std::pow(stream, 4)));
  // When full tiles saturate the grid, the last guarded Q tile still has
  // to finish its KV scan. A scan-limited launch already pays for that tail
  // in critical. A complete Q tile has no drain.
  constexpr double kTailDrain = .25;
  const double drain = upstream_split
      ? kTailDrain * std::min(w.tail_stream, std::max(0., bulk - critical)) : 0.;
  return kLaunch + roof + drain + kGroupLaunch * resources.launched_groups / cores;
}

inline double nax_admission_margin(const NaxPrefillWork& upstream,
                                   const NaxPrefillWork& candidate, int cores) {
  constexpr double kMinimumGain = .10;
  constexpr double kCapacityReserve = .30;
  // A candidate with fewer useful groups than GPU cores has less parallel
  // slack. Apply the reserve to full query work, where upstream is efficient.
  const double idle_fraction = 1. - double(std::min<int64_t>(cores, candidate.groups)) / cores;
  const double full_fraction = 1. - upstream.guarded_q / upstream.issued;
  return kMinimumGain + kCapacityReserve * idle_fraction * full_fraction;
}

}  // namespace vllm_metal::kernels
