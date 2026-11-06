// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

namespace vllm_metal::kernels {

struct NaxSequence {
  int query_tokens;
  int kv_tokens;

  bool operator==(const NaxSequence& other) const {
    return query_tokens == other.query_tokens && kv_tokens == other.kv_tokens;
  }
};

struct NaxSequences {
  const NaxSequence* first;
  const NaxSequence* last;
  const NaxSequence* begin() const { return first; }
  const NaxSequence* end() const { return last; }
};

struct NaxLayout {
  int query_tile;
  int kv_tile;
  int heads_per_group;
  int threads;
  int value_slices = 1;
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
  double qk_flops = 0, value_flops = 0;
  double kv_stream_bytes = 0, page_visits = 0;
  double stream_footprint_bytes = 0, logical_kv_bytes = 0;
  int64_t query_tokens = 0, sequences = 0, launched_groups = 0;
  int64_t logical_state_bytes = 0;
  int threads = 0;
  double longest_stream = 0;
  double full_stream = 0, tail_stream = 0;
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
  work.groups = (heads / layout.heads_per_group) * q_blocks * layout.value_slices;
  work.full_blocks = full;
  return work;
}

// Each sequence keeps its own causal origin, full blocks and final Q/KV
// tails. Concatenating Q lengths would invent complete blocks across requests.
template <typename Sequences>
inline NaxPrefillResources nax_prefill_resources(
    const Sequences& sequences, NaxGeometry geometry, NaxLayout layout, bool split,
    int query_offset = 0) {
  NaxPrefillResources total;
  double streams = 0, sequence_streams = 0;
  for (const auto& sequence : sequences) {
    const auto work = nax_sequence_work(sequence, geometry.heads, layout, split);
    total.work.issued += work.issued;
    total.work.active += work.active;
    total.work.guarded += work.guarded;
    total.work.guarded_q += work.guarded_q;
    total.work.kv_edge += work.kv_edge;
    total.work.fast += work.fast;
    total.work.tail_stream += work.tail_stream;
    total.work.groups += work.groups;
    total.work.full_blocks += work.full_blocks;
    streams += work.stream * work.groups;
    sequence_streams += work.stream * (geometry.heads / layout.heads_per_group);
    const int full_q = (sequence.query_tokens / layout.query_tile) * layout.query_tile;
    if (full_q > 0) {
      const int64_t end = int64_t(sequence.kv_tokens) - sequence.query_tokens + full_q;
      total.full_stream = std::max(total.full_stream,
          double((end + layout.kv_tile - 1) / layout.kv_tile));
    }
    total.tail_stream = std::max(total.tail_stream, work.tail_stream);
    total.longest_stream = std::max(total.longest_stream,
        double((int64_t(sequence.kv_tokens) + layout.kv_tile - 1) / layout.kv_tile));
    total.query_tokens += sequence.query_tokens;
    total.logical_kv_bytes += double(sequence.kv_tokens) * geometry.kv_heads
        * 2 * (geometry.qk_dim + geometry.value_dim);
    ++total.sequences;
  }
  total.work.stream = total.work.groups ? streams / total.work.groups : 0;
  // A mixed suffix starts at a possibly unaligned cumulative Q offset.
  // Useful groups are counted above; this also includes uniform early returns.
  total.launched_groups = ((query_offset + total.query_tokens) / layout.query_tile
      - query_offset / layout.query_tile + total.sequences)
      * (geometry.heads / layout.heads_per_group) * layout.value_slices;
  const double matrix_elements = total.work.issued * 16 * layout.kv_tile;
  total.qk_flops = 2 * matrix_elements * geometry.qk_dim * layout.value_slices;
  total.value_flops = 2 * matrix_elements * geometry.value_dim;
  // One potential shared KV stream per group, not an assertion of cache hits.
  const double group_visits = total.work.issued / (layout.query_tile / 16)
      / layout.heads_per_group;
  total.kv_stream_bytes = 2 * group_visits * layout.kv_tile
      * (geometry.qk_dim * layout.value_slices + geometry.value_dim);
  // Mean stream depth per sequence distinguishes repeated Q-block traffic
  // from its working-set estimate. Shared physical prefixes are not deduplicated.
  total.stream_footprint_bytes = 2 * sequence_streams * layout.kv_tile
      * (geometry.qk_dim * layout.value_slices + geometry.value_dim);
  total.page_visits = group_visits
      * ((layout.kv_tile + geometry.kernel_page - 1) / geometry.kernel_page);
  const int64_t rows = int64_t(layout.query_tile) * layout.heads_per_group;
  total.logical_state_bytes = rows * (
      4 * (geometry.value_dim / layout.value_slices) + 2 * geometry.qk_dim
      + 4 * layout.kv_tile + 8);
  total.threads = layout.threads;
  return total;
}

// Empirical relative costs from the 10/20/40-core calibration described in
// docs/gqa-prefill.md. They rank compiled layouts, not model names or latency
// in milliseconds. Logical state is not hardware register occupancy.
inline double nax_prefill_cost(
    const NaxPrefillResources& resources, NaxGeometry geometry,
    NaxLayout layout, int cores, bool split) {
  const auto& w = resources.work;
  if (cores <= 0 || w.groups == 0) return std::numeric_limits<double>::infinity();
  double rate = 1;
  if (layout.heads_per_group > 1) {
    if (geometry.kernel_page == 16) {
      switch (layout.heads_per_group) {
        case 2: rate = 1.0169304348373918; break;
        case 3: rate = .9432343252369108; break;
        case 6: rate = .7268784060030199; break;
        default: return std::numeric_limits<double>::infinity();
      }
    } else if (geometry.kernel_page == 32) {
      switch (layout.heads_per_group) {
        case 2: rate = .9920785752048182; break;
        case 4: rate = .9680837381817905; break;
        case 8: rate = .4508176063839598; break;
        default: return std::numeric_limits<double>::infinity();
      }
    } else {
      return std::numeric_limits<double>::infinity();
    }
  }
  const bool page16 = geometry.kernel_page == 16;
  const double guarded_cost = page16 ? 1.622904174104084 : 1.5306808620260304;
  const double critical_guard = page16 ? 1.2258362190278242 : 1.4003105500708977;
  const double tail_cost = layout.heads_per_group > 1
      ? 1.34722704155022 : 1.2468577321483032;
  const double inactive = w.issued - w.active;
  const double tail = w.guarded_q - inactive;
  const double guarded = w.active - w.fast - tail;
  const double dimension = double(geometry.qk_dim + geometry.value_dim) / 512;
  // State includes Q, output accumulators, scores and softmax statistics.
  const double state = double(resources.logical_state_bytes) / (64 * 1672);
  const double wide = std::max(0., state - 1);
  double bulk = dimension * (w.fast + guarded_cost * guarded
      + tail_cost * tail + 1.4542458472707103 * inactive) / (9.088170462137231 * cores);
  bulk *= rate * (1 + 1.326972148183182 * wide * wide
      * std::min(1., 29.185172825937 * cores / w.groups));
  double critical = dimension * std::max(
      resources.full_stream * (split ? 1 : critical_guard), resources.tail_stream * tail_cost);
  critical *= 1 + .08081622214154811 * std::max(0., state - .5);
  const double footprint = resources.logical_kv_bytes / (cores * 32768.);
  const double stream_fraction = std::pow(
      footprint / (footprint + .0015885238571067709), .25000080683650183);
  const double memory_parallelism = std::max(1.,
      std::min(double(cores), w.groups / 50.82434564183909));
  const double stream = .022333392978314857 * resources.stream_footprint_bytes
      / 32768 * stream_fraction / memory_parallelism;
  const double roof = std::sqrt(std::sqrt(
      std::pow(bulk, 4) + std::pow(critical, 4) + std::pow(stream, 4)));
  return 6.850719284203303 + roof
      + .00033546262790716194 * resources.launched_groups / cores;
}

inline double nax_replacement_margin(
    const NaxPrefillResources& baseline, int heads, int cores) {
  const double full_simd_groups = double(heads) * baseline.work.full_blocks * 4;
  const double full_fraction = 1 - baseline.work.guarded_q / baseline.work.issued;
  return .15 + .30 * std::min(1., cores / std::max(1., full_simd_groups)) * full_fraction;
}

}  // namespace vllm_metal::kernels
