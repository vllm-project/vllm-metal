// SPDX-License-Identifier: Apache-2.0
// Shared internal types and declarations for the MLX ops translation units.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "metal_device.h"
#include "paged_attention_kernels.h"

#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace kernels = vllm_metal::kernels;
namespace hardware = vllm_metal::hardware;
using namespace mlx::core;

#ifndef VLLM_METAL_PARTITION_SIZE
#define VLLM_METAL_PARTITION_SIZE 512
#endif
constexpr int kPartitionSize = VLLM_METAL_PARTITION_SIZE;
// Process-wide diagnostic for tests; not a per-request trace or routing input.
enum class PagedDispatch {
  None, GqaDecode, PerToken, SplitKv, Window, WindowSplitKv, NaxPrefill,
  TiledPrefill, MixedPrefillDecode, MixedNaxPrefillDecode,
  NaxGqa6Prefill, Count
};
using hardware::detected_gpu_core_count;
using hardware::gpu_core_count;
// One measured dispatch table, also exposed read-only to numerical/routing
// tests. Include the kernel page view so geometry and page admission cannot
// drift independently. Model names are not routing inputs; the calibrated
// device preference below is separate from functional geometry admission.
struct GqaDecodeGeometry {
  int num_heads;
  int num_kv_heads;
  int head_size;
  int block_size;
};
constexpr std::array<GqaDecodeGeometry, 6> kGqaDecodeGeometries = {{
    {32, 8, 128, 16},
    {24, 4, 256, 16},
    {24, 4, 256, 32},
    {16, 2, 128, 16},
    {16, 2, 256, 16},
    {16, 2, 256, 32},
}};

// Shared by production selection, private test dispatch and library checks.
static_assert(kPartitionSize == 512,
              "GQA partition list assumes the established 512-token split");
constexpr std::array<int, 2> kGqaPartitionSizes = {512, 256};
constexpr int64_t kGqaSimdGroupsPerCore = 33;
// The ordinary GQA reducer specializes out the TurboQuant workspace, but
// retains red_smem[2 * NUM_WARPS] (256 threads / 32 SIMD lanes). Reserve its
// 64 bytes as well as the dynamic statistics; dispatch also checks the
// compiled pipeline's actual static allocation before encoding the reducer.
constexpr size_t kGqaReduceStaticMemoryBytes = 2 * (256 / 32) * sizeof(float);

// M3 (10-core, g15g) batched calibration. Tighten short-context admission,
// and avoid the slower P512 producer for measured long head256 layouts.
// Single requests and other devices retain the existing selection.
constexpr int kM3GqaGpuCores = 10;
constexpr int kM3GqaShortHead128Batch = 6;
constexpr int kM3GqaShortHead128MinContext = 704;
constexpr int kM3GqaBlock16MinContext = 4096;
constexpr int kM3GqaBlock32MinContext = 8192;
constexpr size_t kM3GqaReducerMemoryBytes = 32 * 1024;

// Bound partial outputs, both statistics and any compact-layout offsets.
// This is a per-invocation ceiling, not a reservation or a process memory limit.
constexpr size_t kGqaMaxScratchBytes = 512 * 1024 * 1024;
// Immutable CPU metadata shared by every attention layer of one forward.
// No device reads or geometry-dependent policy are captured here.
using GqaPartitionOffsets = std::array<std::vector<int64_t>, 2>;
struct GqaDecodeLengthPlan {
  int num_requests = 0;
  int max_length = 0;
  int64_t total_length = 0;
  std::array<int64_t, 2> full_partitions = {};
  std::shared_ptr<const GqaPartitionOffsets> partition_offsets;

  bool operator==(const GqaDecodeLengthPlan& other) const {
    return num_requests == other.num_requests && max_length == other.max_length &&
        total_length == other.total_length &&
        full_partitions == other.full_partitions &&
        (partition_offsets == other.partition_offsets ||
         (partition_offsets && other.partition_offsets &&
          *partition_offsets == *other.partition_offsets));
  }
};

extern std::atomic<PagedDispatch> g_last_dispatch;
extern std::atomic<int> g_last_gqa_partition;
extern std::atomic<int> g_last_gqa_requests;
extern std::atomic<bool> g_dispatch_diagnostics_enabled;
extern std::atomic<int> g_last_nax_split;
extern std::atomic<int> g_last_nax_heads_per_tg;
extern bool nax_lib_ready_;
extern bool nax_enabled_;
template <typename Lengths>
static GqaDecodeLengthPlan gqa_decode_length_plan(const Lengths& context_lens) {
  GqaDecodeLengthPlan plan;
  auto offsets = context_lens.size() > 1
      ? std::make_shared<GqaPartitionOffsets>() : nullptr;
  if (offsets) {
    for (auto& row : *offsets) {
      row.reserve(context_lens.size() + 1);
      row.push_back(0);
    }
  }
  for (int length : context_lens) {
    if (length <= 0) return {};
    ++plan.num_requests;
    plan.max_length = std::max(plan.max_length, length);
    plan.total_length += length;
    for (size_t i = 0; i < kGqaPartitionSizes.size(); ++i) {
      plan.full_partitions[i] += length / kGqaPartitionSizes[i];
      if (offsets) {
        auto& row = (*offsets)[i];
        row.push_back(row.back() +
            (int64_t(length) + kGqaPartitionSizes[i] - 1) / kGqaPartitionSizes[i]);
      }
    }
  }
  if (plan.num_requests > 0) plan.partition_offsets = std::move(offsets);
  return plan;
}

bool gqa_decode_geometry_supported(int num_heads, int num_kv_heads,
                                   int head_size, int block_size);
std::string gqa_decode_kernel_name(
    const std::string& dtype, int head_size, int block_size, int partition_size);
std::string gqa_decode_tq_kernel_name(
    const std::string& dtype, const std::string& k_cache_dtype, int head_size,
    int block_size, int partition_size);
int gqa_decode_plan_for_lengths(
    int num_heads, int num_kv_heads, int head_size,
    const GqaDecodeLengthPlan& plan, int gpu_cores, int block_size,
    const std::string& gpu_arch = "", int max_seq_len = 0, bool turboquant = false);
void init_library_path(const std::string& name, const std::string& path);
void init_nax_library_path(const std::string& path);
std::string dtype_to_metal(Dtype dt);
array paged_attention_primitive_fn(
    const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float scale, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int block_size, int max_seq_len, int sliding_window,
    bool use_turboquant = false, const std::string& quant_type = "",
    const array* key_scale_cache = nullptr,
    const array* value_scale_cache = nullptr,
    const array* key_zero_cache = nullptr,
    const array* v_centroids = nullptr,
    int v_bits = 3, int window_seqlen_q = 1,
    const array* sinks = nullptr,
    const array* mm_prefix_ranges = nullptr,
    int num_decode_requests = -1,
    int num_decode_tokens = 0,
    int max_decode_context_len = 0, bool gqa_disabled = false,
    int gqa_test_partition = 0,
    const std::optional<GqaDecodeLengthPlan>& gqa_length_plan = std::nullopt,
    int nax_split_hint = -1, int nax_prefill_split_hint = -1);
std::tuple<array, array, array, array, array> tq_encode_primitive_fn(
    const array& key, const array& value,
    const array& key_cache, const array& value_cache,
    const array& key_scale_cache, const array& value_scale_cache,
    const array& key_zero_cache,
    const array& slot_mapping, const array& v_centroids,
    int v_bits, int k_bits, bool k_signed);
std::tuple<array, array> reshape_and_cache_primitive_fn(
    const array& key, const array& value,
    const array& key_cache, const array& value_cache,
    const array& slot_mapping);
array gdn_state_scatter_primitive_fn(
    const array& pool, const array& src, const array& dst_ids,
    bool zero = false, bool paged = false);
array mla_paged_attention_primitive_fn(
    const array& q_nope,
    const array& q_pe,
    const array& latent_cache,
    const array& block_tables,
    const array& context_lens,
    const array& cu_seqlens_q,
    int block_size,
    float scale,
    int heads_per_tg);
std::tuple<array, array> gdn_linear_attention_primitive_fn(
    const array& q, const array& k, const array& v,
    const array& g, const array& beta, const array& state_pool,
    const array& cu_seqlens, const array& slot_mapping,
    int Hk, int Hv, int Dk, int Dv);

struct NaxPrefillDispatchEvent {
  const char* family;
  const char* dtype;
  int q, hq, hkv, d, page, numseq, gridx, gridy, threads, bq, heads_per_tg;
  int32_t q_stride, key_block_stride, key_head_stride;
  int64_t key_stride0, key_stride1, key_stride2;
  int key_page_tokens, sliding_window;
  float softcap;
  bool use_sinks;
  int value_slices, grouped_skip_bounds;
  std::string function;
  bool split_full;
  int max_seq_len, q_block_offset;
};
struct NaxPrefillTraceResult {
  std::vector<NaxPrefillDispatchEvent> events;
  size_t capacity;
  uint64_t dropped;
  uint64_t append_ns;
};
void start_nax_prefill_dispatch_trace();
NaxPrefillTraceResult take_nax_prefill_dispatch_trace();
