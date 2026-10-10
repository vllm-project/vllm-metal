// SPDX-License-Identifier: Apache-2.0
// C++ nanobind bridge for paged attention Metal kernels.
//
// Dispatches the v2 paged-attention / TurboQuant / GDN / MLA kernels through
// MLX's own Metal command encoder, eliminating the PyTorch MPS bridge.
//
// Uses nb::handle + nb::inst_ptr<array>() to extract the C++ array from
// the Python mlx.core.array object, bypassing nanobind's cross-module
// RTTI matching which fails due to hidden symbol visibility in libmlx.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace nb = nanobind;
using namespace mlx::core;

void register_mlx_patch(nb::module_& m);

#ifndef VLLM_METAL_PARTITION_SIZE
#define VLLM_METAL_PARTITION_SIZE 512
#endif

// ---------------------------------------------------------------------------
// Library caching
// ---------------------------------------------------------------------------

static std::string v2_paged_attention_source_;
constexpr int kPartitionSize = VLLM_METAL_PARTITION_SIZE;

// Process-wide diagnostic for tests; not a per-request trace or routing input.
enum class PagedDispatch {
  None, GqaDecode, PerToken, SplitKv, Window, WindowSplitKv, NaxPrefill,
  TiledPrefill, MixedPrefillDecode, MixedNaxPrefillDecode, Count
};
static std::atomic<PagedDispatch> g_last_dispatch{PagedDispatch::None};
static std::atomic<int> g_last_gqa_partition{0};
static std::atomic<int> g_last_gqa_requests{0};
static std::atomic<bool> g_dispatch_diagnostics_enabled{false};
inline void record_paged_dispatch(PagedDispatch family, int gqa_partition = 0,
                                  int gqa_requests = 0) {
  if (!g_dispatch_diagnostics_enabled.load(std::memory_order_relaxed)) return;
  g_last_dispatch.store(family, std::memory_order_relaxed);
  g_last_gqa_partition.store(gqa_partition, std::memory_order_relaxed);
  g_last_gqa_requests.store(gqa_requests, std::memory_order_relaxed);
}

static bool set_paged_dispatch_diagnostics(bool enabled) {
  // Call only while evaluation is idle. Clear stale observations on both
  // transitions so a disabled observer cannot report a previous request.
  const bool previous =
      g_dispatch_diagnostics_enabled.exchange(enabled, std::memory_order_relaxed);
  g_last_dispatch.store(PagedDispatch::None, std::memory_order_relaxed);
  g_last_gqa_partition.store(0, std::memory_order_relaxed);
  g_last_gqa_requests.store(0, std::memory_order_relaxed);
  return previous;
}
// Split only after eight KV partitions amortize the extra dispatch.
constexpr int kMixedDecodeMinPartitions = 8;
constexpr int kMixedDecodeMinContext =
    kMixedDecodeMinPartitions * kPartitionSize;
// Query rows per NAX prefill threadgroup (BQ in pagedattention_nax.metal).
constexpr int kNaxBQ = 64;

// Window mode for spec-decode verification (per-token kernel): query rows
// per threadgroup (2 = the measured register/occupancy sweet spot on Apple
// GPUs; larger windows become several sub-window threadgroups).
// Single-sourced from vllm_metal/metal/constants.py via
// -DVLLM_METAL_PA_WINDOW_ROWS; the shader consumes the same define as
// PA_WINDOW_ROWS, so host and kernel cannot drift.
#ifndef VLLM_METAL_PA_WINDOW_ROWS
#define VLLM_METAL_PA_WINDOW_ROWS 2
#endif
constexpr int kWindowRows = VLLM_METAL_PA_WINDOW_ROWS;

// Largest head size window mode serves: per-thread register state scales
// with kWindowRows * ceil(head_size / 32), and 512 collapses occupancy.
// Single-sourced from constants.py; prepare_unified keeps wider-head models
// on the expanded per-token layout, and the binding rejects wider hints.
#ifndef VLLM_METAL_PA_WINDOW_MAX_HEAD
#define VLLM_METAL_PA_WINDOW_MAX_HEAD 256
#endif
constexpr int kWindowMaxHeadSize = VLLM_METAL_PA_WINDOW_MAX_HEAD;

// ---------------------------------------------------------------------------
// Split-KV (flash-decoding) decode gate
// ---------------------------------------------------------------------------
// Decode runs one threadgroup per (q-head, query-token).  At low concurrency
// with a long context the base grid (num_q_heads * num_decode_tokens) leaves
// GPU cores idle while each threadgroup serially crawls the whole KV.  The
// (already-compiled) paged_attention_v2 split path partitions the KV across
// grid.z + a reduce pass to manufacture the missing parallelism.  Engages only
// when the base grid underfills the GPU, so saturated high-concurrency serving
// is untouched.

// GPU core count via IORegistry.  Metal/MLX expose no core-count API, but the
// split-KV gate needs to scale per machine — a small laptop GPU and a large
// desktop one saturate at very different grid sizes. Read once; zero means
// unknown so the new GQA performance gate can fail closed.
// Tests may inject a non-negative count through
// `_override_detected_gpu_core_count_for_test` so CI hosts without
// IORegistry still exercise default routing. Production must not call it.
static std::atomic<int> g_test_gpu_core_count{-1};

static int hardware_gpu_core_count() {
  static const int v = []() {
    int cores = 0;
    io_iterator_t it;
    if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                     IOServiceMatching("AGXAccelerator"),
                                     &it) == KERN_SUCCESS) {
      io_object_t obj;
      while ((obj = IOIteratorNext(it))) {
        CFTypeRef p = IORegistryEntrySearchCFProperty(
            obj, kIOServicePlane, CFSTR("gpu-core-count"),
            kCFAllocatorDefault, kIORegistryIterateRecursively);
        if (p) {
          if (CFGetTypeID(p) == CFNumberGetTypeID())
            CFNumberGetValue((CFNumberRef)p, kCFNumberIntType, &cores);
          CFRelease(p);
        }
        IOObjectRelease(obj);
        if (cores > 0) break;
      }
      IOObjectRelease(it);
    }
    return cores;
  }();
  return v;
}

static int detected_gpu_core_count() {
  const int override =
      g_test_gpu_core_count.load(std::memory_order_relaxed);
  if (override >= 0)
    return override;
  return hardware_gpu_core_count();
}

static void override_detected_gpu_core_count_for_test(int cores) {
  if (cores < -1)
    throw std::invalid_argument(
        "test GPU core override must be >= -1 (got " +
        std::to_string(cores) + ")");
  g_test_gpu_core_count.store(cores, std::memory_order_relaxed);
}

// Preserve the established split-KV fallback when detection is unavailable.
static int gpu_core_count() {
  const int cores = detected_gpu_core_count();
  return cores > 0 ? cores : 14;
}

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
constexpr std::array<GqaDecodeGeometry, 5> kGqaDecodeGeometries = {{
    {32, 8, 128, 16},
    {24, 4, 256, 16},
    {16, 2, 128, 16},
    {16, 2, 256, 16},
    {16, 2, 256, 32},
}};

static bool gqa_decode_geometry_supported(int num_heads, int num_kv_heads,
                                           int head_size, int block_size) {
  return std::any_of(
      kGqaDecodeGeometries.begin(), kGqaDecodeGeometries.end(),
      [=](const auto& geometry) {
        return geometry.num_heads == num_heads &&
            geometry.num_kv_heads == num_kv_heads &&
            geometry.head_size == head_size && geometry.block_size == block_size;
      });
}

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

static std::string gqa_decode_kernel_name(
    const std::string& dtype, int head_size, int block_size, int partition_size) {
  return "paged_attention_gqa_decode_" + dtype + "_hs" + std::to_string(head_size)
      + "_bs" + std::to_string(block_size) + "_ps" + std::to_string(partition_size);
}

static std::string paged_reduce_kernel_name(
    const std::string& dtype, int head_size, int partition_size) {
  return "paged_attention_v2_reduce_" + dtype + "_hs" + std::to_string(head_size)
      + "_nt256_nsl32_ps" + std::to_string(partition_size);
}

static size_t paged_reduce_threadgroup_bytes(int64_t num_partitions) {
  // Two FP32 statistics per partition; Metal requires 16-byte alignment.
  return (static_cast<size_t>(num_partitions) * 2 * sizeof(float) + 15) &
      ~size_t(15);
}

static bool paged_reduce_memory_fits(
    size_t dynamic_bytes, size_t static_bytes, size_t capacity) {
  return dynamic_bytes <= capacity && static_bytes <= capacity - dynamic_bytes;
}

// Bound all three rectangular temporary arrays, including ragged padding.
// This is a per-invocation ceiling, not a reservation or a process memory limit.
constexpr size_t kGqaMaxScratchBytes = 512 * 1024 * 1024;
static bool gqa_scratch_fits(
    int requests, int heads, int64_t partitions, int head_size, size_t item_size) {
  const size_t bytes_per_partition = head_size * item_size + 2 * sizeof(float);
  size_t remaining = kGqaMaxScratchBytes / bytes_per_partition;
  // Divide the budget instead of multiplying potentially large dimensions.
  for (int64_t count : {int64_t(requests), int64_t(heads), partitions}) {
    if (count <= 0 || static_cast<uint64_t>(count) > remaining) return false;
    remaining /= static_cast<size_t>(count);
  }
  return true;
}

// Compact CPU statistics shared by every attention layer of one forward.
// No arrays, device reads or geometry-dependent policy are captured here.
struct GqaDecodeLengthPlan {
  int num_requests = 0;
  int max_length = 0;
  std::array<int64_t, 2> full_partitions = {};

  bool operator==(const GqaDecodeLengthPlan& other) const {
    return num_requests == other.num_requests && max_length == other.max_length &&
        full_partitions == other.full_partitions;
  }
};

template <typename Lengths>
static GqaDecodeLengthPlan gqa_decode_length_plan(const Lengths& context_lens) {
  GqaDecodeLengthPlan plan;
  for (int length : context_lens) {
    if (length <= 0) return {};
    ++plan.num_requests;
    plan.max_length = std::max(plan.max_length, length);
    for (size_t i = 0; i < kGqaPartitionSizes.size(); ++i)
      plan.full_partitions[i] += length / kGqaPartitionSizes[i];
  }
  return plan;
}

static int gqa_decode_partition_for_lengths(
    int num_heads, int num_kv_heads, int head_size,
    const GqaDecodeLengthPlan& plan, int gpu_cores, int block_size,
    const std::string& gpu_arch = "", int max_seq_len = 0) {
  if (gpu_cores <= 0 || plan.num_requests == 0 ||
      !gqa_decode_geometry_supported(
          num_heads, num_kv_heads, head_size, block_size)) return 0;
  int partition = 0;
  for (size_t i = 0; i < kGqaPartitionSizes.size(); ++i) {
    if (plan.full_partitions[i] * num_heads >= kGqaSimdGroupsPerCore * gpu_cores) {
      partition = kGqaPartitionSizes[i];
      break;
    }
  }
  if (partition == 0 || plan.num_requests == 1 ||
      gpu_cores != kM3GqaGpuCores) return partition;

  const auto& architecture = gpu_arch.empty()
      ? metal::device(Device::gpu).get_architecture() : gpu_arch;
  if (architecture != "applegpu_g15g") return partition;

  if (num_heads == 32 && num_kv_heads == 8 && head_size == 128 &&
      plan.num_requests == kM3GqaShortHead128Batch &&
      plan.max_length < kM3GqaShortHead128MinContext) return 0;
  if (partition != 512 || head_size != 256) return partition;

  const int minimum = block_size == 16
      ? kM3GqaBlock16MinContext : kM3GqaBlock32MinContext;
  if (plan.max_length < minimum) return partition;

  // P512 passed the work gate, so P256 also has enough complete work.
  // Include a caller's larger allocation bound in the reducer resource check.
  const int64_t allocation_length = std::max(max_seq_len, plan.max_length);
  const int64_t partitions256 = (allocation_length + 255) / 256;
  if (!paged_reduce_memory_fits(paged_reduce_threadgroup_bytes(partitions256),
                               kGqaReduceStaticMemoryBytes,
                               kM3GqaReducerMemoryBytes)) return partition;
  return 256;
}

static int gqa_decode_plan_for_lengths(
    int num_heads, int num_kv_heads, int head_size,
    const GqaDecodeLengthPlan& plan, int gpu_cores, int block_size,
    const std::string& gpu_arch = "", int max_seq_len = 0) {
  const int partition = gqa_decode_partition_for_lengths(num_heads, num_kv_heads,
      head_size, plan, gpu_cores, block_size, gpu_arch, max_seq_len);
  if (partition == 0) return 0;
  const int64_t length = std::max(max_seq_len, plan.max_length);
  const int64_t partitions = (length + partition - 1) / partition;
  // Every admitted GQA dtype stores a two-byte partial output.
  return gqa_scratch_fits(plan.num_requests, num_heads, partitions, head_size, 2)
      ? partition : 0;
}

static int gqa_decode_plan_for_shape(int num_heads, int num_kv_heads,
                                     int head_size, int max_seq_len,
                                     int gpu_cores, int block_size) {
  return gqa_decode_plan_for_lengths(num_heads, num_kv_heads, head_size,
      gqa_decode_length_plan(std::array<int, 1>{max_seq_len}), gpu_cores, block_size);
}

static int gqa_decode_batch_plan_for_shape(
    int num_heads, int num_kv_heads, int head_size,
    const std::vector<int>& context_lens, int gpu_cores, int block_size,
    const std::string& gpu_arch = "", int max_seq_len = 0) {
  return gqa_decode_plan_for_lengths(num_heads, num_kv_heads, head_size,
      gqa_decode_length_plan(context_lens), gpu_cores, block_size,
      gpu_arch, max_seq_len);
}

static bool gqa_decode_shape_eligible(int num_heads, int num_kv_heads,
                                     int head_size, int max_seq_len,
                                     int gpu_cores, int block_size) {
  return gqa_decode_plan_for_shape(num_heads, num_kv_heads, head_size,
                                   max_seq_len, gpu_cores, block_size) > 0;
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
static int min_decode_grid() {
  static const int v = gpu_core_count() * 8;
  return v;
}

void init_v2_library(const std::string& v2_src) {
  v2_paged_attention_source_ = v2_src;
  auto& d = metal::device(Device::gpu);
  d.get_library(
      "paged_attention_v2_kern",
      [&]() { return v2_paged_attention_source_; });
}

// Load a precompiled .metallib instead of compiling its source. Uses MLX's
// get_library(name, path) overload, which loads the library at `path` and
// caches it under `name`; a later dispatch's get_library(name) (no path) then
// returns this cached library. One generic loader for every shader library —
// the cache-key name is passed in, so there is no per-library variant.
void init_library_path(const std::string& name, const std::string& path) {
  auto& d = metal::device(Device::gpu);
  d.get_library(name, path);
}

// ---------------------------------------------------------------------------
// NAX (M5 tensor-unit) prefill kernel support
// ---------------------------------------------------------------------------

static std::string nax_source_;
static bool nax_lib_ready_ = false;   // a NAX library was registered
static bool nax_enabled_ = true;      // test / A-B switch

// Match MLX's hardware gate, but ignore MLX_METAL_NO_NAX because this library
// is compiled separately. The 'p' family requires generation 18 rather than 17.
static bool nax_hardware_supported() {
  static const bool v = []() {
    bool ok = false;
    if (__builtin_available(macOS 26.2, *)) {
      ok = true;
    }
    auto& d = metal::device(Device::gpu);
    const auto& arch = d.get_architecture();
    if (arch.empty()) return false;
    ok &= d.get_architecture_gen() >= (arch.back() == 'p' ? 18 : 17);
    return ok;
  }();
  return v;
}

void init_nax_library(const std::string& nax_src) {
  nax_source_ = nax_src;
  auto& d = metal::device(Device::gpu);
  d.get_library("paged_attention_nax_kern", [&]() { return nax_source_; });
  nax_lib_ready_ = true;
}

void init_nax_library_path(const std::string& path) {
  init_library_path("paged_attention_nax_kern", path);
  nax_lib_ready_ = true;
}

// Match the instantiated NAX shapes; other shapes retain the tiled path.
static bool nax_eligible(Dtype dtype, int head_size, int block_size) {
  return nax_lib_ready_ && nax_enabled_ &&
      (dtype == float16 || dtype == bfloat16) &&
      (head_size == 64 || head_size == 96 || head_size == 128 ||
       head_size == 256 || head_size == 512) &&
      (block_size == 8 || block_size == 16 || block_size == 32);
}

// ---------------------------------------------------------------------------
// Helper: dtype → Metal type string
// ---------------------------------------------------------------------------

static std::string dtype_to_metal(Dtype dt) {
  switch (dt) {
    case float16:   return "half";
    case bfloat16:  return "bfloat16_t";
    case float32:   return "float";
    case int8:      return "char";
    case uint8:     return "uchar";
    default:
      throw std::runtime_error(
          "Unsupported dtype for paged attention kernel");
  }
}

// MLA dispatchers pick a single Metal specialization from one tensor's
// dtype (q_nope) but the kernel template binds the same `T` to every
// fp16/bf16 buffer (q_nope, q_pe, latent_cache, out, tmp_out). If they
// disagree the shader will reinterpret bytes — e.g. read a bf16 cache
// as fp16 — and silently corrupt attention. Validate up front.
static void mla_validate_t_dtypes(
    const char* dispatcher_name,
    std::initializer_list<std::pair<const char*, const array*>> tensors) {
  if (tensors.size() == 0) return;
  Dtype expected = tensors.begin()->second->dtype();
  if (expected != float16 && expected != bfloat16) {
    throw std::runtime_error(
        std::string(dispatcher_name) +
        ": T buffers must be fp16 or bf16; got " +
        dtype_to_metal(expected) + " on " + tensors.begin()->first);
  }
  for (const auto& [name, arr] : tensors) {
    if (arr->dtype() != expected) {
      throw std::runtime_error(
          std::string(dispatcher_name) +
          ": all T buffers must share the same dtype; got " +
          dtype_to_metal(expected) + " on " + tensors.begin()->first +
          " but " + dtype_to_metal(arr->dtype()) + " on " + name);
    }
  }
}

static int get_bits(const std::string& quant_type) {
  static const std::unordered_map<std::string, int> BITS = {
      {"q8_0", 8}, {"int8", 8}, {"uint8", 8},
      {"q5_0", 5},
      {"q4_0", 4}, {"int4", 4}, {"uint4", 4},
      {"int2", 2}, {"uint2", 2},
  };
  auto it = BITS.find(quant_type);
  if (it == BITS.end()) {
    throw std::runtime_error("Unknown quant_type: " + quant_type);
  }
  return it->second;
}

// ---------------------------------------------------------------------------
// paged_attention_v2_online — dispatch helper (used by PagedAttentionPrimitive)
// ---------------------------------------------------------------------------

// Both GQA and the established kernels address translated subpages this way.
static void bind_paged_attn_strides(
    metal::CommandEncoder& enc, const array& query,
    const array& key_cache, int block_size) {
  int32_t q_stride = static_cast<int32_t>(query.shape(1) * query.shape(2));
  int32_t kv_block_stride = static_cast<int32_t>(key_cache.strides()[0]);
  if (static_cast<int>(key_cache.shape(1)) > block_size) {
    kv_block_stride = static_cast<int32_t>(block_size * key_cache.strides()[1]);
  }
  int32_t kv_head_stride = static_cast<int32_t>(key_cache.strides()[2]);
  enc.set_bytes(q_stride, 15);
  enc.set_bytes(kv_block_stride, 16);
  enc.set_bytes(kv_head_stride, 17);
}

// Shared buffer binding for paged attention kernels (slots 2-21).
static void bind_paged_attn_buffers(
    metal::CommandEncoder& enc,
    array& out, const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int block_size, int sliding_window) {
  enc.set_output_array(out, 2);
  enc.set_input_array(query,        3);
  enc.set_input_array(key_cache,    4);
  enc.set_input_array(value_cache,  5);

  int32_t nkv = static_cast<int32_t>(num_kv_heads);
  enc.set_bytes(nkv,   8);
  // Slot 9 (scale) is set by the caller — varies per dispatch path.
  enc.set_bytes(softcap, 10);

  enc.set_input_array(block_tables, 11);
  enc.set_input_array(seq_lens,     12);

  int32_t max_blocks_i = static_cast<int32_t>(block_tables.shape(1));
  enc.set_bytes(max_blocks_i, 13);

  bind_paged_attn_strides(enc, query, key_cache, block_size);

  enc.set_input_array(cu_seqlens_q, 19);
  int32_t num_seqs_i = static_cast<int32_t>(cu_seqlens_q.shape(0) - 1);
  enc.set_bytes(num_seqs_i, 20);
  int32_t sliding_window_i = static_cast<int32_t>(sliding_window);
  enc.set_bytes(sliding_window_i, 21);
}

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
static std::optional<TileConfig> select_tile_config(int head_size) {
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

// Same buffer ABI as the tiled kernel, with BQ=64 and no threadgroup memory.
// q_block_offset skips leading q-blocks exactly as in the tiled dispatch.
static void dispatch_paged_attention_nax(
    array& out, const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float scale, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int block_size, int sliding_window, int q_block_offset,
    Stream s, const array* sinks) {
  auto& d = metal::device(s.device);

  constexpr int kNaxThreads = 128;

  int total_q_tokens = static_cast<int>(query.shape(0));
  int head_size  = static_cast<int>(query.shape(2));
  int num_seqs   = static_cast<int>(cu_seqlens_q.shape(0)) - 1;
  int total_q_blocks = total_q_tokens / kNaxBQ + num_seqs;
  if (q_block_offset < 0 || q_block_offset > total_q_blocks) {
    throw std::invalid_argument(
        "dispatch_paged_attention_nax: q_block_offset (" +
        std::to_string(q_block_offset) + ") out of range for total_q_blocks "
        "(" + std::to_string(total_q_blocks) + ")");
  }
  bool use_sinks = sinks != nullptr;

  std::string base_kname =
      "paged_attention_nax_" + dtype_to_metal(query.dtype()) +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size);
  std::string hash_name = base_kname + "_sk" + (use_sinks ? "1" : "0");

  auto* lib = d.get_library("paged_attention_nax_kern");
  auto* kernel = d.get_kernel(
      base_kname, lib, hash_name,
      {{&use_sinks, MTL::DataType::DataTypeBool, NS::UInteger(40)}});

  int num_heads = static_cast<int>(query.shape(1));
  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  bind_paged_attn_buffers(enc, out, query, key_cache, value_cache,
                          num_kv_heads, softcap, block_tables, seq_lens,
                          cu_seqlens_q, block_size, sliding_window);
  enc.set_bytes(scale, 9);
  enc.set_bytes(q_block_offset, 30);
  if (use_sinks) {
    enc.set_input_array(*sinks, 18);
  }

  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, total_q_blocks - q_block_offset, 1),
      MTL::Size::Make(kNaxThreads, 1, 1));
}

static void dispatch_paged_attention_tiled(
    array& out, const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float scale, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int block_size, int max_seq_len, int sliding_window, int q_block_offset,
    TileConfig cfg, Stream s, const array* sinks,
    const array* mm_prefix_ranges) {
  auto& d = metal::device(s.device);

  int total_q_tokens = static_cast<int>(query.shape(0));
  int head_size  = static_cast<int>(query.shape(2));
  int num_seqs   = static_cast<int>(cu_seqlens_q.shape(0)) - 1;

  int total_q_blocks = total_q_tokens / cfg.BQ + num_seqs;
  if (q_block_offset < 0 || q_block_offset > total_q_blocks) {
    throw std::invalid_argument(
        "dispatch_paged_attention_tiled: q_block_offset (" +
        std::to_string(q_block_offset) + ") out of range for total_q_blocks "
        "(" + std::to_string(total_q_blocks) + ")");
  }
  bool use_sinks = sinks != nullptr;
  bool use_mm_prefix = mm_prefix_ranges != nullptr;

  auto dt = dtype_to_metal(query.dtype());
  std::string base_kname =
      "paged_attention_tiled_" + dt +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_bq" + std::to_string(cfg.BQ) +
      "_tk" + std::to_string(cfg.TILE_KV) +
      "_nt" + std::to_string(cfg.NUM_THREADS);
  std::string hash_name = base_kname + "_sk" + (use_sinks ? "1" : "0")
                          + "_mp" + (use_mm_prefix ? "1" : "0");

  auto* lib = d.get_library("paged_attention_v2_kern");
  // Both constants are set on every dispatch: a function that references an
  // unset function constant fails pipeline creation.  MLX caches pipelines
  // by hash_name, so each constant combination needs its own suffix.
  auto* kernel = d.get_kernel(
      base_kname, lib, hash_name,
      {{&use_sinks, MTL::DataType::DataTypeBool, NS::UInteger(40)},
       {&use_mm_prefix, MTL::DataType::DataTypeBool, NS::UInteger(120)}});

  const int t_size = static_cast<int>(query.itemsize());
  // S, O, m, l are register-resident, so no S/O/M/L threadgroup buffers.
  // Output staging reuses Q_smem as fp32 O_smem at exit; fits because
  // BQ*LD*4 <= (BQ+2*TILE_KV)*LD*2  <=>  BQ <= 2*TILE_KV.
  // A1: leading dim padded by 16 B for bank-conflict avoidance —
  // smem_pad/ld MUST match SMEM_PAD/LD in pagedattention_tiled.metal.
  const int smem_pad = 16 / t_size;
  const int ld       = head_size + smem_pad;
  size_t shmem = static_cast<size_t>(
      (cfg.BQ + 2 * cfg.TILE_KV) * ld * t_size);  // Q + K + V, padded

  int num_heads = static_cast<int>(query.shape(1));
  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(shmem, 0);

  bind_paged_attn_buffers(enc, out, query, key_cache, value_cache,
                          num_kv_heads, softcap, block_tables, seq_lens,
                          cu_seqlens_q, block_size, sliding_window);
  enc.set_bytes(scale, 9);
  enc.set_bytes(q_block_offset, 30);
  if (use_sinks) {
    enc.set_input_array(*sinks, 18);
  }
  // Slot 22 is bound here, not in bind_paged_attn_buffers: that helper is
  // shared with the per-token kernel, where slot 22 is TurboQuant's k_codes.
  if (use_mm_prefix) {
    enc.set_input_array(*mm_prefix_ranges, 22);
  }

  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, total_q_blocks - q_block_offset, 1),
      MTL::Size::Make(cfg.NUM_THREADS, 1, 1));
}

// ---------------------------------------------------------------------------
// Shared pass-2 for the split-KV decode paths: merge per-partition partials
// into `out` (log-sum-exp combine).  Partials must follow the ps512 contract:
// log2-space (max, exp-sum) stats plus epsilon-normalised partial outputs in
// tmp_out[token, head, partition, :].
static MTL::ComputePipelineState* paged_attention_v2_reduce_kernel(
    metal::Device& d, const std::string& dt, int head_size,
    int partition_size, bool use_sinks, bool use_tq_fc) {
  const auto rname = paged_reduce_kernel_name(dt, head_size, partition_size);
  // The reduce kernel reads only use_sinks (40) and use_turboquant (50); the
  // other function constants are inert for it.  TurboQuant batches take this
  // path (use_tq_fc varies: the TQ reduce applies the deferred inverse FWHT),
  // and sinks are folded here rather than in the partitioned kernel so the
  // sink logit is counted once globally.  The cache key MUST encode every
  // constant the pipeline is specialized on — otherwise the first compile
  // wins and a later caller with different constants silently reuses the
  // wrong pipeline.
  std::string rhash = rname + "_v2reduce"
      + "_tq" + (use_tq_fc ? "1" : "0")
      + "_sk" + (use_sinks ? "1" : "0");
  auto* lib = d.get_library("paged_attention_v2_kern");
  return d.get_kernel(
      rname, lib, rhash,
      {{&use_sinks, MTL::DataType::DataTypeBool, NS::UInteger(40)},
       {&use_tq_fc, MTL::DataType::DataTypeBool, NS::UInteger(50)}});
}

static void dispatch_paged_attention_v2_reduce(
    metal::Device& d, metal::CommandEncoder& enc, array& out,
    const array& exp_sums, const array& max_logits, const array& tmp_out,
    const array& seq_lens, const array& cu_seqlens_q, int num_seqs,
    int total_q_tokens, int num_heads, int head_size, int max_num_partitions,
    const std::string& dt, bool use_sinks, const array* sinks,
    bool use_tq_fc, int partition_size = kPartitionSize,
    MTL::ComputePipelineState* rkernel = nullptr) {
  if (rkernel == nullptr) {
    rkernel = paged_attention_v2_reduce_kernel(
        d, dt, head_size, partition_size, use_sinks, use_tq_fc);
  }
  const size_t reduce_shmem = paged_reduce_threadgroup_bytes(max_num_partitions);
  const size_t capacity = d.mtl_device()->maxThreadgroupMemoryLength();
  if (!paged_reduce_memory_fits(
          reduce_shmem, rkernel->staticThreadgroupMemoryLength(), capacity)) {
    throw std::invalid_argument(
        "paged attention reduction exceeds the device threadgroup memory limit");
  }
  enc.set_compute_pipeline_state(rkernel);
  enc.set_threadgroup_memory_length(reduce_shmem, 0);
  enc.set_output_array(out, 0);
  enc.set_input_array(exp_sums, 1);
  enc.set_input_array(max_logits, 2);
  enc.set_input_array(tmp_out, 3);
  enc.set_input_array(seq_lens, 4);
  int32_t max_num_partitions_i = static_cast<int32_t>(max_num_partitions);
  enc.set_bytes(max_num_partitions_i, 5);
  if (use_sinks) {
    enc.set_input_array(*sinks, 6);
  }
  enc.set_input_array(cu_seqlens_q, 7);
  int32_t num_seqs_i = static_cast<int32_t>(num_seqs);
  enc.set_bytes(num_seqs_i, 8);
  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, total_q_tokens, 1),
      MTL::Size::Make(256, 1, 1));
}

static void dispatch_paged_attention_v2_online(
    array& out, const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float scale, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int block_size, int max_seq_len, int sliding_window,
    int window_seqlen_q, Stream s,
    // TurboQuant (optional, all nullptr when disabled):
    const array* key_scale_cache = nullptr,
    const array* value_scale_cache = nullptr,
    const array* key_zero_cache = nullptr,
    const array* v_centroids = nullptr,
    bool use_turboquant = false,
    int k_bits = 8,
    int v_bits = 3,
    // Attention sinks (optional): one float per query head, a learned logit
    // that joins the softmax denominator without contributing a value row.
    // nullptr for every model that has no sinks, which is the common case.
    const array* sinks = nullptr,
    // Gemma 4 vision image-block ranges (optional): one inclusive absolute
    // [start, end] key range per query row, implemented in the tiled kernel.
    const array* mm_prefix_ranges = nullptr,
    int num_decode_requests = -1,
    int num_decode_tokens = 0,
    int max_decode_context_len = 0,
    int decode_only_rows = 0, bool gqa_disabled = false,
    int gqa_test_partition = 0,
    const GqaDecodeLengthPlan& gqa_length_plan = {}) {
  int head_size = static_cast<int>(query.shape(2));

  // Tiled kernel for prefill batches, matching vLLM Triton's 2D/3D dispatch
  // split.  Pure-decode batches (every sequence has exactly 1 query token)
  // use the original per-token kernel.
  // A mixed batch's decode prefix (decode_only_rows > 0) is a pure decode
  // batch of its leading num_decode_requests sequences: cu_seqlens_q[0..D]
  // is 0..D, so every per-sequence view below stops at D.
  int total_q_tokens = decode_only_rows > 0
      ? decode_only_rows : static_cast<int>(query.shape(0));
  int num_seqs = decode_only_rows > 0
      ? num_decode_requests : static_cast<int>(cu_seqlens_q.shape(0)) - 1;
  bool has_prefill = decode_only_rows == 0 && total_q_tokens > num_seqs;
  bool dtype_ok = query.dtype() != float32
               && query.dtype() == key_cache.dtype();

  // Spec-decode verification windows: a multi-token batch whose segments
  // are all verification windows runs the per-token kernel in window mode
  // (kWindowRows-row sub-window threadgroups share every KV read), keeping
  // split-KV eligibility.  The tiled kernel is wrong for this shape: its
  // BQ-row MMA tiles waste ~80% of their compute on a K+1 window and it has
  // no KV-length parallelism.  head_size <= kWindowMaxHeadSize holds for
  // every constructed primitive (the binding rejects wider hints), so a
  // verify batch here always takes window mode, never the tiled fallback.
  const bool window_batch =
      has_prefill && window_seqlen_q > 1 && head_size <= kWindowMaxHeadSize;

  const bool use_nax = has_prefill && !window_batch && !use_turboquant
      && dtype_ok && mm_prefix_ranges == nullptr
      && nax_eligible(query.dtype(), head_size, block_size);
  const auto tile_config = has_prefill && !window_batch && !use_turboquant
      && dtype_ok ? select_tile_config(head_size) : std::nullopt;

  // Split long ordinary decode rows from NAX or tiled prefill: a one-row
  // decode sequence would otherwise occupy a whole BQ-row prefill tile with
  // no KV-length parallelism. Spec verify windows keep their whole-batch route.
  const bool split_mixed_batch = num_decode_requests > 0
      && num_decode_requests < num_seqs
      && num_decode_tokens == num_decode_requests
      && max_decode_context_len >= kMixedDecodeMinContext
      && (use_nax || tile_config.has_value());
  if (split_mixed_batch) {
    // The decode prefix reads only the decode rows. Size its split-KV
    // partitions and scratch by their context bound, never by max_seq_len:
    // that bound also covers the prefill rows and may be a whole allocation
    // bound, which the prefill kernels ignore but split-KV would plan for.
    const int decode_max_seq_len = std::min(max_seq_len, max_decode_context_len);
    // A caller-supplied plan for exactly the decode prefix (validated at the
    // binding against max_seq_len and max_decode_context_len) opts the prefix
    // into GQA and tightens the bound to the plan's longest decode row.
    const bool prefix_plan = gqa_length_plan.num_requests == num_decode_requests;
    dispatch_paged_attention_v2_online(
        out, query, key_cache, value_cache,
        num_kv_heads, scale, softcap,
        block_tables, seq_lens, cu_seqlens_q,
        block_size,
        prefix_plan ? gqa_length_plan.max_length : decode_max_seq_len,
        sliding_window, window_seqlen_q, s,
        key_scale_cache, value_scale_cache, key_zero_cache, v_centroids,
        use_turboquant, k_bits, v_bits, sinks, nullptr,
        num_decode_requests, num_decode_tokens, max_decode_context_len,
        num_decode_tokens, gqa_disabled, 0,
        prefix_plan ? gqa_length_plan : GqaDecodeLengthPlan{});
    // Keep the decode prefix's GQA selection visible in the diagnostics.
    const int decode_partition = g_last_gqa_partition.load(std::memory_order_relaxed);
    const int decode_requests = g_last_gqa_requests.load(std::memory_order_relaxed);
    // Decode rows lead the batch, so the first prefill sequence starts at
    // q-block cu_seqlens_q[D] / BQ + D with D = num_decode_requests.
    if (use_nax) {
      dispatch_paged_attention_nax(
          out, query, key_cache, value_cache,
          num_kv_heads, scale, softcap,
          block_tables, seq_lens, cu_seqlens_q,
          block_size, sliding_window,
          num_decode_tokens / kNaxBQ + num_decode_requests, s, sinks);
      record_paged_dispatch(PagedDispatch::MixedNaxPrefillDecode,
                            decode_partition, decode_requests);
      return;
    }
    const int q_block_offset =
        num_decode_tokens / tile_config->BQ + num_decode_requests;
    dispatch_paged_attention_tiled(
        out, query, key_cache, value_cache,
        num_kv_heads, scale, softcap,
        block_tables, seq_lens, cu_seqlens_q,
        block_size, max_seq_len, sliding_window, q_block_offset,
        *tile_config, s, sinks, mm_prefix_ranges);
    record_paged_dispatch(PagedDispatch::MixedPrefillDecode,
                          decode_partition, decode_requests);
    return;
  }

  // TurboQuant stays excluded from the tiled kernel. Sink prefill can use it:
  // pagedattention_tiled.metal folds the sink into each row's denominator-only
  // softmax state before final normalization.
  if (use_nax) {
    record_paged_dispatch(PagedDispatch::NaxPrefill);
    dispatch_paged_attention_nax(
        out, query, key_cache, value_cache,
        num_kv_heads, scale, softcap,
        block_tables, seq_lens, cu_seqlens_q,
        block_size, sliding_window, 0, s, sinks);
    return;
  }
  if (tile_config) {
    record_paged_dispatch(PagedDispatch::TiledPrefill);
    dispatch_paged_attention_tiled(
        out, query, key_cache, value_cache,
        num_kv_heads, scale, softcap,
        block_tables, seq_lens, cu_seqlens_q,
        block_size, max_seq_len, sliding_window, 0,
        *tile_config, s, sinks, mm_prefix_ranges);
    return;
  }
  if (mm_prefix_ranges != nullptr) {
    // paged_attention_primitive_fn already rejects every such batch eagerly;
    // this guard keeps the mask from being dropped silently should a new
    // routing condition appear above.
    throw std::invalid_argument(
        "mm_prefix ranges need the tiled prefill kernel, but this batch was "
        "routed to the per-token kernel");
  }

  // Fallback: original per-token kernel
  auto& d = metal::device(s.device);

  int num_heads  = static_cast<int>(query.shape(1));

  auto dt        = dtype_to_metal(query.dtype());
  auto k_cache_dt = dtype_to_metal(key_cache.dtype());
  auto v_cache_dt = dtype_to_metal(value_cache.dtype());
  // Window mode: one threadgroup per (segment, kWindowRows-row sub-window)
  // instead of per token.  The function constant carries the sub-window
  // count so the kernel's grid.y decomposition folds to constants.
  int window_q_fc = 0;
  if (window_batch) {
    window_q_fc = (window_seqlen_q + kWindowRows - 1) / kWindowRows;
  }
  const int grid_y =
      window_batch ? num_seqs * window_q_fc : total_q_tokens;

  // ----- Split-KV (flash-decoding) occupancy gate ------------------------
  const bool pure_decode = !has_prefill;  // every seq has exactly 1 query token
  const int max_num_partitions =
      (max_seq_len + kPartitionSize - 1) / kPartitionSize;
  // TurboQuant and sliding-window batches take the split too.  TQ partials
  // stay in the rotated domain with one inverse FWHT in the reduce (measured
  // -23% TPOT at conc=1/8K); windowed batches mask per partition, and a
  // fully-masked partition contributes exact zeros — epsilon-normalized
  // partial, zero merge weight (Gemma-4 E2B measured -35% TPOT at conc=1).
  // Window batches keep split-KV eligibility: their per-row partition
  // stats/tmp_out land at the same per-token rows the reduce kernel reads.
  // Gate on the split-equivalent grid (one row per token), not the
  // window-compacted one: window mode halves grid.y, so gating on the
  // dispatch grid flips the partition decision relative to the split
  // path in a band of shapes whose location depends on the core count
  // (min_decode_grid).  Inside that band the two paths run different
  // kernel families (_ps512 vs _ps0) and the bitwise contract breaks --
  // observed on an M4 Pro 16-core for 32q/8kv single-sequence windows,
  // invisible on 10-core M4 / M2 Ultra whose thresholds land elsewhere.
  // (For non-window batches grid_y == total_q_tokens, so this is the
  // same value the gate always used.)
  const int gate_grid = num_heads * total_q_tokens;  // grid.z = 1 occupancy
  const bool partition = (pure_decode || window_batch)
      && gate_grid < min_decode_grid() && max_num_partitions >= 2;

  auto* lib = d.get_library("paged_attention_v2_kern");
  auto& enc = metal::get_command_encoder(s);

  // Split-KV scratch factory shared by the split paths below: partial output
  // + softmax (max, exp-sum) stats.  Tiny (~64 KB tmp_out @ conc=1/8K).
  // add_temporary keeps them alive until the command buffer completes; MLX
  // auto-inserts a barrier between the two dispatches via input/output
  // dependency tracking (set_output here -> set_input in the reduce),
  // mirroring MLX's own sdpa_vector_2pass.
  auto make_temp = [&](Shape shape, Dtype dtype) {
    array a(std::move(shape), dtype, nullptr, {});
    a.set_data(allocator::malloc(a.nbytes()));
    enc.add_temporary(a);
    return a;
  };

  // Ordinary decode has exactly one query row per request. Scheduler counts
  // distinguish it from expanded verification and one-token prefill segments.
  // A mixed batch's decode prefix qualifies only with its own length plan.
  const bool gqa_decode_rows =
      total_q_tokens == num_seqs &&
      ((num_seqs == 1 &&
        (num_decode_requests == -1 || num_decode_requests == 1)) ||
       (num_seqs > 1 && num_decode_requests == num_seqs &&
        num_decode_tokens == num_seqs));
  // Enforce geometry at the actual dispatch boundary, including the private
  // forced-partition entry. That entry bypasses only occupancy selection.
  const bool gqa_supported =
      !gqa_disabled
      && pure_decode && window_seqlen_q <= 1 && gqa_decode_rows
      && (query.dtype() == float16 || query.dtype() == bfloat16)
      && query.dtype() == key_cache.dtype()
      && query.dtype() == value_cache.dtype()
      && !use_turboquant && softcap <= 0.f && sinks == nullptr
      && sliding_window < 0
      && gqa_decode_geometry_supported(
          num_heads, num_kv_heads, head_size, block_size);
  const int gqa_partition_size = !gqa_supported ? 0
      : gqa_test_partition > 0 ? gqa_test_partition
      : gqa_decode_plan_for_lengths(
            num_heads, num_kv_heads, head_size,
            num_seqs == 1 && decode_only_rows == 0
                ? gqa_decode_length_plan(std::array<int, 1>{max_seq_len})
                : gqa_length_plan,
            detected_gpu_core_count(), block_size, d.get_architecture(), max_seq_len);
  const int64_t gqa_partitions = gqa_partition_size > 0
      ? (static_cast<int64_t>(max_seq_len) + gqa_partition_size - 1) /
            gqa_partition_size : 0;
  // Check the actual reducer pipeline before allocating scratch or encoding
  // the GQA producer. Dynamic statistics alone can fit while their sum with
  // static workspace exceeds the device limit. Reuse this pipeline below.
  const bool scratch_fits = gqa_partition_size > 0 && gqa_scratch_fits(
      total_q_tokens, num_heads, gqa_partitions, head_size, query.itemsize());
  auto* gqa_rkernel = scratch_fits
      ? paged_attention_v2_reduce_kernel(
            d, dt, head_size, gqa_partition_size, false, false)
      : nullptr;
  const bool gqa_decode = scratch_fits
      && paged_reduce_memory_fits(
          paged_reduce_threadgroup_bytes(gqa_partitions),
          gqa_rkernel->staticThreadgroupMemoryLength(),
          d.mtl_device()->maxThreadgroupMemoryLength());
  if (gqa_decode) {
    const int gqa_num_partitions = static_cast<int>(gqa_partitions);
    const int gqa_group = num_heads / num_kv_heads;
    record_paged_dispatch(PagedDispatch::GqaDecode, gqa_partition_size, num_seqs);
    const auto gname = gqa_decode_kernel_name(
        dt, head_size, block_size, gqa_partition_size);
    // Instantiated in pagedattention.metal and shipped in the v2 metallib.
    // A missing specialization is a build/packaging bug: fail loudly
    // rather than silently drop eligible long decode back to the slower
    // per-token kernel.
    auto* gkernel = d.get_kernel(gname, lib, gname, {});

    array g_tmp_out = make_temp(
        Shape{total_q_tokens, num_heads, gqa_num_partitions, head_size},
        query.dtype());
    array g_exp_sums =
        make_temp(Shape{total_q_tokens, num_heads, gqa_num_partitions}, float32);
    array g_max_logits =
        make_temp(Shape{total_q_tokens, num_heads, gqa_num_partitions}, float32);

    enc.set_compute_pipeline_state(gkernel);
    enc.set_output_array(g_exp_sums, 0);
    enc.set_output_array(g_max_logits, 1);
    enc.set_output_array(g_tmp_out, 2);
    enc.set_input_array(query, 3);
    enc.set_input_array(key_cache, 4);
    enc.set_input_array(value_cache, 5);
    int32_t g_nkv = static_cast<int32_t>(num_kv_heads);
    enc.set_bytes(g_nkv, 8);
    enc.set_bytes(scale, 9);
    enc.set_input_array(block_tables, 11);
    enc.set_input_array(seq_lens, 12);
    int32_t g_max_blocks = static_cast<int32_t>(block_tables.shape(1));
    enc.set_bytes(g_max_blocks, 13);
    bind_paged_attn_strides(enc, query, key_cache, block_size);
    // One threadgroup per (partition, kv head, sequence); each simdgroup
    // owns one query head of the GQA group.
    enc.dispatch_threadgroups(
        MTL::Size::Make(gqa_num_partitions, num_kv_heads, num_seqs),
        MTL::Size::Make(32 * gqa_group, 1, 1));

    dispatch_paged_attention_v2_reduce(
        d, enc, out, g_exp_sums, g_max_logits, g_tmp_out, seq_lens,
        cu_seqlens_q, num_seqs, total_q_tokens, num_heads, head_size,
        gqa_num_partitions, dt, /*use_sinks=*/false, nullptr,
        /*use_tq_fc=*/false, gqa_partition_size, gqa_rkernel);
    return;
  }

  std::string kname =
      "paged_attention_" + dt + "_cache_" + k_cache_dt + "_" + v_cache_dt +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_nt256_nsl32_ps" + std::to_string(partition ? kPartitionSize : 0);

  bool use_partitioning  = partition;
  bool use_alibi         = false;
  bool use_fp8           = false;
  bool use_sinks         = sinks != nullptr;
  bool use_tq_fc         = use_turboquant;
  int  k_bits_i          = k_bits;
  int  v_bits_i          = v_bits;

  // The hash name is MLX's cache key.  It MUST encode every function-constant
  // value that varies across calls.  ``kname`` already encodes the partition
  // suffix (_ps0 vs _ps512), which is 1:1 with use_partitioning.
  std::string hash_name = kname + "_v2"
      + "_tq" + (use_tq_fc ? "1" : "0")
      + "_kb" + std::to_string(k_bits_i)
      + "_vb" + std::to_string(v_bits_i)
      + "_wq" + std::to_string(window_q_fc)
      + "_sk" + (use_sinks ? "1" : "0");

  auto* kernel = d.get_kernel(
      kname, lib, hash_name,
      {{&use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)},
       {&use_alibi,        MTL::DataType::DataTypeBool, NS::UInteger(20)},
       {&use_fp8,          MTL::DataType::DataTypeBool, NS::UInteger(30)},
       {&use_sinks,        MTL::DataType::DataTypeBool, NS::UInteger(40)},
       {&use_tq_fc,        MTL::DataType::DataTypeBool, NS::UInteger(50)},
       {&k_bits_i,         MTL::DataType::DataTypeInt,  NS::UInteger(60)},
       {&v_bits_i,         MTL::DataType::DataTypeInt,  NS::UInteger(70)},
       {&window_q_fc,      MTL::DataType::DataTypeInt,  NS::UInteger(110)}});

  constexpr int NUM_THREADS    = 256;
  constexpr int NUM_SIMD_LANES = 32;
  constexpr int NUM_WARPS      = NUM_THREADS / NUM_SIMD_LANES;
  // Window mode widens the per-warp score slices to one BLOCK_SIZE slice
  // per row and stages the sub-window's query rows after them; the layout
  // must mirror the kernel's shared_mem carve exactly.
  const int rows_per_tg = window_batch ? kWindowRows : 1;
  int warp_scores_bytes = NUM_WARPS * rows_per_tg * block_size
                          * static_cast<int>(sizeof(float));
  int q_window_bytes = window_batch
      ? kWindowRows * head_size * static_cast<int>(query.itemsize())
      : 0;
  int merge_bytes = (2 * NUM_WARPS + NUM_WARPS * head_size)
                    * static_cast<int>(sizeof(float));
  size_t shmem = static_cast<size_t>(
      std::max(warp_scores_bytes + q_window_bytes, merge_bytes));
  shmem = (shmem + 15) & ~size_t(15);

  // TurboQuant scale/zero/centroid buffers (slots 22-27); shared by both paths.
  auto bind_turboquant = [&]() {
    if (!use_turboquant) return;
    enc.set_input_array(*key_scale_cache,   22);
    enc.set_input_array(*value_scale_cache, 23);
    int32_t v_block_stride_i = static_cast<int32_t>(value_cache.strides()[0]);
    int32_t v_head_stride_i  = static_cast<int32_t>(value_cache.strides()[2]);
    enc.set_bytes(v_block_stride_i, 24);
    enc.set_bytes(v_head_stride_i,  25);
    enc.set_input_array(*key_zero_cache, 26);
    enc.set_input_array(*v_centroids, 27);
    int64_t scale_block_stride = key_scale_cache->strides()[0];
    enc.set_bytes(scale_block_stride, 28);
    int64_t scale_head_stride = key_scale_cache->strides()[2];
    enc.set_bytes(scale_head_stride, 29);
  };

  // Sink logits (slot 18); bound only when the kernel was specialized with
  // use_sinks, since the argument is declared under that function constant.
  auto bind_sinks = [&]() {
    if (!use_sinks) return;
    enc.set_input_array(*sinks, 18);
  };

  if (!partition) {
    // Single-pass path (grid.z = 1): the original decode/per-token kernel.
    record_paged_dispatch(
        window_batch ? PagedDispatch::Window : PagedDispatch::PerToken);
    enc.set_compute_pipeline_state(kernel);
    enc.set_threadgroup_memory_length(shmem, 0);
    bind_paged_attn_buffers(enc, out, query, key_cache, value_cache,
                            num_kv_heads, softcap, block_tables, seq_lens,
                            cu_seqlens_q, block_size, sliding_window);
    enc.set_bytes(scale, 9);
    bind_turboquant();
    bind_sinks();
    enc.dispatch_threadgroups(
        MTL::Size::Make(num_heads, grid_y, 1),
        MTL::Size::Make(NUM_THREADS, 1, 1));
    return;
  }

  // ----- Split-KV path: paged_attention(_ps512) -> paged_attention_v2_reduce.
  record_paged_dispatch(
      window_batch ? PagedDispatch::WindowSplitKv : PagedDispatch::SplitKv);
  array tmp_out = make_temp(
      Shape{total_q_tokens, num_heads, max_num_partitions, head_size},
      query.dtype());
  array exp_sums =
      make_temp(Shape{total_q_tokens, num_heads, max_num_partitions}, float32);
  array max_logits =
      make_temp(Shape{total_q_tokens, num_heads, max_num_partitions}, float32);

  // Pass 1: partitioned kernel writes partials.  Out slot (2) = tmp_out scratch.
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(shmem, 0);
  bind_paged_attn_buffers(enc, tmp_out, query, key_cache, value_cache,
                          num_kv_heads, softcap, block_tables, seq_lens,
                          cu_seqlens_q, block_size, sliding_window);
  enc.set_bytes(scale, 9);
  enc.set_output_array(exp_sums, 0);
  enc.set_output_array(max_logits, 1);
  bind_turboquant();
  // The partitioned kernel does not fold the sink itself (the reduce does, so
  // it is counted once rather than once per partition), but slot 18 is still a
  // declared argument under function constant 40, so it must be bound here.
  bind_sinks();
  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, grid_y, max_num_partitions),
      MTL::Size::Make(NUM_THREADS, 1, 1));

  // Pass 2: reduce per-partition partials -> out (log-sum-exp combine).
  dispatch_paged_attention_v2_reduce(
      d, enc, out, exp_sums, max_logits, tmp_out, seq_lens, cu_seqlens_q,
      num_seqs, total_q_tokens, num_heads, head_size, max_num_partitions, dt,
      use_sinks, sinks, use_tq_fc);
}

// ---------------------------------------------------------------------------
// Paged attention primitive (read-only): paged_attention_v2_online only.
//
// Single output: attention result.  The KV cache is read-only — cache
// writes are handled upstream by MLX-native scatter (pure functional).
// This is a clean pure function: inputs → output, no side effects.
// ---------------------------------------------------------------------------

class PagedAttentionPrimitive : public UnaryPrimitive {
 public:
  PagedAttentionPrimitive(
      Stream stream, int num_kv_heads, float scale, float softcap,
      int block_size, int max_seq_len, int sliding_window,
      bool use_turboquant = false, int k_bits = 8, int v_bits = 3,
      int window_seqlen_q = 1, bool use_sinks = false,
      bool use_mm_prefix = false, int num_decode_requests = -1,
      int num_decode_tokens = 0, int max_decode_context_len = 0,
      bool gqa_disabled = false, int gqa_test_partition = 0,
      const GqaDecodeLengthPlan& gqa_length_plan = {})
      : UnaryPrimitive(stream),
        num_kv_heads_(num_kv_heads), scale_(scale), softcap_(softcap),
        block_size_(block_size), max_seq_len_(max_seq_len),
        sliding_window_(sliding_window),
        use_turboquant_(use_turboquant), k_bits_(k_bits), v_bits_(v_bits),
        window_seqlen_q_(window_seqlen_q), use_sinks_(use_sinks),
        use_mm_prefix_(use_mm_prefix),
        num_decode_requests_(num_decode_requests),
        num_decode_tokens_(num_decode_tokens),
        max_decode_context_len_(max_decode_context_len),
        gqa_disabled_(gqa_disabled), gqa_test_partition_(gqa_test_partition),
        gqa_length_plan_(gqa_length_plan) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error(
        "PagedAttentionPrimitive only supports GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    // Non-TQ inputs: [query, key_cache, value_cache, block_tables, seq_lens, cu_seqlens_q]
    // TQ inputs:     [query, key_cache, value_cache, block_tables, seq_lens, cu_seqlens_q,
    //                 key_scale_cache, value_scale_cache, key_zero_cache, v_centroids]
    // Sinks append one array at slot 6.  TQ and sinks are mutually exclusive
    // (rejected in paged_attention_primitive_fn), so the two never collide.
    // mm_prefix ranges (Gemma 4 vision) append one more array after
    // everything else: slot 6 plain, slot 7 with sinks; TQ + mm_prefix is
    // rejected in paged_attention_primitive_fn, so slot 10 never occurs.
    out.set_data(allocator::malloc(out.nbytes()));
    const array* ks = use_turboquant_ ? &inputs[6] : nullptr;
    const array* vs = use_turboquant_ ? &inputs[7] : nullptr;
    const array* kz = use_turboquant_ ? &inputs[8] : nullptr;
    const array* vc = use_turboquant_ ? &inputs[9] : nullptr;
    const array* sk = use_sinks_ ? &inputs[6] : nullptr;
    const array* mp = use_mm_prefix_ ? &inputs[use_sinks_ ? 7 : 6] : nullptr;
    dispatch_paged_attention_v2_online(
        out,
        inputs[0],               // query
        inputs[1], inputs[2],    // key_cache, value_cache
        num_kv_heads_, scale_, softcap_,
        inputs[3], inputs[4], inputs[5],  // block_tables, seq_lens, cu_seqlens_q
        block_size_, max_seq_len_, sliding_window_, window_seqlen_q_,
        stream(),
        ks, vs, kz, vc, use_turboquant_, k_bits_, v_bits_, sk, mp,
        num_decode_requests_, num_decode_tokens_, max_decode_context_len_,
        0, gqa_disabled_, gqa_test_partition_, gqa_length_plan_);
  }

  const char* name() const override { return "PagedAttention"; }

  bool is_equivalent(const Primitive& other) const override {
    auto* rhs = dynamic_cast<const PagedAttentionPrimitive*>(&other);
    return rhs && rhs->num_kv_heads_ == num_kv_heads_
        && rhs->scale_ == scale_ && rhs->softcap_ == softcap_
        && rhs->block_size_ == block_size_
        && rhs->max_seq_len_ == max_seq_len_
        && rhs->sliding_window_ == sliding_window_
        && rhs->use_turboquant_ == use_turboquant_
        && rhs->k_bits_ == k_bits_
        && rhs->v_bits_ == v_bits_
        && rhs->window_seqlen_q_ == window_seqlen_q_
        && rhs->use_sinks_ == use_sinks_
        && rhs->use_mm_prefix_ == use_mm_prefix_
        && rhs->num_decode_requests_ == num_decode_requests_
        && rhs->num_decode_tokens_ == num_decode_tokens_
        && rhs->max_decode_context_len_ == max_decode_context_len_
        && rhs->gqa_disabled_ == gqa_disabled_
        && rhs->gqa_test_partition_ == gqa_test_partition_
        && rhs->gqa_length_plan_ == gqa_length_plan_;
  }

 private:
  int num_kv_heads_;
  float scale_;
  float softcap_;
  int block_size_;
  int max_seq_len_;
  int sliding_window_;
  bool use_turboquant_;
  int k_bits_;
  int v_bits_;
  int window_seqlen_q_;
  bool use_sinks_;
  bool use_mm_prefix_;
  int num_decode_requests_;
  int num_decode_tokens_;
  int max_decode_context_len_;
  bool gqa_disabled_;
  int gqa_test_partition_;
  GqaDecodeLengthPlan gqa_length_plan_;
};

static array paged_attention_primitive_fn(
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
    const std::vector<int>& gqa_context_lens = {},
    const std::optional<GqaDecodeLengthPlan>& gqa_length_plan = std::nullopt) {
  if (gqa_length_plan && !gqa_context_lens.empty())
    throw std::invalid_argument("pass only one of gqa_length_plan and gqa_context_lens");
  const auto lengths = gqa_length_plan
      ? *gqa_length_plan : gqa_decode_length_plan(gqa_context_lens);
  if (gqa_length_plan || !gqa_context_lens.empty()) {
    const bool shaped = query.ndim() == 3 && seq_lens.ndim() == 1 &&
        cu_seqlens_q.ndim() == 1 && lengths.num_requests > 0 &&
        lengths.max_length <= max_seq_len &&
        seq_lens.shape(0) + 1 == cu_seqlens_q.shape(0);
    const bool whole_decode = shaped &&
        lengths.num_requests == query.shape(0) &&
        lengths.num_requests == seq_lens.shape(0);
    // Or the leading ordinary decode rows of a mixed batch.
    const bool decode_prefix = shaped &&
        lengths.num_requests == num_decode_requests &&
        num_decode_tokens == num_decode_requests &&
        num_decode_requests < seq_lens.shape(0) &&
        lengths.max_length <= max_decode_context_len;
    if (!whole_decode && !decode_prefix) {
      throw std::invalid_argument(
          "gqa_context_lens or gqa_length_plan requires one positive length per "
          "decode row (every row of a decode batch, or the leading ordinary "
          "decode rows of a mixed batch), bounded by max_seq_len and "
          "max_decode_context_len");
    }
  }
  if (sinks != nullptr) {
    // Upstream MLX refuses the same combination
    // (mlx_lm/models/base.py: "Quantized SDPA does not support attention
    // sinks"), and the TurboQuant reduce already owns the deferred inverse
    // FWHT, so folding a sink there would need its own derivation.  Reject
    // rather than silently drop the sink term.
    if (use_turboquant) {
      throw std::invalid_argument(
          "attention sinks are not supported with TurboQuant quantized KV; "
          "pass sinks=None or disable TurboQuant for this layer");
    }
    if (sinks->ndim() != 1) {
      throw std::invalid_argument(
          "sinks must be 1-D with one entry per query head, got ndim=" +
          std::to_string(sinks->ndim()));
    }
    const int num_q_heads = static_cast<int>(query.shape(1));
    if (static_cast<int>(sinks->shape(0)) != num_q_heads) {
      throw std::invalid_argument(
          "sinks must have one entry per query head (" +
          std::to_string(num_q_heads) + "), got " +
          std::to_string(sinks->shape(0)));
    }
    if (sinks->dtype() != float32) {
      throw std::invalid_argument(
          "sinks must be float32; the kernel reads them as device float and "
          "folds them into a float32 softmax accumulator");
    }
  }
  if (mm_prefix_ranges != nullptr) {
    // Every routing condition of dispatch_paged_attention_v2_online that
    // would bypass the tiled kernel is rejected here, eagerly, so a caller
    // gets a ValueError instead of a silently causal image block.
    if (use_turboquant) {
      throw std::invalid_argument(
          "mm_prefix ranges are not supported with TurboQuant quantized KV; "
          "the Gemma 4 vision sidecar refuses TurboQuant at load time");
    }
    if (mm_prefix_ranges->dtype() != int32) {
      throw std::invalid_argument("mm_prefix_ranges must be int32");
    }
    const int total_q = static_cast<int>(query.shape(0));
    if (mm_prefix_ranges->ndim() != 2
        || static_cast<int>(mm_prefix_ranges->shape(0)) != total_q
        || mm_prefix_ranges->shape(1) != 2) {
      throw std::invalid_argument(
          "mm_prefix_ranges must have shape (query rows, 2) = (" +
          std::to_string(total_q) + ", 2)");
    }
    const int num_segments = static_cast<int>(cu_seqlens_q.shape(0)) - 1;
    const int head_size_q = static_cast<int>(query.shape(2));
    if (query.dtype() == float32 || query.dtype() != key_cache.dtype()
        || window_seqlen_q > 1 || total_q <= num_segments
        || !select_tile_config(head_size_q)) {
      throw std::invalid_argument(
          "mm_prefix ranges need the tiled prefill kernel: a non-float32 "
          "query matching the KV cache dtype, at least one multi-token "
          "segment, no spec-decode verification window and a head size in "
          "{64, 96, 128, 256, 512}");
    }
  }
  // window_seqlen_q must equal the longest cu_seqlens_q segment: window-mode
  // threadgroups only exist for ceil(window_seqlen_q / kWindowRows)
  // sub-windows per segment, so an understated value leaves the tail rows of
  // a longer segment unwritten (garbage logits, no error), and an overstated
  // one dispatches empty threadgroups.  Only the spec-verify path passes a
  // window hint and its cu_seqlens_q is a small host-built array, so
  // materializing it to check the real segment lengths is cheap, and the
  // plain-decode path (window_seqlen_q == 1) skips the block entirely.
  if (window_seqlen_q < 1) {
    throw std::invalid_argument(
        "window_seqlen_q must be >= 1 (1 = per-token decode), got " +
        std::to_string(window_seqlen_q) +
        "; a non-positive hint would silently select non-window routing");
  }
  if (window_seqlen_q > 1) {
    const int head_size_q = static_cast<int>(query.shape(2));
    if (head_size_q > kWindowMaxHeadSize) {
      throw std::invalid_argument(
          "window_seqlen_q=" + std::to_string(window_seqlen_q) +
          " requires head_size <= " + std::to_string(kWindowMaxHeadSize) +
          ", got " + std::to_string(head_size_q) +
          "; wider heads must keep the expanded per-token verify layout "
          "(window-mode register state scales with rows * head_size)");
    }
    array cu = cu_seqlens_q;
    cu.eval();
    if (cu.dtype() != int32) {
      throw std::invalid_argument(
          "cu_seqlens_q must be int32 for window-mode validation");
    }
    const int32_t* cu_data = cu.data<int32_t>();
    const int num_segments = static_cast<int>(cu.shape(0)) - 1;
    const int total_q = static_cast<int>(query.shape(0));
    if (num_segments < 1 || cu_data[0] != 0 ||
        cu_data[num_segments] != total_q) {
      throw std::invalid_argument(
          "cu_seqlens_q must start at 0 and end at the query row count (" +
          std::to_string(total_q) + "), got " +
          std::to_string(num_segments) + " segment(s)");
    }
    int max_segment = 0;
    for (int i = 0; i < num_segments; ++i) {
      const int seg = cu_data[i + 1] - cu_data[i];
      if (seg < 1) {
        throw std::invalid_argument(
            "cu_seqlens_q segment " + std::to_string(i) + " is empty (" +
            std::to_string(cu_data[i]) + " -> " + std::to_string(cu_data[i + 1]) +
            "); every window-mode segment needs at least one query row");
      }
      max_segment = std::max(max_segment, seg);
    }
    if (max_segment != window_seqlen_q) {
      throw std::invalid_argument(
          "window_seqlen_q=" + std::to_string(window_seqlen_q) +
          " does not match the longest cu_seqlens_q segment (" +
          std::to_string(max_segment) +
          "); an understated value leaves window rows unwritten");
    }
  }
  int k_bits = use_turboquant ? get_bits(quant_type) : 8;
  auto prim = std::make_shared<PagedAttentionPrimitive>(
      default_stream(Device::gpu),
      num_kv_heads, scale, softcap,
      block_size, max_seq_len, sliding_window,
      use_turboquant, k_bits, v_bits, window_seqlen_q, sinks != nullptr,
      mm_prefix_ranges != nullptr, num_decode_requests, num_decode_tokens,
      max_decode_context_len, gqa_disabled, gqa_test_partition, lengths);
  std::vector<array> inputs = {query, key_cache, value_cache,
                               block_tables, seq_lens, cu_seqlens_q};
  if (use_turboquant) {
    inputs.insert(inputs.end(), {*key_scale_cache, *value_scale_cache,
                                 *key_zero_cache, *v_centroids});
  } else if (sinks != nullptr) {
    inputs.push_back(*sinks);
  }
  if (mm_prefix_ranges != nullptr) {
    inputs.push_back(*mm_prefix_ranges);
  }
  return array(query.shape(), query.dtype(), std::move(prim), std::move(inputs));
}

// ---------------------------------------------------------------------------
// tq_encode — fused TurboQuant encode + paged scatter
//
// Replaces the Python turbo_quant_encode() + 5 MLX scatters on the hot path.
// Lives in the v2 library because turboquant.metal is concatenated there.
// Supports all K quants in QUANT_PARAMS: signed 8-bit (q8_0/int8) and
// unsigned {8,5,4,2}-bit (uint8/q5_0/q4_0/int4/uint4/int2/uint2).  V supports
// any v_bits in [1, 8] via the v_centroids buffer.
//
// Wrapped in a proper MLX Primitive so that the five cache writes become
// new MLX-graph nodes with provenance pointing at this primitive.  This is
// critical: paged_attention_primitive runs on a separate command buffer and
// reads the same cache arrays, and the downstream decode step re-reads
// them too.  Without a real graph edge, MLX's scheduler has no idea this
// op must complete before those readers — the primitives submit to their
// encoders out of order and the reader sees uninitialised / in-flight
// bytes (silent GPU fault → EngineCore crash on first real request).
//
// Each of the five outputs aliases the corresponding input cache buffer
// via copy_shared_buffer, so the kernel writes in place (no extra
// allocation) while MLX still gets clean graph provenance.  The caller
// rebinds kv_cache.key_caches[layer_idx] = new_k_cache so the next decode
// step's tq_encode input reads through this primitive's output.
// ---------------------------------------------------------------------------

class TQEncodePrimitive : public Primitive {
 public:
  TQEncodePrimitive(Stream stream, int v_bits, int k_bits, bool k_signed)
      : Primitive(stream),
        v_bits_(v_bits),
        k_bits_(k_bits),
        k_signed_(k_signed) {}

  void eval_cpu(
      const std::vector<array>&,
      std::vector<array>&) override {
    throw std::runtime_error("TQEncodePrimitive only supports GPU");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    // inputs:  0=key, 1=value,
    //          2=key_cache_in, 3=value_cache_in,
    //          4=key_scale_in, 5=value_scale_in, 6=key_zero_in,
    //          7=slot_mapping, 8=v_centroids
    // outputs: 0=new_key_cache, 1=new_value_cache,
    //          2=new_key_scale, 3=new_value_scale, 4=new_key_zero

    // Alias each output onto the corresponding input cache buffer.  The
    // kernel writes in place — the aliasing simply gives the output a
    // distinct graph identity (new ArrayDesc with primitive = this) while
    // sharing the underlying Metal buffer.  After eval, Python rebinds
    // kv_cache.<cache>[layer_idx] = outputs[i], so subsequent ops naturally
    // depend on this primitive via the MLX graph.
    outputs[0].copy_shared_buffer(inputs[2]);
    outputs[1].copy_shared_buffer(inputs[3]);
    outputs[2].copy_shared_buffer(inputs[4]);
    outputs[3].copy_shared_buffer(inputs[5]);
    outputs[4].copy_shared_buffer(inputs[6]);

    const array& key          = inputs[0];
    const array& value        = inputs[1];
    const array& slot_mapping = inputs[7];
    const array& v_centroids  = inputs[8];

    auto s = stream();
    auto& d = metal::device(s.device);

    // key shape: [num_tokens, num_kv_heads, head_size]
    int num_tokens   = static_cast<int>(key.shape(0));
    int num_kv_heads = static_cast<int>(key.shape(1));
    int head_size    = static_cast<int>(key.shape(2));
    int block_size   = static_cast<int>(inputs[2].shape(1));

    auto kv_dt = dtype_to_metal(key.dtype());
    std::string kname = "tq_encode_" + kv_dt +
                        "_hs" + std::to_string(head_size);

    // Function constants control bit widths + signedness used inside the
    // kernel.  The hash name MUST encode them so MLX caches the right
    // specialization per (k_bits, k_signed, v_bits) tuple.
    int  k_bits_i   = k_bits_;
    int  v_bits_i   = v_bits_;
    bool k_signed_b = k_signed_;
    std::string hash_name = kname +
        "_kb" + std::to_string(k_bits_i) +
        "_ks" + (k_signed_b ? "1" : "0") +
        "_vb" + std::to_string(v_bits_i);

    auto* lib = d.get_library("paged_attention_v2_kern");
    auto* kernel = d.get_kernel(
        kname, lib, hash_name,
        {{&k_bits_i,   MTL::DataType::DataTypeInt,  NS::UInteger(80)},
         {&k_signed_b, MTL::DataType::DataTypeBool, NS::UInteger(81)},
         {&v_bits_i,   MTL::DataType::DataTypeInt,  NS::UInteger(90)}});

    int32_t num_kv_heads_i = static_cast<int32_t>(num_kv_heads);
    int32_t block_size_i   = static_cast<int32_t>(block_size);

    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(key,                0);
    enc.set_input_array(value,              1);
    enc.set_output_array(outputs[0],        2);
    enc.set_output_array(outputs[1],        3);
    enc.set_output_array(outputs[2],        4);
    enc.set_output_array(outputs[3],        5);
    enc.set_output_array(outputs[4],        6);
    enc.set_input_array(slot_mapping,       7);
    enc.set_input_array(v_centroids,        8);
    enc.set_bytes(num_kv_heads_i,           9);
    enc.set_bytes(block_size_i,             10);
    int64_t k_block_stride = inputs[2].strides()[0];
    int64_t v_block_stride = inputs[3].strides()[0];
    int64_t scale_block_stride = inputs[4].strides()[0];
    enc.set_bytes(k_block_stride, 11);
    enc.set_bytes(v_block_stride, 12);
    enc.set_bytes(scale_block_stride, 13);
    int64_t k_head_stride = inputs[2].strides()[2];
    int64_t v_head_stride = inputs[3].strides()[2];
    int64_t scale_head_stride = inputs[4].strides()[2];
    enc.set_bytes(k_head_stride, 14);
    enc.set_bytes(v_head_stride, 15);
    enc.set_bytes(scale_head_stride, 16);

    enc.dispatch_threadgroups(
        MTL::Size::Make(num_tokens, num_kv_heads, 1),
        MTL::Size::Make(head_size, 1, 1));

    // Intentionally no add_temporary: inside a primitive, MLX's evaluator
    // manages array lifetimes via the completion handler.  add_temporary
    // here would strip outputs[0..4] from the encoder's tracking and
    // silently defeat the fence for downstream primitives.
  }

  const char* name() const override { return "TQEncode"; }

  bool is_equivalent(const Primitive& other) const override {
    auto* rhs = dynamic_cast<const TQEncodePrimitive*>(&other);
    return rhs && rhs->v_bits_   == v_bits_
               && rhs->k_bits_   == k_bits_
               && rhs->k_signed_ == k_signed_;
  }

 private:
  int  v_bits_;
  int  k_bits_;
  bool k_signed_;
};

static std::vector<array> tq_encode_primitive_fn(
    const array& key, const array& value,
    const array& key_cache, const array& value_cache,
    const array& key_scale_cache, const array& value_scale_cache,
    const array& key_zero_cache,
    const array& slot_mapping, const array& v_centroids,
    int v_bits, int k_bits, bool k_signed) {
  // Accept every bit width present in QUANT_PARAMS (2/3/4/5/8).  Signed is
  // only legal at bits=8 because Python stores signed sub-8-bit types as
  // unsigned for packability (e.g. int4 is signed:False in QUANT_PARAMS).
  if (k_bits != 2 && k_bits != 3 && k_bits != 4 && k_bits != 5 && k_bits != 8) {
    throw std::runtime_error(
        "tq_encode: k_bits must be 2, 3, 4, 5, or 8 (got " +
        std::to_string(k_bits) + ")");
  }
  if (k_signed && k_bits != 8) {
    throw std::runtime_error(
        "tq_encode: signed K is only supported at k_bits=8 "
        "(matches QUANT_PARAMS in turboquant.py).");
  }
  int head_size = static_cast<int>(key.shape(2));
  if (head_size != 64 && head_size != 128 && head_size != 256 && head_size != 512) {
    throw std::runtime_error(
        "tq_encode: head_size must be 64, 128, 256, or 512 (got " +
        std::to_string(head_size) + ")");
  }

  auto prim = std::make_shared<TQEncodePrimitive>(
      default_stream(Device::gpu), v_bits, k_bits, k_signed);

  return array::make_arrays(
      {key_cache.shape(), value_cache.shape(),
       key_scale_cache.shape(), value_scale_cache.shape(),
       key_zero_cache.shape()},
      {key_cache.dtype(), value_cache.dtype(),
       key_scale_cache.dtype(), value_scale_cache.dtype(),
       key_zero_cache.dtype()},
      prim,
      {key, value,
       key_cache, value_cache,
       key_scale_cache, value_scale_cache, key_zero_cache,
       slot_mapping, v_centroids});
}

// ---------------------------------------------------------------------------
// reshape_and_cache — fused K/V paged scatter (non-quantized fp16/bf16/fp32)
//
// Replaces the two per-layer MLX scatters on the standard decode path
// (flat_k[slot_mapping] = k_3d; flat_v[slot_mapping] = v_3d) with one fused
// Metal dispatch that writes both K and V into the paged cache by slot_mapping,
// wiring up the pre-existing reshape_and_cache.metal kernel. Same in-place
// aliasing pattern as TQEncodePrimitive so the writes carry graph provenance
// for the downstream paged_attention read.
// ---------------------------------------------------------------------------

class ReshapeAndCachePrimitive : public Primitive {
 public:
  explicit ReshapeAndCachePrimitive(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("ReshapeAndCachePrimitive only supports GPU");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    // inputs:  0=key, 1=value, 2=key_cache_in, 3=value_cache_in, 4=slot_mapping
    // outputs: 0=new_key_cache, 1=new_value_cache (alias inputs 2,3 in place)
    outputs[0].copy_shared_buffer(inputs[2]);
    outputs[1].copy_shared_buffer(inputs[3]);

    const array& key          = inputs[0];
    const array& value        = inputs[1];
    const array& slot_mapping = inputs[4];

    auto s = stream();
    auto& d = metal::device(s.device);

    // key shape: [num_tokens, num_kv_heads, head_size]
    int num_tokens   = static_cast<int>(key.shape(0));
    int num_kv_heads = static_cast<int>(key.shape(1));
    int head_size    = static_cast<int>(key.shape(2));
    // cache shape: [num_blocks, block_size, num_kv_heads, head_size]
    int block_size   = static_cast<int>(inputs[2].shape(1));

    auto kv_dt    = dtype_to_metal(key.dtype());
    auto cache_dt = dtype_to_metal(inputs[2].dtype());
    std::string kname = "reshape_and_cache_kv_" + kv_dt + "_cache_" + cache_dt;

    // rac_use_fp8_scales (fc 100) = false: non-quantized cache, so the
    // k_scale/v_scale buffers (5,6) are absent from the signature.
    bool use_fp8 = false;
    std::string hash_name = kname + "_fp8_0";

    auto* lib = d.get_library("paged_attention_v2_kern");
    auto* kernel = d.get_kernel(
        kname, lib, hash_name,
        {{&use_fp8, MTL::DataType::DataTypeBool, NS::UInteger(100)}});

    int32_t key_stride_i   = static_cast<int32_t>(num_kv_heads * head_size);
    int32_t value_stride_i = key_stride_i;
    int32_t num_heads_i    = static_cast<int32_t>(num_kv_heads);
    int32_t head_size_i    = static_cast<int32_t>(head_size);
    int32_t block_size_i   = static_cast<int32_t>(block_size);

    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(key,          0);
    enc.set_input_array(value,        1);
    enc.set_output_array(outputs[0],  2);
    enc.set_output_array(outputs[1],  3);
    enc.set_input_array(slot_mapping, 4);
    enc.set_bytes(key_stride_i,   7);
    enc.set_bytes(value_stride_i, 8);
    enc.set_bytes(num_heads_i,    9);
    enc.set_bytes(head_size_i,    10);
    enc.set_bytes(block_size_i,   11);
    int64_t cache_block_stride = inputs[2].strides()[0];
    int64_t cache_token_stride = inputs[2].strides()[1];
    int64_t cache_head_stride = inputs[2].strides()[2];
    enc.set_bytes(cache_block_stride, 12);
    enc.set_bytes(cache_token_stride, 13);
    enc.set_bytes(cache_head_stride, 14);

    int tg = std::min(num_kv_heads * head_size, 256);
    enc.dispatch_threadgroups(
        MTL::Size::Make(num_tokens, 1, 1),
        MTL::Size::Make(tg, 1, 1));
  }

  const char* name() const override { return "ReshapeAndCache"; }

  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const ReshapeAndCachePrimitive*>(&other) != nullptr;
  }
};

static std::vector<array> reshape_and_cache_primitive_fn(
    const array& key, const array& value,
    const array& key_cache, const array& value_cache,
    const array& slot_mapping) {
  auto prim = std::make_shared<ReshapeAndCachePrimitive>(
      default_stream(Device::gpu));
  return array::make_arrays(
      {key_cache.shape(), value_cache.shape()},
      {key_cache.dtype(), value_cache.dtype()},
      prim,
      {key, value, key_cache, value_cache, slot_mapping});
}

// ---------------------------------------------------------------------------
// GDN linear attention — in-place paged state
// ---------------------------------------------------------------------------

static std::string gdn_source_;

void init_gdn_library(const std::string& src) {
  gdn_source_ = src;
  auto& d = metal::device(Device::gpu);
  d.get_library("gdn_kern", [&]() { return gdn_source_; });
}

// MLX Scatter/SliceUpdate copy their destination unless it is donatable; GDN
// pools are aliased across sibling layers, so pool[ids] = rows copies the full
// pool. mx.fast.metal_kernel allocates fresh outputs, so this Primitive aliases
// the pool with copy_shared_buffer; callers rebind the output so downstream
// readers depend on the write. Destination slots must be distinct.

class GDNStateScatterPrimitive : public Primitive {
 public:
  explicit GDNStateScatterPrimitive(
      Stream stream, bool zero = false, bool paged = false)
      : Primitive(stream), zero_(zero), paged_(paged) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GDNStateScatterPrimitive only supports GPU");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    // inputs:  0=pool_in, 1=src_rows, 2=dst_ids
    // outputs: 0=pool_out (aliases pool_in; the kernel writes in place)
    const array& pool    = inputs[0];
    const array& src     = inputs[1];
    const array& dst_ids = inputs[2];
    outputs[0].copy_shared_buffer(pool);

    int n = static_cast<int>(dst_ids.size());
    if (n == 0) {
      return;
    }
    int tokens_per_page = paged_ ? pool.shape(1) : 1;
    int row_elems = static_cast<int>(
        pool.size() / pool.shape(0) / tokens_per_page);

    // Both GPU kernels scatter the same rows. Select four elements per thread
    // for dense, vector-aligned rows, or one element per thread for other views.
    // This selection happens before launch.
    bool dense_row = true;
    size_t inner_stride = 1;
    for (int axis = pool.ndim() - 1; axis >= (paged_ ? 2 : 1); --axis) {
      dense_row &= pool.shape(axis) == 1 || pool.strides()[axis] == inner_stride;
      inner_stride *= pool.shape(axis);
    }
    // Row sizes alone do not guarantee aligned vector access to sliced views.
    const size_t vector_bytes = 4 * pool.itemsize();
    bool vec4 = dense_row && (row_elems % 4) == 0 && (pool.strides()[0] % 4) == 0
        && (!paged_ || (pool.strides()[1] % 4) == 0)
        && reinterpret_cast<uintptr_t>(pool.data<char>()) % vector_bytes == 0
        && (zero_ || reinterpret_cast<uintptr_t>(src.data<char>()) % vector_bytes == 0);
    int lanes = vec4 ? row_elems / 4 : row_elems;

    auto s = stream();
    auto& d = metal::device(s.device);
    auto dt = dtype_to_metal(pool.dtype());
    std::string kname =
        std::string("gdn_state_scatter_rows_") + (vec4 ? "vec4_" : "") + dt;
    auto* lib = d.get_library("gdn_kern");
    auto* kernel = d.get_kernel(kname, lib, kname, {});

    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);
    enc.set_output_array(outputs[0], 0);
    enc.set_input_array(src,         1);
    enc.set_input_array(dst_ids,     2);
    enc.set_bytes(lanes,             3);
    int64_t row_stride = pool.strides()[0] / (vec4 ? 4 : 1);
    int64_t token_stride = paged_ ? pool.strides()[1] / (vec4 ? 4 : 1) : 0;
    enc.set_bytes(row_stride,        4);
    enc.set_bytes(zero_,             8);
    enc.set_bytes(tokens_per_page,   9);
    enc.set_bytes(token_stride,     10);
    if (!vec4) {
      auto row_shape = pool.shape();
      auto row_strides = pool.strides();
      if (paged_) {
        row_shape.erase(row_shape.begin());
        row_strides.erase(row_strides.begin());
      }
      enc.set_vector_bytes(row_shape, 5);
      enc.set_vector_bytes(row_strides, 6);
      int ndim = dense_row ? 0 : static_cast<int>(row_shape.size());
      enc.set_bytes(ndim, 7);
    }

    // 2D thread grid: x walks the row, y selects the update row. One
    // threadgroup per row leaves most of the GPU idle on a 1 MiB slab.
    //
    // dispatch_threads maps to Metal's non-uniform dispatchThreads, so exactly
    // `lanes` threads run along x and the kernels need no bounds check. This is
    // the only dispatch_threads call in this file; the others round up through
    // dispatch_threadgroups, which here would write past the end of a row.
    int tg_x = std::min<int>(
        lanes, static_cast<int>(kernel->maxTotalThreadsPerThreadgroup()));
    enc.dispatch_threads(
        MTL::Size::Make(lanes, n, 1),
        MTL::Size::Make(tg_x, 1, 1));
  }

  const char* name() const override { return "GDNStateScatter"; }

  bool is_equivalent(const Primitive& other) const override {
    auto* rhs = dynamic_cast<const GDNStateScatterPrimitive*>(&other);
    return rhs && rhs->zero_ == zero_ && rhs->paged_ == paged_;
  }
 private:
  bool zero_;
  bool paged_;
};

static array gdn_state_scatter_primitive_fn(
    const array& pool, const array& src, const array& dst_ids,
    bool zero = false, bool paged = false) {
  if (pool.ndim() < (paged ? 3 : 2)) {
    throw std::runtime_error(
        "gdn_state_scatter: pool must be [num_slots, ...] or "
        "[num_blocks, block_size, ...] with paged=True");
  }
  if (paged && (zero || pool.shape(1) == 0)) {
    throw std::runtime_error(
        "gdn_state_scatter: paged writes require a positive block size and zero=False");
  }
  if (pool.dtype() != float16 &&
      pool.dtype() != bfloat16 &&
        pool.dtype() != float32 && pool.dtype() != uint8 && pool.dtype() != int8) {
    throw std::runtime_error(
        "gdn_state_scatter: pool dtype must be float16, bfloat16 or float32");
  }
  if (src.dtype() != pool.dtype()) {
    throw std::runtime_error("gdn_state_scatter: src and pool dtypes differ");
  }
  if (dst_ids.dtype() != int32) {
    throw std::runtime_error("gdn_state_scatter: dst_ids must be int32");
  }
  if (dst_ids.ndim() != 1) {
    throw std::runtime_error("gdn_state_scatter: dst_ids must be 1-D");
  }
  if (src.ndim() != pool.ndim() - (paged ? 1 : 0) ||
      !std::equal(
          pool.shape().begin() + (paged ? 2 : 1), pool.shape().end(),
          src.shape().begin() + 1)) {
    throw std::runtime_error(
        "gdn_state_scatter: src row shape does not match pool row shape");
  }
  if (!zero && static_cast<size_t>(src.shape(0)) != dst_ids.size()) {
    throw std::runtime_error(
        "gdn_state_scatter: one dst id per src row required");
  }
  // State components have dense inner rows but a padded physical page stride.
  // Never materialize the destination: that would break shared cache storage.
  auto contiguous_src = zero ? pool : contiguous(src);
  auto contiguous_ids = contiguous(dst_ids);
  auto prim = std::make_shared<GDNStateScatterPrimitive>(
      default_stream(Device::gpu), zero, paged);
  return array::make_arrays(
      {pool.shape()}, {pool.dtype()}, prim,
      {pool, contiguous_src, contiguous_ids})[0];
}

// ---------------------------------------------------------------------------
// MLA paged attention (RFC #360)
// ---------------------------------------------------------------------------

static std::string mla_source_;

void init_mla_library(const std::string& src) {
  mla_source_ = src;
  auto& d = metal::device(Device::gpu);
  d.get_library("paged_mla_kern", [&]() { return mla_source_; });
}

// Dispatch the MLA paged attention kernel.
//
// Buffer slot map (must match kernels_v2/mla.metal):
//   2: out          [total_q_tokens, num_heads, KV_LORA_RANK]
//   3: q_nope       [total_q_tokens, num_heads, KV_LORA_RANK]
//   4: q_pe         [total_q_tokens, num_heads, QK_ROPE_HEAD_DIM]
//   5: latent_cache [num_blocks, BLOCK_SIZE, KV_LORA_RANK + QK_ROPE_HEAD_DIM]
//   6: block_tables 7: context_lens 8: cu_seqlens_q
//   9: num_seqs 10: max_num_blocks_per_seq 11: scale
// The instantiated MLA specialization space — one row per distinct shape the
// instantiate_mla call sites in kernels_v2/mla.metal declare. The dispatch
// gate below and the drift test in tests/test_block_size_translation.py both
// read this one table, so keep the row format stable:
//   {kv_lora_rank, qk_rope_head_dim, block_size, heads_per_tg, num_threads,
//    partition_size}
// Each variant keeps the per-thread register footprint roughly constant
// (NUM_THREADS scaled inversely to G):
//   G=1 → NUM_THREADS=1024 (32 simdgroups, current sdpa_vector layout).
//   G=2 → NUM_THREADS=512  (16 simdgroups, 2× cross-head amortization).
struct MlaKernelSpec {
  int kv_lora_rank;
  int qk_rope_head_dim;
  int block_size;
  int heads_per_tg;
  int num_threads;
  int partition_size;
};
static constexpr MlaKernelSpec kMlaKernelSpecs[] = {
    {512, 64, 16, 1, 1024, 0},
    {512, 64, 32, 1, 1024, 0},
    {512, 64, 16, 2, 512, 0},
    {512, 64, 32, 2, 512, 0},
};

static void dispatch_mla_paged_attention(
    array& out,
    const array& q_nope,
    const array& q_pe,
    const array& latent_cache,
    const array& block_tables,
    const array& context_lens,
    const array& cu_seqlens_q,
    int block_size,
    float scale,
    int heads_per_tg,
    Stream s) {
  auto& d = metal::device(s.device);

  int total_q_tokens = static_cast<int>(q_nope.shape(0));
  int num_heads = static_cast<int>(q_nope.shape(1));
  int kv_lora_rank = static_cast<int>(q_nope.shape(2));
  int qk_rope_head_dim = static_cast<int>(q_pe.shape(2));
  int max_num_blocks_per_seq = static_cast<int>(block_tables.shape(1));
  int num_seqs = static_cast<int>(cu_seqlens_q.shape(0)) - 1;

  // The shape must match a kernel template instantiation — see
  // kMlaKernelSpecs for the admitted space.
  const MlaKernelSpec* spec = nullptr;
  for (const auto& candidate : kMlaKernelSpecs) {
    if (candidate.kv_lora_rank == kv_lora_rank &&
        candidate.qk_rope_head_dim == qk_rope_head_dim &&
        candidate.block_size == block_size &&
        candidate.heads_per_tg == heads_per_tg) {
      spec = &candidate;
      break;
    }
  }
  if (spec == nullptr) {
    throw std::runtime_error(
        "MLA kernel: no instantiation for kv_lora_rank=" +
        std::to_string(kv_lora_rank) + " qk_rope_head_dim=" +
        std::to_string(qk_rope_head_dim) + " block_size=" +
        std::to_string(block_size) + " heads_per_tg=" +
        std::to_string(heads_per_tg));
  }

  if (spec->partition_size != 0) {
    throw std::runtime_error(
        "MLA kernel: matched kv_lora_rank=" +
        std::to_string(spec->kv_lora_rank) + " qk_rope_head_dim=" +
        std::to_string(spec->qk_rope_head_dim) + " block_size=" +
        std::to_string(spec->block_size) + " heads_per_tg=" +
        std::to_string(spec->heads_per_tg) + " but partition_size=" +
        std::to_string(spec->partition_size) + " is not dispatchable");
  }

  if (num_heads % heads_per_tg != 0) {
    throw std::runtime_error(
        "MLA kernel: num_heads (" + std::to_string(num_heads) +
        ") must be divisible by heads_per_tg (" +
        std::to_string(heads_per_tg) + ")");
  }
  mla_validate_t_dtypes("MLA kernel", {
      {"q_nope", &q_nope},
      {"q_pe", &q_pe},
      {"latent_cache", &latent_cache},
      {"out", &out},
  });

  auto dt = dtype_to_metal(q_nope.dtype());
  std::string kname = "paged_mla_attention_" + dt + "_kvr" +
                      std::to_string(spec->kv_lora_rank) + "_pe" +
                      std::to_string(spec->qk_rope_head_dim) + "_bs" +
                      std::to_string(spec->block_size) + "_g" +
                      std::to_string(spec->heads_per_tg) + "_nt" +
                      std::to_string(spec->num_threads) + "_nsl32_ps" +
                      std::to_string(spec->partition_size);

  bool use_partitioning = false;

  std::string hash_name = kname + "_part" + (use_partitioning ? "1" : "0");

  auto* lib = d.get_library("paged_mla_kern");
  auto* kernel = d.get_kernel(
      kname,
      lib,
      hash_name,
      {{&use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)}});

  // Threadgroup memory:
  //   max_scores[G * BN] + sum_exp_scores[G * BN] + outputs[BD * BD]
  // The outputs buffer must be sized for the maximum write offset
  // `lane*BD + sg`, which reaches (BD-1)*BD + (BN-1). Using BD*BD always
  // (rather than BN*BD) gives enough room across all G; on G=1 (BN=BD=32)
  // they coincide.
  // For G=1, NT=1024: 2*32 + 32*32 = 1088 fp32 ≈ 4.3 KB.
  // For G=2, NT=512:  2*2*16 + 32*32 = 1088 fp32 ≈ 4.3 KB.
  const int BD = 32;
  const int BN = spec->num_threads / BD;
  size_t shmem =
      static_cast<size_t>((2 * heads_per_tg * BN + BD * BD) * sizeof(float));

  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(shmem, 0);

  enc.set_output_array(out, 2);
  enc.set_input_array(q_nope, 3);
  enc.set_input_array(q_pe, 4);
  enc.set_input_array(latent_cache, 5);
  enc.set_input_array(block_tables, 6);
  enc.set_input_array(context_lens, 7);
  enc.set_input_array(cu_seqlens_q, 8);

  int32_t num_seqs_i = static_cast<int32_t>(num_seqs);
  int32_t max_blocks_i = static_cast<int32_t>(max_num_blocks_per_seq);
  enc.set_bytes(num_seqs_i, 9);
  enc.set_bytes(max_blocks_i, 10);
  enc.set_bytes(scale, 11);

  // Grid: (num_heads / G, total_q_tokens, 1). Each TG owns G consecutive
  // query heads sharing the same latent KV.
  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads / heads_per_tg, total_q_tokens, 1),
      MTL::Size::Make(spec->num_threads, 1, 1));

  // No add_temporary calls: the only caller is MlaPagedAttentionPrimitive,
  // and inside a primitive MLX manages array lifetimes via the completion
  // handler.
}

// MLA single-pass paged attention as an MLX Primitive so the kernel
// dispatch participates in the lazy graph (no per-call mx.eval boundary).
class MlaPagedAttentionPrimitive : public UnaryPrimitive {
 public:
  MlaPagedAttentionPrimitive(
      Stream stream, int block_size, float scale, int heads_per_tg)
      : UnaryPrimitive(stream),
        block_size_(block_size),
        scale_(scale),
        heads_per_tg_(heads_per_tg) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("MlaPagedAttentionPrimitive only supports GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    // Inputs match the single-pass dispatcher's positional order:
    //   [q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q]
    out.set_data(allocator::malloc(out.nbytes()));
    dispatch_mla_paged_attention(
        out,
        inputs[0], inputs[1], inputs[2],
        inputs[3], inputs[4], inputs[5],
        block_size_, scale_, heads_per_tg_,
        stream());
  }

  const char* name() const override { return "MlaPagedAttention"; }

  bool is_equivalent(const Primitive& other) const override {
    auto* rhs = dynamic_cast<const MlaPagedAttentionPrimitive*>(&other);
    return rhs && rhs->block_size_ == block_size_
        && rhs->scale_ == scale_ && rhs->heads_per_tg_ == heads_per_tg_;
  }

 private:
  int block_size_;
  float scale_;
  int heads_per_tg_;
};

static array mla_paged_attention_primitive_fn(
    const array& q_nope,
    const array& q_pe,
    const array& latent_cache,
    const array& block_tables,
    const array& context_lens,
    const array& cu_seqlens_q,
    int block_size,
    float scale,
    int heads_per_tg) {
  auto prim = std::make_shared<MlaPagedAttentionPrimitive>(
      default_stream(Device::gpu), block_size, scale, heads_per_tg);
  // Output shape matches q_nope: (total_q_tokens, num_heads, kv_lora_rank).
  return array(
      q_nope.shape(),
      q_nope.dtype(),
      std::move(prim),
      {q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q});
}

// TODO: Remove the Python per-request convolution once its batched replacement
// covers the required cases and its performance is verified.
// TODO: Once optimized recurrence covers every required case, remove the native
// fallback, this primitive, its kernel binding, and VLLM_METAL_GDN_LAZY_KERNELS.
class GDNLinearAttentionPrimitive : public Primitive {
 public:
  GDNLinearAttentionPrimitive(Stream stream, int Hk, int Hv, int Dk, int Dv)
      : Primitive(stream), Hk_(Hk), Hv_(Hv), Dk_(Dk), Dv_(Dv) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GDNLinearAttentionPrimitive only supports GPU");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    // Each recurrence owns its output; only persistent state aliases an input.
    outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
    outputs[1].copy_shared_buffer(inputs[5]);
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    const auto& g = inputs[3];
    const auto& beta = inputs[4];
    const auto& cu_seqlens = inputs[6];
    const auto& slot_mapping = inputs[7];
    auto& y = outputs[0];
    auto& state_pool = outputs[1];
    int num_requests = static_cast<int>(cu_seqlens.shape(0)) - 1;
    auto s = stream();
    auto& d = metal::device(Device::gpu);

    auto dt = dtype_to_metal(q.dtype());
    std::string kname = "gdn_linear_attention_" + dt;
    auto* lib = d.get_library("gdn_kern");
    auto* kernel = d.get_kernel(kname, lib, kname, {});

    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);

    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(g, 3);
    enc.set_input_array(beta, 4);
    enc.set_output_array(state_pool, 5);
    enc.set_input_array(cu_seqlens, 6);
    enc.set_input_array(slot_mapping, 7);
    enc.set_output_array(y, 8);

    enc.set_bytes(num_requests, 9);
    enc.set_bytes(Hk_, 10);
    enc.set_bytes(Hv_, 11);
    enc.set_bytes(Dk_, 12);
    enc.set_bytes(Dv_, 13);
    int64_t state_stride = state_pool.strides()[0];
    enc.set_bytes(state_stride, 14);

    // Grid: (Dv, 1, num_requests * Hv)  Threadgroup: (32, 1, 1)
    enc.dispatch_threadgroups(
        MTL::Size::Make(Dv_, 1, num_requests * Hv_),
        MTL::Size::Make(32, 1, 1));
  }

  const char* name() const override { return "GDNLinearAttention"; }

  bool is_equivalent(const Primitive& other) const override {
    auto* rhs = dynamic_cast<const GDNLinearAttentionPrimitive*>(&other);
    return rhs && rhs->Hk_ == Hk_ && rhs->Hv_ == Hv_
        && rhs->Dk_ == Dk_ && rhs->Dv_ == Dv_;
  }

 private:
  int Hk_, Hv_, Dk_, Dv_;
};

static std::vector<array> gdn_linear_attention_primitive_fn(
    const array& q, const array& k, const array& v,
    const array& g, const array& beta, const array& state_pool,
    const array& cu_seqlens, const array& slot_mapping,
    int Hk, int Hv, int Dk, int Dv) {
  if (Dk <= 0 || Dk > 256 || Dk % 32 != 0) {
    throw std::runtime_error(
        "GDN kernel requires Dk to be a positive multiple of 32, at most 256. "
        "Got Dk=" + std::to_string(Dk));
  }
  auto prim = std::make_shared<GDNLinearAttentionPrimitive>(
      default_stream(Device::gpu), Hk, Hv, Dk, Dv);
  return array::make_arrays(
      {{q.shape(0), Hv, Dv}, state_pool.shape()},
      {q.dtype(), state_pool.dtype()}, prim,
      {q, k, v, g, beta, state_pool, cu_seqlens, slot_mapping});
}
// ---------------------------------------------------------------------------
// nanobind module
// ---------------------------------------------------------------------------

NB_MODULE(_paged_ops, m) {
  register_mlx_patch(m);
  m.attr("PARTITION_SIZE") = nb::int_(kPartitionSize);
  m.def("detected_gpu_core_count", &detected_gpu_core_count,
        "Detected GPU core count, or zero when detection is unavailable.");
  m.def("_override_detected_gpu_core_count_for_test",
        &override_detected_gpu_core_count_for_test, nb::arg("cores"),
        "Test-only. A non-negative count replaces IORegistry detection; "
        "-1 restores hardware detection. Production routing must not call this.");
  m.def("_set_paged_dispatch_diagnostics", &set_paged_dispatch_diagnostics,
        nb::arg("enabled"),
        "Private process-wide diagnostic opt-in; disabled by default. "
        "Call with evaluation idle, in the worker that executes attention. "
        "Clears the last family/partition and returns the previous enabled state. "
        "This does not change attention routing or numerical computation.");
  nb::class_<GqaDecodeLengthPlan>(m, "GqaDecodeLengthPlan")
      .def_ro("num_requests", &GqaDecodeLengthPlan::num_requests)
      .def_ro("max_length", &GqaDecodeLengthPlan::max_length);
  m.def("gqa_decode_length_plan", &gqa_decode_length_plan<std::vector<int>>,
        nb::arg("context_lens"),
        "Immutable CPU length statistics for one forward, shared across layers. "
        "Does not select a partition or read GPU arrays.");
  m.def("gqa_decode_partition_size", &gqa_decode_plan_for_shape,
        nb::arg("num_heads"), nb::arg("num_kv_heads"), nb::arg("head_size"),
        nb::arg("max_seq_len"), nb::arg("gpu_cores"), nb::arg("block_size") = 16,
        "Default GQA partition for a supported decode geometry, or zero. "
        "Geometry, work and scratch-budget planning; dtype, feature and reducer "
        "eligibility are enforced separately by gqa_decode dispatch.");
  m.def("gqa_decode_batch_partition_size", &gqa_decode_batch_plan_for_shape,
        nb::arg("num_heads"), nb::arg("num_kv_heads"), nb::arg("head_size"),
        nb::arg("context_lens"), nb::arg("gpu_cores"), nb::arg("block_size") = 16,
        nb::arg("gpu_arch") = "", nb::arg("max_seq_len") = 0,
        "GQA partition from per-request KV lengths, or zero. Empty gpu_arch "
        "uses the executing GPU; an explicit architecture is a read-only planning "
        "input and does not override dispatch. max_seq_len optionally supplies "
        "the allocation upper bound for the scratch budget. Dispatch separately "
        "checks features and compiled reducer resources.");
  m.def("last_gqa_partition_size", []() {
    return g_last_gqa_partition.load(std::memory_order_relaxed);
  }, "Partition selected by the most recent recorded paged eval, or zero "
     "when disabled, cleared or on a fallback. "
     "Process-wide diagnostic; not a request trace or routing input.");
  m.def("last_gqa_num_requests", []() {
    return g_last_gqa_requests.load(std::memory_order_relaxed);
  }, "Number of requests in the most recent recorded GQA dispatch, or zero "
     "when disabled, cleared or on a fallback. Not per-request telemetry.");
  m.def("gqa_decode_shape_eligible", &gqa_decode_shape_eligible,
        nb::arg("num_heads"), nb::arg("num_kv_heads"), nb::arg("head_size"),
        nb::arg("max_seq_len"), nb::arg("gpu_cores"), nb::arg("block_size") = 16,
        "Measured default scope with a conservative grid guard; geometry "
        "work and scratch-budget planning. Functional dispatch checks dtype, "
        "feature and resource eligibility separately.");
  m.def("min_decode_grid", &min_decode_grid,
        "Decode-grid threshold (threadgroups) below which split-KV decode "
        "engages on this machine.");
  m.def("_gqa_decode_config_for_test", []() {
    nb::dict config;
    nb::list geometries;
    for (const auto& geometry : kGqaDecodeGeometries) {
      geometries.append(nb::make_tuple(geometry.num_heads, geometry.num_kv_heads,
                                       geometry.head_size, geometry.block_size));
    }
    nb::list partitions;
    for (int partition : kGqaPartitionSizes) partitions.append(nb::int_(partition));
    config["geometries"] = geometries;
    config["partitions"] = partitions;
    config["simd_groups_per_core"] = kGqaSimdGroupsPerCore;
    config["max_scratch_bytes"] = kGqaMaxScratchBytes;
    nb::dict m3;
    m3["architecture"] = "applegpu_g15g";
    m3["gpu_cores"] = kM3GqaGpuCores;
    m3["short_head128_batch"] = kM3GqaShortHead128Batch;
    m3["short_head128_min_context"] = kM3GqaShortHead128MinContext;
    m3["long_head_size"] = 256;
    m3["block16_min_context"] = kM3GqaBlock16MinContext;
    m3["block32_min_context"] = kM3GqaBlock32MinContext;
    m3["preferred_partition"] = 256;
    m3["reducer_memory_bytes"] = kM3GqaReducerMemoryBytes;
    m3["reducer_static_memory_bytes"] = kGqaReduceStaticMemoryBytes;
    config["m3_batched_preference"] = m3;
    return config;
  }, "Read-only test metadata from the native dispatch table and planner. "
     "Does not create pipelines, override policy or inspect hardware.");
  m.def(
      "_has_gqa_decode_kernel",
      []() {
        try {
          auto& d = metal::device(Device::gpu);
          auto* lib = d.get_library("paged_attention_v2_kern");
          // Check every layout admitted by the dispatch table. Duplicate
          // layouts reuse the same cached pipelines.
          for (const char* dtype : {"half", "bfloat16_t"}) {
            for (int part : kGqaPartitionSizes) {
              for (const auto& layout : kGqaDecodeGeometries) {
                const auto reducer = paged_reduce_kernel_name(
                    dtype, layout.head_size, part);
                bool no = false;
                if (d.get_kernel(reducer, lib, reducer + "_gqa_check",
                    {{&no, MTL::DataType::DataTypeBool, NS::UInteger(40)},
                     {&no, MTL::DataType::DataTypeBool, NS::UInteger(50)}})
                    == nullptr) return false;
                const auto name = gqa_decode_kernel_name(
                    dtype, layout.head_size, layout.block_size, part);
                if (d.get_kernel(name, lib, name, {}) == nullptr) return false;
              }
            }
          }
          return true;
        } catch (const std::exception&) {
          return false;
        }
      },
      "Private test-only probe. Creates pipelines for all shipped GQA "
      "specializations; never call from a production latency-sensitive path.");
  m.def("paged_attention_capabilities", []() {
    nb::dict caps;
    caps["gqa_decode"] = true;
    caps["gqa_disable"] = true;
    caps["decode_routing_metadata"] = true;
    caps["gqa_batch_context_lens"] = true;
    caps["gqa_length_plan"] = true;
    caps["gqa_mixed_decode_plan"] = true;
    return caps;
  }, "Capabilities of the public paged attention interface. Private test "
     "kernels do not imply an enabled production route.");

  m.def("tile_config",
        [](int head_size) -> nb::object {
          auto cfg = select_tile_config(head_size);
          if (!cfg) return nb::none();
          return nb::make_tuple(cfg->BQ, cfg->TILE_KV);
        },
        nb::arg("head_size"),
        "The tiled-prefill kernel's (BQ, TILE_KV) tile config for a head "
        "size, or None when that head size has no tiled instantiation.");

  m.def("init_v2_library", &init_v2_library,
        nb::arg("v2_src"),
        "JIT-compile the v2 online-softmax Metal shader.");

  m.def("init_library_path", &init_library_path,
        nb::arg("name"), nb::arg("path"),
        "Load a precompiled .metallib from disk, cached under `name`.");

  m.def("init_gdn_library", &init_gdn_library,
        nb::arg("gdn_src"),
        "JIT-compile the GDN linear attention Metal shader.");

  m.def("init_nax_library", &init_nax_library,
        nb::arg("nax_src"),
        "JIT-compile the NAX prefill attention Metal shader.");

  m.def("init_nax_library_path", &init_nax_library_path,
        nb::arg("path"),
        "Load the precompiled NAX prefill .metallib from disk.");

  m.def("nax_supported", &nax_hardware_supported,
        "True when the OS and GPU expose NAX tensor units.");

  m.def("nax_ready", []() { return nax_lib_ready_ && nax_enabled_; },
        "True when the NAX prefill kernel is loaded and enabled.");

  m.def("set_nax_enabled", [](bool enabled) { nax_enabled_ = enabled; },
        nb::arg("enabled"),
        "Runtime kill-switch for the NAX prefill kernel (tests / A-B runs).");

  m.def("supports_mm_prefix", []() { return true; },
        "True when paged_attention_primitive accepts mm_prefix_ranges "
        "(Gemma 4 vision image-block attention in the tiled prefill kernel).");

  m.def("tq_encode",
        [](nb::handle key_h, nb::handle value_h,
           nb::handle key_cache_h, nb::handle value_cache_h,
           nb::handle key_scale_cache_h, nb::handle value_scale_cache_h,
           nb::handle key_zero_cache_h,
           nb::handle slot_mapping_h,
           nb::handle v_centroids_h,
           int v_bits, int k_bits, bool k_signed) {
          auto results = tq_encode_primitive_fn(
              *nb::inst_ptr<array>(key_h),
              *nb::inst_ptr<array>(value_h),
              *nb::inst_ptr<array>(key_cache_h),
              *nb::inst_ptr<array>(value_cache_h),
              *nb::inst_ptr<array>(key_scale_cache_h),
              *nb::inst_ptr<array>(value_scale_cache_h),
              *nb::inst_ptr<array>(key_zero_cache_h),
              *nb::inst_ptr<array>(slot_mapping_h),
              *nb::inst_ptr<array>(v_centroids_h),
              v_bits, k_bits, k_signed);

          // Mint five Python mx.core.array placeholders inside the binding
          // (callers never see the placeholder dance).  We go through the
          // Python-side mlx.core.array constructor because cross-module
          // nanobind RTTI for nb::class_<array> from libmlx is broken under
          // hidden symbol visibility; overwrite_descriptor is the same
          // escape hatch used by paged_attention_primitive below.
          nb::object mx_core  = nb::module_::import_("mlx.core");
          nb::object arr_cls  = mx_core.attr("array");
          nb::object zero_arg = nb::int_(0);
          nb::object out_k    = arr_cls(zero_arg);
          nb::object out_v    = arr_cls(zero_arg);
          nb::object out_ks   = arr_cls(zero_arg);
          nb::object out_vs   = arr_cls(zero_arg);
          nb::object out_kz   = arr_cls(zero_arg);
          nb::inst_ptr<array>(out_k)->overwrite_descriptor(results[0]);
          nb::inst_ptr<array>(out_v)->overwrite_descriptor(results[1]);
          nb::inst_ptr<array>(out_ks)->overwrite_descriptor(results[2]);
          nb::inst_ptr<array>(out_vs)->overwrite_descriptor(results[3]);
          nb::inst_ptr<array>(out_kz)->overwrite_descriptor(results[4]);
          return nb::make_tuple(out_k, out_v, out_ks, out_vs, out_kz);
        },
        nb::arg("key"), nb::arg("value"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("key_scale_cache"), nb::arg("value_scale_cache"),
        nb::arg("key_zero_cache"),
        nb::arg("slot_mapping"),
        nb::arg("v_centroids"),
        nb::arg("v_bits"),
        nb::arg("k_bits"),
        nb::arg("k_signed"),
        "Fused TurboQuant encode + paged scatter.  Wraps a real MLX "
        "Primitive so its five cache writes carry graph provenance — "
        "downstream paged_attention_primitive and the next decode step's "
        "tq_encode depend on this op through the lazy graph instead of "
        "racing it on a separate command buffer.  Returns a 5-tuple "
        "(new_key_cache, new_value_cache, new_key_scale_cache, "
        "new_value_scale_cache, new_key_zero_cache); each aliases the "
        "corresponding input buffer in place and the caller MUST rebind "
        "kv_cache.<cache>[layer_idx] to the returned value so subsequent "
        "ops see the post-write provenance.  Supports all K quants in "
        "QUANT_PARAMS (signed q8_0/int8 at k_bits=8; unsigned uint8/q5_0/"
        "q4_0/int4/uint4/int2/uint2 at k_bits in {2,3,4,5,8}). V supports "
        "any v_bits in [1, 8] via the v_centroids buffer.");

  m.def("reshape_and_cache",
        [](nb::handle key_h, nb::handle value_h,
           nb::handle key_cache_h, nb::handle value_cache_h,
           nb::handle slot_mapping_h) {
          auto results = reshape_and_cache_primitive_fn(
              *nb::inst_ptr<array>(key_h),
              *nb::inst_ptr<array>(value_h),
              *nb::inst_ptr<array>(key_cache_h),
              *nb::inst_ptr<array>(value_cache_h),
              *nb::inst_ptr<array>(slot_mapping_h));

          // Same placeholder dance as tq_encode: mint mx.core.array objects and
          // overwrite_descriptor to bypass cross-module nanobind RTTI.
          nb::object mx_core  = nb::module_::import_("mlx.core");
          nb::object arr_cls  = mx_core.attr("array");
          nb::object zero_arg = nb::int_(0);
          nb::object out_k    = arr_cls(zero_arg);
          nb::object out_v    = arr_cls(zero_arg);
          nb::inst_ptr<array>(out_k)->overwrite_descriptor(results[0]);
          nb::inst_ptr<array>(out_v)->overwrite_descriptor(results[1]);
          return nb::make_tuple(out_k, out_v);
        },
        nb::arg("key"), nb::arg("value"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("slot_mapping"),
        "Fused K/V paged scatter for non-quantized (fp16/bf16/fp32) caches. "
        "Wraps an MLX Primitive whose two cache writes carry graph provenance "
        "for the downstream paged_attention read; each output aliases its input "
        "cache buffer in place, so the caller MUST rebind "
        "kv_cache.<cache>[layer_idx] to the returned value. key/value are "
        "[num_tokens, num_kv_heads, head_size]; caches are "
        "[num_blocks, block_size, num_kv_heads, head_size].");

  m.def("gdn_state_scatter",
        [](nb::handle pool_h, nb::handle src_h, nb::handle ids_h,
           bool zero, bool paged) {
          // inst_ptr<array> on a non-array is undefined behaviour, so check
          // before dereferencing: a wrong type must raise, not crash.
          nb::object mx_array_cls =
              nb::module_::import_("mlx.core").attr("array");
          for (nb::handle h : {pool_h, src_h, ids_h}) {
            if (!nb::isinstance(h, mx_array_cls)) {
              throw std::runtime_error(
                  "gdn_state_scatter: pool, src and dst_ids must be "
                  "mlx.core.array");
            }
          }
          auto result = gdn_state_scatter_primitive_fn(
              *nb::inst_ptr<array>(pool_h),
              *nb::inst_ptr<array>(src_h),
              *nb::inst_ptr<array>(ids_h), zero, paged);

          // Same placeholder dance as tq_encode / reshape_and_cache: mint an
          // mx.core.array and overwrite_descriptor to bypass cross-module
          // nanobind RTTI.
          nb::object mx_core  = nb::module_::import_("mlx.core");
          nb::object arr_cls  = mx_core.attr("array");
          nb::object zero_arg = nb::int_(0);
          nb::object out      = arr_cls(zero_arg);
          nb::inst_ptr<array>(out)->overwrite_descriptor(result);
          return out;
        },
        nb::arg("pool"), nb::arg("src"), nb::arg("dst_ids"), nb::arg("zero") = false,
        nb::arg("paged") = false,
        "In-place row scatter into a slot-indexed GDN state pool. Writes "
        "src[i] into pool[dst_ids[i]] without MLX's whole-pool copy preamble; "
        "source rows and indices are made contiguous; destination strides and "
        "backing storage are preserved, so the caller MUST rebind its pool "
        "reference. dst_ids must be distinct int32 slots; src is "
        "[n, *pool.shape[1:]] with pool's dtype. With paged=True, pool is "
        "[num_blocks, block_size, ...], src is [n, *pool.shape[2:]], and "
        "dst_ids address flattened token slots while preserving page padding.");

  // Private numerical-test entry: bypass only the performance selector, never
  // mutate process-wide policy or expose an override on the production API.
  m.def("_gqa_paged_attention_for_test",
        [](nb::handle query_h, nb::handle key_h, nb::handle value_h,
           float scale, nb::handle tables_h, nb::handle lengths_h,
           int block_size, int max_seq_len, int partition_size,
           nb::handle out_h) {
          const auto& q = *nb::inst_ptr<array>(query_h);
          const auto& k = *nb::inst_ptr<array>(key_h);
          const auto& v = *nb::inst_ptr<array>(value_h);
          const auto& tables = *nb::inst_ptr<array>(tables_h);
          const auto& lengths = *nb::inst_ptr<array>(lengths_h);
          if (std::find(kGqaPartitionSizes.begin(), kGqaPartitionSizes.end(),
                        partition_size) == kGqaPartitionSizes.end()) {
            throw std::invalid_argument("GQA test partition must be 256/512");
          }
          if (q.ndim() != 3 || q.shape(0) < 1 || k.ndim() != 4 ||
              k.shape() != v.shape() || k.strides() != v.strides() ||
              (q.dtype() != float16 && q.dtype() != bfloat16) ||
              k.dtype() != q.dtype() || v.dtype() != q.dtype() ||
              k.shape(3) != q.shape(2) ||
              !gqa_decode_geometry_supported(q.shape(1), k.shape(2),
                                             q.shape(2), block_size) ||
              k.shape(1) % block_size != 0 ||
              tables.ndim() != 2 || tables.shape(0) != q.shape(0) ||
              tables.dtype() != int32 || lengths.ndim() != 1 ||
              lengths.shape(0) != q.shape(0) || lengths.dtype() != int32 ||
              max_seq_len <= 0 ||
              max_seq_len > static_cast<int64_t>(tables.shape(1)) * block_size) {
            throw std::invalid_argument("GQA test entry requires supported one-token decode rows");
          }
          const int64_t num_partitions =
              (static_cast<int64_t>(max_seq_len) + partition_size - 1) /
              partition_size;
          if (paged_reduce_threadgroup_bytes(num_partitions) >
              metal::device(Device::gpu).mtl_device()->maxThreadgroupMemoryLength()) {
            throw std::invalid_argument(
                "GQA test partition exceeds the device threadgroup memory limit");
          }
          auto cu = arange(q.shape(0) + 1, int32);
          auto result = paged_attention_primitive_fn(
              q, k, v, k.shape(2), scale, 0.f, tables, lengths, cu,
              block_size, max_seq_len, -1, false, "", nullptr, nullptr,
              nullptr, nullptr, 3, 1, nullptr, nullptr, q.shape(0), q.shape(0),
              max_seq_len, false,
              partition_size);
          nb::inst_ptr<array>(out_h)->overwrite_descriptor(result);
        }, nb::arg("query"), nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("scale"), nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("block_size"), nb::arg("max_seq_len"), nb::arg("partition_size"),
        nb::arg("out"),
        "Private test-only one-token-per-request GQA dispatch with an explicit partition. "
        "Page IDs and sequence lengths must describe valid cache contents. "
        "Does not change the default selector or require GPU core detection.");

  // Paged attention primitive (read-only): dispatches paged_attention_v2_online.
  // Cache writes are handled by MLX-native scatter upstream.
  // Uses overwrite_descriptor to bypass cross-module nanobind RTTI.
  m.def("paged_attention_primitive",
        [](nb::handle query_h,
           nb::handle key_cache_h, nb::handle value_cache_h,
           int num_kv_heads, float scale, float softcap,
           nb::handle block_tables_h, nb::handle seq_lens_h,
           nb::handle cu_seqlens_q_h,
           int block_size, int max_seq_len, int sliding_window,
           nb::handle out_h,
           nb::object key_scale_cache_h,
           nb::object value_scale_cache_h,
           nb::object key_zero_cache_h,
           nb::object v_centroids_h,
           bool use_turboquant,
           const std::string& quant_type,
           int v_bits,
           int window_seqlen_q,
           nb::object sinks_h,
           nb::object mm_prefix_ranges_h,
           int num_decode_requests,
           int num_decode_tokens,
           int max_decode_context_len, bool gqa_disabled,
           const std::vector<int>& gqa_context_lens,
           const std::optional<GqaDecodeLengthPlan>& gqa_length_plan) {
          const array* sk = sinks_h.is_none()
              ? nullptr : nb::inst_ptr<array>(sinks_h);
          const array* mp = mm_prefix_ranges_h.is_none()
              ? nullptr : nb::inst_ptr<array>(mm_prefix_ranges_h);
          const array* ks = use_turboquant
              ? nb::inst_ptr<array>(key_scale_cache_h) : nullptr;
          const array* vs = use_turboquant
              ? nb::inst_ptr<array>(value_scale_cache_h) : nullptr;
          const array* kz = use_turboquant
              ? nb::inst_ptr<array>(key_zero_cache_h) : nullptr;
          const array* vc = use_turboquant
              ? nb::inst_ptr<array>(v_centroids_h) : nullptr;
          auto result = paged_attention_primitive_fn(
              *nb::inst_ptr<array>(query_h),
              *nb::inst_ptr<array>(key_cache_h),
              *nb::inst_ptr<array>(value_cache_h),
              num_kv_heads, scale, softcap,
              *nb::inst_ptr<array>(block_tables_h),
              *nb::inst_ptr<array>(seq_lens_h),
              *nb::inst_ptr<array>(cu_seqlens_q_h),
              block_size, max_seq_len, sliding_window,
              use_turboquant, quant_type, ks, vs, kz, vc, v_bits,
              window_seqlen_q, sk, mp, num_decode_requests,
              num_decode_tokens, max_decode_context_len, gqa_disabled,
              0, gqa_context_lens, gqa_length_plan);
          nb::inst_ptr<array>(out_h)->overwrite_descriptor(result);
        },
        nb::arg("query"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("num_kv_heads"), nb::arg("scale"), nb::arg("softcap"),
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("cu_seqlens_q"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        nb::arg("sliding_window"),
        nb::arg("out"),
        nb::arg("key_scale_cache") = nb::none(),
        nb::arg("value_scale_cache") = nb::none(),
        nb::arg("key_zero_cache") = nb::none(),
        nb::arg("v_centroids") = nb::none(),
        nb::arg("use_turboquant") = false,
        nb::arg("quant_type") = "",
        nb::arg("v_bits") = 3,
        nb::arg("window_seqlen_q") = 1,
        nb::arg("sinks") = nb::none(),
        nb::arg("mm_prefix_ranges") = nb::none(),
        nb::arg("num_decode_requests") = -1,
        nb::arg("num_decode_tokens") = 0,
        nb::arg("max_decode_context_len") = 0,
        nb::arg("gqa_disabled") = false,
        nb::arg("gqa_context_lens") = std::vector<int>{},
        nb::arg("gqa_length_plan") = nb::none(),
        "Paged attention primitive (read-only). Cache writes are handled "
        "by MLX-native scatter upstream.  window_seqlen_q must equal the "
        "longest cu_seqlens_q segment (validated when > 1); small "
        "multi-token batches (spec-decode verification windows) route to "
        "the per-token kernel's window mode.  sinks is an optional float32 "
        "array of one learned logit per query head (GPT-OSS style attention "
        "sinks); it joins the softmax denominator without contributing a "
        "value row, and is rejected together with TurboQuant.  "
        "mm_prefix_ranges is an optional (query rows, 2) int32 array of "
        "inclusive absolute image-block bounds per query row, (-1, -1) "
        "elsewhere (Gemma 4 vision): the tiled prefill kernel unmasks the "
        "block on top of the causal rule and ANDs the sliding window; "
        "rejected with TurboQuant, float32 queries, verification windows and "
        "pure-decode batches.  "
        "max_decode_context_len must bound the decode rows' seq_lens: a "
        "mixed batch that splits its decode prefix sizes that prefix's "
        "split-KV work by it rather than by max_seq_len.  "
        "gqa_disabled mirrors VLLM_METAL_DISABLE_GQA_DECODE and keeps "
        "eligible batches off the GQA-shared decode kernel. "
        "gqa_context_lens is the CPU copy of seq_lens for a whole ordinary "
        "decode batch, used for planning without a GPU readback; the caller "
        "must keep both consistent. Alternatively pass a gqa_length_plan "
        "built once per forward; its compact statistics are captured by value. "
        "Omit both to retain legacy batch routing.");

  m.def(
      "last_paged_dispatch",
      []() {
        static constexpr const char* names[] = {
            "", "gqa_decode", "per_token_ps0", "per_token_ps512", "window_ps0",
            "window_ps512", "nax_prefill", "tiled_prefill", "mixed_prefill_decode",
            "mixed_nax_prefill_decode"};
        static_assert(std::size(names) == static_cast<size_t>(PagedDispatch::Count),
                      "PagedDispatch and its diagnostic names must stay in sync");
        return names[static_cast<size_t>(
            g_last_dispatch.load(std::memory_order_relaxed))];
      },
      "Dispatch family chosen by the most recent recorded paged_attention_primitive "
      "eval (\"gqa_decode\", \"per_token_ps0\", \"per_token_ps512\", "
      "\"window_ps0\", \"window_ps512\", \"nax_prefill\", "
      "\"tiled_prefill\", \"mixed_prefill_decode\", \"mixed_nax_prefill_decode\"). "
      "Diagnostic surface for routing tests; empty "
      "when diagnostics are disabled, cleared or before the first recorded eval.");

  m.def("gdn_linear_attention",
        [](nb::handle q, nb::handle k, nb::handle v,
           nb::handle g, nb::handle beta, nb::handle state_pool,
           nb::handle cu_seqlens, nb::handle slot_mapping,
           int Hk, int Hv, int Dk, int Dv) {
          auto results = gdn_linear_attention_primitive_fn(
              *nb::inst_ptr<array>(q), *nb::inst_ptr<array>(k),
              *nb::inst_ptr<array>(v), *nb::inst_ptr<array>(g),
              *nb::inst_ptr<array>(beta), *nb::inst_ptr<array>(state_pool),
              *nb::inst_ptr<array>(cu_seqlens),
              *nb::inst_ptr<array>(slot_mapping),
              Hk, Hv, Dk, Dv);
          auto arr_cls = nb::module_::import_("mlx.core").attr("array");
          nb::object out_y = arr_cls(nb::int_(0));
          nb::object out_state = arr_cls(nb::int_(0));
          nb::inst_ptr<array>(out_y)->overwrite_descriptor(results[0]);
          nb::inst_ptr<array>(out_state)->overwrite_descriptor(results[1]);
          return nb::make_tuple(out_y, out_state);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("g"), nb::arg("beta"),
        nb::arg("state_pool"), nb::arg("cu_seqlens"),
        nb::arg("slot_mapping"),
        nb::arg("Hk"), nb::arg("Hv"), nb::arg("Dk"), nb::arg("Dv"),
        "Lazy GDN recurrence returning (y, updated_state_pool). Only the "
        "state output aliases its input. Callers must use both returned "
        "handles so output reads and state reuse depend on the native writes.");

  m.def("init_mla_library", &init_mla_library,
        nb::arg("src"),
        "JIT-compile the MLA paged attention Metal shader (RFC #360).");

  m.def("mla_paged_attention_primitive",
        [](nb::handle q_nope_h,
           nb::handle q_pe_h,
           nb::handle latent_cache_h,
           nb::handle block_tables_h,
           nb::handle context_lens_h,
           nb::handle cu_seqlens_q_h,
           int block_size, float scale, int heads_per_tg,
           nb::handle out_h) {
          auto result = mla_paged_attention_primitive_fn(
              *nb::inst_ptr<array>(q_nope_h),
              *nb::inst_ptr<array>(q_pe_h),
              *nb::inst_ptr<array>(latent_cache_h),
              *nb::inst_ptr<array>(block_tables_h),
              *nb::inst_ptr<array>(context_lens_h),
              *nb::inst_ptr<array>(cu_seqlens_q_h),
              block_size, scale, heads_per_tg);
          nb::inst_ptr<array>(out_h)->overwrite_descriptor(result);
        },
        nb::arg("q_nope"), nb::arg("q_pe"),
        nb::arg("latent_cache"),
        nb::arg("block_tables"), nb::arg("context_lens"),
        nb::arg("cu_seqlens_q"),
        nb::arg("block_size"), nb::arg("scale"),
        nb::arg("heads_per_tg") = 1,
        nb::arg("out"),
        "Paged MLA (single-pass), wrapped as an MLX Primitive — fills "
        "``out`` with a lazy descriptor so the kernel call participates "
        "in the wrapper's lazy graph and avoids the per-call mx.eval "
        "boundary the eager binding requires. Saves ~200 μs at B=1 "
        "small-H cells where dispatch overhead dominates.");

}
