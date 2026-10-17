// SPDX-License-Identifier: Apache-2.0
// Paged attention dispatch, cache writes, and TurboQuant encoding.
#include "ops_internal.h"

#include <unordered_map>

// ---------------------------------------------------------------------------
// Library caching
// ---------------------------------------------------------------------------

static std::string v2_paged_attention_source_;

std::atomic<PagedDispatch> g_last_dispatch{PagedDispatch::None};
std::atomic<int> g_last_gqa_partition{0};
std::atomic<int> g_last_gqa_requests{0};
std::atomic<bool> g_dispatch_diagnostics_enabled{false};
// NAX prefill pipeline of the most recent recorded NAX launch: 1 split,
// 0 original loop, -1 none (non-NAX dispatch or diagnostics off).
std::atomic<int> g_last_nax_split{-1};
inline void record_paged_dispatch(PagedDispatch family, int gqa_partition = 0,
                                  int gqa_requests = 0) {
  if (!g_dispatch_diagnostics_enabled.load(std::memory_order_relaxed)) return;
  g_last_dispatch.store(family, std::memory_order_relaxed);
  if (family != PagedDispatch::NaxPrefill &&
      family != PagedDispatch::MixedNaxPrefillDecode)
    g_last_nax_split.store(-1, std::memory_order_relaxed);
  g_last_gqa_partition.store(gqa_partition, std::memory_order_relaxed);
  g_last_gqa_requests.store(gqa_requests, std::memory_order_relaxed);
}

// Split only after eight KV partitions amortize the extra dispatch.
constexpr int kMixedDecodeMinPartitions = 8;
constexpr int kMixedDecodeMinContext =
    kMixedDecodeMinPartitions * kPartitionSize;

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


bool gqa_decode_geometry_supported(int num_heads, int num_kv_heads,
                                           int head_size, int block_size) {
  return std::any_of(
      kGqaDecodeGeometries.begin(), kGqaDecodeGeometries.end(),
      [=](const auto& geometry) {
        return geometry.num_heads == num_heads &&
            geometry.num_kv_heads == num_kv_heads &&
            geometry.head_size == head_size && geometry.block_size == block_size;
      });
}

std::string gqa_decode_kernel_name(
    const std::string& dtype, int head_size, int block_size, int partition_size) {
  return "paged_attention_gqa_decode_" + dtype + "_hs" + std::to_string(head_size)
      + "_bs" + std::to_string(block_size) + "_ps" + std::to_string(partition_size);
}

static bool paged_reduce_memory_fits(
    size_t dynamic_bytes, size_t static_bytes, size_t capacity) {
  return dynamic_bytes <= capacity && static_bytes <= capacity - dynamic_bytes;
}

static bool gqa_scratch_fits(
    int requests, int heads, int64_t partitions, int head_size, size_t item_size,
    const std::vector<int64_t>* offsets = nullptr) {
  const size_t bytes_per_partition = head_size * item_size + 2 * sizeof(float);
  const size_t offset_bytes = offsets ? offsets->size() * sizeof(int64_t) : 0;
  if (offset_bytes >= kGqaMaxScratchBytes) return false;
  size_t remaining = (kGqaMaxScratchBytes - offset_bytes) / bytes_per_partition;
  const int64_t slots = offsets ? offsets->back() : int64_t(requests) * partitions;
  // Divide the budget instead of multiplying potentially large dimensions.
  for (int64_t count : {int64_t(heads), slots}) {
    if (count <= 0 || static_cast<uint64_t>(count) > remaining) return false;
    remaining /= static_cast<size_t>(count);
  }
  return true;
}

static const std::vector<int64_t>* gqa_compact_offsets(
    const GqaDecodeLengthPlan& plan, int partition, int64_t max_partitions) {
  if (!plan.partition_offsets) return nullptr;
  const auto& offsets = (*plan.partition_offsets)[partition == 512 ? 0 : 1];
  // Keep the existing layout when every row fills the dispatch stride.
  return offsets.back() < int64_t(plan.num_requests) * max_partitions
      ? &offsets : nullptr;
}

static int gqa_decode_partition_for_arch(
    int partition, int num_heads, int num_kv_heads, int head_size,
    const GqaDecodeLengthPlan& plan, int block_size,
    const std::string& architecture, int max_seq_len) {
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
  if (!paged_reduce_memory_fits(kernels::paged_reduce_threadgroup_bytes(partitions256),
                               kGqaReduceStaticMemoryBytes,
                               kM3GqaReducerMemoryBytes)) return partition;
  return 256;
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
  return gqa_decode_partition_for_arch(
      partition, num_heads, num_kv_heads, head_size, plan, block_size,
      architecture, max_seq_len);
}

int gqa_decode_plan_for_lengths(
    int num_heads, int num_kv_heads, int head_size,
    const GqaDecodeLengthPlan& plan, int gpu_cores, int block_size,
    const std::string& gpu_arch, int max_seq_len) {
  const int partition = gqa_decode_partition_for_lengths(num_heads, num_kv_heads,
      head_size, plan, gpu_cores, block_size, gpu_arch, max_seq_len);
  if (partition == 0) return 0;
  const int64_t length = std::max(max_seq_len, plan.max_length);
  const int64_t partitions = (length + partition - 1) / partition;
  // Every admitted GQA dtype stores a two-byte partial output.
  return gqa_scratch_fits(plan.num_requests, num_heads, partitions, head_size, 2,
                         gqa_compact_offsets(plan, partition, partitions))
      ? partition : 0;
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
bool nax_lib_ready_ = false;   // a NAX library was registered
bool nax_enabled_ = true;      // test / A-B switch

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
static bool nax_eligible(
    Dtype dtype, int head_size, int block_size, bool lib_ready, bool enabled) {
  return lib_ready && enabled &&
      (dtype == float16 || dtype == bfloat16) &&
      (head_size == 64 || head_size == 96 || head_size == 128 ||
       head_size == 256 || head_size == 512) &&
      (block_size == 8 || block_size == 16 || block_size == 32);
}

// ---------------------------------------------------------------------------
// Helper: dtype → Metal type string
// ---------------------------------------------------------------------------

std::string dtype_to_metal(Dtype dt) {
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

struct NaxAttentionPlan {
  std::string base_kname;
  std::string hash_name;
  bool use_sinks;
  bool split_full;
  int q_blocks;
  int num_threads;
};

static NaxAttentionPlan make_nax_attention_plan(
    Dtype dtype, int total_q_tokens, int head_size, int num_seqs,
    int block_size, int q_block_offset, bool use_sinks, bool split_full) {
  using kernels::kNaxBQ;

  int total_q_blocks = total_q_tokens / kNaxBQ + num_seqs;
  if (q_block_offset < 0 || q_block_offset > total_q_blocks) {
    throw std::invalid_argument(
        "dispatch_paged_attention_nax: q_block_offset (" +
        std::to_string(q_block_offset) + ") out of range for total_q_blocks "
        "(" + std::to_string(total_q_blocks) + ")");
  }

  std::string base_kname =
      kernels::nax_name(dtype_to_metal(dtype), head_size, block_size);
  std::string hash_name = base_kname + "_sk" + (use_sinks ? "1" : "0")
                          + "_sf" + (split_full ? "1" : "0");
  return {std::move(base_kname), std::move(hash_name), use_sinks, split_full,
          total_q_blocks - q_block_offset, kernels::kNaxThreads};
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
    Stream s, const array* sinks, bool split_full) {
  auto& d = metal::device(s.device);

  int total_q_tokens = static_cast<int>(query.shape(0));
  int head_size  = static_cast<int>(query.shape(2));
  int num_seqs   = static_cast<int>(cu_seqlens_q.shape(0)) - 1;
  auto plan = make_nax_attention_plan(
      query.dtype(), total_q_tokens, head_size, num_seqs,
      block_size, q_block_offset, sinks != nullptr, split_full);
  if (g_dispatch_diagnostics_enabled.load(std::memory_order_relaxed))
    g_last_nax_split.store(plan.split_full ? 1 : 0, std::memory_order_relaxed);

  auto* lib = d.get_library("paged_attention_nax_kern");
  auto* kernel = d.get_kernel(
      plan.base_kname, lib, plan.hash_name,
      {{&plan.use_sinks, MTL::DataType::DataTypeBool, NS::UInteger(40)},
       {&plan.split_full, MTL::DataType::DataTypeBool, NS::UInteger(41)}});

  int num_heads = static_cast<int>(query.shape(1));
  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  bind_paged_attn_buffers(enc, out, query, key_cache, value_cache,
                          num_kv_heads, softcap, block_tables, seq_lens,
                          cu_seqlens_q, block_size, sliding_window);
  enc.set_bytes(scale, 9);
  enc.set_bytes(q_block_offset, 30);
  if (plan.use_sinks) {
    enc.set_input_array(*sinks, 18);
  }

  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, plan.q_blocks, 1),
      MTL::Size::Make(plan.num_threads, 1, 1));
}

struct TiledAttentionPlan {
  std::string base_kname;
  std::string hash_name;
  bool use_sinks;
  bool use_mm_prefix;
  int q_blocks;
  int num_threads;
  size_t shmem;
};

static TiledAttentionPlan make_tiled_attention_plan(
    Dtype dtype, int total_q_tokens, int head_size, int num_seqs,
    int block_size, int q_block_offset, kernels::TileConfig cfg,
    int t_size, bool use_sinks, bool use_mm_prefix) {
  int total_q_blocks = total_q_tokens / cfg.BQ + num_seqs;
  if (q_block_offset < 0 || q_block_offset > total_q_blocks) {
    throw std::invalid_argument(
        "dispatch_paged_attention_tiled: q_block_offset (" +
        std::to_string(q_block_offset) + ") out of range for total_q_blocks "
        "(" + std::to_string(total_q_blocks) + ")");
  }

  auto dt = dtype_to_metal(dtype);
  std::string base_kname = kernels::tiled_name(dt, head_size, block_size, cfg);
  std::string hash_name = base_kname + "_sk" + (use_sinks ? "1" : "0")
                          + "_mp" + (use_mm_prefix ? "1" : "0");
  return {std::move(base_kname), std::move(hash_name), use_sinks, use_mm_prefix,
          total_q_blocks - q_block_offset, cfg.NUM_THREADS,
          kernels::tiled_shared_bytes(cfg, head_size, t_size)};
}

static void dispatch_paged_attention_tiled(
    array& out, const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float scale, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int block_size, int max_seq_len, int sliding_window, int q_block_offset,
    kernels::TileConfig cfg, Stream s, const array* sinks,
    const array* mm_prefix_ranges) {
  auto& d = metal::device(s.device);

  int total_q_tokens = static_cast<int>(query.shape(0));
  int head_size  = static_cast<int>(query.shape(2));
  int num_seqs   = static_cast<int>(cu_seqlens_q.shape(0)) - 1;
  auto plan = make_tiled_attention_plan(
      query.dtype(), total_q_tokens, head_size, num_seqs,
      block_size, q_block_offset, cfg, static_cast<int>(query.itemsize()),
      sinks != nullptr, mm_prefix_ranges != nullptr);

  auto* lib = d.get_library("paged_attention_v2_kern");
  // Both constants are set on every dispatch: a function that references an
  // unset function constant fails pipeline creation.  MLX caches pipelines
  // by hash_name, so each constant combination needs its own suffix.
  auto* kernel = d.get_kernel(
      plan.base_kname, lib, plan.hash_name,
      {{&plan.use_sinks, MTL::DataType::DataTypeBool, NS::UInteger(40)},
       {&plan.use_mm_prefix, MTL::DataType::DataTypeBool, NS::UInteger(120)}});

  int num_heads = static_cast<int>(query.shape(1));
  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(plan.shmem, 0);

  bind_paged_attn_buffers(enc, out, query, key_cache, value_cache,
                          num_kv_heads, softcap, block_tables, seq_lens,
                          cu_seqlens_q, block_size, sliding_window);
  enc.set_bytes(scale, 9);
  enc.set_bytes(q_block_offset, 30);
  if (plan.use_sinks) {
    enc.set_input_array(*sinks, 18);
  }
  // Slot 22 is bound here, not in bind_paged_attn_buffers: that helper is
  // shared with the per-token kernel, where slot 22 is TurboQuant's k_codes.
  if (plan.use_mm_prefix) {
    enc.set_input_array(*mm_prefix_ranges, 22);
  }

  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, plan.q_blocks, 1),
      MTL::Size::Make(plan.num_threads, 1, 1));
}

// ---------------------------------------------------------------------------
// Shared pass-2 for the split-KV decode paths: merge per-partition partials
// into `out` (log-sum-exp combine).  Partials must follow the ps512 contract:
// log2-space (max, exp-sum) stats plus epsilon-normalised partial outputs in
// tmp_out[token, head, partition, :].
static MTL::ComputePipelineState* paged_attention_v2_reduce_kernel(
    metal::Device& d, const std::string& dt, int head_size,
    int partition_size, bool use_sinks, bool use_tq_fc, bool compact = false) {
  auto rname = kernels::paged_reduce_kernel_name(dt, head_size, partition_size);
  // The reduce kernel reads sinks (40), TurboQuant (50) and compact scratch
  // (130); other function constants are inert for it. TurboQuant batches take this
  // path (use_tq_fc varies: the TQ reduce applies the deferred inverse FWHT),
  // and sinks are folded here rather than in the partitioned kernel so the
  // sink logit is counted once globally.  The cache key MUST encode every
  // constant the pipeline is specialized on — otherwise the first compile
  // wins and a later caller with different constants silently reuses the
  // wrong pipeline.
  std::string rhash = rname + "_v2reduce"
      + "_tq" + (use_tq_fc ? "1" : "0")
      + "_sk" + (use_sinks ? "1" : "0")
      + "_compact" + (compact ? "1" : "0");
  auto* lib = d.get_library("paged_attention_v2_kern");
  return d.get_kernel(
      rname, lib, rhash,
      {{&use_sinks, MTL::DataType::DataTypeBool, NS::UInteger(40)},
       {&use_tq_fc, MTL::DataType::DataTypeBool, NS::UInteger(50)},
       {&compact, MTL::DataType::DataTypeBool, NS::UInteger(130)}});
}

static void dispatch_paged_attention_v2_reduce(
    metal::Device& d, metal::CommandEncoder& enc, array& out,
    const array& exp_sums, const array& max_logits, const array& tmp_out,
    const array& seq_lens, const array& cu_seqlens_q, int num_seqs,
    int total_q_tokens, int num_heads, int head_size, int max_num_partitions,
    const std::string& dt, bool use_sinks, const array* sinks,
    bool use_tq_fc, int partition_size = kPartitionSize,
    MTL::ComputePipelineState* rkernel = nullptr,
    const array* partition_offsets = nullptr) {
  if (rkernel == nullptr) {
    rkernel = paged_attention_v2_reduce_kernel(
        d, dt, head_size, partition_size, use_sinks, use_tq_fc,
        partition_offsets != nullptr);
  }
  const size_t reduce_shmem = kernels::paged_reduce_threadgroup_bytes(max_num_partitions);
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
  if (partition_offsets != nullptr) enc.set_input_array(*partition_offsets, 9);
  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, total_q_tokens, 1),
      MTL::Size::Make(256, 1, 1));
}

struct PagedAttentionMetadata {
  const GqaDecodeLengthPlan& gqa_length_plan;
  Dtype query_dtype;
  Dtype key_dtype;
  Dtype value_dtype;
  int query_tokens;
  int query_sequences;
  int head_size;
  int num_heads;
  int num_kv_heads;
  int block_size;
  int max_seq_len;
  int sliding_window;
  int window_seqlen_q;
  size_t item_size;
  float scale;
  float softcap;
  bool use_turboquant;
  int k_bits;
  int v_bits;
  bool use_mm_prefix;
  bool use_sinks;
  bool nax_ready;
  bool nax_enabled;
  int num_decode_requests;
  int num_decode_tokens;
  int max_decode_context_len;
  int decode_only_rows;
  bool gqa_disabled;
  int gqa_test_partition;
  int nax_split_hint;
  int nax_prefill_split_hint;
};

// The single routing point selects execution paths from CPU metadata and lazy providers.
enum class PagedAttentionRoute {
  MixedNaxSplit, MixedNaxUnsplit, MixedTiled,
  NaxSplit, NaxUnsplit, Tiled,
  GqaDecode, PerToken, SplitKv, Window, WindowSplitKv
};

struct PagedAttentionPlan {
  PagedAttentionRoute route = PagedAttentionRoute::PerToken;
  PagedAttentionRoute fallback = PagedAttentionRoute::PerToken;
  int total_q_tokens = 0;
  int num_seqs = 0;
  std::optional<kernels::TileConfig> tile_config;
  bool window_batch = false;
  int window_q = 0;
  int grid_y = 0;
  int max_num_partitions = 0;
  bool partition = false;
  int gqa_partition_size = 0;
  int64_t gqa_partitions = 0;
  const std::vector<int64_t>* offsets = nullptr;
  bool compact = false;
};

template <typename PrepareDecode, typename PrepareEncoder,
          typename GetGpuCores, typename GetDetectedGpuCores, typename GetArchitecture>
static PagedAttentionPlan select_paged_attention_route(
    const PagedAttentionMetadata& metadata,
    PrepareDecode prepare_decode, PrepareEncoder prepare_encoder,
    GetGpuCores get_gpu_cores, GetDetectedGpuCores get_detected_gpu_cores,
    GetArchitecture get_architecture) {
  const auto& m = metadata;

  // Tiled kernel for prefill batches, matching vLLM Triton's 2D/3D dispatch
  // split.  Pure-decode batches (every sequence has exactly 1 query token)
  // use the original per-token kernel.
  // A mixed batch's decode prefix (decode_only_rows > 0) is a pure decode
  // batch of its leading num_decode_requests sequences: cu_seqlens_q[0..D]
  // is 0..D, so every per-sequence view below stops at D.
  int total_q_tokens = m.decode_only_rows > 0
      ? m.decode_only_rows : m.query_tokens;
  int num_seqs = m.decode_only_rows > 0
      ? m.num_decode_requests : m.query_sequences;
  bool has_prefill = m.decode_only_rows == 0 && total_q_tokens > num_seqs;
  bool dtype_ok = m.query_dtype != float32
               && m.query_dtype == m.key_dtype;

  // Spec-decode verification windows: a multi-token batch whose segments
  // are all verification windows runs the per-token kernel in window mode
  // (kWindowRows-row sub-window threadgroups share every KV read), keeping
  // split-KV eligibility.  The tiled kernel is wrong for this shape: its
  // BQ-row MMA tiles waste ~80% of their compute on a K+1 window and it has
  // no KV-length parallelism.  head_size <= kWindowMaxHeadSize holds for
  // every constructed primitive (the binding rejects wider hints), so a
  // verify batch here always takes window mode, never the tiled fallback.
  const bool window_batch =
      has_prefill && m.window_seqlen_q > 1 && m.head_size <= kWindowMaxHeadSize;

  const bool use_nax = has_prefill && !window_batch && !m.use_turboquant
      && dtype_ok && !m.use_mm_prefix
      && nax_eligible(m.query_dtype, m.head_size, m.block_size, m.nax_ready, m.nax_enabled);
  const auto tile_config = has_prefill && !window_batch && !m.use_turboquant
      && dtype_ok ? kernels::select_tile_config(m.head_size) : std::nullopt;

  // Split long ordinary decode rows from NAX or tiled prefill: a one-row
  // decode sequence would otherwise occupy a whole BQ-row prefill tile with
  // no KV-length parallelism. Spec verify windows keep their whole-batch route.
  const bool split_mixed_batch = m.num_decode_requests > 0
      && m.num_decode_requests < num_seqs
      && m.num_decode_tokens == m.num_decode_requests
      && m.max_decode_context_len >= kMixedDecodeMinContext
      && (use_nax || tile_config.has_value());
  PagedAttentionPlan plan;
  plan.total_q_tokens = total_q_tokens;
  plan.num_seqs = num_seqs;
  plan.tile_config = tile_config;
  plan.window_batch = window_batch;
  if (use_nax) {
    int split_hint = m.nax_split_hint;
    if (split_mixed_batch) {
      // Only this route removes the ordinary decode prefix from NAX work.
      const int prefill_hint = m.nax_prefill_split_hint >= 0
          ? m.nax_prefill_split_hint : m.nax_split_hint;
      split_hint = prefill_hint;
    }
    using kernels::kNaxBQ;
    constexpr int kNaxSplitMinAvgQ = 2 * kNaxBQ;
    // Two pipelines of the same kernel, chosen per launch by function
    // constant 41 (nax_split_full):
    //  - split:   complete KV tiles of full 64-row Q blocks skip bounds checks;
    //  - unsplit: keeps the original single guarded loop structure.
    // Containing both tile bodies changes codegen enough to slow the guarded
    // path, so launches dominated by partial Q blocks (short prefill, cached
    // prefix + few new tokens, small chunks) take the unsplit pipeline.
    // nax_split_hint (0/1) is the caller's per-sequence choice, computed on the
    // CPU from the lengths it already holds (vllm_metal.attention.nax_split).
    // Without it (-1) the host only knows totals here, so it falls back to an
    // average of at least 128 query tokens (two Q blocks) per sequence.
    // head_size 512 stays unsplit either way: conservative enablement scope for
    // this change to avoid regressing measured shapes (the split was slower on
    // long D512 prefill; register pressure is suspected but unverified). D512
    // can be optimized separately later.
    bool split_full = m.head_size < 512
        && (split_hint >= 0
                ? split_hint == 1
                : m.query_tokens >= kNaxSplitMinAvgQ * m.query_sequences);
    plan.route = split_mixed_batch
        ? (split_full ? PagedAttentionRoute::MixedNaxSplit
                      : PagedAttentionRoute::MixedNaxUnsplit)
        : (split_full ? PagedAttentionRoute::NaxSplit
                      : PagedAttentionRoute::NaxUnsplit);
    return plan;
  }
  if (tile_config) {
    plan.route = split_mixed_batch
        ? PagedAttentionRoute::MixedTiled : PagedAttentionRoute::Tiled;
    return plan;
  }
  if (m.use_mm_prefix) {
    // paged_attention_primitive_fn already rejects every such batch eagerly;
    // this guard keeps the mask from being dropped silently should a new
    // routing condition appear above.
    throw std::invalid_argument(
        "mm_prefix ranges need the tiled prefill kernel, but this batch was "
        "routed to the per-token kernel");
  }

  prepare_decode();

  // Window mode: one threadgroup per (segment, kWindowRows-row sub-window)
  // instead of per token.  The function constant carries the sub-window
  // count so the kernel's grid.y decomposition folds to constants.
  int window_q = 0;
  if (window_batch) {
    window_q = (m.window_seqlen_q + kWindowRows - 1) / kWindowRows;
  }
  const int grid_y = window_batch ? num_seqs * window_q : total_q_tokens;
  const int max_num_partitions =
      (m.max_seq_len + kPartitionSize - 1) / kPartitionSize;
  // ----- Split-KV (flash-decoding) occupancy gate ------------------------
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
  const bool pure_decode = !has_prefill;  // every seq has exactly 1 query token
  const bool partition = (pure_decode || window_batch)
      && kernels::should_split_decode(
          m.num_heads, total_q_tokens, max_num_partitions, get_gpu_cores());

  prepare_encoder();

  // Ordinary decode has exactly one query row per request. Scheduler counts
  // distinguish it from expanded verification and one-token prefill segments.
  // A mixed batch's decode prefix qualifies only with its own length plan.
  const bool gqa_decode_rows =
      total_q_tokens == num_seqs &&
      ((num_seqs == 1 &&
        (m.num_decode_requests == -1 || m.num_decode_requests == 1)) ||
       (num_seqs > 1 && m.num_decode_requests == num_seqs &&
        m.num_decode_tokens == num_seqs));
  // Enforce geometry at the actual dispatch boundary, including the private
  // forced-partition entry. That entry bypasses only occupancy selection.
  const bool gqa_supported = !m.gqa_disabled
      && !has_prefill && m.window_seqlen_q <= 1 && gqa_decode_rows
      && (m.query_dtype == float16 || m.query_dtype == bfloat16)
      && m.query_dtype == m.key_dtype
      && m.query_dtype == m.value_dtype
      && !m.use_turboquant && m.softcap <= 0.f && !m.use_sinks
      && m.sliding_window < 0
      && gqa_decode_geometry_supported(
          m.num_heads, m.num_kv_heads,
          m.head_size, m.block_size);

  const int gqa_partition_size = !gqa_supported ? 0
      : m.gqa_test_partition > 0 ? m.gqa_test_partition
      : gqa_decode_plan_for_lengths(
            m.num_heads, m.num_kv_heads, m.head_size,
            num_seqs == 1 && m.decode_only_rows == 0
                ? gqa_decode_length_plan(std::array<int, 1>{m.max_seq_len})
                : m.gqa_length_plan,
            get_detected_gpu_cores(), m.block_size, get_architecture(), m.max_seq_len);

  const int64_t partitions = gqa_partition_size > 0
      ? (static_cast<int64_t>(m.max_seq_len) + gqa_partition_size - 1) /
            gqa_partition_size : 0;
  const auto* offsets = gqa_partition_size > 0
      ? gqa_compact_offsets(m.gqa_length_plan, gqa_partition_size, partitions) : nullptr;
  const bool compact = offsets != nullptr;
  const bool scratch_fits = gqa_partition_size > 0 && gqa_scratch_fits(
      total_q_tokens, m.num_heads, partitions, m.head_size, m.item_size, offsets);
  plan.window_q = window_q;
  plan.grid_y = grid_y;
  plan.max_num_partitions = max_num_partitions;
  plan.partition = partition;
  plan.gqa_partition_size = gqa_partition_size;
  plan.gqa_partitions = partitions;
  plan.offsets = offsets;
  plan.compact = compact;
  plan.fallback = window_batch
      ? (partition ? PagedAttentionRoute::WindowSplitKv : PagedAttentionRoute::Window)
      : (partition ? PagedAttentionRoute::SplitKv : PagedAttentionRoute::PerToken);
  plan.route = scratch_fits ? PagedAttentionRoute::GqaDecode : plan.fallback;
  return plan;
}

struct PagedAttentionKernelPlan {
  std::string kernel_name;
  std::string hash_name;
  bool use_partitioning;
  bool use_alibi;
  bool use_fp8;
  bool use_sinks;
  bool use_tq;
  int k_bits;
  int v_bits;
  int window_q;
};

static PagedAttentionKernelPlan make_paged_attention_kernel_plan(
    const std::string& dt, const std::string& k_cache_dt,
    const std::string& v_cache_dt, int head_size, int block_size,
    bool partition, bool use_sinks, bool use_turboquant,
    int k_bits, int v_bits, int window_q) {
  std::string kname = kernels::paged_name(
      dt, k_cache_dt, v_cache_dt, head_size, block_size,
      partition ? kPartitionSize : 0);
  // The hash name is MLX's cache key.  It MUST encode every function-constant
  // value that varies across calls.  ``kname`` already encodes the partition
  // suffix (_ps0 vs _ps512), which is 1:1 with use_partitioning.
  std::string hash_name = kname + "_v2"
      + "_tq" + (use_turboquant ? "1" : "0")
      + "_kb" + std::to_string(k_bits)
      + "_vb" + std::to_string(v_bits)
      + "_wq" + std::to_string(window_q)
      + "_sk" + (use_sinks ? "1" : "0");
  return {std::move(kname), std::move(hash_name), partition, false, false,
          use_sinks, use_turboquant, k_bits, v_bits, window_q};
}

struct PagedAttentionArrays {
  array& out;
  const array& query;
  const array& key_cache;
  const array& value_cache;
  const array& block_tables;
  const array& seq_lens;
  const array& cu_seqlens_q;
  const array* key_scale_cache;
  const array* value_scale_cache;
  const array* key_zero_cache;
  const array* v_centroids;
  const array* sinks;
  const array* mm_prefix_ranges;
  Stream s;
};

struct PagedAttentionEncoding {
  metal::Device* device = nullptr;
  MTL::Library* library = nullptr;
  metal::CommandEncoder* encoder = nullptr;
  std::string dt;
  std::string k_cache_dt;
  std::string v_cache_dt;
};

// Split-KV scratch factory shared by the split paths below: partial output
// + softmax (max, exp-sum) stats.  Tiny (~64 KB tmp_out @ conc=1/8K).
// add_temporary keeps them alive until the command buffer completes; MLX
// auto-inserts a barrier between the two dispatches via input/output
// dependency tracking (set_output here -> set_input in the reduce),
// mirroring MLX's own sdpa_vector_2pass.
static array make_paged_attention_temp(
    metal::CommandEncoder& enc, Shape shape, Dtype dtype) {
  array a(std::move(shape), dtype, nullptr, {});
  a.set_data(allocator::malloc(a.nbytes()));
  enc.add_temporary(a);
  return a;
}

static void dispatch_paged_attention_gqa_decode(
    const PagedAttentionArrays& arrays, const PagedAttentionMetadata& metadata,
    const PagedAttentionPlan& plan, const PagedAttentionEncoding& encoding,
    MTL::ComputePipelineState* gqa_rkernel) {
  auto& enc = *encoding.encoder;
  const int gqa_num_partitions = static_cast<int>(plan.gqa_partitions);
  record_paged_dispatch(
      PagedDispatch::GqaDecode, plan.gqa_partition_size, plan.num_seqs);
  std::string name = gqa_decode_kernel_name(
      encoding.dt, metadata.head_size, metadata.block_size, plan.gqa_partition_size);
  std::string hash = name + (plan.compact ? "_compact1" : "_compact0");
  const int num_threads = 32 * (metadata.num_heads / metadata.num_kv_heads);
  // Instantiated in pagedattention.metal and shipped in the v2 metallib.
  // A missing specialization is a build/packaging bug: fail loudly
  // rather than silently drop eligible long decode back to the slower
  // per-token kernel.
  auto* gkernel = encoding.device->get_kernel(
      name, encoding.library, hash,
      {{&plan.compact, MTL::DataType::DataTypeBool, NS::UInteger(130)}});

  Shape stats_shape = plan.offsets
      ? Shape{static_cast<int>(plan.offsets->back()), metadata.num_heads}
      : Shape{plan.total_q_tokens, metadata.num_heads, gqa_num_partitions};
  Shape output_shape = stats_shape;
  output_shape.push_back(metadata.head_size);
  array g_tmp_out = make_paged_attention_temp(enc, output_shape, arrays.query.dtype());
  array g_exp_sums = make_paged_attention_temp(enc, stats_shape, float32);
  array g_max_logits = make_paged_attention_temp(enc, stats_shape, float32);
  std::optional<array> g_offsets;
  if (plan.compact) {
    g_offsets.emplace(
        plan.offsets->data(), Shape{static_cast<int>(plan.offsets->size())}, int64);
    enc.add_temporary(*g_offsets);
  }

  enc.set_compute_pipeline_state(gkernel);
  enc.set_output_array(g_exp_sums, 0);
  enc.set_output_array(g_max_logits, 1);
  enc.set_output_array(g_tmp_out, 2);
  enc.set_input_array(arrays.query, 3);
  enc.set_input_array(arrays.key_cache, 4);
  enc.set_input_array(arrays.value_cache, 5);
  int32_t g_nkv = static_cast<int32_t>(metadata.num_kv_heads);
  enc.set_bytes(g_nkv, 8);
  enc.set_bytes(metadata.scale, 9);
  enc.set_input_array(arrays.block_tables, 11);
  enc.set_input_array(arrays.seq_lens, 12);
  int32_t g_max_blocks = static_cast<int32_t>(arrays.block_tables.shape(1));
  enc.set_bytes(g_max_blocks, 13);
  bind_paged_attn_strides(enc, arrays.query, arrays.key_cache, metadata.block_size);
  if (plan.compact) enc.set_input_array(*g_offsets, 18);
  // One threadgroup per (partition, kv head, sequence); each simdgroup
  // owns one query head of the GQA group.
  enc.dispatch_threadgroups(
      MTL::Size::Make(gqa_num_partitions, metadata.num_kv_heads, plan.num_seqs),
      MTL::Size::Make(num_threads, 1, 1));

  dispatch_paged_attention_v2_reduce(
      *encoding.device, enc, arrays.out, g_exp_sums, g_max_logits, g_tmp_out,
      arrays.seq_lens, arrays.cu_seqlens_q, plan.num_seqs, plan.total_q_tokens,
      metadata.num_heads, metadata.head_size,
      gqa_num_partitions, encoding.dt, /*use_sinks=*/false, nullptr,
      /*use_tq_fc=*/false, plan.gqa_partition_size, gqa_rkernel,
      plan.compact ? &*g_offsets : nullptr);
  return;
}

static void dispatch_paged_attention_per_token(
    const PagedAttentionArrays& arrays, const PagedAttentionMetadata& metadata,
    const PagedAttentionPlan& plan, const PagedAttentionEncoding& encoding) {
  auto& enc = *encoding.encoder;
  const auto kernel_plan = make_paged_attention_kernel_plan(
      encoding.dt, encoding.k_cache_dt, encoding.v_cache_dt,
      metadata.head_size, metadata.block_size, plan.partition,
      arrays.sinks != nullptr, metadata.use_turboquant,
      metadata.k_bits, metadata.v_bits, plan.window_q);
  auto* kernel = encoding.device->get_kernel(
      kernel_plan.kernel_name, encoding.library, kernel_plan.hash_name,
      {{&kernel_plan.use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)},
       {&kernel_plan.use_alibi,        MTL::DataType::DataTypeBool, NS::UInteger(20)},
       {&kernel_plan.use_fp8,          MTL::DataType::DataTypeBool, NS::UInteger(30)},
       {&kernel_plan.use_sinks,        MTL::DataType::DataTypeBool, NS::UInteger(40)},
       {&kernel_plan.use_tq,           MTL::DataType::DataTypeBool, NS::UInteger(50)},
       {&kernel_plan.k_bits,           MTL::DataType::DataTypeInt,  NS::UInteger(60)},
       {&kernel_plan.v_bits,           MTL::DataType::DataTypeInt,  NS::UInteger(70)},
       {&kernel_plan.window_q,         MTL::DataType::DataTypeInt,  NS::UInteger(110)}});

  const size_t shmem = kernels::paged_threadgroup_bytes(
      metadata.head_size, metadata.block_size, static_cast<int>(arrays.query.itemsize()),
      plan.window_batch, kWindowRows);
  const int grid_z = plan.partition ? plan.max_num_partitions : 1;
  constexpr int num_threads = 256;

  // TurboQuant scale/zero/centroid buffers (slots 22-27); shared by both paths.
  auto bind_turboquant = [&]() {
    if (!metadata.use_turboquant) return;
    enc.set_input_array(*arrays.key_scale_cache,   22);
    enc.set_input_array(*arrays.value_scale_cache, 23);
    int32_t v_block_stride_i = static_cast<int32_t>(arrays.value_cache.strides()[0]);
    int32_t v_head_stride_i  = static_cast<int32_t>(arrays.value_cache.strides()[2]);
    enc.set_bytes(v_block_stride_i, 24);
    enc.set_bytes(v_head_stride_i,  25);
    enc.set_input_array(*arrays.key_zero_cache, 26);
    enc.set_input_array(*arrays.v_centroids, 27);
    int64_t scale_block_stride = arrays.key_scale_cache->strides()[0];
    enc.set_bytes(scale_block_stride, 28);
    int64_t scale_head_stride = arrays.key_scale_cache->strides()[2];
    enc.set_bytes(scale_head_stride, 29);
  };

  // Sink logits (slot 18); bound only when the kernel was specialized with
  // use_sinks, since the argument is declared under that function constant.
  auto bind_sinks = [&]() {
    if (!kernel_plan.use_sinks) return;
    enc.set_input_array(*arrays.sinks, 18);
  };

  if (!plan.partition) {
    // Single-pass path (grid.z = 1): the original decode/per-token kernel.
    record_paged_dispatch(
        plan.window_batch ? PagedDispatch::Window : PagedDispatch::PerToken);
    enc.set_compute_pipeline_state(kernel);
    enc.set_threadgroup_memory_length(shmem, 0);
    bind_paged_attn_buffers(
        enc, arrays.out, arrays.query, arrays.key_cache, arrays.value_cache,
        metadata.num_kv_heads, metadata.softcap, arrays.block_tables,
        arrays.seq_lens, arrays.cu_seqlens_q,
        metadata.block_size, metadata.sliding_window);
    enc.set_bytes(metadata.scale, 9);
    bind_turboquant();
    bind_sinks();
    enc.dispatch_threadgroups(
        MTL::Size::Make(metadata.num_heads, plan.grid_y, grid_z),
        MTL::Size::Make(num_threads, 1, 1));
    return;
  }

  // ----- Split-KV path: paged_attention(_ps512) -> paged_attention_v2_reduce.
  record_paged_dispatch(
      plan.window_batch ? PagedDispatch::WindowSplitKv : PagedDispatch::SplitKv);
  Shape stats_shape{plan.total_q_tokens, metadata.num_heads, plan.max_num_partitions};
  Shape output_shape = stats_shape;
  output_shape.push_back(metadata.head_size);
  array tmp_out = make_paged_attention_temp(enc, output_shape, arrays.query.dtype());
  array exp_sums = make_paged_attention_temp(enc, stats_shape, float32);
  array max_logits = make_paged_attention_temp(enc, stats_shape, float32);

  // Pass 1: partitioned kernel writes partials.  Out slot (2) = tmp_out scratch.
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(shmem, 0);
  bind_paged_attn_buffers(
      enc, tmp_out, arrays.query, arrays.key_cache, arrays.value_cache,
      metadata.num_kv_heads, metadata.softcap, arrays.block_tables,
      arrays.seq_lens, arrays.cu_seqlens_q,
      metadata.block_size, metadata.sliding_window);
  enc.set_bytes(metadata.scale, 9);
  enc.set_output_array(exp_sums, 0);
  enc.set_output_array(max_logits, 1);
  bind_turboquant();
  // The partitioned kernel does not fold the sink itself (the reduce does, so
  // it is counted once rather than once per partition), but slot 18 is still a
  // declared argument under function constant 40, so it must be bound here.
  bind_sinks();
  enc.dispatch_threadgroups(
      MTL::Size::Make(metadata.num_heads, plan.grid_y, grid_z),
      MTL::Size::Make(num_threads, 1, 1));

  // Pass 2: reduce per-partition partials -> out (log-sum-exp combine).
  dispatch_paged_attention_v2_reduce(
      *encoding.device, enc, arrays.out, exp_sums, max_logits, tmp_out,
      arrays.seq_lens, arrays.cu_seqlens_q, plan.num_seqs, plan.total_q_tokens,
      metadata.num_heads, metadata.head_size, plan.max_num_partitions, encoding.dt,
      kernel_plan.use_sinks, arrays.sinks, kernel_plan.use_tq);
}

static void dispatch_paged_attention_mixed(
    const PagedAttentionArrays& arrays, const PagedAttentionMetadata& metadata,
    const PagedAttentionPlan& plan);

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
    const GqaDecodeLengthPlan& gqa_length_plan = {},
    int nax_split_hint = -1, int nax_prefill_split_hint = -1) {
  int head_size = static_cast<int>(query.shape(2));
  PagedAttentionMetadata metadata{
      gqa_length_plan, query.dtype(), key_cache.dtype(), value_cache.dtype()};
  metadata.head_size = head_size;
  metadata.query_tokens = static_cast<int>(query.shape(0));
  metadata.query_sequences = static_cast<int>(cu_seqlens_q.shape(0)) - 1;
  metadata.num_heads = static_cast<int>(query.shape(1));
  metadata.num_kv_heads = num_kv_heads;
  metadata.block_size = block_size;
  metadata.max_seq_len = max_seq_len;
  metadata.sliding_window = sliding_window;
  metadata.window_seqlen_q = window_seqlen_q;
  metadata.item_size = query.itemsize();
  metadata.scale = scale;
  metadata.softcap = softcap;
  metadata.use_turboquant = use_turboquant;
  metadata.k_bits = k_bits;
  metadata.v_bits = v_bits;
  metadata.use_mm_prefix = mm_prefix_ranges != nullptr;
  metadata.use_sinks = sinks != nullptr;
  metadata.nax_ready = nax_lib_ready_;
  metadata.nax_enabled = nax_enabled_;
  metadata.num_decode_requests = num_decode_requests;
  metadata.num_decode_tokens = num_decode_tokens;
  metadata.max_decode_context_len = max_decode_context_len;
  metadata.decode_only_rows = decode_only_rows;
  metadata.gqa_disabled = gqa_disabled;
  metadata.gqa_test_partition = gqa_test_partition;
  metadata.nax_split_hint = nax_split_hint;
  metadata.nax_prefill_split_hint = nax_prefill_split_hint;
  const PagedAttentionArrays arrays{
      out, query, key_cache, value_cache, block_tables, seq_lens, cu_seqlens_q,
      key_scale_cache, value_scale_cache, key_zero_cache, v_centroids,
      sinks, mm_prefix_ranges, s};
  PagedAttentionEncoding encoding;
  auto plan = select_paged_attention_route(
      metadata,
      [&]() {
        // Fallback: original per-token kernel
        encoding.device = &metal::device(s.device);
        encoding.dt = dtype_to_metal(query.dtype());
        encoding.k_cache_dt = dtype_to_metal(key_cache.dtype());
        encoding.v_cache_dt = dtype_to_metal(value_cache.dtype());
      },
      [&]() {
        encoding.library = encoding.device->get_library("paged_attention_v2_kern");
        encoding.encoder = &metal::get_command_encoder(s);
      },
      gpu_core_count, detected_gpu_core_count,
      [&]() -> const std::string& { return encoding.device->get_architecture(); });

  MTL::ComputePipelineState* gqa_rkernel = nullptr;
  if (plan.route == PagedAttentionRoute::GqaDecode) {
    // Check the actual reducer pipeline before allocating scratch or encoding
    // the GQA producer. Dynamic statistics alone can fit while their sum with
    // static workspace exceeds the device limit. Reuse this pipeline below.
    gqa_rkernel = paged_attention_v2_reduce_kernel(
        *encoding.device, encoding.dt, metadata.head_size,
        plan.gqa_partition_size, false, false, plan.compact);
    // This compiled-pipeline check is the only GQA-to-per-token fallback.
    if (!paged_reduce_memory_fits(
            kernels::paged_reduce_threadgroup_bytes(plan.gqa_partitions),
            gqa_rkernel->staticThreadgroupMemoryLength(),
            encoding.device->mtl_device()->maxThreadgroupMemoryLength())) {
      plan.route = plan.fallback;
    }
  }

  switch (plan.route) {
    case PagedAttentionRoute::MixedNaxSplit:
    case PagedAttentionRoute::MixedNaxUnsplit:
    case PagedAttentionRoute::MixedTiled:
      dispatch_paged_attention_mixed(arrays, metadata, plan);
      return;
    case PagedAttentionRoute::NaxSplit:
    case PagedAttentionRoute::NaxUnsplit:
      record_paged_dispatch(PagedDispatch::NaxPrefill);
      dispatch_paged_attention_nax(
          out, query, key_cache, value_cache, num_kv_heads, scale, softcap,
          block_tables, seq_lens, cu_seqlens_q, block_size, sliding_window,
          0, s, sinks, plan.route == PagedAttentionRoute::NaxSplit);
      return;
    // TurboQuant stays excluded from the tiled kernel. Sink prefill can use it:
    // pagedattention_tiled.metal folds the sink into each row's denominator-only
    // softmax state before final normalization.
    case PagedAttentionRoute::Tiled:
      record_paged_dispatch(PagedDispatch::TiledPrefill);
      dispatch_paged_attention_tiled(
          out, query, key_cache, value_cache, num_kv_heads, scale, softcap,
          block_tables, seq_lens, cu_seqlens_q, block_size, max_seq_len,
          sliding_window, 0, *plan.tile_config, s, sinks, mm_prefix_ranges);
      return;
    case PagedAttentionRoute::GqaDecode:
      dispatch_paged_attention_gqa_decode(arrays, metadata, plan, encoding, gqa_rkernel);
      return;
    case PagedAttentionRoute::PerToken:
    case PagedAttentionRoute::SplitKv:
    case PagedAttentionRoute::Window:
    case PagedAttentionRoute::WindowSplitKv:
      dispatch_paged_attention_per_token(arrays, metadata, plan, encoding);
      return;
  }
}

static void dispatch_paged_attention_mixed(
    const PagedAttentionArrays& arrays, const PagedAttentionMetadata& metadata,
    const PagedAttentionPlan& plan) {
  // The decode prefix reads only the decode rows. Size its split-KV
  // partitions and scratch by their context bound, never by max_seq_len:
  // that bound also covers the prefill rows and may be a whole allocation
  // bound, which the prefill kernels ignore but split-KV would plan for.
  const int decode_max_seq_len =
      std::min(metadata.max_seq_len, metadata.max_decode_context_len);
  // A caller-supplied plan for exactly the decode prefix (validated at the
  // binding against max_seq_len and max_decode_context_len) opts the prefix
  // into GQA and tightens the bound to the plan's longest decode row.
  const bool prefix_plan =
      metadata.gqa_length_plan.num_requests == metadata.num_decode_requests;
  dispatch_paged_attention_v2_online(
      arrays.out, arrays.query, arrays.key_cache, arrays.value_cache,
      metadata.num_kv_heads, metadata.scale, metadata.softcap,
      arrays.block_tables, arrays.seq_lens, arrays.cu_seqlens_q,
      metadata.block_size,
      prefix_plan ? metadata.gqa_length_plan.max_length : decode_max_seq_len,
      metadata.sliding_window, metadata.window_seqlen_q, arrays.s,
      arrays.key_scale_cache, arrays.value_scale_cache,
      arrays.key_zero_cache, arrays.v_centroids,
      metadata.use_turboquant, metadata.k_bits, metadata.v_bits, arrays.sinks, nullptr,
      metadata.num_decode_requests, metadata.num_decode_tokens,
      metadata.max_decode_context_len,
      metadata.num_decode_tokens, metadata.gqa_disabled, 0,
      prefix_plan ? metadata.gqa_length_plan : GqaDecodeLengthPlan{});
  // Keep the decode prefix's GQA selection visible in the diagnostics.
  const int decode_partition = g_last_gqa_partition.load(std::memory_order_relaxed);
  const int decode_requests = g_last_gqa_requests.load(std::memory_order_relaxed);
  // Decode rows lead the batch, so the first prefill sequence starts at
  // q-block cu_seqlens_q[D] / BQ + D with D = num_decode_requests.
  if (plan.route == PagedAttentionRoute::MixedNaxSplit ||
      plan.route == PagedAttentionRoute::MixedNaxUnsplit) {
    dispatch_paged_attention_nax(
        arrays.out, arrays.query, arrays.key_cache, arrays.value_cache,
        metadata.num_kv_heads, metadata.scale, metadata.softcap,
        arrays.block_tables, arrays.seq_lens, arrays.cu_seqlens_q,
        metadata.block_size, metadata.sliding_window,
        metadata.num_decode_tokens / kernels::kNaxBQ + metadata.num_decode_requests,
        arrays.s, arrays.sinks,
        plan.route == PagedAttentionRoute::MixedNaxSplit);
    record_paged_dispatch(PagedDispatch::MixedNaxPrefillDecode,
                          decode_partition, decode_requests);
    return;
  }
  const int q_block_offset =
      metadata.num_decode_tokens / plan.tile_config->BQ + metadata.num_decode_requests;
  dispatch_paged_attention_tiled(
      arrays.out, arrays.query, arrays.key_cache, arrays.value_cache,
      metadata.num_kv_heads, metadata.scale, metadata.softcap,
      arrays.block_tables, arrays.seq_lens, arrays.cu_seqlens_q,
      metadata.block_size, metadata.max_seq_len, metadata.sliding_window, q_block_offset,
      *plan.tile_config, arrays.s, arrays.sinks, arrays.mm_prefix_ranges);
  record_paged_dispatch(PagedDispatch::MixedPrefillDecode,
                        decode_partition, decode_requests);
  return;
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
      const GqaDecodeLengthPlan& gqa_length_plan = {},
      int nax_split_hint = -1, int nax_prefill_split_hint = -1)
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
        gqa_length_plan_(gqa_length_plan), nax_split_hint_(nax_split_hint),
        nax_prefill_split_hint_(nax_prefill_split_hint) {}

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
        0, gqa_disabled_, gqa_test_partition_, gqa_length_plan_,
        nax_split_hint_, nax_prefill_split_hint_);
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
        && rhs->gqa_length_plan_ == gqa_length_plan_
        && rhs->nax_split_hint_ == nax_split_hint_
        && rhs->nax_prefill_split_hint_ == nax_prefill_split_hint_;
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
  int nax_split_hint_;
  int nax_prefill_split_hint_;
};

array paged_attention_primitive_fn(
    const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float scale, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int block_size, int max_seq_len, int sliding_window,
    bool use_turboquant, const std::string& quant_type,
    const array* key_scale_cache,
    const array* value_scale_cache,
    const array* key_zero_cache,
    const array* v_centroids,
    int v_bits, int window_seqlen_q,
    const array* sinks,
    const array* mm_prefix_ranges,
    int num_decode_requests,
    int num_decode_tokens,
    int max_decode_context_len, bool gqa_disabled,
    int gqa_test_partition,
    const std::vector<int>& gqa_context_lens,
    const std::optional<GqaDecodeLengthPlan>& gqa_length_plan,
    int nax_split_hint, int nax_prefill_split_hint) {
  if (nax_split_hint < -1 || nax_split_hint > 1)
    throw std::invalid_argument("nax_split_hint must be -1, 0 or 1");
  if (nax_prefill_split_hint < -1 || nax_prefill_split_hint > 1)
    throw std::invalid_argument("nax_prefill_split_hint must be -1, 0 or 1");
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
        || !kernels::select_tile_config(head_size_q)) {
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
      max_decode_context_len, gqa_disabled, gqa_test_partition, lengths,
      nax_split_hint, nax_prefill_split_hint);
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

struct TQEncodePlan {
  std::string kernel_name;
  std::string hash_name;
  int k_bits;
  int v_bits;
  bool k_signed;
  int grid_x;
  int grid_y;
  int threads_x;
};

static TQEncodePlan make_tq_encode_plan(
    Dtype dtype, int num_tokens, int num_kv_heads, int head_size,
    int k_bits, int v_bits, bool k_signed) {
  auto kv_dt = dtype_to_metal(dtype);
  std::string kname = "tq_encode_" + kv_dt +
                      "_hs" + std::to_string(head_size);
  // Function constants control bit widths + signedness used inside the
  // kernel.  The hash name MUST encode them so MLX caches the right
  // specialization per (k_bits, k_signed, v_bits) tuple.
  std::string hash_name = kname +
      "_kb" + std::to_string(k_bits) +
      "_ks" + (k_signed ? "1" : "0") +
      "_vb" + std::to_string(v_bits);
  return {std::move(kname), std::move(hash_name), k_bits, v_bits, k_signed,
          num_tokens, num_kv_heads, head_size};
}

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

    auto plan = make_tq_encode_plan(
        key.dtype(), num_tokens, num_kv_heads, head_size,
        k_bits_, v_bits_, k_signed_);

    auto* lib = d.get_library("paged_attention_v2_kern");
    auto* kernel = d.get_kernel(
        plan.kernel_name, lib, plan.hash_name,
        {{&plan.k_bits,   MTL::DataType::DataTypeInt,  NS::UInteger(80)},
         {&plan.k_signed, MTL::DataType::DataTypeBool, NS::UInteger(81)},
         {&plan.v_bits,   MTL::DataType::DataTypeInt,  NS::UInteger(90)}});

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
        MTL::Size::Make(plan.grid_x, plan.grid_y, 1),
        MTL::Size::Make(plan.threads_x, 1, 1));

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

std::vector<array> tq_encode_primitive_fn(
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

struct ReshapeAndCachePlan {
  std::string kernel_name;
  std::string hash_name;
  bool use_fp8;
  int grid_x;
  int threads_x;
};

static ReshapeAndCachePlan make_reshape_and_cache_plan(
    Dtype dtype, Dtype cache_dtype,
    int num_tokens, int num_kv_heads, int head_size) {
  auto kv_dt = dtype_to_metal(dtype);
  auto cache_dt = dtype_to_metal(cache_dtype);
  std::string kname = kernels::reshape_and_cache_name(kv_dt, cache_dt);
  // rac_use_fp8_scales (fc 100) = false: non-quantized cache, so the
  // k_scale/v_scale buffers (5,6) are absent from the signature.
  std::string hash_name = kname + "_fp8_0";
  return {std::move(kname), std::move(hash_name), false,
          num_tokens, std::min(num_kv_heads * head_size, 256)};
}

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

    auto plan = make_reshape_and_cache_plan(
        key.dtype(), inputs[2].dtype(), num_tokens, num_kv_heads, head_size);

    auto* lib = d.get_library("paged_attention_v2_kern");
    auto* kernel = d.get_kernel(
        plan.kernel_name, lib, plan.hash_name,
        {{&plan.use_fp8, MTL::DataType::DataTypeBool, NS::UInteger(100)}});

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

    enc.dispatch_threadgroups(
        MTL::Size::Make(plan.grid_x, 1, 1),
        MTL::Size::Make(plan.threads_x, 1, 1));
  }

  const char* name() const override { return "ReshapeAndCache"; }

  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const ReshapeAndCachePrimitive*>(&other) != nullptr;
  }
};

std::vector<array> reshape_and_cache_primitive_fn(
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
