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
#include <optional>
#include <string>
#include <unordered_map>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace nb = nanobind;
using namespace mlx::core;

#ifndef VLLM_METAL_PARTITION_SIZE
#define VLLM_METAL_PARTITION_SIZE 512
#endif

// ---------------------------------------------------------------------------
// Library caching
// ---------------------------------------------------------------------------

static std::string v2_paged_attention_source_;
constexpr int kPartitionSize = VLLM_METAL_PARTITION_SIZE;

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
// desktop one saturate at very different grid sizes.  Read once; falls back to
// a modest default if the query ever fails (future macOS / service tree).
static int gpu_core_count() {
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
    return cores > 0 ? cores : 14;
  }();
  return v;
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

// MLX 0.31.2 moved stream-scoped encoder/temporary management from
// ``metal::Device`` onto ``metal::CommandEncoder`` plus a free
// ``metal::get_command_encoder(Stream)`` helper.  Keep a tiny shim here so the
// paged-kernel bridge compiles against both 0.31.1 and 0.31.2.
#if MLX_VERSION_NUMERIC >= 31002
static metal::CommandEncoder& get_command_encoder_compat(
    metal::Device& d,
    Stream s) {
  (void)d;
  return metal::get_command_encoder(s);
}

static void add_temporary_compat(
    metal::CommandEncoder& enc,
    const array& arr,
    metal::Device& d,
    Stream s) {
  (void)d;
  (void)s;
  enc.add_temporary(arr);
}
#else
static metal::CommandEncoder& get_command_encoder_compat(
    metal::Device& d,
    Stream s) {
  return d.get_command_encoder(s.index);
}

static void add_temporary_compat(
    metal::CommandEncoder& enc,
    const array& arr,
    metal::Device& d,
    Stream s) {
  (void)enc;
  d.add_temporary(arr, s.index);
}
#endif

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

// Shared buffer binding for paged attention kernels (slots 2-21).
static void bind_paged_attn_buffers(
    metal::CommandEncoder& enc,
    array& out, const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int sliding_window) {
  int num_heads = static_cast<int>(query.shape(1));
  int head_size = static_cast<int>(query.shape(2));

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

  int32_t q_stride        = static_cast<int32_t>(num_heads * head_size);
  int32_t kv_block_stride = static_cast<int32_t>(key_cache.strides()[0]);
  int32_t kv_head_stride  = static_cast<int32_t>(key_cache.strides()[2]);
  enc.set_bytes(q_stride,        15);
  enc.set_bytes(kv_block_stride, 16);
  enc.set_bytes(kv_head_stride,  17);

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

static void dispatch_paged_attention_tiled(
    array& out, const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float scale, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int block_size, int max_seq_len, int sliding_window,
    TileConfig cfg, Stream s) {
  auto& d = metal::device(s.device);

  int total_q_tokens = static_cast<int>(query.shape(0));
  int head_size  = static_cast<int>(query.shape(2));
  int num_seqs   = static_cast<int>(cu_seqlens_q.shape(0)) - 1;

  int total_q_blocks = total_q_tokens / cfg.BQ + num_seqs;

  auto dt = dtype_to_metal(query.dtype());
  std::string kname =
      "paged_attention_tiled_" + dt +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_bq" + std::to_string(cfg.BQ) +
      "_tk" + std::to_string(cfg.TILE_KV) +
      "_nt" + std::to_string(cfg.NUM_THREADS);

  auto* lib = d.get_library("paged_attention_v2_kern");
  auto* kernel = d.get_kernel(kname, lib, kname, {});

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
  auto& enc = get_command_encoder_compat(d, s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(shmem, 0);

  bind_paged_attn_buffers(enc, out, query, key_cache, value_cache,
                          num_kv_heads, softcap, block_tables, seq_lens,
                          cu_seqlens_q, sliding_window);
  enc.set_bytes(scale, 9);

  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, total_q_blocks, 1),
      MTL::Size::Make(cfg.NUM_THREADS, 1, 1));
}

static void dispatch_paged_attention_v2_online(
    array& out, const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float scale, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int block_size, int max_seq_len, int sliding_window, Stream s,
    // TurboQuant (optional, all nullptr when disabled):
    const array* key_scale_cache = nullptr,
    const array* value_scale_cache = nullptr,
    const array* key_zero_cache = nullptr,
    const array* v_centroids = nullptr,
    bool use_turboquant = false,
    int k_bits = 8,
    int v_bits = 3) {
  int head_size = static_cast<int>(query.shape(2));

  // Tiled kernel for prefill batches (max_seqlen_q > 1), matching vLLM
  // Triton's 2D/3D dispatch split.  Pure-decode batches (every sequence has
  // exactly 1 query token) use the original per-token kernel.
  int total_q_tokens = static_cast<int>(query.shape(0));
  int num_seqs = static_cast<int>(cu_seqlens_q.shape(0)) - 1;
  bool has_prefill = total_q_tokens > num_seqs;
  bool dtype_ok = query.dtype() != float32
               && query.dtype() == key_cache.dtype();
  if (has_prefill && !use_turboquant && dtype_ok) {
    if (auto cfg = select_tile_config(head_size)) {
      dispatch_paged_attention_tiled(
          out, query, key_cache, value_cache,
          num_kv_heads, scale, softcap,
          block_tables, seq_lens, cu_seqlens_q,
          block_size, max_seq_len, sliding_window, *cfg, s);
      return;
    }
  }

  // Fallback: original per-token kernel
  auto& d = metal::device(s.device);

  int num_heads  = static_cast<int>(query.shape(1));

  auto dt        = dtype_to_metal(query.dtype());
  auto k_cache_dt = dtype_to_metal(key_cache.dtype());
  auto v_cache_dt = dtype_to_metal(value_cache.dtype());
  // ----- Split-KV (flash-decoding) occupancy gate ------------------------
  const bool pure_decode = !has_prefill;  // every seq has exactly 1 query token
  const int base_grid = num_heads * total_q_tokens;  // grid.z = 1 occupancy
  const int max_num_partitions =
      (max_seq_len + kPartitionSize - 1) / kPartitionSize;
  // TurboQuant and sliding-window batches take the split too.  TQ partials
  // stay in the rotated domain with one inverse FWHT in the reduce (measured
  // -23% TPOT at conc=1/8K); windowed batches mask per partition, and a
  // fully-masked partition contributes exact zeros — epsilon-normalized
  // partial, zero merge weight (Gemma-4 E2B measured -35% TPOT at conc=1).
  const bool partition = pure_decode
      && base_grid < min_decode_grid() && max_num_partitions >= 2;

  std::string kname =
      "paged_attention_" + dt + "_cache_" + k_cache_dt + "_" + v_cache_dt +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_nt256_nsl32_ps" + std::to_string(partition ? kPartitionSize : 0);

  bool use_partitioning  = partition;
  bool use_alibi         = false;
  bool use_fp8           = false;
  bool use_sinks         = false;
  bool use_tq_fc         = use_turboquant;
  int  k_bits_i          = k_bits;
  int  v_bits_i          = v_bits;

  // The hash name is MLX's cache key.  It MUST encode every function-constant
  // value that varies across calls.  ``kname`` already encodes the partition
  // suffix (_ps0 vs _ps512), which is 1:1 with use_partitioning.
  std::string hash_name = kname + "_v2"
      + "_tq" + (use_tq_fc ? "1" : "0")
      + "_kb" + std::to_string(k_bits_i)
      + "_vb" + std::to_string(v_bits_i);

  auto* lib = d.get_library("paged_attention_v2_kern");
  auto* kernel = d.get_kernel(
      kname, lib, hash_name,
      {{&use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)},
       {&use_alibi,        MTL::DataType::DataTypeBool, NS::UInteger(20)},
       {&use_fp8,          MTL::DataType::DataTypeBool, NS::UInteger(30)},
       {&use_sinks,        MTL::DataType::DataTypeBool, NS::UInteger(40)},
       {&use_tq_fc,        MTL::DataType::DataTypeBool, NS::UInteger(50)},
       {&k_bits_i,         MTL::DataType::DataTypeInt,  NS::UInteger(60)},
       {&v_bits_i,         MTL::DataType::DataTypeInt,  NS::UInteger(70)}});

  constexpr int NUM_THREADS    = 256;
  constexpr int NUM_SIMD_LANES = 32;
  constexpr int NUM_WARPS      = NUM_THREADS / NUM_SIMD_LANES;
  int warp_scores_bytes = NUM_WARPS * block_size
                          * static_cast<int>(sizeof(float));
  int merge_bytes = (2 * NUM_WARPS + NUM_WARPS * head_size)
                    * static_cast<int>(sizeof(float));
  size_t shmem = static_cast<size_t>(std::max(warp_scores_bytes, merge_bytes));

  auto& enc = get_command_encoder_compat(d, s);

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
  };

  if (!partition) {
    // Single-pass path (grid.z = 1): the original decode/per-token kernel.
    enc.set_compute_pipeline_state(kernel);
    enc.set_threadgroup_memory_length(shmem, 0);
    bind_paged_attn_buffers(enc, out, query, key_cache, value_cache,
                            num_kv_heads, softcap, block_tables, seq_lens,
                            cu_seqlens_q, sliding_window);
    enc.set_bytes(scale, 9);
    bind_turboquant();
    enc.dispatch_threadgroups(
        MTL::Size::Make(num_heads, total_q_tokens, 1),
        MTL::Size::Make(NUM_THREADS, 1, 1));
    return;
  }

  // ----- Split-KV path: paged_attention(_ps512) -> paged_attention_v2_reduce.
  // Per-partition scratch: partial output + softmax (max, exp-sum) stats.
  // Tiny (~64 KB tmp_out @ conc=1/8K).  add_temporary keeps them alive until
  // the command buffer completes; MLX auto-inserts a barrier between the two
  // dispatches via input/output dependency tracking (set_output here ->
  // set_input below), mirroring MLX's own sdpa_vector_2pass.
  auto make_temp = [&](Shape shape, Dtype dtype) {
    array a(std::move(shape), dtype, nullptr, {});
    a.set_data(allocator::malloc(a.nbytes()));
    add_temporary_compat(enc, a, d, s);
    return a;
  };
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
                          cu_seqlens_q, sliding_window);
  enc.set_bytes(scale, 9);
  enc.set_output_array(exp_sums, 0);
  enc.set_output_array(max_logits, 1);
  bind_turboquant();
  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, total_q_tokens, max_num_partitions),
      MTL::Size::Make(NUM_THREADS, 1, 1));

  // Pass 2: reduce per-partition partials -> out (log-sum-exp combine).
  std::string rname =
      "paged_attention_v2_reduce_" + dt + "_hs" + std::to_string(head_size) +
      "_nt256_nsl32_ps" + std::to_string(kPartitionSize);
  // The reduce kernel reads only use_sinks (40) and use_turboquant (50); the
  // other function constants are inert for it.  TurboQuant batches take this
  // path (use_tq_fc varies: the TQ reduce applies the deferred inverse FWHT),
  // sinks are unsupported and stay false.  The cache key MUST encode every
  // constant the pipeline is specialized on — otherwise the first compile
  // wins and a later caller with different constants silently reuses the
  // wrong pipeline.
  std::string rhash = rname + "_v2reduce"
      + "_tq" + (use_tq_fc ? "1" : "0")
      + "_sk" + (use_sinks ? "1" : "0");
  auto* rkernel = d.get_kernel(
      rname, lib, rhash,
      {{&use_sinks, MTL::DataType::DataTypeBool, NS::UInteger(40)},
       {&use_tq_fc, MTL::DataType::DataTypeBool, NS::UInteger(50)}});
  enc.set_compute_pipeline_state(rkernel);
  // Metal requires setThreadgroupMemoryLength to be a multiple of 16 bytes
  // (odd partition counts would yield 8 mod 16 and trip the API-validation
  // layer).  The kernel reads exactly 2*num_partitions floats; the padding
  // is never touched.
  size_t reduce_shmem =
      static_cast<size_t>(2 * max_num_partitions) * sizeof(float);
  enc.set_threadgroup_memory_length((reduce_shmem + 15) & ~size_t(15), 0);
  enc.set_output_array(out, 0);
  enc.set_input_array(exp_sums, 1);
  enc.set_input_array(max_logits, 2);
  enc.set_input_array(tmp_out, 3);
  enc.set_input_array(seq_lens, 4);
  int32_t max_num_partitions_i = static_cast<int32_t>(max_num_partitions);
  enc.set_bytes(max_num_partitions_i, 5);
  enc.set_input_array(cu_seqlens_q, 7);
  int32_t num_seqs_i = static_cast<int32_t>(num_seqs);
  enc.set_bytes(num_seqs_i, 8);
  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, total_q_tokens, 1),
      MTL::Size::Make(NUM_THREADS, 1, 1));
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
      bool use_turboquant = false, int k_bits = 8, int v_bits = 3)
      : UnaryPrimitive(stream),
        num_kv_heads_(num_kv_heads), scale_(scale), softcap_(softcap),
        block_size_(block_size), max_seq_len_(max_seq_len),
        sliding_window_(sliding_window),
        use_turboquant_(use_turboquant), k_bits_(k_bits), v_bits_(v_bits) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error(
        "PagedAttentionPrimitive only supports GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    // Non-TQ inputs: [query, key_cache, value_cache, block_tables, seq_lens, cu_seqlens_q]
    // TQ inputs:     [query, key_cache, value_cache, block_tables, seq_lens, cu_seqlens_q,
    //                 key_scale_cache, value_scale_cache, key_zero_cache, v_centroids]
    out.set_data(allocator::malloc(out.nbytes()));
    const array* ks = use_turboquant_ ? &inputs[6] : nullptr;
    const array* vs = use_turboquant_ ? &inputs[7] : nullptr;
    const array* kz = use_turboquant_ ? &inputs[8] : nullptr;
    const array* vc = use_turboquant_ ? &inputs[9] : nullptr;
    dispatch_paged_attention_v2_online(
        out,
        inputs[0],               // query
        inputs[1], inputs[2],    // key_cache, value_cache
        num_kv_heads_, scale_, softcap_,
        inputs[3], inputs[4], inputs[5],  // block_tables, seq_lens, cu_seqlens_q
        block_size_, max_seq_len_, sliding_window_,
        stream(),
        ks, vs, kz, vc, use_turboquant_, k_bits_, v_bits_);
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
        && rhs->v_bits_ == v_bits_;
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
    int v_bits = 3) {
  int k_bits = use_turboquant ? get_bits(quant_type) : 8;
  auto prim = std::make_shared<PagedAttentionPrimitive>(
      default_stream(Device::gpu),
      num_kv_heads, scale, softcap,
      block_size, max_seq_len, sliding_window,
      use_turboquant, k_bits, v_bits);
  if (use_turboquant) {
    return array(
        query.shape(), query.dtype(), std::move(prim),
        {query, key_cache, value_cache, block_tables, seq_lens, cu_seqlens_q,
         *key_scale_cache, *value_scale_cache, *key_zero_cache, *v_centroids});
  }
  return array(
      query.shape(), query.dtype(), std::move(prim),
      {query, key_cache, value_cache, block_tables, seq_lens, cu_seqlens_q});
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

    auto& enc = get_command_encoder_compat(d, s);
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

    auto& enc = get_command_encoder_compat(d, s);
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
// Map heads_per_tg → NUM_THREADS. Each variant keeps the per-thread register
// footprint roughly constant (NUM_THREADS scaled inversely to G).
//   G=1 → NUM_THREADS=1024 (32 simdgroups, current sdpa_vector layout).
//   G=2 → NUM_THREADS=512  (16 simdgroups, 2× cross-head amortization).
// Returns 0 for an unsupported G so callers can validate.
static int mla_num_threads_for_g(int heads_per_tg) {
  switch (heads_per_tg) {
    case 1: return 1024;
    case 2: return 512;
    default: return 0;
  }
}

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

  // Shape sanity — these must match what the kernel template was instantiated for.
  if (kv_lora_rank != 512) {
    throw std::runtime_error(
        "MLA kernel: only kv_lora_rank=512 is instantiated; got " +
        std::to_string(kv_lora_rank));
  }
  if (qk_rope_head_dim != 64) {
    throw std::runtime_error(
        "MLA kernel: only qk_rope_head_dim=64 is instantiated; got " +
        std::to_string(qk_rope_head_dim));
  }
  if (block_size != 16 && block_size != 32) {
    throw std::runtime_error(
        "MLA kernel: only block_size in {16, 32} is instantiated; got " +
        std::to_string(block_size));
  }
  int num_threads = mla_num_threads_for_g(heads_per_tg);
  if (num_threads == 0) {
    throw std::runtime_error(
        "MLA kernel: heads_per_tg must be in {1, 2}; got " +
        std::to_string(heads_per_tg));
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
                      std::to_string(kv_lora_rank) + "_pe" +
                      std::to_string(qk_rope_head_dim) + "_bs" +
                      std::to_string(block_size) + "_g" +
                      std::to_string(heads_per_tg) + "_nt" +
                      std::to_string(num_threads) + "_nsl32_ps0";

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
  const int BN = num_threads / BD;
  size_t shmem =
      static_cast<size_t>((2 * heads_per_tg * BN + BD * BD) * sizeof(float));

  auto& enc = get_command_encoder_compat(d, s);
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
      MTL::Size::Make(num_threads, 1, 1));

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

void gdn_linear_attention_impl(
    nb::handle q_h, nb::handle k_h, nb::handle v_h,
    nb::handle g_h, nb::handle beta_h,
    nb::handle state_pool_h,
    nb::handle cu_seqlens_h, nb::handle slot_mapping_h,
    nb::handle y_h,
    int Hk, int Hv, int Dk, int Dv
) {
  auto& q           = *nb::inst_ptr<array>(q_h);
  auto& k           = *nb::inst_ptr<array>(k_h);
  auto& v           = *nb::inst_ptr<array>(v_h);
  auto& g           = *nb::inst_ptr<array>(g_h);
  auto& beta        = *nb::inst_ptr<array>(beta_h);
  auto& state_pool  = *nb::inst_ptr<array>(state_pool_h);
  auto& cu_seqlens  = *nb::inst_ptr<array>(cu_seqlens_h);
  auto& slot_mapping = *nb::inst_ptr<array>(slot_mapping_h);
  auto& y           = *nb::inst_ptr<array>(y_h);

  int num_requests = static_cast<int>(cu_seqlens.shape(0)) - 1;

  if (Dk > 256) {
    throw std::runtime_error(
        "GDN kernel supports Dk <= 256 (state[8] * 32 threads). "
        "Got Dk=" + std::to_string(Dk));
  }

  auto s = default_stream(Device::gpu);
  auto& d = metal::device(Device::gpu);

  auto dt = dtype_to_metal(q.dtype());
  std::string kname = "gdn_linear_attention_" + dt;
  auto* lib = d.get_library("gdn_kern");
  auto* kernel = d.get_kernel(kname, lib, kname, {});

  auto& enc = get_command_encoder_compat(d, s);
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
  enc.set_bytes(Hk, 10);
  enc.set_bytes(Hv, 11);
  enc.set_bytes(Dk, 12);
  enc.set_bytes(Dv, 13);

  // Grid: (Dv, 1, num_requests * Hv)  Threadgroup: (32, 1, 1)
  enc.dispatch_threadgroups(
      MTL::Size::Make(Dv, 1, num_requests * Hv),
      MTL::Size::Make(32, 1, 1));

  add_temporary_compat(enc, q, d, s);
  add_temporary_compat(enc, k, d, s);
  add_temporary_compat(enc, v, d, s);
  add_temporary_compat(enc, g, d, s);
  add_temporary_compat(enc, beta, d, s);
  add_temporary_compat(enc, state_pool, d, s);
  add_temporary_compat(enc, cu_seqlens, d, s);
  add_temporary_compat(enc, slot_mapping, d, s);
  add_temporary_compat(enc, y, d, s);
}
// ---------------------------------------------------------------------------
// nanobind module
// ---------------------------------------------------------------------------

NB_MODULE(_paged_ops, m) {
  m.attr("PARTITION_SIZE") = nb::int_(kPartitionSize);
  m.def("min_decode_grid", &min_decode_grid,
        "Decode-grid threshold (threadgroups) below which split-KV decode "
        "engages on this machine.");

  m.def("init_v2_library", &init_v2_library,
        nb::arg("v2_src"),
        "JIT-compile the v2 online-softmax Metal shader.");

  m.def("init_library_path", &init_library_path,
        nb::arg("name"), nb::arg("path"),
        "Load a precompiled .metallib from disk, cached under `name`.");

  m.def("init_gdn_library", &init_gdn_library,
        nb::arg("gdn_src"),
        "JIT-compile the GDN linear attention Metal shader.");

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
           int v_bits) {
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
              use_turboquant, quant_type, ks, vs, kz, vc, v_bits);
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
        "Paged attention primitive (read-only). Cache writes are handled "
        "by MLX-native scatter upstream.");

  m.def("gdn_linear_attention", &gdn_linear_attention_impl,
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("g"), nb::arg("beta"),
        nb::arg("state_pool"), nb::arg("cu_seqlens"),
        nb::arg("slot_mapping"), nb::arg("y"),
        nb::arg("Hk"), nb::arg("Hv"), nb::arg("Dk"), nb::arg("Dv"),
        "GDN linear attention with in-place paged state management.");

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
