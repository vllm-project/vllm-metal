// SPDX-License-Identifier: Apache-2.0
// C++ nanobind bridge for paged attention Metal kernels.
//
// Dispatches reshape_and_cache and paged_attention_v1 through MLX's own
// Metal command encoder, eliminating the PyTorch MPS bridge.
//
// Uses nb::handle + nb::inst_ptr<array>() to extract the C++ array from
// the Python mlx.core.array object, bypassing nanobind's cross-module
// RTTI matching which fails due to hidden symbol visibility in libmlx.

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>

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

static std::string reshape_cache_source_;
static std::string paged_attention_source_;
static std::string v2_paged_attention_source_;
constexpr int kPartitionSize = VLLM_METAL_PARTITION_SIZE;

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

void init_libraries(
    const std::string& reshape_src,
    const std::string& paged_attn_src) {
  reshape_cache_source_ = reshape_src;
  paged_attention_source_ = paged_attn_src;

  auto& d = metal::device(Device::gpu);
  d.get_library(
      "paged_reshape_cache",
      [&]() { return reshape_cache_source_; });
  d.get_library(
      "paged_attention_kern",
      [&]() { return paged_attention_source_; });
}

void init_v2_library(const std::string& v2_src) {
  v2_paged_attention_source_ = v2_src;
  auto& d = metal::device(Device::gpu);
  d.get_library(
      "paged_attention_v2_kern",
      [&]() { return v2_paged_attention_source_; });
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
// reshape_and_cache — dispatch helper + eager binding
// ---------------------------------------------------------------------------

// When called from a primitive's eval_gpu, from_primitive should be true
// to skip ALL add_temporary calls.  add_temporary removes buffer pointers
// from the encoder's input/output tracking, defeating fence-based
// synchronisation across command buffer boundaries.  Inside a primitive,
// MLX's evaluator already manages array lifetimes via the completion
// handler.  In the eager path, add_temporary is needed to keep
// Python-owned arrays alive until the command buffer completes.
static void dispatch_reshape_and_cache(
    const array& key, const array& value,
    array& key_cache, array& value_cache,
    const array& slot_mapping, Stream s,
    bool from_primitive = false) {
  auto& d = metal::device(s.device);

  int num_tokens  = static_cast<int>(key.shape(0));
  int num_heads   = static_cast<int>(key.shape(1));
  int head_size   = static_cast<int>(key.shape(2));
  int block_size  = static_cast<int>(key_cache.shape(1));

  int32_t key_stride   = static_cast<int32_t>(num_heads * head_size);
  int32_t value_stride = static_cast<int32_t>(num_heads * head_size);
  int32_t num_heads_i  = static_cast<int32_t>(num_heads);
  int32_t head_size_i  = static_cast<int32_t>(head_size);
  int32_t block_size_i = static_cast<int32_t>(block_size);

  auto dt = dtype_to_metal(key.dtype());
  std::string kname = "reshape_and_cache_kv_" + dt + "_cache_" + dt;

  auto* lib = d.get_library("paged_reshape_cache");
  bool use_fp8 = false;
  auto* kernel = d.get_kernel(
      kname, lib, kname,
      {{&use_fp8, MTL::DataType::DataTypeBool, NS::UInteger(10)}});

  auto& enc = get_command_encoder_compat(d, s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(key, 0);
  enc.set_input_array(value, 1);
  enc.set_output_array(key_cache, 2);
  enc.set_output_array(value_cache, 3);
  enc.set_input_array(slot_mapping, 4);
  enc.set_bytes(key_stride,   7);
  enc.set_bytes(value_stride, 8);
  enc.set_bytes(num_heads_i,  9);
  enc.set_bytes(head_size_i,  10);
  enc.set_bytes(block_size_i, 11);

  int tpg = std::min(512, num_heads * head_size);
  enc.dispatch_threadgroups(
      MTL::Size::Make(num_tokens, 1, 1),
      MTL::Size::Make(tpg, 1, 1));

  if (!from_primitive) {
    add_temporary_compat(enc, key, d, s);
    add_temporary_compat(enc, value, d, s);
    add_temporary_compat(enc, key_cache, d, s);
    add_temporary_compat(enc, value_cache, d, s);
    add_temporary_compat(enc, slot_mapping, d, s);
  }
}

void reshape_and_cache_impl(
    nb::handle key_h, nb::handle value_h,
    nb::handle key_cache_h, nb::handle value_cache_h,
    nb::handle slot_mapping_h) {
  dispatch_reshape_and_cache(
      *nb::inst_ptr<array>(key_h), *nb::inst_ptr<array>(value_h),
      *nb::inst_ptr<array>(key_cache_h), *nb::inst_ptr<array>(value_cache_h),
      *nb::inst_ptr<array>(slot_mapping_h), default_stream(Device::gpu));
}

// ---------------------------------------------------------------------------
// paged_attention_v1
// ---------------------------------------------------------------------------

void paged_attention_v1_impl(
    nb::handle out_h,
    nb::handle query_h,
    nb::handle key_cache_h,
    nb::handle value_cache_h,
    int num_kv_heads,
    float scale,
    nb::handle block_tables_h,
    nb::handle seq_lens_h,
    int block_size,
    int max_seq_len
) {
  auto& out          = *nb::inst_ptr<array>(out_h);
  auto& query        = *nb::inst_ptr<array>(query_h);
  auto& key_cache    = *nb::inst_ptr<array>(key_cache_h);
  auto& value_cache  = *nb::inst_ptr<array>(value_cache_h);
  auto& block_tables = *nb::inst_ptr<array>(block_tables_h);
  auto& seq_lens     = *nb::inst_ptr<array>(seq_lens_h);

  auto s = default_stream(Device::gpu);
  auto& d = metal::device(Device::gpu);

  int num_seqs   = static_cast<int>(query.shape(0));
  int num_heads  = static_cast<int>(query.shape(1));
  int head_size  = static_cast<int>(query.shape(2));
  int max_blocks = static_cast<int>(block_tables.shape(1));

  // Kernel name
  auto dt = dtype_to_metal(query.dtype());
  std::string kname =
      "paged_attention_" + dt + "_cache_" + dt +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_nt256_nsl32_ps0";

  // Function constants
  bool use_partitioning = false;
  bool use_alibi        = false;
  bool use_fp8          = false;
  bool use_sinks        = false;

  auto* lib = d.get_library("paged_attention_kern");
  auto* kernel = d.get_kernel(
      kname, lib, kname,
      {{&use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)},
       {&use_alibi,        MTL::DataType::DataTypeBool, NS::UInteger(20)},
       {&use_fp8,          MTL::DataType::DataTypeBool, NS::UInteger(30)},
       {&use_sinks,        MTL::DataType::DataTypeBool, NS::UInteger(40)}});

  // Threadgroup shared memory
  constexpr int NUM_THREADS    = 256;
  constexpr int NUM_SIMD_LANES = 32;
  int padded_ctx = ((max_seq_len + block_size - 1) / block_size) * block_size;
  int logits_bytes  = padded_ctx * static_cast<int>(sizeof(float));
  int outputs_bytes = (NUM_THREADS / NUM_SIMD_LANES / 2)
                      * head_size * static_cast<int>(sizeof(float));
  size_t shmem = static_cast<size_t>(std::max(logits_bytes, outputs_bytes));

  // Dispatch
  auto& enc = get_command_encoder_compat(d, s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(shmem, 0);

  // Buffer bindings (match paged_attention.metal signature)
  // 0: exp_sums   — skipped (v1, no partitioning)
  // 1: max_logits — skipped
  enc.set_output_array(out,         2);
  enc.set_input_array(query,        3);
  enc.set_input_array(key_cache,    4);
  enc.set_input_array(value_cache,  5);
  // 6: k_scale    — skipped (no FP8)
  // 7: v_scale    — skipped

  int32_t nkv = static_cast<int32_t>(num_kv_heads);
  enc.set_bytes(nkv,   8);
  enc.set_bytes(scale, 9);
  float softcapping = 1.0f;
  enc.set_bytes(softcapping, 10);

  enc.set_input_array(block_tables, 11);
  enc.set_input_array(seq_lens,     12);

  int32_t max_blocks_i = static_cast<int32_t>(max_blocks);
  enc.set_bytes(max_blocks_i, 13);
  // 14: alibi_slopes — skipped

  // Strides (contiguous row-major)
  // Cache layout: [num_blocks, block_size, num_kv_heads, head_size]
  int32_t q_stride        = static_cast<int32_t>(num_heads * head_size);
  int32_t kv_block_stride = static_cast<int32_t>(key_cache.strides()[0]);
  int32_t kv_head_stride  = static_cast<int32_t>(key_cache.strides()[2]);
  enc.set_bytes(q_stride,        15);
  enc.set_bytes(kv_block_stride, 16);
  enc.set_bytes(kv_head_stride,  17);

  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, num_seqs, 1),
      MTL::Size::Make(NUM_THREADS, 1, 1));

  // Keep ALL referenced arrays alive until the command buffer completes
  add_temporary_compat(enc, out, d, s);
  add_temporary_compat(enc, query, d, s);
  add_temporary_compat(enc, key_cache, d, s);
  add_temporary_compat(enc, value_cache, d, s);
  add_temporary_compat(enc, block_tables, d, s);
  add_temporary_compat(enc, seq_lens, d, s);
}

// ---------------------------------------------------------------------------
// paged_attention_v2_online — dispatch helper + eager wrappers
// ---------------------------------------------------------------------------

static void dispatch_paged_attention_v2_online(
    array& out, const array& query,
    const array& key_cache, const array& value_cache,
    int num_kv_heads, float scale, float softcap,
    const array& block_tables, const array& seq_lens,
    const array& cu_seqlens_q,
    int block_size, int max_seq_len, int sliding_window, Stream s,
    bool from_primitive = false,
    // TurboQuant (optional, all nullptr when disabled):
    const array* key_scale_cache = nullptr,
    const array* value_scale_cache = nullptr,
    const array* key_zero_cache = nullptr,
    const array* v_centroids = nullptr,
    bool use_turboquant = false,
    int k_bits = 8,
    int v_bits = 3) {
  auto& d = metal::device(s.device);

  int total_q_tokens = static_cast<int>(query.shape(0));
  int num_heads  = static_cast<int>(query.shape(1));
  int head_size  = static_cast<int>(query.shape(2));
  int max_blocks = static_cast<int>(block_tables.shape(1));
  int num_seqs   = static_cast<int>(cu_seqlens_q.shape(0)) - 1;

  // 3-part kernel name: paged_attention_<T>_cache_<K>_<V>_hs...
  auto dt        = dtype_to_metal(query.dtype());
  auto k_cache_dt = dtype_to_metal(key_cache.dtype());
  auto v_cache_dt = dtype_to_metal(value_cache.dtype());
  std::string kname =
      "paged_attention_" + dt + "_cache_" + k_cache_dt + "_" + v_cache_dt +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_nt256_nsl32_ps0";

  bool use_partitioning  = false;
  bool use_alibi         = false;
  bool use_fp8           = false;
  bool use_sinks         = false;
  bool use_tq_fc         = use_turboquant;
  int  k_bits_i          = k_bits;
  int  v_bits_i          = v_bits;

  // The hash name is MLX's cache key. It MUST encode every function-constant
  // value that varies across calls — otherwise the first compilation wins and
  // subsequent calls with different constants silently reuse the wrong kernel.
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
  // TurboQuant V path is now register-only (inverse FWHT runs entirely on
  // the simdgroup register file via simd_shuffle_xor).  The previous
  // per-warp NUM_WARPS * head_size FWHT workspace in shared_mem has been
  // removed from the kernel, so no TQ bonus is needed here.
  (void)use_turboquant;
  size_t shmem = static_cast<size_t>(std::max(warp_scores_bytes, merge_bytes));

  auto& enc = get_command_encoder_compat(d, s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(shmem, 0);

  enc.set_output_array(out, 2);
  enc.set_input_array(query,        3);
  enc.set_input_array(key_cache,    4);
  enc.set_input_array(value_cache,  5);

  int32_t nkv = static_cast<int32_t>(num_kv_heads);
  enc.set_bytes(nkv,   8);
  enc.set_bytes(scale, 9);
  float softcapping = softcap;
  enc.set_bytes(softcapping, 10);

  enc.set_input_array(block_tables, 11);
  enc.set_input_array(seq_lens,     12);

  int32_t max_blocks_i = static_cast<int32_t>(max_blocks);
  enc.set_bytes(max_blocks_i, 13);

  int32_t q_stride        = static_cast<int32_t>(num_heads * head_size);
  int32_t kv_block_stride = static_cast<int32_t>(key_cache.strides()[0]);
  int32_t kv_head_stride  = static_cast<int32_t>(key_cache.strides()[2]);
  enc.set_bytes(q_stride,        15);
  enc.set_bytes(kv_block_stride, 16);
  enc.set_bytes(kv_head_stride,  17);

  enc.set_input_array(cu_seqlens_q, 19);
  int32_t num_seqs_i = static_cast<int32_t>(num_seqs);
  enc.set_bytes(num_seqs_i, 20);
  int32_t sliding_window_i = static_cast<int32_t>(sliding_window);
  enc.set_bytes(sliding_window_i, 21);

  // TurboQuant buffers (slots 22-27, only bound when use_turboquant)
  if (use_turboquant) {
    enc.set_input_array(*key_scale_cache,   22);
    enc.set_input_array(*value_scale_cache, 23);
    // v_block_stride = value_cache.strides()[0], v_head_stride = value_cache.strides()[2]
    int32_t v_block_stride_i = static_cast<int32_t>(value_cache.strides()[0]);
    int32_t v_head_stride_i  = static_cast<int32_t>(value_cache.strides()[2]);
    enc.set_bytes(v_block_stride_i, 24);
    enc.set_bytes(v_head_stride_i,  25);
    enc.set_input_array(*key_zero_cache, 26);
    enc.set_input_array(*v_centroids, 27);
  }

  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, total_q_tokens, 1),
      MTL::Size::Make(NUM_THREADS, 1, 1));

  if (!from_primitive) {
    add_temporary_compat(enc, out, d, s);
    add_temporary_compat(enc, query, d, s);
    add_temporary_compat(enc, key_cache, d, s);
    add_temporary_compat(enc, value_cache, d, s);
    add_temporary_compat(enc, block_tables, d, s);
    add_temporary_compat(enc, seq_lens, d, s);
    add_temporary_compat(enc, cu_seqlens_q, d, s);
    if (use_turboquant) {
      add_temporary_compat(enc, *key_scale_cache, d, s);
      add_temporary_compat(enc, *value_scale_cache, d, s);
      add_temporary_compat(enc, *key_zero_cache, d, s);
      add_temporary_compat(enc, *v_centroids, d, s);
    }
  }
}

// Eager wrapper — keeps the old handle-based API for metal_unified_attention.
// Non-partitioned case delegates to the dispatch helper above;
// partitioned case is handled inline (same as original code on main).
void paged_attention_v2_online_impl_common(
    nb::handle out_h,
    nb::handle query_h,
    nb::handle key_cache_h,
    nb::handle value_cache_h,
    int num_kv_heads,
    float scale,
    float softcap,
    nb::handle block_tables_h,
    nb::handle seq_lens_h,
    nb::handle cu_seqlens_q_h,
    int block_size,
    int max_seq_len,
    int sliding_window,
    array* exp_sums,
    array* max_logits,
    array* tmp_out,
    array* sinks
) {
  auto& out          = *nb::inst_ptr<array>(out_h);
  auto& query        = *nb::inst_ptr<array>(query_h);
  auto& key_cache    = *nb::inst_ptr<array>(key_cache_h);
  auto& value_cache  = *nb::inst_ptr<array>(value_cache_h);
  auto& block_tables = *nb::inst_ptr<array>(block_tables_h);
  auto& seq_lens     = *nb::inst_ptr<array>(seq_lens_h);
  auto& cu_seqlens_q = *nb::inst_ptr<array>(cu_seqlens_q_h);

  // Non-partitioned case: delegate to the shared dispatch helper
  bool needs_partitioning =
      exp_sums != nullptr && max_logits != nullptr && tmp_out != nullptr;
  if (!needs_partitioning) {
    dispatch_paged_attention_v2_online(
        out, query, key_cache, value_cache,
        num_kv_heads, scale, softcap,
        block_tables, seq_lens, cu_seqlens_q,
        block_size, max_seq_len, sliding_window,
        default_stream(Device::gpu));
    return;
  }

  // Partitioned path (unchanged from main)
  auto s = default_stream(Device::gpu);
  auto& d = metal::device(Device::gpu);

  int total_q_tokens = static_cast<int>(query.shape(0));
  int num_heads  = static_cast<int>(query.shape(1));
  int head_size  = static_cast<int>(query.shape(2));
  int max_blocks = static_cast<int>(block_tables.shape(1));
  int num_seqs   = static_cast<int>(cu_seqlens_q.shape(0)) - 1;
  int max_num_partitions =
      std::max(1, (max_seq_len + kPartitionSize - 1) / kPartitionSize);
  bool use_partitioning =
      kPartitionSize % block_size == 0 && max_num_partitions > 1;

  auto dt        = dtype_to_metal(query.dtype());
  auto k_cache_dt = dtype_to_metal(key_cache.dtype());
  auto v_cache_dt = dtype_to_metal(value_cache.dtype());
  std::string kname =
      "paged_attention_" + dt + "_cache_" + k_cache_dt + "_" + v_cache_dt +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_nt256_nsl32_ps" +
      std::to_string(use_partitioning ? kPartitionSize : 0);

  bool use_alibi        = false;
  bool use_fp8          = false;
  bool use_sinks        = sinks != nullptr;
  bool use_tq_fc        = false;
  int  k_bits_i         = 8;

  // Same hash-name discipline as v2: encode every varying function constant
  // (use_sinks here) so MLX doesn't reuse a stale specialization.
  std::string hash_name = kname + "_v2"
      + "_sinks" + (use_sinks ? "1" : "0");

  auto* lib = d.get_library("paged_attention_v2_kern");
  auto* kernel = d.get_kernel(
      kname, lib, hash_name,
      {{&use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)},
       {&use_alibi,        MTL::DataType::DataTypeBool, NS::UInteger(20)},
       {&use_fp8,          MTL::DataType::DataTypeBool, NS::UInteger(30)},
       {&use_sinks,        MTL::DataType::DataTypeBool, NS::UInteger(40)},
       {&use_tq_fc,        MTL::DataType::DataTypeBool, NS::UInteger(50)},
       {&k_bits_i,         MTL::DataType::DataTypeInt,  NS::UInteger(60)}});

  constexpr int NUM_THREADS    = 256;
  constexpr int NUM_SIMD_LANES = 32;
  constexpr int NUM_WARPS      = NUM_THREADS / NUM_SIMD_LANES;
  int warp_scores_bytes = NUM_WARPS * block_size
                          * static_cast<int>(sizeof(float));
  int merge_bytes = (2 * NUM_WARPS + NUM_WARPS * head_size)
                    * static_cast<int>(sizeof(float));
  size_t shmem = static_cast<size_t>(std::max(warp_scores_bytes, merge_bytes));

  auto& enc = get_command_encoder_compat(d, s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(shmem, 0);

  if (use_partitioning) {
    enc.set_output_array(*exp_sums, 0);
    enc.set_output_array(*max_logits, 1);
    enc.set_output_array(*tmp_out, 2);
  } else {
    enc.set_output_array(out, 2);
  }
  enc.set_input_array(query,        3);
  enc.set_input_array(key_cache,    4);
  enc.set_input_array(value_cache,  5);

  int32_t nkv = static_cast<int32_t>(num_kv_heads);
  enc.set_bytes(nkv,   8);
  enc.set_bytes(scale, 9);
  float softcapping = softcap;
  enc.set_bytes(softcapping, 10);

  enc.set_input_array(block_tables, 11);
  enc.set_input_array(seq_lens,     12);

  int32_t max_blocks_i = static_cast<int32_t>(max_blocks);
  enc.set_bytes(max_blocks_i, 13);

  int32_t q_stride        = static_cast<int32_t>(num_heads * head_size);
  int32_t kv_block_stride = static_cast<int32_t>(key_cache.strides()[0]);
  int32_t kv_head_stride  = static_cast<int32_t>(key_cache.strides()[2]);
  enc.set_bytes(q_stride,        15);
  enc.set_bytes(kv_block_stride, 16);
  enc.set_bytes(kv_head_stride,  17);
  if (use_sinks) {
    enc.set_input_array(*sinks, 18);
  }

  enc.set_input_array(cu_seqlens_q, 19);
  int32_t num_seqs_i = static_cast<int32_t>(num_seqs);
  enc.set_bytes(num_seqs_i, 20);
  int32_t sliding_window_i = static_cast<int32_t>(sliding_window);
  enc.set_bytes(sliding_window_i, 21);

  const int32_t grid_z =
      static_cast<int32_t>(use_partitioning ? max_num_partitions : 1);
  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, total_q_tokens, grid_z),
      MTL::Size::Make(NUM_THREADS, 1, 1));

  if (use_partitioning) {
    std::string reduce_kname =
        "paged_attention_v2_reduce_" + dt +
        "_hs" + std::to_string(head_size) +
        "_nt256_nsl32_ps" + std::to_string(kPartitionSize);
    // The reduce kernel now references function_constant(50) (use_turboquant)
    // to gate the deferred-FWHT path.  Reaching this dispatch from the
    // non-TQ wrapper means use_turboquant is always false here; encode it
    // in the hash name anyway so a future TQ-partitioned dispatch picks up
    // its own specialisation instead of reusing the non-TQ one.
    bool reduce_use_tq = false;
    auto* reduce_kernel = d.get_kernel(
        reduce_kname,
        lib,
        reduce_kname + "_v2_reduce_tq" + (reduce_use_tq ? "1" : "0"),
        {{&use_sinks,       MTL::DataType::DataTypeBool, NS::UInteger(40)},
         {&reduce_use_tq,   MTL::DataType::DataTypeBool, NS::UInteger(50)}});
    size_t reduce_shmem =
        static_cast<size_t>(2 * max_num_partitions * sizeof(float));
    enc.set_compute_pipeline_state(reduce_kernel);
    enc.set_threadgroup_memory_length(reduce_shmem, 0);

    enc.set_output_array(out,         0);
    enc.set_input_array(*exp_sums,    1);
    enc.set_input_array(*max_logits,  2);
    enc.set_input_array(*tmp_out,     3);
    enc.set_input_array(seq_lens,     4);
    int32_t max_num_partitions_i = static_cast<int32_t>(max_num_partitions);
    enc.set_bytes(max_num_partitions_i, 5);
    if (use_sinks) {
      enc.set_input_array(*sinks, 6);
    }
    enc.set_input_array(cu_seqlens_q, 7);
    enc.set_bytes(num_seqs_i, 8);
    enc.dispatch_threadgroups(
        MTL::Size::Make(num_heads, total_q_tokens, 1),
        MTL::Size::Make(NUM_THREADS, 1, 1));
  }

  add_temporary_compat(enc, out, d, s);
  add_temporary_compat(enc, query, d, s);
  add_temporary_compat(enc, key_cache, d, s);
  add_temporary_compat(enc, value_cache, d, s);
  add_temporary_compat(enc, block_tables, d, s);
  add_temporary_compat(enc, seq_lens, d, s);
  add_temporary_compat(enc, cu_seqlens_q, d, s);
  if (use_partitioning) {
    add_temporary_compat(enc, *exp_sums, d, s);
    add_temporary_compat(enc, *max_logits, d, s);
    add_temporary_compat(enc, *tmp_out, d, s);
  }
  if (use_sinks) {
    add_temporary_compat(enc, *sinks, d, s);
  }
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
        /*from_primitive=*/true,
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

void paged_attention_v2_online_impl(
    nb::handle out_h,
    nb::handle query_h,
    nb::handle key_cache_h,
    nb::handle value_cache_h,
    int num_kv_heads,
    float scale,
    float softcap,
    nb::handle block_tables_h,
    nb::handle seq_lens_h,
    nb::handle cu_seqlens_q_h,
    int block_size,
    int max_seq_len,
    int sliding_window
) {
  paged_attention_v2_online_impl_common(
      out_h,
      query_h,
      key_cache_h,
      value_cache_h,
      num_kv_heads,
      scale,
      softcap,
      block_tables_h,
      seq_lens_h,
      cu_seqlens_q_h,
      block_size,
      max_seq_len,
      sliding_window,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
}

void paged_attention_v2_online_partitioned_impl(
    nb::handle out_h,
    nb::handle query_h,
    nb::handle key_cache_h,
    nb::handle value_cache_h,
    int num_kv_heads,
    float scale,
    float softcap,
    nb::handle block_tables_h,
    nb::handle seq_lens_h,
    nb::handle cu_seqlens_q_h,
    int block_size,
    int max_seq_len,
    int sliding_window,
    nb::handle exp_sums_h,
    nb::handle max_logits_h,
    nb::handle tmp_out_h
) {
  auto& exp_sums = *nb::inst_ptr<array>(exp_sums_h);
  auto& max_logits = *nb::inst_ptr<array>(max_logits_h);
  auto& tmp_out = *nb::inst_ptr<array>(tmp_out_h);
  paged_attention_v2_online_impl_common(
      out_h,
      query_h,
      key_cache_h,
      value_cache_h,
      num_kv_heads,
      scale,
      softcap,
      block_tables_h,
      seq_lens_h,
      cu_seqlens_q_h,
      block_size,
      max_seq_len,
      sliding_window,
      &exp_sums,
      &max_logits,
      &tmp_out,
      nullptr);
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
// GDN linear attention — in-place paged state
// ---------------------------------------------------------------------------

static std::string gdn_source_;

void init_gdn_library(const std::string& src) {
  gdn_source_ = src;
  auto& d = metal::device(Device::gpu);
  d.get_library("gdn_kern", [&]() { return gdn_source_; });
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
// Encoder varlen attention — dense non-causal varlen attention.
//
// Wrapped as a real MLX UnaryPrimitive so the kernel dispatch carries graph
// provenance through the lazy graph (mirroring the PagedAttentionPrimitive
// class above).  The primitive is read-only: Q/K/V/cu_seqlens go in, a fresh
// output array comes out.  Validation lives in two layers: the Python
// helper at vllm_metal.metal.encoder_varlen_attention rejects bad shapes /
// dtypes / cu_seqlens before launch; eval_gpu enforces row-contiguity on
// every input array (the only layer that can — the Python mx.array surface
// does not expose flags().row_contiguous).
// ---------------------------------------------------------------------------

static std::string encoder_varlen_source_;
static bool encoder_varlen_library_initialized_ = false;

void init_encoder_varlen_library(const std::string& src) {
  // The MLX library cache (Device::library_map_) is name-keyed: a second
  // get_library("encoder_varlen_kern", ...) call hits the cache and never
  // re-reads the source.  The Python layer's _encoder_varlen_loaded flag
  // gates Python callers, but this binding is part of the public C++ API
  // and a direct caller bypasses that gate.  Guard explicitly here so the
  // actual behavior is visible at the call site.
  if (encoder_varlen_library_initialized_) {
    if (src != encoder_varlen_source_) {
      throw std::runtime_error(
          "init_encoder_varlen_library called twice with different source; "
          "MLX library cache is name-keyed and will not pick up the new "
          "source. Restart the interpreter to recompile.");
    }
    return;
  }
  encoder_varlen_source_ = src;
  auto& d = metal::device(Device::gpu);
  d.get_library(
      "encoder_varlen_kern",
      [&]() { return encoder_varlen_source_; });
  // Set the flag only after get_library returns successfully so a Metal
  // compile failure leaves the slot retryable rather than wedged.
  encoder_varlen_library_initialized_ = true;
}

// Hand-mirrored on the .metal side at
// kernels_v2/encoder_varlen_attention.metal.  Field order is part of the
// ABI; do not reorder without updating both files in lockstep.  We do not
// stuff this into a project-local header because needs_rebuild() in
// build.py only stats paged_ops.cpp / build.py / constants.py — header-only
// edits would silently fail to trigger an .so rebuild.
struct EncoderVarlenParams {
  int   num_q_heads;
  int   num_kv_heads;
  int   total_tokens;
  int   num_segments;   // == cu_seqlens.shape[0] - 1; binary-search bound.
  // Reserved ABI: launch hint only; not used to bound indexing in the
  // current kernel.  Intentionally excluded from is_equivalent (see below).
  // If a future kernel reads this in a way that affects output, dispatch
  // shape, or scratch sizing, it MUST be added to is_equivalent — otherwise
  // two calls with different max_seqlen values will alias in the lazy graph.
  int   max_seqlen;
  float softmax_scale;
};

// ABI tripwires for the .metal-side struct mirror.  Per-field offsetof is
// stronger than size-only: a same-size reorder (e.g. swapping two int fields)
// would slip past sizeof but break field decoding in-kernel.  needs_rebuild()
// in build.py only stats the .cpp, so a struct edit here is the only way to
// force an .so rebuild and surface this check at compile time.
static_assert(offsetof(EncoderVarlenParams, num_q_heads)   ==  0,
              "EncoderVarlenParams ABI drift: num_q_heads offset changed");
static_assert(offsetof(EncoderVarlenParams, num_kv_heads)  ==  4,
              "EncoderVarlenParams ABI drift: num_kv_heads offset changed");
static_assert(offsetof(EncoderVarlenParams, total_tokens)  ==  8,
              "EncoderVarlenParams ABI drift: total_tokens offset changed");
static_assert(offsetof(EncoderVarlenParams, num_segments)  == 12,
              "EncoderVarlenParams ABI drift: num_segments offset changed");
static_assert(offsetof(EncoderVarlenParams, max_seqlen)    == 16,
              "EncoderVarlenParams ABI drift: max_seqlen offset changed");
static_assert(offsetof(EncoderVarlenParams, softmax_scale) == 20,
              "EncoderVarlenParams ABI drift: softmax_scale offset changed");
static_assert(sizeof(EncoderVarlenParams) == 24,
              "EncoderVarlenParams ABI drift: struct size changed; "
              "update kernels_v2/encoder_varlen_attention.metal in lockstep");

class EncoderVarlenAttentionPrimitive : public UnaryPrimitive {
 public:
  EncoderVarlenAttentionPrimitive(
      Stream stream,
      int num_kv_heads,
      float softmax_scale,
      int dtype_tag,
      int head_dim,
      int max_seqlen)
      : UnaryPrimitive(stream),
        num_kv_heads_(num_kv_heads),
        softmax_scale_(softmax_scale),
        dtype_tag_(dtype_tag),
        head_dim_(head_dim),
        max_seqlen_(max_seqlen) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error(
        "EncoderVarlenAttentionPrimitive only supports GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    const auto& q          = inputs[0];
    const auto& k          = inputs[1];
    const auto& v          = inputs[2];
    const auto& cu_seqlens = inputs[3];

    // Defense-in-depth: the Python helper does not (and cannot) check
    // contiguity, so this is the only layer that catches non-row-contiguous
    // inputs.  Run before set_data so a rejected call does not leak a
    // freshly-allocated output buffer.
    if (!q.flags().row_contiguous) {
      throw std::invalid_argument(
          "encoder_varlen_attention: q must be row-contiguous "
          "(wrap upstream in mx.contiguous(...) before calling).");
    }
    if (!k.flags().row_contiguous) {
      throw std::invalid_argument(
          "encoder_varlen_attention: k must be row-contiguous.");
    }
    if (!v.flags().row_contiguous) {
      throw std::invalid_argument(
          "encoder_varlen_attention: v must be row-contiguous.");
    }
    if (!cu_seqlens.flags().row_contiguous) {
      throw std::invalid_argument(
          "encoder_varlen_attention: cu_seqlens must be row-contiguous.");
    }

    // Defense-in-depth: the Python helper validates shape / dtype /
    // num_kv_heads, but a caller that drives the underscore-prefixed
    // _encoder_varlen_attention_primitive binding directly bypasses that.
    // Re-check the structural invariants the kernel relies on so a
    // bypass surfaces as a clean throw rather than silent K/V misindexing
    // in-kernel.  Value-level checks (cu_seqlens monotonicity /
    // final-token-count) are intentionally not re-checked here — those
    // belong in the helper, and the binding's docstring marks it internal.
    // cu_seqlens dtype / ndim / shape and q/k/v dtype-match ARE re-checked
    // below: all of them are load-fatal failure modes through the kernel's
    // typed bindings (int* for cu_seqlens; a single template parameter T
    // for q/k/v, selected from q.dtype() only — a mismatched k/v would
    // mis-stride the load against the resource binding).
    if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3) {
      throw std::invalid_argument(
          "encoder_varlen_attention: q, k, v must each be 3-D "
          "[total_tokens, num_heads, head_dim].");
    }
    if (k.shape(0) != q.shape(0) || v.shape(0) != q.shape(0) ||
        k.shape(2) != q.shape(2) || v.shape(2) != q.shape(2) ||
        k.shape(1) != q.shape(1) || v.shape(1) != q.shape(1)) {
      throw std::invalid_argument(
          "encoder_varlen_attention: q, k, v must share the same shape "
          "(v1 requires num_q_heads == num_kv_heads).");
    }
    if (k.dtype() != q.dtype() || v.dtype() != q.dtype()) {
      throw std::invalid_argument(
          "encoder_varlen_attention: q, k, v must share dtype "
          "(kernel template binds all three to the same T, selected from "
          "q.dtype() only; a mismatched k/v dtype mis-strides the load "
          "against the resource binding and may read past the K/V buffer).");
    }
    if (static_cast<int>(k.shape(1)) != num_kv_heads_) {
      throw std::invalid_argument(
          "encoder_varlen_attention: num_kv_heads parameter does not "
          "match k.shape(1); kernel would silently misindex K/V.");
    }
    if (static_cast<int>(q.shape(2)) != head_dim_) {
      throw std::invalid_argument(
          "encoder_varlen_attention: head_dim parameter does not "
          "match q.shape(2).");
    }
    // cu_seqlens dtype / shape / rank: the kernel binding is
    // `device const int*`, and find_segment dereferences cu_seqlens[seg + 1]
    // up through cu_seqlens[num_segments].  A non-int32 dtype mis-strides
    // the load (4-byte stride read as 8 / 2 bytes); ndim != 1 silently
    // mis-strides the load; shape(0) < 2 → num_segments <= 0 and the very
    // first dereference is out-of-bounds.  The Python helper enforces all
    // three on the raw and ValidatedSeqlens paths; these guards catch
    // direct callers of the underscore binding that bypass those layers.
    if (cu_seqlens.dtype() != int32) {
      throw std::invalid_argument(
          "encoder_varlen_attention: cu_seqlens must have dtype int32 "
          "(kernel binding is device const int*; any other element width "
          "would mis-stride the load).");
    }
    if (cu_seqlens.ndim() != 1) {
      throw std::invalid_argument(
          "encoder_varlen_attention: cu_seqlens must be 1-D "
          "(kernel binding is device const int*; any other rank "
          "would mis-stride the load).");
    }
    if (cu_seqlens.shape(0) < 2) {
      throw std::invalid_argument(
          "encoder_varlen_attention: cu_seqlens must have at least 2 "
          "entries (num_segments + 1); the kernel reads cu_seqlens[seg + 1] "
          "in find_segment and would dereference out of bounds.");
    }

    // Allocate output via the MLX allocator (mirroring
    // PagedAttentionPrimitive::eval_gpu's set_data + allocator::malloc
    // pattern).  This is the array whose descriptor the
    // _encoder_varlen_attention_primitive binding's overwrite_descriptor
    // call publishes back to Python — the cross-module RTTI bypass path
    // documented at the top of this file.
    out.set_data(allocator::malloc(out.nbytes()));

    const int total_tokens = static_cast<int>(q.shape(0));
    const int num_q_heads  = static_cast<int>(q.shape(1));
    const int num_segments = static_cast<int>(cu_seqlens.shape(0)) - 1;

    EncoderVarlenParams params{};
    params.num_q_heads   = num_q_heads;
    params.num_kv_heads  = num_kv_heads_;
    params.total_tokens  = total_tokens;
    params.num_segments  = num_segments;
    params.max_seqlen    = max_seqlen_;
    params.softmax_scale = softmax_scale_;

    const char* dtype_tag = nullptr;
    switch (dtype_tag_) {
      case 0: dtype_tag = "f16";  break;
      case 1: dtype_tag = "bf16"; break;
      case 2: dtype_tag = "f32";  break;
      default:
        throw std::runtime_error(
            "Unknown dtype_tag in EncoderVarlenAttentionPrimitive");
    }
    const std::string kname =
        "encoder_varlen_attention_" + std::string(dtype_tag) + "_" +
        std::to_string(head_dim_);

    auto s = stream();
    auto& d = metal::device(s.device);
    auto* lib = d.get_library("encoder_varlen_kern");
    auto* kernel = d.get_kernel(kname, lib, kname, {});

    auto& enc = get_command_encoder_compat(d, s);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(cu_seqlens, 3);
    enc.set_output_array(out, 4);
    enc.set_bytes(params, 5);

    // Grid: (total_tokens, num_q_heads, 1).  One threadgroup per
    // (query_token, q_head) output vector.  Threadgroup width 32 == one
    // SIMD-group on Apple Silicon, so simd_max / simd_sum reduce across the
    // whole threadgroup in a single instruction with no inter-SIMD barriers.
    enc.dispatch_threadgroups(
        MTL::Size::Make(total_tokens, num_q_heads, 1),
        MTL::Size::Make(32, 1, 1));

    // No add_temporary calls: inside a primitive, MLX's evaluator manages
    // array lifetimes via the completion handler, and add_temporary would
    // strip buffer pointers from the encoder's input/output tracking and
    // silently defeat the fence for downstream primitives.
  }

  const char* name() const override { return "EncoderVarlenAttention"; }

  bool is_equivalent(const Primitive& other) const override {
    auto* rhs = dynamic_cast<const EncoderVarlenAttentionPrimitive*>(&other);
    return rhs
        && rhs->num_kv_heads_  == num_kv_heads_
        && rhs->softmax_scale_ == softmax_scale_
        && rhs->dtype_tag_     == dtype_tag_
        && rhs->head_dim_      == head_dim_;
    // max_seqlen_ is intentionally excluded: it is a launch hint only,
    // never used to bound cu_seqlens indexing in-kernel.
  }

 private:
  int   num_kv_heads_;
  float softmax_scale_;
  int   dtype_tag_;
  int   head_dim_;
  int   max_seqlen_;
};

static int encoder_varlen_dtype_to_tag(Dtype dt) {
  switch (dt) {
    case float16:  return 0;
    case bfloat16: return 1;
    case float32:  return 2;
    default:
      throw std::runtime_error(
          "Unsupported dtype for encoder_varlen_attention "
          "(supported: float16, bfloat16, float32)");
  }
}

static array encoder_varlen_attention_primitive_fn(
    const array& q,
    const array& k,
    const array& v,
    const array& cu_seqlens,
    int num_kv_heads,
    float softmax_scale,
    int max_seqlen) {
  // Rank guard before q.shape(2): a direct underscore-binding caller with
  // ndim < 3 would otherwise hit MLX's low-level out-of-range from shape()
  // here, before eval_gpu's "must each be 3-D" check could fire.  Same
  // message as eval_gpu so callers see one consistent error regardless of
  // path.  k/v rank stays in eval_gpu since they aren't accessed here.
  if (q.ndim() != 3) {
    throw std::invalid_argument(
        "encoder_varlen_attention: q, k, v must each be 3-D "
        "[total_tokens, num_heads, head_dim].");
  }
  const int dtype_tag = encoder_varlen_dtype_to_tag(q.dtype());
  const int head_dim  = static_cast<int>(q.shape(2));
  auto prim = std::make_shared<EncoderVarlenAttentionPrimitive>(
      default_stream(Device::gpu),
      num_kv_heads,
      softmax_scale,
      dtype_tag,
      head_dim,
      max_seqlen);
  return array(
      q.shape(),
      q.dtype(),
      std::move(prim),
      {q, k, v, cu_seqlens});
}

// ---------------------------------------------------------------------------
// nanobind module
// ---------------------------------------------------------------------------

NB_MODULE(_paged_ops, m) {
  m.attr("PARTITION_SIZE") = nb::int_(kPartitionSize);

  m.def("init_libraries", &init_libraries,
        nb::arg("reshape_src"), nb::arg("paged_attn_src"),
        "JIT-compile the vendored Metal shaders.");

  m.def("init_v2_library", &init_v2_library,
        nb::arg("v2_src"),
        "JIT-compile the v2 online-softmax Metal shader.");

  m.def("init_gdn_library", &init_gdn_library,
        nb::arg("gdn_src"),
        "JIT-compile the GDN linear attention Metal shader.");

  m.def("init_encoder_varlen_library", &init_encoder_varlen_library,
        nb::arg("src"),
        "JIT-compile the dense non-causal varlen encoder attention shader.");

  m.def("reshape_and_cache", &reshape_and_cache_impl,
        nb::arg("key"), nb::arg("value"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("slot_mapping"),
        "Write projected K/V into the paged cache.");

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

  m.def("paged_attention_v1", &paged_attention_v1_impl,
        nb::arg("out"), nb::arg("query"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("num_kv_heads"), nb::arg("scale"),
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        "Zero-copy paged attention (v1, no partitioning).");

  m.def("paged_attention_v2_online", &paged_attention_v2_online_impl,
        nb::arg("out"), nb::arg("query"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("num_kv_heads"), nb::arg("scale"),
        nb::arg("softcap"),
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("cu_seqlens_q"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        nb::arg("sliding_window"),
        "Online-softmax varlen paged attention (v2, unified prefill+decode).");

  m.def("paged_attention_v2_online_partitioned",
        &paged_attention_v2_online_partitioned_impl,
        nb::arg("out"), nb::arg("query"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("num_kv_heads"), nb::arg("scale"),
        nb::arg("softcap"),
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("cu_seqlens_q"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        nb::arg("sliding_window"),
        nb::arg("exp_sums"), nb::arg("max_logits"), nb::arg("tmp_out"),
        "Online-softmax varlen paged attention (v2) with caller-provided "
        "partition scratch buffers.");

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

  // Dense non-causal varlen encoder attention.  Caller supplies a
  // placeholder mx.array(0); the binding publishes the result via
  // overwrite_descriptor (mirroring the paged_attention_primitive binding
  // above) so the returned array carries the kernel dependency through
  // MLX's lazy graph.  Validation of shapes / dtypes / cu_seqlens lives
  // in the Python helper (vllm_metal.metal.encoder_varlen_attention);
  // contiguity is enforced inside eval_gpu.
  m.def("_encoder_varlen_attention_primitive",
        [](nb::handle q_h,
           nb::handle k_h,
           nb::handle v_h,
           nb::handle cu_seqlens_h,
           int num_kv_heads,
           float softmax_scale,
           int max_seqlen,
           nb::handle out_h) {
          auto result = encoder_varlen_attention_primitive_fn(
              *nb::inst_ptr<array>(q_h),
              *nb::inst_ptr<array>(k_h),
              *nb::inst_ptr<array>(v_h),
              *nb::inst_ptr<array>(cu_seqlens_h),
              num_kv_heads,
              softmax_scale,
              max_seqlen);
          nb::inst_ptr<array>(out_h)->overwrite_descriptor(result);
        },
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("v"),
        nb::arg("cu_seqlens"),
        nb::arg("num_kv_heads"),
        nb::arg("softmax_scale"),
        nb::arg("max_seqlen"),
        nb::arg("out"),
        "Internal: call via vllm_metal.metal.encoder_varlen_attention. "
        "Direct calls bypass cu_seqlens monotonicity / final-token-count "
        "validation; contiguity, basic Q/K/V shape / num_kv_heads / "
        "head_dim invariants, q/k/v dtype-match, and cu_seqlens dtype / "
        "ndim / shape are enforced inside eval_gpu.");
}
