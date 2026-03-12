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
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"

namespace nb = nanobind;
using namespace mlx::core;

// ---------------------------------------------------------------------------
// Library caching
// ---------------------------------------------------------------------------

static std::string reshape_cache_source_;
static std::string paged_attention_source_;
static std::string v2_paged_attention_source_;

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
    default:
      throw std::runtime_error(
          "Unsupported dtype for paged attention kernel");
  }
}

// ---------------------------------------------------------------------------
// reshape_and_cache
// ---------------------------------------------------------------------------

void reshape_and_cache_impl(
    nb::handle key_h,
    nb::handle value_h,
    nb::handle key_cache_h,
    nb::handle value_cache_h,
    nb::handle slot_mapping_h
) {
  // Extract C++ arrays from Python handles
  auto& key         = *nb::inst_ptr<array>(key_h);
  auto& value       = *nb::inst_ptr<array>(value_h);
  auto& key_cache   = *nb::inst_ptr<array>(key_cache_h);
  auto& value_cache = *nb::inst_ptr<array>(value_cache_h);
  auto& slot_mapping = *nb::inst_ptr<array>(slot_mapping_h);

  auto s = default_stream(Device::gpu);
  auto& d = metal::device(Device::gpu);

  int num_tokens  = static_cast<int>(key.shape(0));
  int num_heads   = static_cast<int>(key.shape(1));
  int head_size   = static_cast<int>(key.shape(2));
  // Cache layout: [num_blocks, block_size, num_kv_heads, head_size]
  int block_size  = static_cast<int>(key_cache.shape(1));

  // Contiguous strides (arrays must be row-major after mx.eval)
  int32_t key_stride   = static_cast<int32_t>(num_heads * head_size);
  int32_t value_stride = static_cast<int32_t>(num_heads * head_size);
  int32_t num_heads_i  = static_cast<int32_t>(num_heads);
  int32_t head_size_i  = static_cast<int32_t>(head_size);
  int32_t block_size_i = static_cast<int32_t>(block_size);

  // Kernel name: same kv and cache dtype (no FP8)
  auto dt = dtype_to_metal(key.dtype());
  std::string kname =
      "reshape_and_cache_kv_" + dt + "_cache_" + dt;

  // Get library & specialise kernel with function constants
  auto* lib = d.get_library("paged_reshape_cache");
  bool use_fp8 = false;
  auto* kernel = d.get_kernel(
      kname, lib, kname,
      {{&use_fp8, MTL::DataType::DataTypeBool, NS::UInteger(10)}});

  // Dispatch on the current MLX command encoder
  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  // Buffer bindings (match reshape_and_cache.metal signature)
  enc.set_input_array(key, 0);
  enc.set_input_array(value, 1);
  enc.set_output_array(key_cache, 2);
  enc.set_output_array(value_cache, 3);
  enc.set_input_array(slot_mapping, 4);
  // 5, 6: k_scale / v_scale — unused (use_fp8_scales=false)
  enc.set_bytes(key_stride,   7);
  enc.set_bytes(value_stride, 8);
  enc.set_bytes(num_heads_i,  9);
  enc.set_bytes(head_size_i,  10);
  enc.set_bytes(block_size_i, 11);

  int tpg = std::min(512, num_heads * head_size);
  enc.dispatch_threadgroups(
      MTL::Size::Make(num_tokens, 1, 1),
      MTL::Size::Make(tpg, 1, 1));

  // Keep ALL referenced arrays alive until the command buffer completes
  d.add_temporary(key, s.index);
  d.add_temporary(value, s.index);
  d.add_temporary(key_cache, s.index);
  d.add_temporary(value_cache, s.index);
  d.add_temporary(slot_mapping, s.index);
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
  auto& enc = d.get_command_encoder(s.index);
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
  d.add_temporary(out, s.index);
  d.add_temporary(query, s.index);
  d.add_temporary(key_cache, s.index);
  d.add_temporary(value_cache, s.index);
  d.add_temporary(block_tables, s.index);
  d.add_temporary(seq_lens, s.index);
}

// ---------------------------------------------------------------------------
// paged_attention_v2_online — dispatches the online-softmax v2 kernel
// ---------------------------------------------------------------------------

void paged_attention_v2_online_impl(
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

  // Same kernel name format as v1 — the template instantiation is identical.
  auto dt = dtype_to_metal(query.dtype());
  std::string kname =
      "paged_attention_" + dt + "_cache_" + dt +
      "_hs" + std::to_string(head_size) +
      "_bs" + std::to_string(block_size) +
      "_nt256_nsl32_ps0";

  bool use_partitioning = false;
  bool use_alibi        = false;
  bool use_fp8          = false;
  bool use_sinks        = false;

  // Use the v2 library (online softmax kernel).
  // get_kernel(function_name, library, cache_key, constants)
  // Cache key must differ from v1 to avoid collision in MLX's kernel cache.
  auto* lib = d.get_library("paged_attention_v2_kern");
  auto* kernel = d.get_kernel(
      kname, lib, kname + "_v2",
      {{&use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)},
       {&use_alibi,        MTL::DataType::DataTypeBool, NS::UInteger(20)},
       {&use_fp8,          MTL::DataType::DataTypeBool, NS::UInteger(30)},
       {&use_sinks,        MTL::DataType::DataTypeBool, NS::UInteger(40)}});

  // Threadgroup shared memory for online softmax:
  // During KV loop: NUM_WARPS * BLOCK_SIZE floats (per-warp score buffer)
  // During merge: 2*NUM_WARPS floats (m, l) + NUM_WARPS * HEAD_SIZE floats (O)
  constexpr int NUM_THREADS    = 256;
  constexpr int NUM_SIMD_LANES = 32;
  constexpr int NUM_WARPS      = NUM_THREADS / NUM_SIMD_LANES;
  int warp_scores_bytes = NUM_WARPS * block_size
                          * static_cast<int>(sizeof(float));
  int merge_bytes = (2 * NUM_WARPS + NUM_WARPS * head_size)
                    * static_cast<int>(sizeof(float));
  size_t shmem = static_cast<size_t>(std::max(warp_scores_bytes, merge_bytes));

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(shmem, 0);

  // Buffer bindings — identical to v1.
  enc.set_output_array(out,         2);
  enc.set_input_array(query,        3);
  enc.set_input_array(key_cache,    4);
  enc.set_input_array(value_cache,  5);

  int32_t nkv = static_cast<int32_t>(num_kv_heads);
  enc.set_bytes(nkv,   8);
  enc.set_bytes(scale, 9);
  float softcapping = 1.0f;
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

  enc.dispatch_threadgroups(
      MTL::Size::Make(num_heads, num_seqs, 1),
      MTL::Size::Make(NUM_THREADS, 1, 1));

  d.add_temporary(out, s.index);
  d.add_temporary(query, s.index);
  d.add_temporary(key_cache, s.index);
  d.add_temporary(value_cache, s.index);
  d.add_temporary(block_tables, s.index);
  d.add_temporary(seq_lens, s.index);
}

// ---------------------------------------------------------------------------
// nanobind module
// ---------------------------------------------------------------------------

NB_MODULE(_paged_ops, m) {
  m.def("init_libraries", &init_libraries,
        nb::arg("reshape_src"), nb::arg("paged_attn_src"),
        "JIT-compile the vendored Metal shaders.");

  m.def("init_v2_library", &init_v2_library,
        nb::arg("v2_src"),
        "JIT-compile the v2 online-softmax Metal shader.");

  m.def("reshape_and_cache", &reshape_and_cache_impl,
        nb::arg("key"), nb::arg("value"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("slot_mapping"),
        "Write projected K/V into the paged cache.");

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
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        "Online-softmax paged attention (v2, decode-only).");
}
