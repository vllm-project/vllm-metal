// SPDX-License-Identifier: Apache-2.0
// C++ implementation of the MLX FFI shim for paged_ops.rs.
//
// Wraps MLX's C++ Metal API behind a stable C ABI so Rust (via PyO3) can
// dispatch Metal kernels without requiring C++ interop in the Rust toolchain.

#include "mlx_ffi.h"

#include <algorithm>
#include <string>

#include <nanobind/nanobind.h>
#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"

namespace nb = nanobind;
using namespace mlx::core;

// ---------------------------------------------------------------------------
// Cached shader sources (same lifetime as the process)
// ---------------------------------------------------------------------------

static std::string reshape_cache_source_;
static std::string paged_attention_source_;
static std::string v2_paged_attention_source_;

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
// Library init
// ---------------------------------------------------------------------------

extern "C" void mlx_ffi_init_libraries(
    const char* reshape_src,
    const char* paged_attn_src
) {
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

extern "C" void mlx_ffi_init_v2_library(const char* v2_src) {
    v2_paged_attention_source_ = v2_src;
    auto& d = metal::device(Device::gpu);
    d.get_library(
        "paged_attention_v2_kern",
        [&]() { return v2_paged_attention_source_; });
}

// ---------------------------------------------------------------------------
// Array extraction
// ---------------------------------------------------------------------------

extern "C" MlxArrayHandle mlx_ffi_extract_array(void* py_object) {
    nb::handle h(reinterpret_cast<PyObject*>(py_object));
    return static_cast<MlxArrayHandle>(nb::inst_ptr<array>(h));
}

// ---------------------------------------------------------------------------
// reshape_and_cache
// ---------------------------------------------------------------------------

extern "C" void mlx_ffi_reshape_and_cache(
    MlxArrayHandle key_h,
    MlxArrayHandle value_h,
    MlxArrayHandle key_cache_h,
    MlxArrayHandle value_cache_h,
    MlxArrayHandle slot_mapping_h
) {
    auto& key         = *static_cast<array*>(key_h);
    auto& value       = *static_cast<array*>(value_h);
    auto& key_cache   = *static_cast<array*>(key_cache_h);
    auto& value_cache = *static_cast<array*>(value_cache_h);
    auto& slot_mapping = *static_cast<array*>(slot_mapping_h);

    auto s = default_stream(Device::gpu);
    auto& d = metal::device(Device::gpu);

    int num_tokens = static_cast<int>(key.shape(0));
    int num_heads  = static_cast<int>(key.shape(1));
    int head_size  = static_cast<int>(key.shape(2));
    int block_size = static_cast<int>(key_cache.shape(1));

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

    auto& enc = d.get_command_encoder(s.index);
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

    d.add_temporary(key, s.index);
    d.add_temporary(value, s.index);
    d.add_temporary(key_cache, s.index);
    d.add_temporary(value_cache, s.index);
    d.add_temporary(slot_mapping, s.index);
}

// ---------------------------------------------------------------------------
// paged_attention_v1
// ---------------------------------------------------------------------------

extern "C" void mlx_ffi_paged_attention_v1(
    MlxArrayHandle out_h,
    MlxArrayHandle query_h,
    MlxArrayHandle key_cache_h,
    MlxArrayHandle value_cache_h,
    int32_t num_kv_heads,
    float scale,
    MlxArrayHandle block_tables_h,
    MlxArrayHandle seq_lens_h,
    int32_t block_size,
    int32_t max_seq_len
) {
    auto& out          = *static_cast<array*>(out_h);
    auto& query        = *static_cast<array*>(query_h);
    auto& key_cache    = *static_cast<array*>(key_cache_h);
    auto& value_cache  = *static_cast<array*>(value_cache_h);
    auto& block_tables = *static_cast<array*>(block_tables_h);
    auto& seq_lens     = *static_cast<array*>(seq_lens_h);

    auto s = default_stream(Device::gpu);
    auto& d = metal::device(Device::gpu);

    int num_seqs   = static_cast<int>(query.shape(0));
    int num_heads  = static_cast<int>(query.shape(1));
    int head_size  = static_cast<int>(query.shape(2));
    int max_blocks = static_cast<int>(block_tables.shape(1));

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

    auto* lib = d.get_library("paged_attention_kern");
    auto* kernel = d.get_kernel(
        kname, lib, kname,
        {{&use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)},
         {&use_alibi,        MTL::DataType::DataTypeBool, NS::UInteger(20)},
         {&use_fp8,          MTL::DataType::DataTypeBool, NS::UInteger(30)},
         {&use_sinks,        MTL::DataType::DataTypeBool, NS::UInteger(40)}});

    constexpr int NUM_THREADS    = 256;
    constexpr int NUM_SIMD_LANES = 32;
    int padded_ctx = ((max_seq_len + block_size - 1) / block_size) * block_size;
    int logits_bytes  = padded_ctx * static_cast<int>(sizeof(float));
    int outputs_bytes = (NUM_THREADS / NUM_SIMD_LANES / 2)
                        * head_size * static_cast<int>(sizeof(float));
    size_t shmem = static_cast<size_t>(std::max(logits_bytes, outputs_bytes));

    auto& enc = d.get_command_encoder(s.index);
    enc.set_compute_pipeline_state(kernel);
    enc.set_threadgroup_memory_length(shmem, 0);

    enc.set_output_array(out,         2);
    enc.set_input_array(query,        3);
    enc.set_input_array(key_cache,    4);
    enc.set_input_array(value_cache,  5);

    int32_t nkv = num_kv_heads;
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
// paged_attention_v2_online
// ---------------------------------------------------------------------------

extern "C" void mlx_ffi_paged_attention_v2_online(
    MlxArrayHandle out_h,
    MlxArrayHandle query_h,
    MlxArrayHandle key_cache_h,
    MlxArrayHandle value_cache_h,
    int32_t num_kv_heads,
    float scale,
    MlxArrayHandle block_tables_h,
    MlxArrayHandle seq_lens_h,
    int32_t block_size,
    int32_t max_seq_len
) {
    auto& out          = *static_cast<array*>(out_h);
    auto& query        = *static_cast<array*>(query_h);
    auto& key_cache    = *static_cast<array*>(key_cache_h);
    auto& value_cache  = *static_cast<array*>(value_cache_h);
    auto& block_tables = *static_cast<array*>(block_tables_h);
    auto& seq_lens     = *static_cast<array*>(seq_lens_h);

    auto s = default_stream(Device::gpu);
    auto& d = metal::device(Device::gpu);

    int num_seqs   = static_cast<int>(query.shape(0));
    int num_heads  = static_cast<int>(query.shape(1));
    int head_size  = static_cast<int>(query.shape(2));
    int max_blocks = static_cast<int>(block_tables.shape(1));

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

    auto* lib = d.get_library("paged_attention_v2_kern");
    auto* kernel = d.get_kernel(
        kname, lib, kname + "_v2",
        {{&use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)},
         {&use_alibi,        MTL::DataType::DataTypeBool, NS::UInteger(20)},
         {&use_fp8,          MTL::DataType::DataTypeBool, NS::UInteger(30)},
         {&use_sinks,        MTL::DataType::DataTypeBool, NS::UInteger(40)}});

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

    enc.set_output_array(out,         2);
    enc.set_input_array(query,        3);
    enc.set_input_array(key_cache,    4);
    enc.set_input_array(value_cache,  5);

    int32_t nkv = num_kv_heads;
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
