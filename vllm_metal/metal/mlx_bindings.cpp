// MLX nanobind bindings and binding-only helpers.
// SPDX-License-Identifier: Apache-2.0
// C++ nanobind bridge for paged attention Metal kernels.
//
// Dispatches the v2 paged-attention / TurboQuant / GDN / MLA kernels through
// MLX's own Metal command encoder, eliminating the PyTorch MPS bridge.

#include "metal_device.h"
#include "ops_internal.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
void register_mlx_patch(nb::module_& m);
using hardware::override_detected_gpu_core_count_for_test;

static bool set_paged_dispatch_diagnostics(bool enabled) {
  // Call only while evaluation is idle. Clear stale observations on both
  // transitions so a disabled observer cannot report a previous request.
  const bool previous =
      g_dispatch_diagnostics_enabled.exchange(enabled, std::memory_order_relaxed);
  g_last_dispatch.store(PagedDispatch::None, std::memory_order_relaxed);
  g_last_gqa_partition.store(0, std::memory_order_relaxed);
  g_last_gqa_requests.store(0, std::memory_order_relaxed);
  g_last_nax_split.store(-1, std::memory_order_relaxed);
  g_last_nax_heads_per_tg.store(0, std::memory_order_relaxed);
  return previous;
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

static int min_decode_grid() {
  return gpu_core_count() * kernels::kDecodeGroupsPerCore;
}

static bool nax_hardware_supported() {
  auto* device = metal::device(Device::gpu).mtl_device();
  return hardware::nax_supported(
      device->supportsFamily(static_cast<MTL::GPUFamily>(hardware::kApple10)));
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
        "Test-only. A positive count replaces hardware detection; zero simulates "
        "failure and -1 restores detection. Production must not call this.");
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
        "Immutable CPU length statistics and partition offsets shared across layers. "
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
        "the dispatch bound; compact scratch uses actual lengths. Dispatch separately "
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
                const auto reducer = kernels::paged_reduce_kernel_name(
                    dtype, layout.head_size, part);
                bool no = false;
                if (d.get_kernel(reducer, lib, reducer + "_gqa_check",
                    {{&no, MTL::DataType::DataTypeBool, NS::UInteger(40)},
                     {&no, MTL::DataType::DataTypeBool, NS::UInteger(50)},
                     {&no, MTL::DataType::DataTypeBool, NS::UInteger(130)}})
                    == nullptr) return false;
                const auto name = gqa_decode_kernel_name(
                    dtype, layout.head_size, layout.block_size, part);
                if (d.get_kernel(name, lib, name + "_compact0",
                    {{&no, MTL::DataType::DataTypeBool, NS::UInteger(130)}})
                    == nullptr) return false;
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
    caps["gqa_length_plan"] = true;
    caps["gqa_mixed_decode_plan"] = true;
    caps["nax_split_hint"] = true;
    caps["nax_prefill_split_hint"] = true;
    return caps;
  }, "Capabilities of the public paged attention interface. Private test "
     "kernels do not imply an enabled production route.");

  m.def("tile_config",
        [](int head_size) -> nb::object {
          auto cfg = kernels::select_tile_config(head_size);
          if (!cfg) return nb::none();
          return nb::make_tuple(cfg->BQ, cfg->TILE_KV);
        },
        nb::arg("head_size"),
        "The tiled-prefill kernel's (BQ, TILE_KV) tile config for a head "
        "size, or None when that head size has no tiled instantiation.");

  m.def("init_library_path", &init_library_path,
        nb::arg("name"), nb::arg("path"),
        "Load a precompiled .metallib from disk, cached under `name`.");

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

  m.def("tq_encode", &tq_encode_primitive_fn,
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

  m.def("reshape_and_cache", &reshape_and_cache_primitive_fn,
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

  m.def("gdn_state_scatter", &gdn_state_scatter_primitive_fn,
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
        [](const array& q, const array& k, const array& v,
           float scale, const array& tables, const array& lengths,
           int block_size, int max_seq_len, int partition_size,
           const std::optional<GqaDecodeLengthPlan>& gqa_length_plan) {
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
          if (kernels::paged_reduce_threadgroup_bytes(num_partitions) >
              metal::device(Device::gpu).mtl_device()->maxThreadgroupMemoryLength()) {
            throw std::invalid_argument(
                "GQA test partition exceeds the device threadgroup memory limit");
          }
          auto cu = arange(q.shape(0) + 1, int32);
          return paged_attention_primitive_fn(
              q, k, v, k.shape(2), scale, 0.f, tables, lengths, cu,
              block_size, max_seq_len, -1, false, "", nullptr, nullptr,
              nullptr, nullptr, 3, 1, nullptr, nullptr, q.shape(0), q.shape(0),
              max_seq_len, false,
              partition_size, gqa_length_plan);
        }, nb::arg("query"), nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("scale"), nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("block_size"), nb::arg("max_seq_len"), nb::arg("partition_size"),
        nb::arg("gqa_length_plan") = nb::none(),
        "Private test-only one-token-per-request GQA dispatch with an explicit partition. "
        "Page IDs and sequence lengths must describe valid cache contents. "
        "Does not change the default selector or require GPU core detection.");

  // TurboQuant counterpart of the entry above. Default routing never sends
  // TurboQuant caches to the GQA producer; this entry is its only caller.
  m.def("_gqa_turboquant_paged_attention_for_test",
        [](const array& q, const array& k, const array& v,
           const array& key_scale_cache, const array& value_scale_cache,
           const array& key_zero_cache, const array& v_centroids,
           const std::string& quant_type, int v_bits,
           float scale, const array& tables, const array& lengths,
           int block_size, int max_seq_len, int partition_size,
           const std::optional<GqaDecodeLengthPlan>& gqa_length_plan) {
          if (std::find(kGqaPartitionSizes.begin(), kGqaPartitionSizes.end(),
                        partition_size) == kGqaPartitionSizes.end()) {
            throw std::invalid_argument("GQA test partition must be 256/512");
          }
          if (q.ndim() != 3 || q.shape(0) < 1 || k.ndim() != 4 || v.ndim() != 4 ||
              (q.dtype() != float16 && q.dtype() != bfloat16) ||
              (k.dtype() != int8 && k.dtype() != uint8) || v.dtype() != uint8 ||
              k.shape(0) != v.shape(0) || k.shape(1) != v.shape(1) ||
              k.shape(2) != v.shape(2) ||
              !gqa_decode_geometry_supported(q.shape(1), k.shape(2),
                                             q.shape(2), block_size) ||
              k.shape(1) % block_size != 0 ||
              key_scale_cache.ndim() != 4 ||
              key_scale_cache.shape() != value_scale_cache.shape() ||
              key_scale_cache.shape() != key_zero_cache.shape() ||
              key_scale_cache.strides() != value_scale_cache.strides() ||
              key_scale_cache.shape(3) * 32 != q.shape(2) ||
              tables.ndim() != 2 || tables.shape(0) != q.shape(0) ||
              tables.dtype() != int32 || lengths.ndim() != 1 ||
              lengths.shape(0) != q.shape(0) || lengths.dtype() != int32 ||
              max_seq_len <= 0 ||
              max_seq_len > static_cast<int64_t>(tables.shape(1)) * block_size) {
            throw std::invalid_argument(
                "GQA TurboQuant test entry requires supported one-token decode rows");
          }
          const int64_t num_partitions =
              (static_cast<int64_t>(max_seq_len) + partition_size - 1) /
              partition_size;
          if (kernels::paged_reduce_threadgroup_bytes(num_partitions) >
              metal::device(Device::gpu).mtl_device()->maxThreadgroupMemoryLength()) {
            throw std::invalid_argument(
                "GQA test partition exceeds the device threadgroup memory limit");
          }
          auto cu = arange(q.shape(0) + 1, int32);
          return paged_attention_primitive_fn(
              q, k, v, k.shape(2), scale, 0.f, tables, lengths, cu,
              block_size, max_seq_len, -1, true, quant_type, &key_scale_cache,
              &value_scale_cache, &key_zero_cache, &v_centroids, v_bits, 1,
              nullptr, nullptr, q.shape(0), q.shape(0), max_seq_len, false,
              partition_size, gqa_length_plan);
        }, nb::arg("query"), nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("key_scale_cache"), nb::arg("value_scale_cache"),
        nb::arg("key_zero_cache"), nb::arg("v_centroids"),
        nb::arg("quant_type"), nb::arg("v_bits"),
        nb::arg("scale"), nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("block_size"), nb::arg("max_seq_len"), nb::arg("partition_size"),
        nb::arg("gqa_length_plan") = nb::none(),
        "Private test-only TurboQuant GQA dispatch with an explicit partition. "
        "Same contract as _gqa_paged_attention_for_test, with packed TurboQuant "
        "K/V and their scale, zero-point and centroid arrays.");

  // Paged attention primitive (read-only): dispatches paged_attention_v2_online.
  // Cache writes are handled by MLX-native scatter upstream.
  m.def("paged_attention_primitive",
        [](const array& query,
           const array& key_cache, const array& value_cache,
           int num_kv_heads, float scale, float softcap,
           const array& block_tables, const array& seq_lens,
           const array& cu_seqlens_q,
           int block_size, int max_seq_len, int sliding_window,
           const array* key_scale_cache,
           const array* value_scale_cache,
           const array* key_zero_cache,
           const array* v_centroids,
           bool use_turboquant,
           const std::string& quant_type,
           int v_bits,
           int window_seqlen_q,
           const array* sinks,
           const array* mm_prefix_ranges,
           int num_decode_requests,
           int num_decode_tokens,
           int max_decode_context_len, bool gqa_disabled,
           const std::optional<GqaDecodeLengthPlan>& gqa_length_plan,
           int nax_split_hint, int nax_prefill_split_hint) {
          return paged_attention_primitive_fn(
              query, key_cache, value_cache,
              num_kv_heads, scale, softcap,
              block_tables, seq_lens, cu_seqlens_q,
              block_size, max_seq_len, sliding_window,
              use_turboquant, quant_type, key_scale_cache, value_scale_cache,
              key_zero_cache, v_centroids, v_bits,
              window_seqlen_q, sinks, mm_prefix_ranges, num_decode_requests,
              num_decode_tokens, max_decode_context_len, gqa_disabled,
              0, gqa_length_plan, nax_split_hint,
              nax_prefill_split_hint);
        },
        nb::arg("query"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("num_kv_heads"), nb::arg("scale"), nb::arg("softcap"),
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("cu_seqlens_q"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        nb::arg("sliding_window"),
        nb::arg("key_scale_cache").none() = nb::none(),
        nb::arg("value_scale_cache").none() = nb::none(),
        nb::arg("key_zero_cache").none() = nb::none(),
        nb::arg("v_centroids").none() = nb::none(),
        nb::arg("use_turboquant") = false,
        nb::arg("quant_type") = "",
        nb::arg("v_bits") = 3,
        nb::arg("window_seqlen_q") = 1,
        nb::arg("sinks").none() = nb::none(),
        nb::arg("mm_prefix_ranges").none() = nb::none(),
        nb::arg("num_decode_requests") = -1,
        nb::arg("num_decode_tokens") = 0,
        nb::arg("max_decode_context_len") = 0,
        nb::arg("gqa_disabled") = false,
        nb::arg("gqa_length_plan") = nb::none(),
        nb::arg("nax_split_hint") = -1,
        nb::arg("nax_prefill_split_hint") = -1,
        // Window validation may evaluate metadata with a Python-backed reader.
        nb::call_guard<nb::gil_scoped_release>(),
        "Return a lazy paged attention array (read-only). Cache writes are handled "
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
        "gqa_length_plan is per-request KV lengths for ordinary decode rows, "
        "built once per forward with gqa_decode_length_plan; its immutable "
        "offsets are shared by the nodes. Omit it to retain legacy batch routing. "
        "nax_split_hint (M5 NAX prefill only) selects the split full-tile "
        "pipeline (1) or the original loop (0); -1 uses the batch-average "
        "rule for head_size below 512. nax_prefill_split_hint optionally "
        "overrides it only when native dispatch separates the ordinary "
        "decode prefix from NAX prefill; -1 retains nax_split_hint. The "
        "override describes only the remaining prefill sequences. "
        "Outputs are identical either way. For head_size 512, an effective "
        "hint of 1 selects two 256-column output slices; 0 or -1 keeps the "
        "original loop.");

  m.def(
      "last_nax_heads_per_threadgroup",
      []() { return g_last_nax_heads_per_tg.load(std::memory_order_relaxed); },
      "Heads per threadgroup of the most recent recorded NAX launch, after "
      "pipeline-capacity fallback: 1 upstream, 3 or 6 grouped, 0 for another "
      "family or when diagnostics are cleared/disabled. Process-wide test "
      "diagnostic; inspect only after evaluation completes, not a request trace.");

  m.def(
      "last_nax_split_pipeline",
      []() { return g_last_nax_split.load(std::memory_order_relaxed); },
      "NAX prefill pipeline used by the most recent recorded NAX launch: 1 for "
      "the split full-tile pipeline, 0 for the original loop, -1 when the last "
      "recorded dispatch was not NAX or diagnostics are disabled.");

  m.def(
      "last_paged_dispatch",
      []() {
        static constexpr const char* names[] = {
            "", "gqa_decode", "per_token_ps0", "per_token_ps512", "window_ps0",
            "window_ps512", "nax_prefill", "tiled_prefill", "mixed_prefill_decode",
            "mixed_nax_prefill_decode", "nax_gqa6_prefill"};
        static_assert(std::size(names) == static_cast<size_t>(PagedDispatch::Count),
                      "PagedDispatch and its diagnostic names must stay in sync");
        return names[static_cast<size_t>(
            g_last_dispatch.load(std::memory_order_relaxed))];
      },
      "Dispatch family chosen by the most recent recorded paged_attention_primitive "
      "eval (\"gqa_decode\", \"per_token_ps0\", \"per_token_ps512\", "
      "\"window_ps0\", \"window_ps512\", \"nax_prefill\", "
      "\"tiled_prefill\", \"mixed_prefill_decode\", \"mixed_nax_prefill_decode\", "
      "\"nax_gqa6_prefill\"). "
      "Diagnostic surface for routing tests; empty "
      "when diagnostics are disabled, cleared or before the first recorded eval.");

  m.def("gdn_linear_attention", &gdn_linear_attention_primitive_fn,
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("g"), nb::arg("beta"),
        nb::arg("state_pool"), nb::arg("cu_seqlens"),
        nb::arg("slot_mapping"),
        nb::arg("Hk"), nb::arg("Hv"), nb::arg("Dk"), nb::arg("Dv"),
        "Lazy GDN recurrence returning (y, updated_state_pool). Only the "
        "state output aliases its input. Callers must use both returned "
        "handles so output reads and state reuse depend on the native writes.");

  m.def("mla_paged_attention_primitive", &mla_paged_attention_primitive_fn,
        nb::arg("q_nope"), nb::arg("q_pe"),
        nb::arg("latent_cache"),
        nb::arg("block_tables"), nb::arg("context_lens"),
        nb::arg("cu_seqlens_q"),
        nb::arg("block_size"), nb::arg("scale"),
        nb::arg("heads_per_tg") = 1,
        "Return a lazy MLA attention array.");

}
