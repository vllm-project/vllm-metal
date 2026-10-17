// MLX nanobind bindings and binding-only helpers.
// SPDX-License-Identifier: Apache-2.0
// C++ nanobind bridge for paged attention Metal kernels.
//
// Dispatches the v2 paged-attention / TurboQuant / GDN / MLA kernels through
// MLX's own Metal command encoder, eliminating the PyTorch MPS bridge.
//
// Uses nb::handle + nb::inst_ptr<array>() to extract the C++ array from
// the Python mlx.core.array object, bypassing nanobind's cross-module
// RTTI matching which fails due to hidden symbol visibility in libmlx.

#include "metal_device.h"
#include "ops_internal.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
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
    caps["gqa_batch_context_lens"] = true;
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
           nb::handle out_h, const std::vector<int>& context_lens) {
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
          if (kernels::paged_reduce_threadgroup_bytes(num_partitions) >
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
              partition_size, context_lens);
          nb::inst_ptr<array>(out_h)->overwrite_descriptor(result);
        }, nb::arg("query"), nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("scale"), nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("block_size"), nb::arg("max_seq_len"), nb::arg("partition_size"),
        nb::arg("out"), nb::arg("gqa_context_lens") = std::vector<int>{},
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
           const std::optional<GqaDecodeLengthPlan>& gqa_length_plan,
           int nax_split_hint, int nax_prefill_split_hint) {
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
              0, gqa_context_lens, gqa_length_plan, nax_split_hint,
              nax_prefill_split_hint);
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
        nb::arg("nax_split_hint") = -1,
        nb::arg("nax_prefill_split_hint") = -1,
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
        "built once per forward; its immutable offsets are shared by the nodes. "
        "Omit both to retain legacy batch routing. "
        "nax_split_hint (M5 NAX prefill only) selects the split full-tile "
        "pipeline (1) or the original loop (0); -1 uses the batch-average "
        "rule. nax_prefill_split_hint optionally overrides it only when "
        "native dispatch separates the ordinary decode prefix from NAX "
        "prefill; -1 retains nax_split_hint. The override describes only "
        "the remaining prefill sequences. Outputs are identical either "
        "way; head_size 512 ignores both hints.");

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
