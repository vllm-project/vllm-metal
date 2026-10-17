// SPDX-License-Identifier: Apache-2.0
// GDN state scatter and linear attention primitives.
#include "ops_internal.h"
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

array gdn_state_scatter_primitive_fn(
    const array& pool, const array& src, const array& dst_ids,
    bool zero, bool paged) {
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

struct GDNLinearAttentionPlan {
  std::string kernel_name;
  int grid_x;
  int grid_z;
  int threads_x;
};

static GDNLinearAttentionPlan make_gdn_linear_attention_plan(
    Dtype dtype, int num_requests, int Hv, int Dv) {
  auto dt = dtype_to_metal(dtype);
  return {"gdn_linear_attention_" + dt, Dv, num_requests * Hv, 32};
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

    auto plan = make_gdn_linear_attention_plan(
        q.dtype(), num_requests, Hv_, Dv_);
    auto* lib = d.get_library("gdn_kern");
    auto* kernel = d.get_kernel(
        plan.kernel_name, lib, plan.kernel_name, {});

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
        MTL::Size::Make(plan.grid_x, 1, plan.grid_z),
        MTL::Size::Make(plan.threads_x, 1, 1));
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

std::vector<array> gdn_linear_attention_primitive_fn(
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
