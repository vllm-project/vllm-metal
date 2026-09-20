// SPDX-License-Identifier: Apache-2.0
// Shared-buffer as_strided for the shapes, strides, and offsets supplied by
// vLLM's cache layout. A view describes existing storage; it allocates no cache
// data. Strides and offsets are measured in elements of the backing's dtype.
//
// Example:
//   import mlx.core as mx
//   from vllm_metal.metal import get_ops
//   ops = get_ops()
//   backing = mx.arange(64, dtype=mx.float32).reshape(4, 16)
//   public_view = mx.as_strided(backing, (4, 8), (16, 1), 0)
//   state = ops.as_strided(backing, (4, 8), (16, 1), 0)
// Both views select the first eight values of each sixteen-element row:
// state[r, c] addresses backing storage at r * 16 + c. For this simple layout,
// backing[:, :8] also works. The cache adapter transfers upstream descriptors
// directly so it does not need a slicing recipe for every cache layout.
//
// MLX 0.32.1: backing -> flatten(backing) -> AsStrided -> view
// This helper: backing -> CacheViewPrimitive -> view
// Flattening fails when the total element count exceeds INT32_MAX, even if
// every original dimension fits. The internal AsStrided primitive avoids that
// flatten, but the released wheel does not export its vtable for extensions.
// For example, a recorded Qwen3.5-0.8B shared cache uses 5.24 GiB. Its BF16
// backing shape (5046, 557056) has 2810904576 elements: both axes fit int32,
// but flattening the backing does not.
//
// This primitive shares the input buffer directly and retains its lazy
// dependency. It consumes valid cache-layout descriptors supplied by vLLM:
// row-contiguous backing, positive dimensions, and nonnegative strides.
// Callers are responsible for in-bounds addresses and non-overlapping writes.
//
// Upstream plan: contribute the no-flatten view path to MLX's public as_strided,
// with a regression test for backing larger than INT32_MAX elements. Preserve
// buffer aliasing and lazy dependencies. Once that fix ships in the MLX version
// we depend on, use mx.as_strided and remove this helper and its build entry.
// See mlx/ops.cpp:as_strided and mlx/backend/common/common.cpp:AsStrided::eval
// in https://github.com/ml-explore/mlx/tree/v0.32.1.

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace nb = nanobind;
using namespace mlx::core;

class CacheViewPrimitive : public Primitive {
 public:
  CacheViewPrimitive(Stream stream, Shape shape, Strides strides, size_t offset)
      : Primitive(stream), shape_(std::move(shape)),
        strides_(std::move(strides)), offset_(offset) {}
  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    alias(inputs[0], outputs[0]);
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    alias(inputs[0], outputs[0]);
  }
  const char* name() const override { return "CacheView"; }
  bool is_equivalent(const Primitive& other) const override {
    auto* rhs = dynamic_cast<const CacheViewPrimitive*>(&other);
    return rhs && shape_ == rhs->shape_ && strides_ == rhs->strides_
               && offset_ == rhs->offset_;
  }
 private:
  void alias(const array& input, array& output) {
    auto flags = output.flags();
    flags.row_contiguous = flags.col_contiguous = true;
    size_t row = 1, col = 1, span = 1;
    for (int i = 0; i < output.ndim(); ++i) {
      int j = output.ndim() - 1 - i;
      flags.row_contiguous &= output.shape(j) == 1 || strides_[j] == row;
      flags.col_contiguous &= output.shape(i) == 1 || strides_[i] == col;
      row *= output.shape(j);
      col *= output.shape(i);
      span += (output.shape(i) - 1) * strides_[i];
    }
    flags.contiguous = span == output.size();
    output.copy_shared_buffer(input, strides_, flags, span, offset_);
  }
  Shape shape_;
  Strides strides_;
  size_t offset_;
};

void register_mlx_patch(nb::module_& m) {
  m.def("as_strided", [](nb::handle buffer_h, const std::vector<int>& shape,
                          const std::vector<size_t>& strides, size_t offset) {
    const auto& buffer = *nb::inst_ptr<array>(buffer_h);
    if (shape.size() != strides.size()) {
      throw std::invalid_argument("as_strided: shape/stride rank mismatch");
    }
    for (int dim : shape) {
      if (dim <= 0) throw std::invalid_argument("as_strided: non-positive dimension");
    }
    // mx.as_strided flattens first, exceeding MLX's int32 dimension limit
    // for large arenas. Bind the existing backing without that flatten.
    Shape view_shape(shape.begin(), shape.end());
    Strides view_strides(strides.begin(), strides.end());
    auto result = array::make_arrays(
        {view_shape}, {buffer.dtype()},
        std::make_shared<CacheViewPrimitive>(
            default_stream(Device::gpu), view_shape, view_strides, offset),
        {buffer})[0];
    nb::object out = nb::module_::import_("mlx.core").attr("array")(0);
    nb::inst_ptr<array>(out)->overwrite_descriptor(result);
    return out;
  }, nb::arg(), nb::arg("shape"), nb::arg("strides"), nb::arg("offset") = 0,
     "Create a shared cache view without flattening or copying its backing. "
     "Shape is explicit; strides and offset are in elements. The backing must "
     "be row-contiguous, dimensions positive, and strides nonnegative. Callers "
     "supply valid upstream cache metadata and ensure in-bounds addresses and "
     "non-overlapping writes.");
}
