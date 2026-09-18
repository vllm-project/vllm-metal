// SPDX-License-Identifier: Apache-2.0
// Temporary MLX compatibility code for zero-copy views of large cache buffers.
//
// Upstream replacement: mlx.core.as_strided / mlx::core::as_strided should
// accept an already row-contiguous multidimensional backing without flattening
// or copying it. The shape, strides, offset, aliasing, and lazy input dependency
// must be preserved even when the backing has more than INT32_MAX elements.
// Once a released MLX supports that contract, use its public API and remove
// this file and its build entry. This describes the desired upstream change;
// it is not a new API that MLX has already promised.
//
// MLX 0.32.1's public helper flattens first, exceeding its int32 dimension limit
// for large byte buffers. Its internal AsStrided primitive avoids that flatten,
// but the released wheel does not export its vtable for external extensions.
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
  m.def("cache_view", [](nb::handle buffer_h, const std::vector<int>& shape,
                          const std::vector<size_t>& strides, size_t offset) {
    const auto& buffer = *nb::inst_ptr<array>(buffer_h);
    if (shape.size() != strides.size()) {
      throw std::invalid_argument("cache_view: shape/stride rank mismatch");
    }
    size_t end = offset;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] <= 0) throw std::invalid_argument("cache_view: non-positive dimension");
      end += (shape[i] - 1) * strides[i];
    }
    if (end >= buffer.size()) throw std::invalid_argument("cache_view: out of bounds");
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
  });
}
