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

#include <algorithm>
#include <limits>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace nb = nanobind;
using namespace mlx::core;

class CacheViewPrimitive : public Primitive {
 public:
  CacheViewPrimitive(
      Stream stream, Shape shape, Strides strides, int64_t offset, size_t span)
      : Primitive(stream), shape_(std::move(shape)),
        strides_(std::move(strides)), offset_(offset), span_(span) {}
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
    // Lazy View/Depends inputs acquire their actual strides during evaluation.
    // Flattening or copying here would silently break the shared cache backing.
    if (!input.flags().row_contiguous) {
      throw std::invalid_argument("cache_view: backing must be row-contiguous");
    }
    auto flags = output.flags();
    flags.row_contiguous = flags.col_contiguous = true;
    size_t row = 1, col = 1, no_broadcast_size = 1;
    for (int i = 0; i < output.ndim(); ++i) {
      int j = output.ndim() - 1 - i;
      flags.row_contiguous &= output.shape(j) == 1 || strides_[j] == row;
      flags.col_contiguous &= output.shape(i) == 1 || strides_[i] == col;
      row *= output.shape(j);
      col *= output.shape(i);
      if (strides_[i] != 0) no_broadcast_size *= output.shape(i);
    }
    flags.contiguous = span_ == no_broadcast_size;
    output.copy_shared_buffer(input, strides_, flags, span_, offset_);
  }
  Shape shape_;
  Strides strides_;
  int64_t offset_;
  size_t span_;
};

void register_mlx_patch(nb::module_& m) {
  m.def("cache_view", [](nb::handle buffer_h, const std::vector<int>& shape,
                          const std::vector<size_t>& strides, size_t offset) {
    nb::object array_cls = nb::module_::import_("mlx.core").attr("array");
    if (!nb::isinstance(buffer_h, array_cls)) {
      throw nb::type_error("cache_view: buffer must be mlx.core.array");
    }
    const auto& buffer = *nb::inst_ptr<array>(buffer_h);
    if (shape.size() != strides.size()) {
      throw std::invalid_argument("cache_view: shape/stride rank mismatch");
    }
    // MLX stores strides and byte offsets in int64_t. Also bound the logical
    // size before ArrayDesc computes its own contiguous strides and nbytes.
    const size_t max_stride = std::numeric_limits<int64_t>::max();
    const size_t max_elements = max_stride / buffer.itemsize();
    const size_t capacity = std::min(buffer.size(), max_elements);
    if (offset >= capacity) {
      throw std::invalid_argument("cache_view: offset out of bounds");
    }
    size_t end = offset, size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] <= 0) throw std::invalid_argument("cache_view: non-positive dimension");
      if (size > max_elements / shape[i]) {
        throw std::invalid_argument("cache_view: shape size overflow");
      }
      size *= shape[i];
      if (strides[i] > max_stride) {
        throw std::invalid_argument("cache_view: stride exceeds int64 range");
      }
      const size_t extent = shape[i] - 1;
      // Check with division before multiplying or adding: a wrapped span can
      // otherwise appear to fit even a tiny backing allocation.
      if (extent != 0 && strides[i] > (capacity - 1 - end) / extent) {
        throw std::invalid_argument("cache_view: out of bounds");
      }
      end += extent * strides[i];
    }
    // mx.as_strided flattens first, exceeding MLX's int32 dimension limit
    // for large arenas. Bind the existing backing without that flatten.
    Shape view_shape(shape.begin(), shape.end());
    Strides view_strides(strides.begin(), strides.end());
    auto result = array::make_arrays(
        {view_shape}, {buffer.dtype()},
        std::make_shared<CacheViewPrimitive>(
            default_stream(Device::gpu), view_shape, view_strides,
            static_cast<int64_t>(offset), end - offset + 1),
        {buffer})[0];
    nb::object out = array_cls(0);
    nb::inst_ptr<array>(out)->overwrite_descriptor(result);
    return out;
  });
}
