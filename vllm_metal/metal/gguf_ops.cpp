// SPDX-License-Identifier: Apache-2.0
// C++ nanobind bridge for raw GGUF quantized-weight Metal primitives.

#include "gguf_ops.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace nb = nanobind;
using namespace mlx::core;

namespace {

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

static std::string dtype_to_metal(Dtype dt) {
  switch (dt) {
    case float16: return "half";
    case bfloat16: return "bfloat16_t";
    case float32: return "float";
    case uint8: return "uchar";
    default:
      throw std::runtime_error("Unsupported dtype for GGUF Metal kernel");
  }
}

static std::string gguf_source_;
constexpr int kGGUFQK8_0 = 32;
constexpr int kGGUFQ8_0BlockBytes = 34;
constexpr int kGGUFQ8_1BlockBytes = 36;
constexpr int kGGUFQTypeQ8_0 = 8;

void init_gguf_library(const std::string& src) {
  gguf_source_ = src;
  auto& d = metal::device(Device::gpu);
  d.get_library("gguf_kern", [&]() { return gguf_source_; });
}

static void validate_gguf_q8_0_weight(
    const char* op_name,
    const array& qweight) {
  if (qweight.dtype() != uint8) {
    throw std::runtime_error(
        std::string(op_name) + ": qweight must be uint8 raw GGUF bytes");
  }
  if (qweight.ndim() != 2) {
    throw std::runtime_error(
        std::string(op_name) + ": qweight must be 2D [rows, raw_bytes]");
  }
  int bytes_per_row = static_cast<int>(qweight.shape(1));
  if (bytes_per_row <= 0 || bytes_per_row % kGGUFQ8_0BlockBytes != 0) {
    throw std::runtime_error(
        std::string(op_name) +
        ": qweight raw-byte dimension must be a positive multiple of 34");
  }
}

class GGUFQ8_0MulMatPrimitive : public UnaryPrimitive {
 public:
  GGUFQ8_0MulMatPrimitive(
      Stream stream,
      int input_dims,
      int rows,
      int blocks_per_row,
      int bytes_per_row,
      int padded_blocks)
      : UnaryPrimitive(stream),
        input_dims_(input_dims),
        rows_(rows),
        blocks_per_row_(blocks_per_row),
        bytes_per_row_(bytes_per_row),
        padded_blocks_(padded_blocks) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("GGUFQ8_0MulMatPrimitive only supports GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    out.set_data(allocator::malloc(out.nbytes()));

    const array& x = inputs[0];
    const array& qweight = inputs[1];
    int num_tokens = static_cast<int>(x.shape(0));

    Shape qx_shape{
        static_cast<ShapeElem>(num_tokens),
        static_cast<ShapeElem>(padded_blocks_ * kGGUFQ8_1BlockBytes)};
    array qx(
        allocator::malloc(
            static_cast<size_t>(num_tokens) * padded_blocks_ *
            kGGUFQ8_1BlockBytes),
        qx_shape,
        uint8);

    auto s = stream();
    auto& d = metal::device(s.device);
    auto& enc = get_command_encoder_compat(d, s);
    auto* lib = d.get_library("gguf_kern");
    auto dt = dtype_to_metal(x.dtype());

    std::string qname = "gguf_q8_1_quantize_" + dt;
    auto* qkernel = d.get_kernel(qname, lib, qname, {});
    enc.set_compute_pipeline_state(qkernel);
    enc.set_input_array(x, 0);
    enc.set_output_array(qx, 1);
    enc.set_bytes(input_dims_, 2);
    enc.set_bytes(padded_blocks_, 3);
    enc.dispatch_threadgroups(
        MTL::Size::Make(padded_blocks_, num_tokens, 1),
        MTL::Size::Make(kGGUFQK8_0, 1, 1));
    enc.barrier();

    std::string mname = "gguf_q8_0_matvec_" + dt;
    auto* mkernel = d.get_kernel(mname, lib, mname, {});
    enc.set_compute_pipeline_state(mkernel);
    enc.set_input_array(qweight, 0);
    enc.set_input_array(qx, 1);
    enc.set_output_array(out, 2);
    enc.set_bytes(rows_, 3);
    enc.set_bytes(blocks_per_row_, 4);
    enc.set_bytes(bytes_per_row_, 5);
    enc.set_bytes(padded_blocks_, 6);
    enc.dispatch_threadgroups(
        MTL::Size::Make(rows_, num_tokens, 1),
        MTL::Size::Make(kGGUFQK8_0, 1, 1));

    add_temporary_compat(enc, x, d, s);
    add_temporary_compat(enc, qweight, d, s);
    add_temporary_compat(enc, out, d, s);
    add_temporary_compat(enc, qx, d, s);
  }

  const char* name() const override { return "GGUFQ8_0MulMat"; }

  bool is_equivalent(const Primitive& other) const override {
    auto* rhs = dynamic_cast<const GGUFQ8_0MulMatPrimitive*>(&other);
    return rhs && rhs->input_dims_ == input_dims_ &&
        rhs->rows_ == rows_ &&
        rhs->blocks_per_row_ == blocks_per_row_ &&
        rhs->bytes_per_row_ == bytes_per_row_ &&
        rhs->padded_blocks_ == padded_blocks_;
  }

 private:
  int input_dims_;
  int rows_;
  int blocks_per_row_;
  int bytes_per_row_;
  int padded_blocks_;
};

static array gguf_q8_0_mul_mat_primitive_fn(
    const array& x,
    const array& qweight,
    int qweight_type) {
  if (qweight_type != kGGUFQTypeQ8_0) {
    throw std::runtime_error(
        "gguf_q8_0_mul_mat: qweight_type must be GGUF Q8_0");
  }
  if (x.ndim() != 2) {
    throw std::runtime_error("gguf_q8_0_mul_mat: x must be 2D");
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16 && x.dtype() != float32) {
    throw std::runtime_error(
        "gguf_q8_0_mul_mat: x must be fp16, bf16, or fp32");
  }
  validate_gguf_q8_0_weight("gguf_q8_0_mul_mat", qweight);

  int input_dims = static_cast<int>(x.shape(1));
  if (input_dims <= 0 || input_dims % kGGUFQK8_0 != 0) {
    throw std::runtime_error(
        "gguf_q8_0_mul_mat: input dimension must be a positive multiple of 32");
  }
  int rows = static_cast<int>(qweight.shape(0));
  int bytes_per_row = static_cast<int>(qweight.shape(1));
  int blocks_per_row = bytes_per_row / kGGUFQ8_0BlockBytes;
  if (blocks_per_row * kGGUFQK8_0 != input_dims) {
    throw std::runtime_error(
        "gguf_q8_0_mul_mat: qweight shape does not match x input dimension");
  }
  int padded_cols = ((input_dims + 511) / 512) * 512;
  int padded_blocks = padded_cols / kGGUFQK8_0;

  auto prim = std::make_shared<GGUFQ8_0MulMatPrimitive>(
      default_stream(Device::gpu),
      input_dims,
      rows,
      blocks_per_row,
      bytes_per_row,
      padded_blocks);
  return array(
      Shape{static_cast<ShapeElem>(x.shape(0)), static_cast<ShapeElem>(rows)},
      x.dtype(),
      std::move(prim),
      {x, qweight});
}

}  // namespace

void register_gguf_ops(nb::module_& m) {
  m.def("init_gguf_library", &init_gguf_library,
        nb::arg("src"),
        "JIT-compile the GGUF raw-weight Metal shaders.");

  m.def("gguf_q8_0_mul_mat",
        [](nb::handle x_h,
           nb::handle qweight_h,
           int qweight_type,
           nb::handle out_h) {
          auto result = gguf_q8_0_mul_mat_primitive_fn(
              *nb::inst_ptr<array>(x_h),
              *nb::inst_ptr<array>(qweight_h),
              qweight_type);
          nb::inst_ptr<array>(out_h)->overwrite_descriptor(result);
        },
        nb::arg("x"),
        nb::arg("qweight"),
        nb::arg("qweight_type"),
        nb::arg("out"),
        "Raw GGUF Q8_0 linear matvec/matmul. Wraps an MLX Primitive that "
        "internally quantizes activations to Q8_1 and multiplies against raw "
        "Q8_0 GGUF blocks.");
}
