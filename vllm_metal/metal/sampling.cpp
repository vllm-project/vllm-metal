// SPDX-License-Identifier: Apache-2.0
// MLX graph adapter for the same Gumbel-max shader used by PyTorch MPS.
#include "ops_internal.h"

constexpr int64_t kSamplingBlockSize = 1024;

class GumbelSamplePrimitive : public Primitive {
 public:
  GumbelSamplePrimitive(Stream stream, bool indexed)
      : Primitive(stream), indexed_(indexed) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GumbelSamplePrimitive only supports GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    for (auto& out : outputs) out.set_data(allocator::malloc(out.nbytes()));
    auto s = stream();
    auto& device = metal::device(s.device);
    auto* lib = device.get_library("sampling_kern");
    auto* kernel = device.get_kernel("gumbel_sample", lib, "gumbel_sample", {});
    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 5; ++i) enc.set_input_array(inputs[i], i);
    enc.set_output_array(outputs[0], 5);
    enc.set_output_array(outputs[1], 6);
    int64_t vocab = inputs[0].shape(1), blocks = outputs[0].shape(1);
    int64_t apply_temperature = 1, indexed = indexed_;
    enc.set_bytes(vocab, 7);
    enc.set_bytes(kSamplingBlockSize, 8);
    enc.set_bytes(blocks, 9);
    enc.set_bytes(apply_temperature, 10);
    enc.set_input_array(inputs[5], 11);
    enc.set_bytes(indexed, 12);
    enc.dispatch_threadgroups(
        MTL::Size::Make(blocks, inputs[0].shape(0), 1), MTL::Size::Make(256, 1, 1));
  }

  const char* name() const override { return "GumbelSamplePrimitive"; }
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const GumbelSamplePrimitive&>(other);
    return indexed_ == rhs.indexed_;
  }

 private:
  bool indexed_;
};

array gumbel_sample_primitive_fn(
    const array& logits, const array& seeds, const array& positions,
    const array& temperatures, const std::optional<array>& token_ids) {
  if (logits.ndim() != 2 || logits.dtype() != float32 || logits.shape(0) == 0 || logits.shape(1) == 0)
    throw std::invalid_argument("gumbel_sample: expected nonempty float32 [rows, vocab] logits");
  int rows = logits.shape(0);
  int blocks = (logits.shape(1) + kSamplingBlockSize - 1) / kSamplingBlockSize;
  for (const auto* value : {&seeds, &positions, &temperatures})
    if (value->ndim() != 1 || value->shape(0) != rows)
      throw std::invalid_argument("gumbel_sample: one seed, position and temperature per row required");
  if (seeds.dtype() != int64 || positions.dtype() != int64 || temperatures.dtype() != float32)
    throw std::invalid_argument("gumbel_sample: expected int64 seeds/positions and float32 temperatures");
  if (token_ids && (token_ids->shape() != logits.shape() || token_ids->dtype() != uint32))
    throw std::invalid_argument("gumbel_sample: candidate IDs must be uint32 with the logits shape");
  auto s = default_stream(Device::gpu);
  auto primitive = std::make_shared<GumbelSamplePrimitive>(s, token_ids.has_value());
  auto outputs = array::make_arrays(
      {{rows, blocks}, {rows, blocks}}, {int64, float32}, primitive,
      {contiguous(logits), arange(rows, int64), contiguous(seeds),
       contiguous(positions), contiguous(temperatures),
       token_ids ? contiguous(*token_ids) : zeros({1}, uint32)});
  auto& ids = outputs[0];
  auto& scores = outputs[1];
  auto best = max(scores, 1, true, s);
  // Candidate order may differ from vocabulary order. Break ties by token ID.
  auto winning_ids = where(equal(scores, best, s), ids, array(INT64_MAX), s);
  return min(winning_ids, 1, false, s);
}
