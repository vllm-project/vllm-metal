// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace vllm_metal::kernels {

// The shader has no function constants or dynamic threadgroup memory. Inputs
// are packed; only the state pool's outer slot stride is configurable.
struct GDNLinearAttentionPlan {
  std::string kernel_name;
  int grid_x, grid_z, threads_x;
  int num_requests, Hk, Hv, Dk, Dv;
  int64_t state_stride;
};

inline void validate_gdn_key_dim(int Dk) {
  if (Dk <= 0 || Dk > 256 || Dk % 32 != 0)
    throw std::runtime_error(
        "GDN kernel requires Dk to be a positive multiple of 32, at most 256. "
        "Got Dk=" + std::to_string(Dk));
}

inline GDNLinearAttentionPlan make_gdn_linear_attention_plan(
    const std::string& dtype, const std::string& state_dtype,
    int num_requests, int Hk, int Hv, int Dk, int Dv,
    int64_t state_stride, bool use_initial_state_mask = false,
    const std::string& output_dtype = "") {
  validate_gdn_key_dim(Dk);
  if (num_requests < 0 || Hk <= 0 || Hv <= 0 || Hv % Hk || Dv <= 0 ||
      int64_t(num_requests) * Hv > std::numeric_limits<int>::max() ||
      int64_t(Hv) * Dv * Dk > std::numeric_limits<int>::max() ||
      state_stride < int64_t(Hv) * Dv * Dk)
    throw std::invalid_argument("Invalid GDN heads, batch size or state stride");
  if ((dtype != "float" && dtype != "half" && dtype != "bfloat16_t") ||
      (state_dtype != dtype && state_dtype != "float"))
    throw std::invalid_argument("Unsupported GDN input/state dtype pair");
  auto name = "gdn_linear_attention_" + dtype;
  if (state_dtype != dtype) name += "_state_" + state_dtype;
  if (!output_dtype.empty() && output_dtype != dtype) {
    if (dtype != "float" || state_dtype != "float" || !use_initial_state_mask ||
        (output_dtype != "half" && output_dtype != "bfloat16_t"))
      throw std::invalid_argument("Unsupported GDN output dtype");
    name += "_out_" + output_dtype;
  }
  if (use_initial_state_mask) name += "_initial";
  return {name, Dv, num_requests * Hv, 32,
          num_requests, Hk, Hv, Dk, Dv, state_stride};
}

template <typename SetBytes>
inline void bind_gdn_linear_attention_params(
    const GDNLinearAttentionPlan& plan, SetBytes&& set_bytes) {
  set_bytes(9, plan.num_requests);
  set_bytes(10, plan.Hk);
  set_bytes(11, plan.Hv);
  set_bytes(12, plan.Dk);
  set_bytes(13, plan.Dv);
  set_bytes(14, plan.state_stride);
}

}  // namespace vllm_metal::kernels
