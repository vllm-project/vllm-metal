// SPDX-License-Identifier: Apache-2.0
// MLA paged attention validation, planning, and dispatch.
#include "ops_internal.h"

// MLA dispatchers pick a single Metal specialization from one tensor's
// dtype (q_nope) but the kernel template binds the same `T` to every
// fp16/bf16 buffer (q_nope, q_pe, latent_cache, out, tmp_out). If they
// disagree the shader will reinterpret bytes — e.g. read a bf16 cache
// as fp16 — and silently corrupt attention. Validate up front.
static void mla_validate_t_dtypes(
    const char* dispatcher_name,
    std::initializer_list<std::pair<const char*, const array*>> tensors) {
  if (tensors.size() == 0) return;
  Dtype expected = tensors.begin()->second->dtype();
  if (expected != float16 && expected != bfloat16) {
    throw std::runtime_error(
        std::string(dispatcher_name) +
        ": T buffers must be fp16 or bf16; got " +
        dtype_to_metal(expected) + " on " + tensors.begin()->first);
  }
  for (const auto& [name, arr] : tensors) {
    if (arr->dtype() != expected) {
      throw std::runtime_error(
          std::string(dispatcher_name) +
          ": all T buffers must share the same dtype; got " +
          dtype_to_metal(expected) + " on " + tensors.begin()->first +
          " but " + dtype_to_metal(arr->dtype()) + " on " + name);
    }
  }
}

// ---------------------------------------------------------------------------
// MLA paged attention (RFC #360)
// ---------------------------------------------------------------------------

static std::string mla_source_;

void init_mla_library(const std::string& src) {
  mla_source_ = src;
  auto& d = metal::device(Device::gpu);
  d.get_library("paged_mla_kern", [&]() { return mla_source_; });
}

// Dispatch the MLA paged attention kernel.
//
// Buffer slot map (must match kernels_v2/mla.metal):
//   2: out          [total_q_tokens, num_heads, KV_LORA_RANK]
//   3: q_nope       [total_q_tokens, num_heads, KV_LORA_RANK]
//   4: q_pe         [total_q_tokens, num_heads, QK_ROPE_HEAD_DIM]
//   5: latent_cache [num_blocks, BLOCK_SIZE, KV_LORA_RANK + QK_ROPE_HEAD_DIM]
//   6: block_tables 7: context_lens 8: cu_seqlens_q
//   9: num_seqs 10: max_num_blocks_per_seq 11: scale
// The instantiated MLA specialization space — one row per distinct shape the
// instantiate_mla call sites in kernels_v2/mla.metal declare. The dispatch
// gate below and the drift test in tests/test_block_size_translation.py both
// read this one table, so keep the row format stable:
//   {kv_lora_rank, qk_rope_head_dim, block_size, heads_per_tg, num_threads,
//    partition_size}
// Each variant keeps the per-thread register footprint roughly constant
// (NUM_THREADS scaled inversely to G):
//   G=1 → NUM_THREADS=1024 (32 simdgroups, current sdpa_vector layout).
//   G=2 → NUM_THREADS=512  (16 simdgroups, 2× cross-head amortization).
struct MlaKernelSpec {
  int kv_lora_rank;
  int qk_rope_head_dim;
  int block_size;
  int heads_per_tg;
  int num_threads;
  int partition_size;
};
static constexpr MlaKernelSpec kMlaKernelSpecs[] = {
    {512, 64, 16, 1, 1024, 0},
    {512, 64, 32, 1, 1024, 0},
    {512, 64, 16, 2, 512, 0},
    {512, 64, 32, 2, 512, 0},
};

static const MlaKernelSpec& select_mla_kernel_spec(
    int kv_lora_rank, int qk_rope_head_dim, int block_size,
    int heads_per_tg, int num_heads) {
  // The shape must match a kernel template instantiation — see
  // kMlaKernelSpecs for the admitted space.
  const MlaKernelSpec* spec = nullptr;
  for (const auto& candidate : kMlaKernelSpecs) {
    if (candidate.kv_lora_rank == kv_lora_rank &&
        candidate.qk_rope_head_dim == qk_rope_head_dim &&
        candidate.block_size == block_size &&
        candidate.heads_per_tg == heads_per_tg) {
      spec = &candidate;
      break;
    }
  }
  if (spec == nullptr) {
    throw std::runtime_error(
        "MLA kernel: no instantiation for kv_lora_rank=" +
        std::to_string(kv_lora_rank) + " qk_rope_head_dim=" +
        std::to_string(qk_rope_head_dim) + " block_size=" +
        std::to_string(block_size) + " heads_per_tg=" +
        std::to_string(heads_per_tg));
  }

  if (spec->partition_size != 0) {
    throw std::runtime_error(
        "MLA kernel: matched kv_lora_rank=" +
        std::to_string(spec->kv_lora_rank) + " qk_rope_head_dim=" +
        std::to_string(spec->qk_rope_head_dim) + " block_size=" +
        std::to_string(spec->block_size) + " heads_per_tg=" +
        std::to_string(spec->heads_per_tg) + " but partition_size=" +
        std::to_string(spec->partition_size) + " is not dispatchable");
  }

  if (num_heads % heads_per_tg != 0) {
    throw std::runtime_error(
        "MLA kernel: num_heads (" + std::to_string(num_heads) +
        ") must be divisible by heads_per_tg (" +
        std::to_string(heads_per_tg) + ")");
  }
  return *spec;
}

struct MlaAttentionPlan {
  std::string kernel_name;
  std::string hash_name;
  bool use_partitioning;
  size_t shmem;
  int grid_x;
  int grid_y;
  int threads_x;
};

static MlaAttentionPlan make_mla_attention_plan(
    const MlaKernelSpec& spec, Dtype dtype,
    int num_heads, int heads_per_tg, int total_q_tokens) {
  auto dt = dtype_to_metal(dtype);
  std::string kname = "paged_mla_attention_" + dt + "_kvr" +
                      std::to_string(spec.kv_lora_rank) + "_pe" +
                      std::to_string(spec.qk_rope_head_dim) + "_bs" +
                      std::to_string(spec.block_size) + "_g" +
                      std::to_string(spec.heads_per_tg) + "_nt" +
                      std::to_string(spec.num_threads) + "_nsl32_ps" +
                      std::to_string(spec.partition_size);
  bool use_partitioning = false;
  std::string hash_name = kname + "_part" + (use_partitioning ? "1" : "0");
  // Threadgroup memory:
  //   max_scores[G * BN] + sum_exp_scores[G * BN] + outputs[BD * BD]
  // The outputs buffer must be sized for the maximum write offset
  // `lane*BD + sg`, which reaches (BD-1)*BD + (BN-1). Using BD*BD always
  // (rather than BN*BD) gives enough room across all G; on G=1 (BN=BD=32)
  // they coincide.
  // For G=1, NT=1024: 2*32 + 32*32 = 1088 fp32 ≈ 4.3 KB.
  // For G=2, NT=512:  2*2*16 + 32*32 = 1088 fp32 ≈ 4.3 KB.
  const int BD = 32;
  const int BN = spec.num_threads / BD;
  size_t shmem =
      static_cast<size_t>((2 * heads_per_tg * BN + BD * BD) * sizeof(float));
  return {std::move(kname), std::move(hash_name), use_partitioning, shmem,
          num_heads / heads_per_tg, total_q_tokens, spec.num_threads};
}

static void dispatch_mla_paged_attention(
    array& out,
    const array& q_nope,
    const array& q_pe,
    const array& latent_cache,
    const array& block_tables,
    const array& context_lens,
    const array& cu_seqlens_q,
    int block_size,
    float scale,
    int heads_per_tg,
    Stream s) {
  auto& d = metal::device(s.device);

  int total_q_tokens = static_cast<int>(q_nope.shape(0));
  int num_heads = static_cast<int>(q_nope.shape(1));
  int kv_lora_rank = static_cast<int>(q_nope.shape(2));
  int qk_rope_head_dim = static_cast<int>(q_pe.shape(2));
  int max_num_blocks_per_seq = static_cast<int>(block_tables.shape(1));
  int num_seqs = static_cast<int>(cu_seqlens_q.shape(0)) - 1;

  const auto& spec = select_mla_kernel_spec(
      kv_lora_rank, qk_rope_head_dim, block_size, heads_per_tg, num_heads);
  mla_validate_t_dtypes("MLA kernel", {
      {"q_nope", &q_nope},
      {"q_pe", &q_pe},
      {"latent_cache", &latent_cache},
      {"out", &out},
  });
  auto plan = make_mla_attention_plan(
      spec, q_nope.dtype(), num_heads, heads_per_tg, total_q_tokens);

  auto* lib = d.get_library("paged_mla_kern");
  auto* kernel = d.get_kernel(
      plan.kernel_name,
      lib,
      plan.hash_name,
      {{&plan.use_partitioning, MTL::DataType::DataTypeBool, NS::UInteger(10)}});

  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_threadgroup_memory_length(plan.shmem, 0);

  enc.set_output_array(out, 2);
  enc.set_input_array(q_nope, 3);
  enc.set_input_array(q_pe, 4);
  enc.set_input_array(latent_cache, 5);
  enc.set_input_array(block_tables, 6);
  enc.set_input_array(context_lens, 7);
  enc.set_input_array(cu_seqlens_q, 8);

  int32_t num_seqs_i = static_cast<int32_t>(num_seqs);
  int32_t max_blocks_i = static_cast<int32_t>(max_num_blocks_per_seq);
  enc.set_bytes(num_seqs_i, 9);
  enc.set_bytes(max_blocks_i, 10);
  enc.set_bytes(scale, 11);

  // Grid: (num_heads / G, total_q_tokens, 1). Each TG owns G consecutive
  // query heads sharing the same latent KV.
  enc.dispatch_threadgroups(
      MTL::Size::Make(plan.grid_x, plan.grid_y, 1),
      MTL::Size::Make(plan.threads_x, 1, 1));

  // No add_temporary calls: the only caller is MlaPagedAttentionPrimitive,
  // and inside a primitive MLX manages array lifetimes via the completion
  // handler.
}

// MLA single-pass paged attention as an MLX Primitive so the kernel
// dispatch participates in the lazy graph (no per-call mx.eval boundary).
class MlaPagedAttentionPrimitive : public UnaryPrimitive {
 public:
  MlaPagedAttentionPrimitive(
      Stream stream, int block_size, float scale, int heads_per_tg)
      : UnaryPrimitive(stream),
        block_size_(block_size),
        scale_(scale),
        heads_per_tg_(heads_per_tg) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("MlaPagedAttentionPrimitive only supports GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    // Inputs match the single-pass dispatcher's positional order:
    //   [q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q]
    out.set_data(allocator::malloc(out.nbytes()));
    dispatch_mla_paged_attention(
        out,
        inputs[0], inputs[1], inputs[2],
        inputs[3], inputs[4], inputs[5],
        block_size_, scale_, heads_per_tg_,
        stream());
  }

  const char* name() const override { return "MlaPagedAttention"; }

  bool is_equivalent(const Primitive& other) const override {
    auto* rhs = dynamic_cast<const MlaPagedAttentionPrimitive*>(&other);
    return rhs && rhs->block_size_ == block_size_
        && rhs->scale_ == scale_ && rhs->heads_per_tg_ == heads_per_tg_;
  }

 private:
  int block_size_;
  float scale_;
  int heads_per_tg_;
};

array mla_paged_attention_primitive_fn(
    const array& q_nope,
    const array& q_pe,
    const array& latent_cache,
    const array& block_tables,
    const array& context_lens,
    const array& cu_seqlens_q,
    int block_size,
    float scale,
    int heads_per_tg) {
  auto prim = std::make_shared<MlaPagedAttentionPrimitive>(
      default_stream(Device::gpu), block_size, scale, heads_per_tg);
  // Output shape matches q_nope: (total_q_tokens, num_heads, kv_lora_rank).
  return array(
      q_nope.shape(),
      q_nope.dtype(),
      std::move(prim),
      {q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q});
}
