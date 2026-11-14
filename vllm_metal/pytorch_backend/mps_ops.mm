// SPDX-License-Identifier: Apache-2.0
// Host-only launcher for the SAME shaders used by metal/paged_attention.cpp.
// Encode on PyTorch's stream so projections, KV writes and attention are ordered
// without a framework bridge, a CPU round trip, or a per-layer GPU wait.
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <atomic>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../metal/metal_device.h"
#include "../metal/paged_attention_kernels.h"

using at::Tensor;
namespace kernels = vllm_metal::kernels;
namespace hardware = vllm_metal::hardware;

constexpr int kPartitionSize = VLLM_METAL_PARTITION_SIZE;

static bool nax_supported() {
  auto device = at::mps::getCurrentMPSStream()->device();
  return hardware::nax_supported(
      [device supportsFamily:static_cast<MTLGPUFamily>(hardware::kApple10)]);
}

// Tests inject aligned and unaligned limits through
// `_override_max_buffer_length_for_test`; -1 restores the device value.
static std::atomic<int64_t> g_test_max_buffer_length{-1};

static size_t max_buffer_length() {
  const int64_t override =
      g_test_max_buffer_length.load(std::memory_order_relaxed);
  if (override > 0) return static_cast<size_t>(override);
  return [at::mps::getCurrentMPSStream()->device() maxBufferLength];
}

static void override_max_buffer_length_for_test(int64_t nbytes) {
  if (nbytes != -1 && nbytes <= 0)
    throw std::invalid_argument(
        "test max buffer length override must be -1 or positive (got " +
        std::to_string(nbytes) + ")");
  g_test_max_buffer_length.store(nbytes, std::memory_order_relaxed);
}

class PagedAttention {
  id<MTLLibrary> library_;
  id<MTLLibrary> nax_library_ = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
  int gpu_cores_;

  id<MTLComputePipelineState> pipeline(const std::string& name, bool split = false,
                                      int window_q = 0, bool block_attention = false) {
    const auto key = name + "_wq" + std::to_string(window_q)
        + "_block" + std::to_string(block_attention);
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;
    auto constants = [[MTLFunctionConstantValues alloc] init];
    bool no = false;
    for (NSUInteger i : {20, 30, 40, 50, 100, 130})
      [constants setConstantValue:&no type:MTLDataTypeBool atIndex:i];
    // NAX prefill: 41 (nax_split_full) = false keeps the original guarded
    // loop. The split pipeline is only chosen by the MLX launcher.
    [constants setConstantValue:&no type:MTLDataTypeBool atIndex:41];
    [constants setConstantValue:&split type:MTLDataTypeBool atIndex:10];
    [constants setConstantValue:&block_attention type:MTLDataTypeBool atIndex:120];
    int kbits = 8, vbits = 3;
    [constants setConstantValue:&kbits type:MTLDataTypeInt atIndex:60];
    [constants setConstantValue:&vbits type:MTLDataTypeInt atIndex:70];
    [constants setConstantValue:&window_q type:MTLDataTypeInt atIndex:110];
    auto lib = name.find("paged_attention_nax_") == 0 ? nax_library_ : library_;
    NSError* error = nil;
    auto fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]
                       constantValues:constants error:&error];
    [constants release];
    TORCH_CHECK(fn, "MPS attention function ", name, ": ",
                error ? error.localizedDescription.UTF8String : "missing function");
    auto pso = [at::mps::getCurrentMPSStream()->device()
        newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    TORCH_CHECK(pso, "MPS attention pipeline: ", error.localizedDescription.UTF8String);
    pipelines_[key] = pso;
    return pso;
  }

  static void buffer(id<MTLComputeCommandEncoder> enc, int i, const Tensor& t) {
    [enc setBuffer:at::native::mps::getMTLBufferStorage(t)
            offset:t.storage_offset() * t.element_size() atIndex:i];
  }
  template<typename T>
  static void scalar(id<MTLComputeCommandEncoder> enc, int i, T value) {
    [enc setBytes:&value length:sizeof(T) atIndex:i];
  }

  static void check_cache(const Tensor& k, const Tensor& v, const Tensor& kc,
                          const Tensor& vc, const Tensor& slots) {
    for (auto* t : {&k, &v, &kc, &vc, &slots})
      TORCH_CHECK(t->is_mps(), "All cache buffers must be on MPS");
    TORCH_CHECK(kc.dim() == 4 &&
                (kc.size(1) == 8 || kc.size(1) == 16 || kc.size(1) == 32) &&
                kc.stride(3) == 1 && vc.strides() == kc.strides() && kc.sizes() == vc.sizes(),
                "MPS attention requires [blocks,8/16/32,kv_heads,head_dim] KV");
    TORCH_CHECK(k.scalar_type() == at::kHalf || k.scalar_type() == at::kBFloat16,
                "MPS cache writes require fp16 or bf16");
    for (auto* t : {&v, &kc, &vc})
      TORCH_CHECK(t->scalar_type() == k.scalar_type(), "K/V/cache dtype mismatch");
    TORCH_CHECK(k.sizes() == v.sizes() && k.dim() == 3 &&
                k.size(1) == kc.size(2) && k.size(2) == kc.size(3) &&
                k.stride(2) == 1 && k.stride(1) == k.size(2) &&
                v.stride(2) == 1 && v.stride(1) == v.size(2) &&
                slots.numel() == k.size(0) && slots.scalar_type() == at::kLong &&
                slots.is_contiguous(), "Inconsistent cache shapes/strides");
  }

  void encode_cache(id<MTLComputeCommandEncoder> enc, const Tensor& k,
                    const Tensor& v, const Tensor& kc, const Tensor& vc,
                    const Tensor& slots) {
    const std::string dt = k.scalar_type() == at::kHalf ? "half" : "bfloat16_t";
    [enc setComputePipelineState:pipeline(kernels::reshape_and_cache_name(dt, dt))];
    buffer(enc, 0, k); buffer(enc, 1, v);
    buffer(enc, 2, kc); buffer(enc, 3, vc); buffer(enc, 4, slots);
    scalar<int>(enc, 7, k.stride(0)); scalar<int>(enc, 8, v.stride(0));
    scalar<int>(enc, 9, kc.size(2)); scalar<int>(enc, 10, kc.size(3));
    scalar<int>(enc, 11, kc.size(1));
    scalar<int64_t>(enc, 12, kc.stride(0));
    scalar<int64_t>(enc, 13, kc.stride(1));
    scalar<int64_t>(enc, 14, kc.stride(2));
    [enc dispatchThreadgroups:MTLSizeMake(k.size(0), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }

 public:
  PagedAttention(const std::string& path, const std::string& nax_path, int gpu_cores)
      : gpu_cores_(gpu_cores) {
    @autoreleasepool {
      auto device = at::mps::getCurrentMPSStream()->device();
      NSError* error = nil;
      library_ = [device newLibraryWithURL:[NSURL fileURLWithPath:
          [NSString stringWithUTF8String:path.c_str()]] error:&error];
      TORCH_CHECK(library_, "Cannot load paged attention metallib: ",
                  error.localizedDescription.UTF8String);
      if (!nax_path.empty()) {
        nax_library_ = [device newLibraryWithURL:[NSURL fileURLWithPath:
            [NSString stringWithUTF8String:nax_path.c_str()]] error:&error];
        if (!nax_library_)
          TORCH_WARN("Cannot load NAX metallib; using the non-NAX fallback: ",
                     error.localizedDescription.UTF8String);
      }
    }
  }
  ~PagedAttention() {
    for (auto& p : pipelines_) [p.second release];
    [library_ release];
    [nax_library_ release];
  }

  void reshape_and_cache(const Tensor& k, const Tensor& v, const Tensor& kc,
                         const Tensor& vc, const Tensor& slots) {
    check_cache(k, v, kc, vc, slots);
    if (k.size(0) == 0) return;
    auto stream = at::mps::getCurrentMPSStream();
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
      @autoreleasepool {
        encode_cache(stream->commandEncoder(), k, v, kc, vc, slots);
      }
    });
  }

  void forward(const Tensor& q, const Tensor& k, const Tensor& v,
               const Tensor& kc, const Tensor& vc, const Tensor& slots,
               const Tensor& blocks, const Tensor& lens, const Tensor& cu,
               const Tensor& out, int max_seq_len, double scale,
               int sliding_window, double softcap, int window_seqlen_q,
               const std::optional<Tensor>& block_ranges) {
    for (auto* t : {&q, &blocks, &lens, &cu, &out})
      TORCH_CHECK(t->is_mps(), "All paged attention buffers must be on MPS");
    TORCH_CHECK(q.dim() == 3 && q.is_contiguous() &&
                (q.size(2) == 64 || q.size(2) == 96 || q.size(2) == 128 ||
                 q.size(2) == 256 || q.size(2) == 512),
                "MPS attention requires contiguous Q with head_dim=64/96/128/256/512");
    const int head_size = q.size(2);
    check_cache(k, v, kc, vc, slots);
    TORCH_CHECK(kc.size(3) == head_size, "Query and cache head sizes must match");
    const int block_size = kc.size(1);
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && out.scalar_type() == q.scalar_type(),
                "Q/K/V/output dtype mismatch");
    TORCH_CHECK(blocks.scalar_type() == at::kInt &&
                lens.scalar_type() == at::kInt && cu.scalar_type() == at::kInt,
                "Invalid attention metadata dtype");
    TORCH_CHECK(blocks.is_contiguous() &&
                lens.is_contiguous() && cu.is_contiguous(),
                "Attention metadata must be contiguous");
    const int tokens = q.size(0), heads = q.size(1), kv_heads = kc.size(2);
    const int seqs = lens.numel();
    const int parts = (max_seq_len + kPartitionSize - 1) / kPartitionSize;
    TORCH_CHECK(k.size(0) == tokens &&
                out.sizes() == q.sizes() && out.is_contiguous() &&
                blocks.dim() == 2 && blocks.size(0) == seqs &&
                cu.numel() == seqs + 1 && heads % kv_heads == 0,
                "Inconsistent attention shapes/strides");
    // Reuse the same short verification-window shader mode as the MLX launcher.
    const bool block_attention = block_ranges.has_value();
    if (block_attention) {
      const auto& ranges = *block_ranges;
      TORCH_CHECK(ranges.is_mps() && ranges.scalar_type() == at::kInt &&
                  ranges.is_contiguous() && ranges.dim() == 2 &&
                  ranges.size(0) == tokens && ranges.size(1) == 2 && tokens > seqs,
                  "Block attention requires int32 [query tokens,2] ranges");
    }
    const bool window = !block_attention && tokens > seqs && window_seqlen_q > 1 &&
        head_size <= VLLM_METAL_PA_WINDOW_MAX_HEAD;
    const int window_q = window ?
        (window_seqlen_q + VLLM_METAL_PA_WINDOW_ROWS - 1) / VLLM_METAL_PA_WINDOW_ROWS : 0;
    const bool prefill = tokens > seqs && !window;
    const bool nax = prefill && !block_attention && nax_library_ != nil;
    const bool split = !prefill
        && kernels::should_split_decode(heads, tokens, parts, gpu_cores_);
    const std::string dt = q.scalar_type() == at::kHalf ? "half" : "bfloat16_t";
    const auto cfg = kernels::select_tile_config(head_size).value();
    auto name = nax ? kernels::nax_name(dt, head_size, block_size) :
        prefill ? kernels::tiled_name(dt, head_size, block_size, cfg) :
        kernels::paged_name(
            dt, dt, dt, head_size, block_size, split ? kPartitionSize : 0);
    auto attn = pipeline(name, split, window_q, block_attention);
    id<MTLComputePipelineState> reduce = nil;
    Tensor tmp, sums, maxes;
    if (split) {
      tmp = at::empty({tokens, heads, parts, head_size}, q.options());
      sums = at::empty({tokens, heads, parts}, q.options().dtype(at::kFloat));
      maxes = at::empty_like(sums);
      reduce = pipeline(
          kernels::paged_reduce_kernel_name(dt, head_size, kPartitionSize));
    }
    auto stream = at::mps::getCurrentMPSStream();
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
      @autoreleasepool {
        auto enc = stream->commandEncoder();
        encode_cache(enc, k, v, kc, vc, slots);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:attn];
        buffer(enc, 2, split ? tmp : out); buffer(enc, 3, q);
        buffer(enc, 4, kc); buffer(enc, 5, vc);
        scalar<int>(enc, 8, kv_heads); scalar<float>(enc, 9, scale);
        scalar<float>(enc, 10, softcap);
        buffer(enc, 11, blocks); buffer(enc, 12, lens);
        scalar<int>(enc, 13, blocks.size(1));
        scalar<int>(enc, 15, heads * head_size);
        scalar<int>(enc, 16, kc.stride(0)); scalar<int>(enc, 17, kc.stride(2));
        buffer(enc, 19, cu); scalar<int>(enc, 20, seqs);
        scalar<int>(enc, 21, sliding_window);
        if (block_attention) buffer(enc, 22, *block_ranges);
        int grid_y = window ? seqs * window_q : tokens, threads = 256;
        if (prefill) {
          grid_y = tokens / (nax ? kernels::kNaxBQ : cfg.BQ) + seqs;
          threads = nax ? kernels::kNaxThreads : cfg.NUM_THREADS;
          scalar<int>(enc, 30, 0);  // q_block_offset: this launch covers every block.
          if (!nax) {
            [enc setThreadgroupMemoryLength:
                kernels::tiled_shared_bytes(cfg, head_size, q.element_size()) atIndex:0];
          }
        } else {
          [enc setThreadgroupMemoryLength:kernels::paged_threadgroup_bytes(
              head_size, block_size, static_cast<int>(q.element_size()),
              window, VLLM_METAL_PA_WINDOW_ROWS) atIndex:0];
        }
        if (split) { buffer(enc, 0, sums); buffer(enc, 1, maxes); }
        [enc dispatchThreadgroups:MTLSizeMake(heads, grid_y, split ? parts : 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        if (split) {
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
          [enc setComputePipelineState:reduce];
          [enc setThreadgroupMemoryLength:
              kernels::paged_reduce_threadgroup_bytes(parts) atIndex:0];
          buffer(enc, 0, out); buffer(enc, 1, sums); buffer(enc, 2, maxes);
          buffer(enc, 3, tmp); buffer(enc, 4, lens); scalar<int>(enc, 5, parts);
          buffer(enc, 7, cu); scalar<int>(enc, 8, seqs);
          [enc dispatchThreadgroups:MTLSizeMake(heads, tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
      }
    });
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("detected_gpu_core_count", &hardware::detected_gpu_core_count);
  m.def("gpu_core_count", &hardware::gpu_core_count);
  m.def("_override_detected_gpu_core_count_for_test",
        &hardware::override_detected_gpu_core_count_for_test);
  m.def("nax_supported", &nax_supported);
  m.def("max_buffer_length", &max_buffer_length);
  m.def("_override_max_buffer_length_for_test",
        &override_max_buffer_length_for_test);
  pybind11::class_<PagedAttention>(m, "PagedAttention")
      .def(pybind11::init<const std::string&, const std::string&, int>())
      .def("reshape_and_cache", &PagedAttention::reshape_and_cache)
      .def("forward", &PagedAttention::forward);
}
