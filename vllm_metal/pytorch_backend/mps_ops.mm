// SPDX-License-Identifier: Apache-2.0
// Host-only launcher for the SAME shaders used by metal/paged_ops.cpp.
// Encode on PyTorch's stream so projections, KV writes and attention are ordered
// without a framework bridge, a CPU round trip, or a per-layer GPU wait.
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <unordered_map>

using at::Tensor;

class PagedAttention {
  id<MTLLibrary> library_;
  id<MTLLibrary> nax_library_ = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
  int min_decode_grid_;

  id<MTLComputePipelineState> pipeline(const std::string& name, bool split = false) {
    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) return it->second;
    auto constants = [[MTLFunctionConstantValues alloc] init];
    bool no = false;
    for (NSUInteger i : {20, 30, 40, 50, 100, 120})
      [constants setConstantValue:&no type:MTLDataTypeBool atIndex:i];
    [constants setConstantValue:&split type:MTLDataTypeBool atIndex:10];
    int zero = 0, kbits = 8, vbits = 3;
    [constants setConstantValue:&kbits type:MTLDataTypeInt atIndex:60];
    [constants setConstantValue:&vbits type:MTLDataTypeInt atIndex:70];
    [constants setConstantValue:&zero type:MTLDataTypeInt atIndex:110];
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
    pipelines_[name] = pso;
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

 public:
  PagedAttention(const std::string& path, const std::string& nax_path, int gpu_cores)
      : min_decode_grid_(gpu_cores * 8) {
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
        TORCH_CHECK(nax_library_, "Cannot load NAX metallib: ",
                    error.localizedDescription.UTF8String);
      }
    }
  }
  ~PagedAttention() {
    for (auto& p : pipelines_) [p.second release];
    [library_ release];
    [nax_library_ release];
  }

  void forward(const Tensor& q, const Tensor& k, const Tensor& v,
               const Tensor& kc, const Tensor& vc, const Tensor& slots,
               const Tensor& blocks, const Tensor& lens, const Tensor& cu,
               const Tensor& out, int max_seq_len, double scale) {
    for (auto* t : {&q, &k, &v, &kc, &vc, &slots, &blocks, &lens, &cu, &out})
      TORCH_CHECK(t->is_mps(), "All paged attention buffers must be on MPS");
    TORCH_CHECK(q.dim() == 3 && q.size(2) == 128 && q.is_contiguous(),
                "Experimental MPS attention requires contiguous Q with head_dim=128");
    TORCH_CHECK(kc.dim() == 4 && kc.size(1) == 16 && kc.size(3) == 128 &&
                kc.stride(3) == 1 && vc.strides() == kc.strides() && kc.sizes() == vc.sizes(),
                "Experimental MPS attention requires [blocks,16,kv_heads,128] KV");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "Experimental MPS attention requires fp16 or bf16");
    for (auto* t : {&k, &v, &kc, &vc, &out})
      TORCH_CHECK(t->scalar_type() == q.scalar_type(), "Q/K/V/cache/output dtype mismatch");
    TORCH_CHECK(slots.scalar_type() == at::kLong && blocks.scalar_type() == at::kInt &&
                lens.scalar_type() == at::kInt && cu.scalar_type() == at::kInt,
                "Invalid attention metadata dtype");
    TORCH_CHECK(slots.is_contiguous() && blocks.is_contiguous() &&
                lens.is_contiguous() && cu.is_contiguous(),
                "Attention metadata must be contiguous");
    const int tokens = q.size(0), heads = q.size(1), kv_heads = kc.size(2);
    const int seqs = lens.numel(), parts = (max_seq_len + 511) / 512;
    TORCH_CHECK(k.sizes() == v.sizes() && k.dim() == 3 && k.size(0) == tokens &&
                k.size(1) == kv_heads && k.size(2) == 128 &&
                k.stride(2) == 1 && k.stride(1) == 128 &&
                v.stride(2) == 1 && v.stride(1) == 128 &&
                out.sizes() == q.sizes() && out.is_contiguous() &&
                slots.numel() == tokens && blocks.dim() == 2 && blocks.size(0) == seqs &&
                cu.numel() == seqs + 1 && heads % kv_heads == 0,
                "Inconsistent attention shapes/strides");
    const bool prefill = tokens > seqs;
    const bool nax = prefill && nax_library_ != nil;
    const bool split = !prefill && heads * tokens < min_decode_grid_ && parts >= 2;
    const std::string dt = q.scalar_type() == at::kHalf ? "half" : "bfloat16_t";
    auto scatter = pipeline("reshape_and_cache_kv_" + dt + "_cache_" + dt);
    auto name = nax ? "paged_attention_nax_" + dt + "_hs128_bs16" :
        prefill ? "paged_attention_tiled_" + dt + "_hs128_bs16_bq32_tk32_nt128" :
        "paged_attention_" + dt + "_cache_" + dt + "_" + dt +
            "_hs128_bs16_nt256_nsl32_ps" + (split ? "512" : "0");
    auto attn = pipeline(name, split);
    id<MTLComputePipelineState> reduce = nil;
    Tensor tmp, sums, maxes;
    if (split) {
      tmp = at::empty({tokens, heads, parts, 128}, q.options());
      sums = at::empty({tokens, heads, parts}, q.options().dtype(at::kFloat));
      maxes = at::empty_like(sums);
      reduce = pipeline("paged_attention_v2_reduce_" + dt + "_hs128_nt256_nsl32_ps512");
    }
    auto stream = at::mps::getCurrentMPSStream();
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
      @autoreleasepool {
        auto enc = stream->commandEncoder();
        [enc setComputePipelineState:scatter];
        buffer(enc, 0, k); buffer(enc, 1, v);
        buffer(enc, 2, kc); buffer(enc, 3, vc); buffer(enc, 4, slots);
        scalar<int>(enc, 7, k.stride(0)); scalar<int>(enc, 8, v.stride(0));
        scalar<int>(enc, 9, kv_heads); scalar<int>(enc, 10, 128);
        scalar<int>(enc, 11, 16);
        scalar<int64_t>(enc, 12, kc.stride(0));
        scalar<int64_t>(enc, 13, kc.stride(1));
        scalar<int64_t>(enc, 14, kc.stride(2));
        [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:attn];
        buffer(enc, 2, split ? tmp : out); buffer(enc, 3, q);
        buffer(enc, 4, kc); buffer(enc, 5, vc);
        scalar<int>(enc, 8, kv_heads); scalar<float>(enc, 9, scale);
        scalar<float>(enc, 10, 0.0f);
        buffer(enc, 11, blocks); buffer(enc, 12, lens);
        scalar<int>(enc, 13, blocks.size(1));
        scalar<int>(enc, 15, heads * 128);
        scalar<int>(enc, 16, kc.stride(0)); scalar<int>(enc, 17, kc.stride(2));
        buffer(enc, 19, cu); scalar<int>(enc, 20, seqs);
        scalar<int>(enc, 21, -1);
        int grid_y = tokens, threads = 256;
        if (prefill) {
          grid_y = tokens / (nax ? 64 : 32) + seqs;
          threads = 128;
          if (!nax) [enc setThreadgroupMemoryLength:96 * 136 * 2 atIndex:0];
        } else {
          [enc setThreadgroupMemoryLength:(16 + 8 * 128) * 4 atIndex:0];
        }
        if (split) { buffer(enc, 0, sums); buffer(enc, 1, maxes); }
        [enc dispatchThreadgroups:MTLSizeMake(heads, grid_y, split ? parts : 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        if (split) {
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
          [enc setComputePipelineState:reduce];
          [enc setThreadgroupMemoryLength:((2 * parts * 4 + 15) & ~15) atIndex:0];
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
  pybind11::class_<PagedAttention>(m, "PagedAttention")
      .def(pybind11::init<const std::string&, const std::string&, int>())
      .def("forward", &PagedAttention::forward);
}
