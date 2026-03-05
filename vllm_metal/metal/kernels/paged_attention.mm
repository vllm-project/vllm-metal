#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <torch/torch.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <string>
#include <vector>

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static std::string getModuleDirectory() {
  Dl_info dl_info;
  if (dladdr((void *)getModuleDirectory, &dl_info)) {
    std::string path(dl_info.dli_fname);
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
      return path.substr(0, pos);
    }
  }
  return ".";
}

// Helper function to get kernel name based on dtype and parameters
static std::string getKernelName(const std::string &base_name,
                                 torch::ScalarType dtype, 
                                 torch::ScalarType cache_dtype,
                                 int head_size,
                                 int block_size, int num_threads,
                                 int num_simd_lanes, int partition_size = 0) {
  std::string dtype_str;
  switch (dtype) {
  case torch::kFloat:
    dtype_str = "float";
    break;
  case torch::kHalf:
    dtype_str = "half";
    break;
  case torch::kBFloat16:
    dtype_str = "bfloat16_t";
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype for paged attention: ", dtype);
  }

  std::string cache_dtype_str;
  switch (cache_dtype) {
  case torch::kFloat:
    cache_dtype_str = "float";
    break;
  case torch::kHalf:
    cache_dtype_str = "half";
    break;
  case torch::kBFloat16:
    cache_dtype_str = "bfloat16_t";
    break;
  case torch::kUInt8:
    cache_dtype_str = "uchar";
    break;
  default:
    TORCH_CHECK(false, "Unsupported cache dtype for paged attention: ", cache_dtype);
  }

  std::string kernel_name =
      base_name + "_" + dtype_str + "_cache_" + cache_dtype_str + "_hs" + std::to_string(head_size) + "_bs" +
      std::to_string(block_size) + "_nt" + std::to_string(num_threads) +
      "_nsl" + std::to_string(num_simd_lanes);

  if (partition_size >= 0) {
    kernel_name += "_ps" + std::to_string(partition_size);
  }

  return kernel_name;
}

// Helper function to calculate shared memory size
static size_t calculateSharedMemorySize(int max_seq_len, int head_size,
                                        int num_threads, int num_simd_lanes) {
  // Logits storage: max_seq_len * sizeof(float)
  size_t logits_size = max_seq_len * sizeof(float);

  // Reduction workspace: 2 * (num_threads / num_simd_lanes) * sizeof(float)
  size_t reduction_size = 2 * (num_threads / num_simd_lanes) * sizeof(float);

  // Output workspace for cross-warp reduction: head_size * sizeof(float)
  size_t output_size = head_size * sizeof(float);
  return std::max(logits_size + reduction_size, output_size);
}

// Helper function to get supported configurations
static bool isValidConfiguration(int head_size, int block_size) {
  // Supported head sizes from the Metal kernel instantiations
  std::vector<int> supported_head_sizes = {32,  64,  80,  96, 112,
                                           120, 128, 192, 256};
  std::vector<int> supported_block_sizes = {8, 16, 32};

  return std::find(supported_head_sizes.begin(), supported_head_sizes.end(),
                   head_size) != supported_head_sizes.end() &&
         std::find(supported_block_sizes.begin(), supported_block_sizes.end(),
                   block_size) != supported_block_sizes.end();
}

void paged_attention_v1(
    torch::Tensor &out,   // [num_seqs, num_heads, head_size]
    torch::Tensor &query, // [num_seqs, num_heads, head_size]
    torch::Tensor
        &key_cache, // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor
        &value_cache,     // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads, // [num_heads]
    double scale,
    torch::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor &seq_lens,     // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::Tensor> &alibi_slopes,
    const std::string &kv_cache_dtype, torch::Tensor &k_scale,
    torch::Tensor &v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);

  // Validate block sparse is not supported yet
  // TODO: support blocksparse.
  TORCH_CHECK(
      !is_block_sparse,
      "Block sparse attention is not yet supported in Metal implementation");
  
  // Determine cache dtype based on kv_cache_dtype
  torch::ScalarType cache_dtype = key_cache.scalar_type();
  bool use_fp8_scales = (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3");
  if (use_fp8_scales) {
    TORCH_CHECK(cache_dtype == torch::kUInt8, "FP8 cache requires UInt8 tensor type");
    TORCH_CHECK(k_scale.numel() == 1 && v_scale.numel() == 1, "FP8 scales must be scalars");
    TORCH_CHECK(k_scale.scalar_type() == torch::kFloat32 && v_scale.scalar_type() == torch::kFloat32,
                "FP8 scales must be float32");
  }

  // Validate input tensors
  TORCH_CHECK(out.device().is_mps() && query.device().is_mps() &&
                  key_cache.device().is_mps() &&
                  value_cache.device().is_mps() &&
                  block_tables.device().is_mps() && seq_lens.device().is_mps(),
              "All tensors must be on MPS device");

  const int64_t num_seqs = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_size = query.size(2);
  const int64_t max_num_blocks_per_seq = block_tables.size(1);

  // Validate configurations
  TORCH_CHECK(isValidConfiguration(head_size, block_size),
              "Unsupported head_size/block_size combination: ", head_size, "/",
              block_size);

  // For v1, no partitioning - each sequence processed by one threadgroup
  // Kernel configuration (should match the instantiated kernels)
  const int num_threads = 256;
  const int num_simd_lanes = 32;
  const int partition_size = 0; // v1 doesn't use partitioning

  // Calculate shared memory requirements (from mistral.rs)
  const int num_simds = num_threads / num_simd_lanes;
  const int padded_max_context_len =
      ((max_seq_len + block_size - 1) / block_size) * block_size;
  const int logits_size = padded_max_context_len * sizeof(float);
  const int outputs_size = (num_simds / 2) * head_size * sizeof(float);
  const size_t shared_memory_size = std::max(logits_size, outputs_size);

  // Get kernel name - v1 kernels have partition_size=0 in their name
  std::string kernel_name =
      getKernelName("paged_attention", query.scalar_type(), cache_dtype, head_size,
                    block_size, num_threads, num_simd_lanes, partition_size);

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    // Load Metal library
    std::string moduleDir = getModuleDirectory();
    std::string metallibPath = moduleDir + "/" + METALLIB_PATH;
    NSString *metallibPathStr =
        [NSString stringWithUTF8String:metallibPath.c_str()];
    NSURL *metallibURL = [NSURL fileURLWithPath:metallibPathStr];
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:metallibURL error:&error];
    TORCH_CHECK(lib, "Failed to load Metal library at ", metallibPath, ": ",
                error ? error.localizedDescription.UTF8String
                      : "unknown error");

    // Create function constants for conditional compilation
    MTLFunctionConstantValues *constants =
        [[MTLFunctionConstantValues alloc] init];
    bool use_partitioning = false;
    bool use_alibi = alibi_slopes.has_value();
    [constants setConstantValue:&use_partitioning
                           type:MTLDataTypeBool
                        atIndex:10];
    [constants setConstantValue:&use_alibi type:MTLDataTypeBool atIndex:20];
    [constants setConstantValue:&use_fp8_scales type:MTLDataTypeBool atIndex:30];

    NSString *kernelNameStr =
        [NSString stringWithUTF8String:kernel_name.c_str()];
    id<MTLFunction> fn = [lib newFunctionWithName:kernelNameStr
                                   constantValues:constants
                                            error:&error];
    TORCH_CHECK(
        fn, "Failed to create Metal function '", kernel_name,
        "': ", error ? error.localizedDescription.UTF8String : "unknown error");

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&error];
    TORCH_CHECK(pso, "Failed to create compute pipeline state: ",
                error ? error.localizedDescription.UTF8String
                      : "unknown error");

    // Setup command buffer and encoder
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream, "Failed to get current MPS stream");

    id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
    TORCH_CHECK(cmdBuf, "Failed to get MPS command buffer");

    dispatch_queue_t q = stream->queue();
    dispatch_sync(q, ^{
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      TORCH_CHECK(enc, "Failed to create compute command encoder");

      [enc setComputePipelineState:pso];

      // Set threadgroup memory
      [enc setThreadgroupMemoryLength:shared_memory_size atIndex:0];

      // Buffer arguments (matching the Metal kernel signature)
      int buffer_idx = 0;

      // Skip exp_sums and max_logits for v1 (buffers 0, 1)
      buffer_idx = 2;

      // out buffer
      [enc setBuffer:getMTLBufferStorage(out)
              offset:out.storage_offset() * out.element_size()
             atIndex:buffer_idx++];

      // query buffer
      [enc setBuffer:getMTLBufferStorage(query)
              offset:query.storage_offset() * query.element_size()
             atIndex:buffer_idx++];

      // key_cache buffer
      [enc setBuffer:getMTLBufferStorage(key_cache)
              offset:key_cache.storage_offset() * key_cache.element_size()
             atIndex:buffer_idx++];

      // value_cache buffer
      [enc setBuffer:getMTLBufferStorage(value_cache)
              offset:value_cache.storage_offset() * value_cache.element_size()
             atIndex:buffer_idx++];

      // k_scale and v_scale (for FP8)
      if (use_fp8_scales) {
        [enc setBuffer:getMTLBufferStorage(k_scale)
                offset:k_scale.storage_offset() * k_scale.element_size()
               atIndex:buffer_idx++];
        [enc setBuffer:getMTLBufferStorage(v_scale)
                offset:v_scale.storage_offset() * v_scale.element_size()
               atIndex:buffer_idx++];
      } else {
        buffer_idx += 2; // Skip k_scale and v_scale buffer slots
      }

      // num_kv_heads
      int32_t num_kv_heads_i32 = static_cast<int32_t>(num_kv_heads);
      [enc setBytes:&num_kv_heads_i32
             length:sizeof(int32_t)
            atIndex:buffer_idx++];

      // scale
      float scale_f32 = static_cast<float>(scale);
      [enc setBytes:&scale_f32 length:sizeof(float) atIndex:buffer_idx++];

      // softcapping (default to 1.0 for no capping)
      float softcapping = 1.0f;
      [enc setBytes:&softcapping length:sizeof(float) atIndex:buffer_idx++];

      // block_tables buffer
      [enc setBuffer:getMTLBufferStorage(block_tables)
              offset:block_tables.storage_offset() * block_tables.element_size()
             atIndex:buffer_idx++];

      // seq_lens buffer (context_lens in kernel)
      [enc setBuffer:getMTLBufferStorage(seq_lens)
              offset:seq_lens.storage_offset() * seq_lens.element_size()
             atIndex:buffer_idx++];

      // max_num_blocks_per_seq
      int32_t max_num_blocks_per_seq_i32 =
          static_cast<int32_t>(max_num_blocks_per_seq);
      [enc setBytes:&max_num_blocks_per_seq_i32
             length:sizeof(int32_t)
            atIndex:buffer_idx++];

      // alibi_slopes (optional)
      if (use_alibi) {
        [enc setBuffer:getMTLBufferStorage(alibi_slopes.value())
                offset:alibi_slopes.value().storage_offset() *
                       alibi_slopes.value().element_size()
               atIndex:buffer_idx++];
      } else {
        buffer_idx++; // Skip this buffer slot
      }

      // Stride parameters
      int32_t q_stride = static_cast<int32_t>(query.stride(0));
      int32_t kv_block_stride = static_cast<int32_t>(key_cache.stride(0));
      int32_t kv_head_stride = static_cast<int32_t>(key_cache.stride(1));

      [enc setBytes:&q_stride length:sizeof(int32_t) atIndex:buffer_idx++];
      [enc setBytes:&kv_block_stride
             length:sizeof(int32_t)
            atIndex:buffer_idx++];
      [enc setBytes:&kv_head_stride
             length:sizeof(int32_t)
            atIndex:buffer_idx++];

      // Dispatch configuration
      // Grid: (num_heads, num_seqs, 1) - no partitioning for v1
      MTLSize grid = MTLSizeMake(num_heads, num_seqs, 1);
      MTLSize threadgroup = MTLSizeMake(num_threads, 1, 1);

      [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
      [enc endEncoding];

      stream->synchronize(at::mps::SyncType::COMMIT);
    });
  }
}

void paged_attention_v2(
    torch::Tensor &out,        // [num_seqs, num_heads, head_size]
    torch::Tensor &exp_sums,   // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor &max_logits, // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor
        &tmp_out, // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor &query, // [num_seqs, num_heads, head_size]
    torch::Tensor
        &key_cache, // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor
        &value_cache,     // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads, // [num_heads]
    double scale,
    torch::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor &seq_lens,     // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::Tensor> &alibi_slopes,
    const std::string &kv_cache_dtype, torch::Tensor &k_scale,
    torch::Tensor &v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);

  // TODO: support blocksparse.
  // Validate block sparse is not supported yet
  TORCH_CHECK(
      !is_block_sparse,
      "Block sparse attention is not yet supported in Metal implementation");
  
  // Determine cache dtype based on kv_cache_dtype
  torch::ScalarType cache_dtype = key_cache.scalar_type();
  bool use_fp8_scales = (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3");
  if (use_fp8_scales) {
    TORCH_CHECK(cache_dtype == torch::kUInt8, "FP8 cache requires UInt8 tensor type");
    TORCH_CHECK(k_scale.numel() == 1 && v_scale.numel() == 1, "FP8 scales must be scalars");
    TORCH_CHECK(k_scale.scalar_type() == torch::kFloat32 && v_scale.scalar_type() == torch::kFloat32,
                "FP8 scales must be float32");
  }

  // Validate input tensors
  TORCH_CHECK(out.device().is_mps() && query.device().is_mps() &&
                  key_cache.device().is_mps() &&
                  value_cache.device().is_mps() && exp_sums.device().is_mps() &&
                  max_logits.device().is_mps() && tmp_out.device().is_mps() &&
                  block_tables.device().is_mps() && seq_lens.device().is_mps(),
              "All tensors must be on MPS device");

  const int64_t num_seqs = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_size = query.size(2);
  const int64_t max_num_blocks_per_seq = block_tables.size(1);
  const int64_t max_num_partitions = exp_sums.size(2);

  // Validate configurations
  TORCH_CHECK(isValidConfiguration(head_size, block_size),
              "Unsupported head_size/block_size combination: ", head_size, "/",
              block_size);

  // For v2, use partitioning (matching the instantiated kernels)
  const int num_threads = 256;
  const int num_simd_lanes = 32;
  const int partition_size = 512; // v2 uses partitioning

  // Calculate shared memory requirements (from mistral.rs)
  const int num_simds = num_threads / num_simd_lanes;
  const int logits_size = partition_size * sizeof(float);
  const int outputs_size = (num_simds / 2) * head_size * sizeof(float);
  const size_t shared_memory_size = std::max(logits_size, outputs_size);

  // Get kernel names
  std::string kernel_name =
      getKernelName("paged_attention", query.scalar_type(), cache_dtype, head_size,
                    block_size, num_threads, num_simd_lanes, partition_size);
  // Reduce kernel doesn't have block_size in its name
  std::string reduce_kernel_name = "paged_attention_v2_reduce";
  switch (query.scalar_type()) {
  case torch::kFloat:
    reduce_kernel_name += "_float";
    break;
  case torch::kHalf:
    reduce_kernel_name += "_half";
    break;
  case torch::kBFloat16:
    reduce_kernel_name += "_bfloat16_t";
    break;
  default:
    TORCH_CHECK(false,
                "Unsupported dtype for paged attention: ", query.scalar_type());
  }
  reduce_kernel_name += "_hs" + std::to_string(head_size) + "_nt" +
                        std::to_string(num_threads) + "_nsl" +
                        std::to_string(num_simd_lanes) + "_ps" +
                        std::to_string(partition_size);

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    // Load Metal library
    std::string moduleDir = getModuleDirectory();
    std::string metallibPath = moduleDir + "/" + METALLIB_PATH;
    NSString *metallibPathStr =
        [NSString stringWithUTF8String:metallibPath.c_str()];
    NSURL *metallibURL = [NSURL fileURLWithPath:metallibPathStr];
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:metallibURL error:&error];
    TORCH_CHECK(lib, "Failed to load Metal library at ", metallibPath, ": ",
                error ? error.localizedDescription.UTF8String
                      : "unknown error");

    // Setup command buffer and queue
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream, "Failed to get current MPS stream");

    id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
    TORCH_CHECK(cmdBuf, "Failed to get MPS command buffer");

    dispatch_queue_t q = stream->queue();
    dispatch_sync(q, ^{
      // ==================================================================
      // Phase 1: Main paged attention kernel with partitioning
      // ==================================================================

      // Create function constants for main kernel
      MTLFunctionConstantValues *mainConstants =
          [[MTLFunctionConstantValues alloc] init];
      bool use_partitioning = true;
      bool use_alibi = alibi_slopes.has_value();
      [mainConstants setConstantValue:&use_partitioning
                                 type:MTLDataTypeBool
                              atIndex:10];
      [mainConstants setConstantValue:&use_alibi
                                 type:MTLDataTypeBool
                              atIndex:20];
      [mainConstants setConstantValue:&use_fp8_scales
                                 type:MTLDataTypeBool
                              atIndex:30];

      NSString *kernelNameStr =
          [NSString stringWithUTF8String:kernel_name.c_str()];
      NSError *mainError = nil;
      id<MTLFunction> mainFn = [lib newFunctionWithName:kernelNameStr
                                         constantValues:mainConstants
                                                  error:&mainError];
      TORCH_CHECK(mainFn, "Failed to create Metal function '", kernel_name,
                  "': ",
                  mainError ? mainError.localizedDescription.UTF8String
                            : "unknown error");

      NSError *psoError = nil;
      id<MTLComputePipelineState> mainPso =
          [device newComputePipelineStateWithFunction:mainFn error:&psoError];
      TORCH_CHECK(mainPso, "Failed to create compute pipeline state: ",
                  psoError ? psoError.localizedDescription.UTF8String
                           : "unknown error");

      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      TORCH_CHECK(enc, "Failed to create compute command encoder");

      [enc setComputePipelineState:mainPso];
      [enc setThreadgroupMemoryLength:shared_memory_size atIndex:0];

      // Set buffers for main kernel
      int buffer_idx = 0;

      // exp_sums buffer
      [enc setBuffer:getMTLBufferStorage(exp_sums)
              offset:exp_sums.storage_offset() * exp_sums.element_size()
             atIndex:buffer_idx++];

      // max_logits buffer
      [enc setBuffer:getMTLBufferStorage(max_logits)
              offset:max_logits.storage_offset() * max_logits.element_size()
             atIndex:buffer_idx++];

      // tmp_out buffer
      [enc setBuffer:getMTLBufferStorage(tmp_out)
              offset:tmp_out.storage_offset() * tmp_out.element_size()
             atIndex:buffer_idx++];

      // query buffer
      [enc setBuffer:getMTLBufferStorage(query)
              offset:query.storage_offset() * query.element_size()
             atIndex:buffer_idx++];

      // key_cache buffer
      [enc setBuffer:getMTLBufferStorage(key_cache)
              offset:key_cache.storage_offset() * key_cache.element_size()
             atIndex:buffer_idx++];

      // value_cache buffer
      [enc setBuffer:getMTLBufferStorage(value_cache)
              offset:value_cache.storage_offset() * value_cache.element_size()
             atIndex:buffer_idx++];

      // k_scale and v_scale (for FP8)
      if (use_fp8_scales) {
        [enc setBuffer:getMTLBufferStorage(k_scale)
                offset:k_scale.storage_offset() * k_scale.element_size()
               atIndex:buffer_idx++];
        [enc setBuffer:getMTLBufferStorage(v_scale)
                offset:v_scale.storage_offset() * v_scale.element_size()
               atIndex:buffer_idx++];
      } else {
        buffer_idx += 2; // Skip k_scale and v_scale buffer slots
      }

      // num_kv_heads
      int32_t num_kv_heads_i32 = static_cast<int32_t>(num_kv_heads);
      [enc setBytes:&num_kv_heads_i32
             length:sizeof(int32_t)
            atIndex:buffer_idx++];

      // scale
      float scale_f32 = static_cast<float>(scale);
      [enc setBytes:&scale_f32 length:sizeof(float) atIndex:buffer_idx++];

      // softcapping (default to 1.0 for no capping)
      float softcapping = 1.0f;
      [enc setBytes:&softcapping length:sizeof(float) atIndex:buffer_idx++];

      // block_tables buffer
      [enc setBuffer:getMTLBufferStorage(block_tables)
              offset:block_tables.storage_offset() * block_tables.element_size()
             atIndex:buffer_idx++];

      // seq_lens buffer (context_lens in kernel)
      [enc setBuffer:getMTLBufferStorage(seq_lens)
              offset:seq_lens.storage_offset() * seq_lens.element_size()
             atIndex:buffer_idx++];

      // max_num_blocks_per_seq
      int32_t max_num_blocks_per_seq_i32 =
          static_cast<int32_t>(max_num_blocks_per_seq);
      [enc setBytes:&max_num_blocks_per_seq_i32
             length:sizeof(int32_t)
            atIndex:buffer_idx++];

      // alibi_slopes (optional)
      if (use_alibi) {
        [enc setBuffer:getMTLBufferStorage(alibi_slopes.value())
                offset:alibi_slopes.value().storage_offset() *
                       alibi_slopes.value().element_size()
               atIndex:buffer_idx++];
      } else {
        buffer_idx++; // Skip this buffer slot
      }

      // Stride parameters
      int32_t q_stride = static_cast<int32_t>(query.stride(0));
      int32_t kv_block_stride = static_cast<int32_t>(key_cache.stride(0));
      int32_t kv_head_stride = static_cast<int32_t>(key_cache.stride(1));

      [enc setBytes:&q_stride length:sizeof(int32_t) atIndex:buffer_idx++];
      [enc setBytes:&kv_block_stride
             length:sizeof(int32_t)
            atIndex:buffer_idx++];
      [enc setBytes:&kv_head_stride
             length:sizeof(int32_t)
            atIndex:buffer_idx++];

      // Dispatch main kernel
      // Grid: (num_heads, num_seqs, max_num_partitions) - with partitioning for
      // v2
      MTLSize mainGrid = MTLSizeMake(num_heads, num_seqs, max_num_partitions);
      MTLSize mainThreadgroup = MTLSizeMake(num_threads, 1, 1);

      [enc dispatchThreadgroups:mainGrid threadsPerThreadgroup:mainThreadgroup];
      [enc endEncoding];

      // ==================================================================
      // Phase 2: Reduction kernel to combine partitions
      // ==================================================================

      // Create reduction kernel
      NSString *reduceKernelNameStr =
          [NSString stringWithUTF8String:reduce_kernel_name.c_str()];
      id<MTLFunction> reduceFn = [lib newFunctionWithName:reduceKernelNameStr];
      TORCH_CHECK(reduceFn, "Failed to create Metal function '",
                  reduce_kernel_name, "'");

      NSError *reducePsoError = nil;
      id<MTLComputePipelineState> reducePso =
          [device newComputePipelineStateWithFunction:reduceFn
                                                error:&reducePsoError];
      TORCH_CHECK(
          reducePso, "Failed to create compute pipeline state for reduction: ",
          reducePsoError ? reducePsoError.localizedDescription.UTF8String
                         : "unknown error");

      // Calculate shared memory for reduction kernel
      size_t reduce_shared_memory_size =
          max_num_partitions * sizeof(float) * 2; // max_logits + exp_sums

      id<MTLComputeCommandEncoder> reduceEnc = [cmdBuf computeCommandEncoder];
      TORCH_CHECK(reduceEnc,
                  "Failed to create compute command encoder for reduction");

      [reduceEnc setComputePipelineState:reducePso];
      [reduceEnc setThreadgroupMemoryLength:reduce_shared_memory_size
                                    atIndex:0];

      // Set buffers for reduction kernel
      buffer_idx = 0;

      // out buffer (final output)
      [reduceEnc setBuffer:getMTLBufferStorage(out)
                    offset:out.storage_offset() * out.element_size()
                   atIndex:buffer_idx++];

      // exp_sums buffer
      [reduceEnc setBuffer:getMTLBufferStorage(exp_sums)
                    offset:exp_sums.storage_offset() * exp_sums.element_size()
                   atIndex:buffer_idx++];

      // max_logits buffer
      [reduceEnc
          setBuffer:getMTLBufferStorage(max_logits)
             offset:max_logits.storage_offset() * max_logits.element_size()
            atIndex:buffer_idx++];

      // tmp_out buffer
      [reduceEnc setBuffer:getMTLBufferStorage(tmp_out)
                    offset:tmp_out.storage_offset() * tmp_out.element_size()
                   atIndex:buffer_idx++];

      // seq_lens buffer (context_lens in kernel)
      [reduceEnc setBuffer:getMTLBufferStorage(seq_lens)
                    offset:seq_lens.storage_offset() * seq_lens.element_size()
                   atIndex:buffer_idx++];

      // max_num_partitions
      int32_t max_num_partitions_i32 = static_cast<int32_t>(max_num_partitions);
      [reduceEnc setBytes:&max_num_partitions_i32
                   length:sizeof(int32_t)
                  atIndex:buffer_idx++];

      // Dispatch reduction kernel
      // Grid: (num_heads, num_seqs) - one threadgroup per sequence/head
      // combination
      MTLSize reduceGrid = MTLSizeMake(num_heads, num_seqs, 1);
      MTLSize reduceThreadgroup = MTLSizeMake(num_threads, 1, 1);

      [reduceEnc dispatchThreadgroups:reduceGrid
                threadsPerThreadgroup:reduceThreadgroup];
      [reduceEnc endEncoding];

      stream->synchronize(at::mps::SyncType::COMMIT);
    });
  }
}