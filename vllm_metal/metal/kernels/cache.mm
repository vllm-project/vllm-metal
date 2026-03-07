#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <torch/torch.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <string>

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

void swap_blocks(torch::Tensor &src, torch::Tensor &dst,
                 const torch::Tensor &block_mapping) {
  TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const int64_t num_blocks = block_mapping.size(0);

  // Handle different device combinations
  if (src.device().is_mps() && dst.device().is_mps()) {
    // MPS to MPS: Use Metal blit encoder
    @autoreleasepool {
      at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
      TORCH_CHECK(stream, "Failed to get current MPS stream");

      id<MTLCommandBuffer> commandBuffer = stream->commandBuffer();
      TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer");

      dispatch_queue_t serialQueue = stream->queue();

      dispatch_sync(serialQueue, ^{
        id<MTLBlitCommandEncoder> blitEncoder =
            [commandBuffer blitCommandEncoder];
        TORCH_CHECK(blitEncoder, "Failed to create blit command encoder");

        id<MTLBuffer> srcBuf = getMTLBufferStorage(src);
        id<MTLBuffer> dstBuf = getMTLBufferStorage(dst);

        for (int64_t i = 0; i < num_blocks; ++i) {
          int64_t src_block_number = block_mapping[i][0].item<int64_t>();
          int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
          NSUInteger src_offset = src_block_number * block_size_in_bytes;
          NSUInteger dst_offset = dst_block_number * block_size_in_bytes;

          [blitEncoder copyFromBuffer:srcBuf
                         sourceOffset:src_offset
                             toBuffer:dstBuf
                    destinationOffset:dst_offset
                                 size:block_size_in_bytes];
        }

        [blitEncoder endEncoding];
        stream->synchronize(at::mps::SyncType::COMMIT);
      });
    }
  } else {
    // Cross-device transfers (MPS-CPU, CPU-MPS, CPU-CPU): Use PyTorch's copy
    for (int64_t i = 0; i < num_blocks; ++i) {
      int64_t src_block_number = block_mapping[i][0].item<int64_t>();
      int64_t dst_block_number = block_mapping[i][1].item<int64_t>();

      // Copy the entire block
      dst[dst_block_number].copy_(src[src_block_number]);
    }
  }
}

void copy_blocks(const std::vector<torch::Tensor> &key_caches,
                 const std::vector<torch::Tensor> &value_caches,
                 const torch::Tensor &block_mapping) {
  const int64_t num_layers = key_caches.size();
  TORCH_CHECK(num_layers == static_cast<int64_t>(value_caches.size()),
              "key_caches and value_caches must have the same length");
  if (num_layers == 0) {
    return;
  }

  // --- Preconditions --------------------------------------------------
  torch::Device dev = key_caches[0].device();
  TORCH_CHECK(dev.is_mps(), "copy_blocks: expected MPS tensors");

  // Move block_mapping to CPU if it's on MPS
  torch::Tensor block_mapping_cpu = block_mapping;
  if (block_mapping.device().is_mps()) {
    block_mapping_cpu = block_mapping.cpu();
  }

  for (int64_t i = 0; i < num_layers; ++i) {
    TORCH_CHECK(key_caches[i].device() == dev &&
                    value_caches[i].device() == dev,
                "All cache tensors must be on the same MPS device");
    TORCH_CHECK(key_caches[i].dtype() == value_caches[i].dtype(),
                "Key/value cache dtype mismatch at layer ", i);
  }

  const int64_t num_pairs = block_mapping.size(0);
  const int32_t numel_per_block =
      static_cast<int32_t>(key_caches[0][0].numel());

  @autoreleasepool {
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream, "Failed to get current MPS stream");

    id<MTLDevice> device = stream->device();
    id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
    TORCH_CHECK(cmdBuf, "Failed to get command buffer");

    // Construct the full path to the metallib file
    std::string moduleDir = getModuleDirectory();
    std::string metallibPath = moduleDir + "/" + METALLIB_PATH;

    NSString *metallibPathStr =
        [NSString stringWithUTF8String:metallibPath.c_str()];
    NSURL *metallibURL = [NSURL fileURLWithPath:metallibPathStr];
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:metallibURL error:&error];
    if (!lib) {
      NSLog(@"[cache.mm] Failed to load pre-compiled Metal library at %@: %@",
            metallibPathStr, error.localizedDescription);
    }

    // Process each layer separately
    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
      NSString *kernName = nil;
      switch (key_caches[layer_idx].scalar_type()) {
      case torch::kFloat:
        kernName = @"copy_blocks_float";
        break;
      case torch::kHalf:
        kernName = @"copy_blocks_half";
        break;
      case torch::kBFloat16:
        kernName = @"copy_blocks_bfloat16_t";
        break;
      case torch::kUInt8:
        kernName = @"copy_blocks_uchar";
        break;
      default:
        TORCH_CHECK(false, "Unsupported dtype for copy_blocks");
      }

      id<MTLFunction> fn = [lib newFunctionWithName:kernName];
      TORCH_CHECK(fn, "Missing Metal kernel function: ", kernName.UTF8String);

      id<MTLComputePipelineState> pso =
          [device newComputePipelineStateWithFunction:fn error:&error];
      TORCH_CHECK(pso, error.localizedDescription.UTF8String);

      dispatch_queue_t q = stream->queue();
      dispatch_sync(q, ^{
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        TORCH_CHECK(enc, "Failed to create compute encoder");

        [enc setComputePipelineState:pso];

        // Set key and value cache buffers
        [enc setBuffer:getMTLBufferStorage(key_caches[layer_idx])
                offset:key_caches[layer_idx].storage_offset() *
                       key_caches[layer_idx].element_size()
               atIndex:0];
        [enc setBuffer:getMTLBufferStorage(value_caches[layer_idx])
                offset:value_caches[layer_idx].storage_offset() *
                       value_caches[layer_idx].element_size()
               atIndex:1];

        // Set block mapping buffer
        id<MTLBuffer> mappingBuf =
            [device newBufferWithBytes:block_mapping_cpu.data_ptr<int64_t>()
                                length:num_pairs * 2 * sizeof(int64_t)
                               options:MTLResourceStorageModeShared];
        [enc setBuffer:mappingBuf offset:0 atIndex:2];

        // Set numel_per_block as buffer
        id<MTLBuffer> numelBuf =
            [device newBufferWithBytes:&numel_per_block
                                length:sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        [enc setBuffer:numelBuf offset:0 atIndex:3];

        const uint32_t threadsPerThreadgroup =
            std::min<uint32_t>(256, numel_per_block);
        MTLSize tg = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        MTLSize grid = MTLSizeMake(threadsPerThreadgroup * num_pairs, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
      });
    }

    stream->synchronize(at::mps::SyncType::COMMIT);
  }
}

void reshape_and_cache(
    torch::Tensor &key,   // [num_tokens, num_heads, head_size]
    torch::Tensor &value, // [num_tokens, num_heads, head_size]
    torch::Tensor
        &key_cache, // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor
        &value_cache, // [num_blocks, num_heads, head_size,    block_size]
    torch::Tensor &slot_mapping, // [num_tokens]
    const std::string &kv_cache_dtype, torch::Tensor &k_scale,
    torch::Tensor &v_scale) {

  // Determine cache dtype and FP8 usage
  torch::ScalarType cache_dtype = key_cache.scalar_type();
  bool use_fp8_scales = (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3");
  if (use_fp8_scales) {
    TORCH_CHECK(cache_dtype == torch::kUInt8, "FP8 cache requires UInt8 tensor type");
    TORCH_CHECK(k_scale.numel() == 1 && v_scale.numel() == 1, "FP8 scales must be scalars");
    TORCH_CHECK(k_scale.scalar_type() == torch::kFloat32 && v_scale.scalar_type() == torch::kFloat32,
                "FP8 scales must be float32");
  }

  TORCH_CHECK(key.device().is_mps() && value.device().is_mps() &&
                  key_cache.device().is_mps() && value_cache.device().is_mps(),
              "All tensors must be on MPS device");

  // Move slot_mapping to CPU if it's on MPS
  torch::Tensor slot_mapping_cpu = slot_mapping;
  if (slot_mapping.device().is_mps()) {
    slot_mapping_cpu = slot_mapping.cpu();
  }

  const int64_t num_tokens = key.size(0);
  const int64_t num_heads = key.size(1);
  const int64_t head_size = key.size(2);
  const int64_t block_size = key_cache.size(3);
  const int64_t x = key_cache.size(4);

  const int32_t key_stride = key.stride(0);
  const int32_t value_stride = value.stride(0);

  @autoreleasepool {
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream, "Failed to get current MPS stream");

    id<MTLDevice> device = stream->device();
    id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
    TORCH_CHECK(cmdBuf, "Failed to get command buffer");

    // Construct the full path to the metallib file
    std::string moduleDir = getModuleDirectory();
    std::string metallibPath = moduleDir + "/" + METALLIB_PATH;

    NSString *metallibPathStr =
        [NSString stringWithUTF8String:metallibPath.c_str()];
    NSURL *metallibURL = [NSURL fileURLWithPath:metallibPathStr];
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:metallibURL error:&error];
    if (!lib) {
      NSLog(@"[cache.mm] Failed to load pre-compiled Metal library at %@: %@",
            metallibPathStr, error.localizedDescription);
    }

    NSString *kernName = nil;
    std::string kv_dtype_str, cache_dtype_str;
    
    // Get KV dtype string
    switch (key.scalar_type()) {
    case torch::kFloat:
      kv_dtype_str = "float";
      break;
    case torch::kHalf:
      kv_dtype_str = "half";
      break;
    case torch::kBFloat16:
      kv_dtype_str = "bfloat16_t";
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for reshape_and_cache");
    }
    
    // Get cache dtype string
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
      TORCH_CHECK(false, "Unsupported cache dtype for reshape_and_cache");
    }
    
    std::string kernName_str = "reshape_and_cache_kv_" + kv_dtype_str + "_cache_" + cache_dtype_str;
    kernName = [NSString stringWithUTF8String:kernName_str.c_str()];

    // Create function constants for FP8 support
    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&use_fp8_scales type:MTLDataTypeBool atIndex:10];
    
    id<MTLFunction> fn = [lib newFunctionWithName:kernName constantValues:constants error:&error];
    TORCH_CHECK(fn, "Missing Metal kernel function: ", kernName.UTF8String, 
                error ? [NSString stringWithFormat:@": %@", error.localizedDescription].UTF8String : "");

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&error];
    TORCH_CHECK(pso, error.localizedDescription.UTF8String);

    dispatch_queue_t q = stream->queue();
    dispatch_sync(q, ^{
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      TORCH_CHECK(enc, "Failed to create compute encoder");

      [enc setComputePipelineState:pso];

      // Set tensor buffers
      [enc setBuffer:getMTLBufferStorage(key)
              offset:key.storage_offset() * key.element_size()
             atIndex:0];
      [enc setBuffer:getMTLBufferStorage(value)
              offset:value.storage_offset() * value.element_size()
             atIndex:1];
      [enc setBuffer:getMTLBufferStorage(key_cache)
              offset:key_cache.storage_offset() * key_cache.element_size()
             atIndex:2];
      [enc setBuffer:getMTLBufferStorage(value_cache)
              offset:value_cache.storage_offset() * value_cache.element_size()
             atIndex:3];

      // Set slot mapping buffer
      id<MTLBuffer> slotMappingBuf =
          [device newBufferWithBytes:slot_mapping_cpu.data_ptr<int64_t>()
                              length:num_tokens * sizeof(int64_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:slotMappingBuf offset:0 atIndex:4];

      // k_scale and v_scale buffers (for FP8)
      if (use_fp8_scales) {
        [enc setBuffer:getMTLBufferStorage(k_scale)
                offset:k_scale.storage_offset() * k_scale.element_size()
               atIndex:5];
        [enc setBuffer:getMTLBufferStorage(v_scale)
                offset:v_scale.storage_offset() * v_scale.element_size()
               atIndex:6];
      } else {
        // For non-FP8, we still need to increment buffer indices
        // The Metal kernel expects buffers at indices 5 and 6 even if unused
      }

      // Set parameters as individual buffers (matching mistralrs pattern)
      id<MTLBuffer> keyStrideBuf =
          [device newBufferWithBytes:&key_stride
                              length:sizeof(int32_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:keyStrideBuf offset:0 atIndex:7];

      id<MTLBuffer> valueStrideBuf =
          [device newBufferWithBytes:&value_stride
                              length:sizeof(int32_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:valueStrideBuf offset:0 atIndex:8];

      const int32_t num_heads_i32 = static_cast<int32_t>(num_heads);
      id<MTLBuffer> numHeadsBuf =
          [device newBufferWithBytes:&num_heads_i32
                              length:sizeof(int32_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:numHeadsBuf offset:0 atIndex:9];

      const int32_t head_size_i32 = static_cast<int32_t>(head_size);
      id<MTLBuffer> headSizeBuf =
          [device newBufferWithBytes:&head_size_i32
                              length:sizeof(int32_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:headSizeBuf offset:0 atIndex:10];

      const int32_t block_size_i32 = static_cast<int32_t>(block_size);
      id<MTLBuffer> blockSizeBuf =
          [device newBufferWithBytes:&block_size_i32
                              length:sizeof(int32_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:blockSizeBuf offset:0 atIndex:11];

      const int32_t x_i32 = static_cast<int32_t>(x);
      id<MTLBuffer> xBuf =
          [device newBufferWithBytes:&x_i32
                              length:sizeof(int32_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:xBuf offset:0 atIndex:12];

      const uint64_t threads_per_threadgroup =
          std::min<uint64_t>(512, num_heads * head_size);
      MTLSize tg = MTLSizeMake(threads_per_threadgroup, 1, 1);
      MTLSize grid = MTLSizeMake(num_tokens, 1, 1);

      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
    });

    stream->synchronize(at::mps::SyncType::COMMIT);
  }
}

void reshape_and_cache_flash(
    torch::Tensor &key,       // [num_tokens, num_heads, head_size]
    torch::Tensor &value,     // [num_tokens, num_heads, head_size]
    torch::Tensor &key_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor
        &value_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping, // [num_tokens]
    const std::string &kv_cache_dtype, torch::Tensor &k_scale,
    torch::Tensor &v_scale) {

  TORCH_CHECK(key.device().is_mps() && value.device().is_mps() &&
                  key_cache.device().is_mps() && value_cache.device().is_mps(),
              "All tensors must be on MPS device");

  // Move slot_mapping to CPU if it's on MPS
  torch::Tensor slot_mapping_cpu = slot_mapping;
  if (slot_mapping.device().is_mps()) {
    slot_mapping_cpu = slot_mapping.cpu();
  }

  const int64_t num_tokens = key.size(0);
  const int64_t num_heads = key.size(1);
  const int64_t head_size = key.size(2);
  const int64_t block_size = key_cache.size(1);

  const int32_t key_stride = key.stride(0);
  const int32_t value_stride = value.stride(0);

  @autoreleasepool {
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream, "Failed to get current MPS stream");

    id<MTLDevice> device = stream->device();
    id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
    TORCH_CHECK(cmdBuf, "Failed to get command buffer");

    // Construct the full path to the metallib file
    std::string moduleDir = getModuleDirectory();
    std::string metallibPath = moduleDir + "/" + METALLIB_PATH;

    NSString *metallibPathStr =
        [NSString stringWithUTF8String:metallibPath.c_str()];
    NSURL *metallibURL = [NSURL fileURLWithPath:metallibPathStr];
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:metallibURL error:&error];
    if (!lib) {
      NSLog(@"[cache.mm] Failed to load pre-compiled Metal library at %@: %@",
            metallibPathStr, error.localizedDescription);
    }

    NSString *kernName = nil;
    switch (key.scalar_type()) {
    case torch::kFloat:
      kernName = @"reshape_and_cache_flash_float";
      break;
    case torch::kHalf:
      kernName = @"reshape_and_cache_flash_half";
      break;
    case torch::kBFloat16:
      kernName = @"reshape_and_cache_flash_bfloat16_t";
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for reshape_and_cache_flash");
    }

    id<MTLFunction> fn = [lib newFunctionWithName:kernName];
    TORCH_CHECK(fn, "Missing Metal kernel function: ", kernName.UTF8String);

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&error];
    TORCH_CHECK(pso, error.localizedDescription.UTF8String);

    dispatch_queue_t q = stream->queue();
    dispatch_sync(q, ^{
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      TORCH_CHECK(enc, "Failed to create compute encoder");

      [enc setComputePipelineState:pso];

      // Set tensor buffers
      [enc setBuffer:getMTLBufferStorage(key)
              offset:key.storage_offset() * key.element_size()
             atIndex:0];
      [enc setBuffer:getMTLBufferStorage(value)
              offset:value.storage_offset() * value.element_size()
             atIndex:1];
      [enc setBuffer:getMTLBufferStorage(key_cache)
              offset:key_cache.storage_offset() * key_cache.element_size()
             atIndex:2];
      [enc setBuffer:getMTLBufferStorage(value_cache)
              offset:value_cache.storage_offset() * value_cache.element_size()
             atIndex:3];

      // Set slot mapping buffer
      id<MTLBuffer> slotMappingBuf =
          [device newBufferWithBytes:slot_mapping_cpu.data_ptr<int64_t>()
                              length:num_tokens * sizeof(int64_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:slotMappingBuf offset:0 atIndex:4];

      // Set parameters as individual buffers
      id<MTLBuffer> keyStrideBuf =
          [device newBufferWithBytes:&key_stride
                              length:sizeof(int32_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:keyStrideBuf offset:0 atIndex:5];

      id<MTLBuffer> valueStrideBuf =
          [device newBufferWithBytes:&value_stride
                              length:sizeof(int32_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:valueStrideBuf offset:0 atIndex:6];

      const int32_t num_heads_i32 = static_cast<int32_t>(num_heads);
      id<MTLBuffer> numHeadsBuf =
          [device newBufferWithBytes:&num_heads_i32
                              length:sizeof(int32_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:numHeadsBuf offset:0 atIndex:7];

      const int32_t head_size_i32 = static_cast<int32_t>(head_size);
      id<MTLBuffer> headSizeBuf =
          [device newBufferWithBytes:&head_size_i32
                              length:sizeof(int32_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:headSizeBuf offset:0 atIndex:8];

      const int32_t block_size_i32 = static_cast<int32_t>(block_size);
      id<MTLBuffer> blockSizeBuf =
          [device newBufferWithBytes:&block_size_i32
                              length:sizeof(int32_t)
                             options:MTLResourceStorageModeShared];
      [enc setBuffer:blockSizeBuf offset:0 atIndex:9];

      const uint64_t threads_per_threadgroup =
          std::min<uint64_t>(512, num_heads * head_size);
      MTLSize tg = MTLSizeMake(threads_per_threadgroup, 1, 1);
      MTLSize grid = MTLSizeMake(num_tokens, 1, 1);

      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
    });

    stream->synchronize(at::mps::SyncType::COMMIT);
  }
}