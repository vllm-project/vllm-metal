#include "float8.metal"
#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

// Convert between different precision formats for cache tensors
// This kernel handles conversions like float->fp8, fp8->float, etc.

template <typename SRC_T, typename DST_T>
[[kernel]] void convert_fp8_kernel(
    const device SRC_T *__restrict__ src [[buffer(0)]],
    device DST_T *__restrict__ dst [[buffer(1)]],
    const device float &scale [[buffer(2)]],
    const device uint32_t &num_elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid >= num_elements) {
        return;
    }
    
    // Load source value
    SRC_T src_val = src[gid];
    
    // Convert based on source and destination types
    if constexpr (is_same_v<SRC_T, uchar> && !is_same_v<DST_T, uchar>) {
        // FP8 -> higher precision (dequantization)
        float fp32_val = fp8_e4m3_to_float(src_val) * scale;
        dst[gid] = static_cast<DST_T>(fp32_val);
    } else if constexpr (!is_same_v<SRC_T, uchar> && is_same_v<DST_T, uchar>) {
        // Higher precision -> FP8 (quantization)
        float fp32_val = static_cast<float>(src_val) / scale;
        dst[gid] = float_to_fp8_e4m3(fp32_val);
    } else if constexpr (is_same_v<SRC_T, uchar> && is_same_v<DST_T, uchar>) {
        // FP8 -> FP8 (with rescaling)
        float fp32_val = fp8_e4m3_to_float(src_val) * scale;
        dst[gid] = float_to_fp8_e4m3(fp32_val);
    } else {
        // Regular precision -> regular precision (with scaling)
        float fp32_val = static_cast<float>(src_val) * scale;
        dst[gid] = static_cast<DST_T>(fp32_val);
    }
}

// Instantiate all required combinations
#define INSTANTIATE_CONVERT_FP8(src_type, dst_type) \
    template [[host_name("convert_fp8_" #src_type "_to_" #dst_type)]] \
    [[kernel]] void convert_fp8_kernel<src_type, dst_type>( \
        const device src_type *__restrict__ src [[buffer(0)]], \
        device dst_type *__restrict__ dst [[buffer(1)]], \
        const device float &scale [[buffer(2)]], \
        const device uint32_t &num_elements [[buffer(3)]], \
        uint gid [[thread_position_in_grid]]);

// FP8 to other formats (dequantization)
INSTANTIATE_CONVERT_FP8(uchar, float);
INSTANTIATE_CONVERT_FP8(uchar, half);
INSTANTIATE_CONVERT_FP8(uchar, bfloat16_t);

// Other formats to FP8 (quantization)
INSTANTIATE_CONVERT_FP8(float, uchar);
INSTANTIATE_CONVERT_FP8(half, uchar);
INSTANTIATE_CONVERT_FP8(bfloat16_t, uchar);

// FP8 to FP8 (rescaling)
INSTANTIATE_CONVERT_FP8(uchar, uchar);

// Regular precision conversions with scaling
INSTANTIATE_CONVERT_FP8(float, float);
INSTANTIATE_CONVERT_FP8(float, half);
INSTANTIATE_CONVERT_FP8(float, bfloat16_t);
INSTANTIATE_CONVERT_FP8(half, float);
INSTANTIATE_CONVERT_FP8(half, half);
INSTANTIATE_CONVERT_FP8(half, bfloat16_t);
INSTANTIATE_CONVERT_FP8(bfloat16_t, float);
INSTANTIATE_CONVERT_FP8(bfloat16_t, half);
INSTANTIATE_CONVERT_FP8(bfloat16_t, bfloat16_t);