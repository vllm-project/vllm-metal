// SPDX-License-Identifier: Apache-2.0
// Raw GGUF Q8_0 quantized-weight kernels.

constant int GGUF_QK8_0 = 32;
constant int GGUF_Q8_0_BLOCK_BYTES = 34;  // half d + int8 qs[32]
constant int GGUF_Q8_1_BLOCK_BYTES = 36;  // half2(d, sum) + int8 qs[32]

template <typename T>
[[kernel]] void gguf_q8_1_quantize(
    const device T* __restrict__ x [[buffer(0)]],
    device uchar* __restrict__ qx [[buffer(1)]],
    constant int& input_dims [[buffer(2)]],
    constant int& padded_blocks [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int block = int(tgid.x);
  const int token = int(tgid.y);
  const int col = block * GGUF_QK8_0 + int(lane);

  const float xi = (col < input_dims)
      ? static_cast<float>(x[token * input_dims + col])
      : 0.0f;
  const float amax = simd_max(fabs(xi));
  const float sum = simd_sum(xi);
  const float d = amax / 127.0f;
  const float qf = (amax == 0.0f) ? 0.0f : round(xi / d);
  const int qi = int(clamp(qf, -127.0f, 127.0f));

  const int base = (token * padded_blocks + block) * GGUF_Q8_1_BLOCK_BYTES;
  if (lane == 0) {
    device half* ds = reinterpret_cast<device half*>(qx + base);
    ds[0] = half(d);
    ds[1] = half(sum);
  }
  qx[base + 4 + int(lane)] = as_type<uchar>(char(qi));
}

template <typename T>
[[kernel]] void gguf_q8_0_matvec(
    const device uchar* __restrict__ qweight [[buffer(0)]],
    const device uchar* __restrict__ qx [[buffer(1)]],
    device T* __restrict__ out [[buffer(2)]],
    constant int& rows [[buffer(3)]],
    constant int& blocks_per_row [[buffer(4)]],
    constant int& bytes_per_row [[buffer(5)]],
    constant int& padded_blocks [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int row = int(tgid.x);
  const int token = int(tgid.y);
  if (row >= rows) {
    return;
  }

  float acc = 0.0f;
  for (int block = 0; block < blocks_per_row; ++block) {
    const int w_base = row * bytes_per_row
        + block * GGUF_Q8_0_BLOCK_BYTES;
    const int x_base = (token * padded_blocks + block)
        * GGUF_Q8_1_BLOCK_BYTES;

    const device half* wd_ptr =
        reinterpret_cast<const device half*>(qweight + w_base);
    const device half* xd_ptr =
        reinterpret_cast<const device half*>(qx + x_base);
    const float wd = static_cast<float>(wd_ptr[0]);
    const float xd = static_cast<float>(xd_ptr[0]);

    const char wq = as_type<char>(qweight[w_base + 2 + int(lane)]);
    const char xq = as_type<char>(qx[x_base + 4 + int(lane)]);
    const float partial = static_cast<float>(int(wq) * int(xq));
    acc += simd_sum(partial) * wd * xd;
  }

  if (lane == 0) {
    out[token * rows + row] = static_cast<T>(acc);
  }
}

#define instantiate_gguf_q8_1_quantize(type)                         \
  template [[host_name("gguf_q8_1_quantize_" #type)]]                \
  [[kernel]] void gguf_q8_1_quantize<type>(                          \
      const device type*, device uchar*, constant int&,               \
      constant int&, uint3, uint);

#define instantiate_gguf_q8_0_matvec(type)                            \
  template [[host_name("gguf_q8_0_matvec_" #type)]]                  \
  [[kernel]] void gguf_q8_0_matvec<type>(                             \
      const device uchar*, const device uchar*, device type*,         \
      constant int&, constant int&, constant int&, constant int&,      \
      uint3, uint);

instantiate_gguf_q8_1_quantize(float);
instantiate_gguf_q8_1_quantize(half);
instantiate_gguf_q8_1_quantize(bfloat16_t);

instantiate_gguf_q8_0_matvec(float);
instantiate_gguf_q8_0_matvec(half);
instantiate_gguf_q8_0_matvec(bfloat16_t);
