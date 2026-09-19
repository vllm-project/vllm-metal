#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

// Scatter compact update rows into a slot-indexed GDN state pool, in place.
//
//   pool:    [num_slots, row_elems]   flattened; written in place
//   src:     [n, row_elems]           compact update rows
//   dst_ids: [n]                      destination slot for each update row
//
// A 2D grid walks elements on x and update rows on y. Destination slots must
// be distinct because duplicate rows would race between threadgroups.
//
// Both kernels assume an exact-size grid and therefore carry no bounds check:
// the host dispatches through MLX's dispatch_threads, which maps to Metal's
// non-uniform dispatchThreads and launches exactly grid_dims threads. Moving
// to dispatch_threadgroups -- the style the rest of paged_ops.cpp uses --
// would round the grid up to whole threadgroups and write past the end of a
// row. Add the bounds checks back first if you ever change the dispatch.
template <typename T>
[[kernel]] void gdn_state_scatter_rows(
    device T *pool [[buffer(0)]], const device T *src [[buffer(1)]],
    const device int *dst_ids [[buffer(2)]],
    device const int &row_elems [[buffer(3)]],
    constant int64_t &row_stride [[buffer(4)]],
    constant int *shape [[buffer(5)]],
    constant size_t *strides [[buffer(6)]],
    constant int &ndim [[buffer(7)]],
    constant bool &zero [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x);
  const int64_t row = int64_t(gid.y);
  const int64_t dst = int64_t(dst_ids[gid.y]) * row_stride;
  int64_t offset = i;
  if (ndim > 0) {
    int index = i;
    offset = 0;
    for (int axis = ndim - 1; axis > 0; --axis) {
      offset += (index % shape[axis]) * strides[axis];
      index /= shape[axis];
    }
  }
  pool[dst + offset] = zero ? T(0) : src[row * row_elems + i];
}

// Same scatter, four elements per thread. The host selects this kernel for
// dense rows whose lengths, row strides, and source/destination pointers meet
// vec<T, 4> alignment requirements; other supported views use the scalar kernel.
template <typename T>
[[kernel]] void gdn_state_scatter_rows_vec4(
    device T *pool [[buffer(0)]], const device T *src [[buffer(1)]],
    const device int *dst_ids [[buffer(2)]],
    device const int &row_vec4s [[buffer(3)]],
    constant int64_t &row_stride [[buffer(4)]],
    constant bool &zero [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x);
  const int64_t row = int64_t(gid.y);
  const int64_t dst = int64_t(dst_ids[gid.y]) * row_stride;
  reinterpret_cast<device vec<T, 4> *>(pool)[dst + i] =
      zero ? vec<T, 4>(0) :
      reinterpret_cast<const device vec<T, 4> *>(src)[row * row_vec4s + i];
}

#define instantiate_gdn_state_scatter_rows(type)                          \
  template [[host_name("gdn_state_scatter_rows_" #type)]] [[kernel]] void \
  gdn_state_scatter_rows<type>(                                           \
      device type *, const device type *, const device int *,             \
      device const int &, constant int64_t &, constant int *,              \
      constant size_t *, constant int &, constant bool &, uint2);          \
  template [[host_name("gdn_state_scatter_rows_vec4_" #type)]] [[kernel]] \
  void gdn_state_scatter_rows_vec4<type>(                                 \
      device type *, const device type *, const device int *,             \
      device const int &, constant int64_t &, constant bool &, uint2);

instantiate_gdn_state_scatter_rows(float);
instantiate_gdn_state_scatter_rows(bfloat16_t);
instantiate_gdn_state_scatter_rows(half);
instantiate_gdn_state_scatter_rows(uchar);
instantiate_gdn_state_scatter_rows(char);
